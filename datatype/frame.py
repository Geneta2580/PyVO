import numpy as np
import cv2
import gtsam
import threading
import math
from typing import List, Dict, Optional, Tuple, Set

# 假設你已經定義了 CameraCalibration 類
# from camera_calibration import CameraCalibration

class Keypoint:
    """
    單目版本 Keypoint 結構
    """
    def __init__(self, lmid: int = -1):
        self.lmid = lmid           # Landmark ID (地圖點ID)
        self.px = np.zeros(2)      # 原始像素坐標 [x, y]
        self.unpx = np.zeros(2)    # 去畸變後像素坐標 [x, y]
        self.bv = np.zeros(3)      # 歸一化相機坐標系下的方向向量 (Bearing Vector)
        self.desc = None           # 描述子 (numpy array)
        self.scale = 0             # 金字塔層級
        self.angle = 0.0           # 特征點角度

        self.is3d = False          # 是否已三角化為3D點
        self.is_retracked = False  # 是否是上一幀跟蹤過來的點 (Retracked)

class Frame:
    """
    OV2SLAM Frame 類的 Python 復刻 (僅單目 Monocular)
    """
    def __init__(self, pcalib, ncellsize: int = 35, frame_id: int = -1, timestamp: float = 0.0):
        self.id = frame_id
        self.kfid = 0
        self.img_time = timestamp

        # 統計計數
        self.nbkps = 0
        self.nb2dkps = 0
        self.nb3dkps = 0

        # 相機參數 (僅保留左目/主相機)
        self.pcalib = pcalib

        # 位姿 (GTSAM Pose3)
        # Twc: World -> Camera (相機在世界坐標系下的位姿)
        # Tcw: Camera -> World (世界點投影到相機坐標系的變換)
        self.T_w_c = gtsam.Pose3()
        self.T_c_w = gtsam.Pose3()

        # 存儲關鍵點: {landmark_id: KeypointObj}
        self.map_kps: Dict[int, Keypoint] = {}

        # 共視關鍵幀: {kf_id: weight}
        self.map_covkfs: Dict[int, int] = {}
        self.set_local_mapids: Set[int] = set()

        # 图像网格
        self.ncellsize = ncellsize
        self.nbwcells = int(math.ceil(pcalib.img_w / ncellsize))
        self.nbhcells = int(math.ceil(pcalib.img_h / ncellsize))
        self.ngridcells = self.nbwcells * self.nbhcells
        self.noccupcells = 0

        # Grid: list of lists containing landmark IDs
        self.vgridkps: List[List[int]] = [[] for _ in range(self.ngridcells)]

        # 互斥鎖 (確保線程安全)
        self.kps_mutex = threading.Lock()
        self.grid_mutex = threading.Lock()
        self.pose_mutex = threading.Lock()
        self.cokfs_mutex = threading.Lock()

    # 更新帧参数
    def update_frame(self, frame_id: int, time: float):
        self.id = frame_id
        self.img_time = time

    # =========================================================================
    # Keypoint Accessors (獲取器)
    # =========================================================================

    def get_keypoints(self) -> List[Keypoint]:
        with self.kps_mutex:
            return list(self.map_kps.values())

    def get_keypoints_2d(self) -> List[Keypoint]:
        with self.kps_mutex:
            return [kp for kp in self.map_kps.values() if not kp.is3d]

    def get_keypoints_3d(self) -> List[Keypoint]:
        with self.kps_mutex:
            return [kp for kp in self.map_kps.values() if kp.is3d]

    def get_keypoints_px(self) -> List[np.ndarray]:
        with self.kps_mutex:
            return [kp.px for kp in self.map_kps.values()]

    def get_keypoints_unpx(self) -> List[np.ndarray]:
        with self.kps_mutex:
            return [kp.unpx for kp in self.map_kps.values()]

    def get_keypoints_id(self) -> List[int]:
        with self.kps_mutex:
            return list(self.map_kps.keys())

    def get_keypoint_by_id(self, lmid: int) -> Optional[Keypoint]:
        with self.kps_mutex:
            return self.map_kps.get(lmid)

    def get_keypoints_by_ids(self, lmids: List[int]) -> List[Keypoint]:
        with self.kps_mutex:
            res = []
            for lmid in lmids:
                if lmid in self.map_kps:
                    res.append(self.map_kps[lmid])
            return res

    def get_keypoints_desc(self) -> List[np.ndarray]:
        with self.kps_mutex:
            return [kp.desc for kp in self.map_kps.values()]

    # =========================================================================
    # Keypoint Modifiers (增刪改)
    # =========================================================================

    def _compute_keypoint(self, pt: np.ndarray, kp: Keypoint):
        """內部函數：根據像素坐標計算去畸變坐標和 Bearing Vector"""
        kp.px = pt
        # 調用 CameraCalibration 的 undistort
        kp.unpx = self.pcalib.undistort_image_point(pt)

        # 計算 Bearing Vector: K_inv * [u, v, 1]
        hunpx = np.array([kp.unpx[0], kp.unpx[1], 1.0])
        bv = self.pcalib.iK @ hunpx
        kp.bv = bv / np.linalg.norm(bv) # 歸一化

    def compute_keypoint_obj(self, pt: np.ndarray, lmid: int) -> Keypoint:
        kp = Keypoint(lmid)
        self._compute_keypoint(pt, kp)
        return kp

    def add_keypoint(self, kp: Keypoint):
        with self.kps_mutex:
            if kp.lmid in self.map_kps:
                print(f"[Frame] Warning: Keypoint {kp.lmid} already exists!")
                return

            self.map_kps[kp.lmid] = kp
            self._add_keypoint_to_grid(kp)

            self.nbkps += 1
            if kp.is3d:
                self.nb3dkps += 1
            else:
                self.nb2dkps += 1

    def add_keypoint_from_pt(self, pt: np.ndarray, lmid: int, desc: Optional[np.ndarray] = None,
                             scale: int = 0, angle: float = 0.0):
        kp = self.compute_keypoint_obj(pt, lmid)
        kp.desc = desc
        kp.scale = scale
        kp.angle = angle
        self.add_keypoint(kp)

    def update_keypoint(self, lmid: int, pt: np.ndarray):
        with self.kps_mutex:
            if lmid not in self.map_kps:
                return

            old_kp = self.map_kps[lmid]

            # 記錄舊位置用於 Grid 更新
            old_px = old_kp.px.copy()

            # 更新數值
            self._compute_keypoint(pt, old_kp)

            # Grid 更新
            self._update_keypoint_in_grid(old_px, old_kp)

    def update_keypoint_id(self, prev_lmid: int, new_lmid: int, is3d: bool) -> bool:
        """更新關鍵點 ID (通常用於從局部地圖匹配到全局地圖點時)"""
        with self.kps_mutex:
            if new_lmid in self.map_kps:
                return False
            if prev_lmid not in self.map_kps:
                return False

            kp = self.map_kps[prev_lmid]

            # 1. 從 Grid 移除舊引用
            self._remove_keypoint_from_grid(kp)
            # 2. 從 Map 移除舊 Key
            del self.map_kps[prev_lmid]

            # 3. 修改屬性
            kp.lmid = new_lmid
            kp.is_retracked = True
            kp.is3d = is3d

            # 4. 重新添加
            self.map_kps[new_lmid] = kp
            self._add_keypoint_to_grid(kp)

            return True

    def remove_keypoint_by_id(self, lmid: int):
        with self.kps_mutex:
            if lmid not in self.map_kps:
                return

            kp = self.map_kps[lmid]
            self._remove_keypoint_from_grid(kp)

            if kp.is3d:
                self.nb3dkps -= 1
            else:
                self.nb2dkps -= 1

            self.nbkps -= 1
            del self.map_kps[lmid]

    def turn_keypoint_3d(self, lmid: int):
        """將 2D 關鍵點標記為 3D (通常在三角化成功後)"""
        with self.kps_mutex:
            if lmid in self.map_kps:
                kp = self.map_kps[lmid]
                if not kp.is3d:
                    kp.is3d = True
                    self.nb3dkps += 1
                    self.nb2dkps -= 1

    def is_observing_kp(self, lmid: int) -> bool:
        with self.kps_mutex:
            return lmid in self.map_kps

    # =========================================================================
    # Grid Logic (空間哈希加速)
    # =========================================================================

    def _get_keypoint_cell_idx(self, pt: np.ndarray) -> int:
        r = int(math.floor(pt[1] / self.ncellsize))
        c = int(math.floor(pt[0] / self.ncellsize))
        return r * self.nbwcells + c

    def _add_keypoint_to_grid(self, kp: Keypoint):
        with self.grid_mutex:
            idx = self._get_keypoint_cell_idx(kp.px)
            if 0 <= idx < len(self.vgridkps):
                if not self.vgridkps[idx]: # is empty
                    self.noccupcells += 1
                self.vgridkps[idx].append(kp.lmid)

    def _remove_keypoint_from_grid(self, kp: Keypoint):
        with self.grid_mutex:
            idx = self._get_keypoint_cell_idx(kp.px)
            if 0 <= idx < len(self.vgridkps):
                cell = self.vgridkps[idx]
                if kp.lmid in cell:
                    cell.remove(kp.lmid)
                    if not cell:
                        self.noccupcells -= 1

    def _update_keypoint_in_grid(self, old_px: np.ndarray, kp: Keypoint):
        old_idx = self._get_keypoint_cell_idx(old_px)
        new_idx = self._get_keypoint_cell_idx(kp.px)

        if old_idx == new_idx:
            return

        with self.grid_mutex:
            # Remove
            if 0 <= old_idx < len(self.vgridkps):
                if kp.lmid in self.vgridkps[old_idx]:
                    self.vgridkps[old_idx].remove(kp.lmid)
                    if not self.vgridkps[old_idx]:
                        self.noccupcells -= 1

            # Add
            if 0 <= new_idx < len(self.vgridkps):
                if not self.vgridkps[new_idx]:
                    self.noccupcells += 1
                self.vgridkps[new_idx].append(kp.lmid)

    def get_keypoints_in_cell(self, pt: np.ndarray) -> List[Keypoint]:
        # 獲取特定單元格內的關鍵點
        with self.grid_mutex:
            idx = self._get_keypoint_cell_idx(pt)
            if idx < 0 or idx >= len(self.vgridkps):
                return []

            lmids = self.vgridkps[idx]

        return self.get_keypoints_by_ids(lmids) # 內部加了 kps_mutex 鎖

    def get_surrounding_keypoints(self, pt: np.ndarray, radius_cells: int = 1) -> List[Keypoint]:
        """
        獲取點 pt 周圍的所有關鍵點 (用於特徵匹配加速)
        """
        rkp = int(math.floor(pt[1] / self.ncellsize))
        ckp = int(math.floor(pt[0] / self.ncellsize))

        found_kps = []

        with self.grid_mutex, self.kps_mutex:
            for r in range(rkp - radius_cells, rkp + radius_cells + 1):
                for c in range(ckp - radius_cells, ckp + radius_cells + 1):
                    idx = r * self.nbwcells + c

                    if r < 0 or c < 0 or idx >= len(self.vgridkps):
                        continue

                    for lmid in self.vgridkps[idx]:
                        if lmid in self.map_kps:
                            found_kps.append(self.map_kps[lmid])

        return found_kps

    # =========================================================================
    # Pose & Geometry (位姿與幾何)
    # =========================================================================

    def get_T_c_w(self) -> gtsam.Pose3:
        with self.pose_mutex:
            return self.T_c_w

    def get_T_w_c(self) -> gtsam.Pose3:
        with self.pose_mutex:
            return self.T_w_c

    def set_T_w_c(self, T_w_c: gtsam.Pose3):
        with self.pose_mutex:
            self.T_w_c = T_w_c
            self.T_c_w = T_w_c.inverse()

    def set_T_c_w(self, T_c_w: gtsam.Pose3):
        with self.pose_mutex:
            self.T_c_w = T_c_w
            self.T_w_c = T_c_w.inverse()

    def proj_world_to_cam(self, pt_world: np.ndarray) -> np.ndarray:
        # P_c = T_cw * P_w
        with self.pose_mutex:
            return self.T_c_w.transformFrom(pt_world)

    def proj_cam_to_world(self, pt_cam: np.ndarray) -> np.ndarray:
        # P_w = T_wc * P_c
        with self.pose_mutex:
            return self.T_w_c.transformFrom(pt_cam)

    def proj_world_to_image(self, pt_world: np.ndarray) -> np.ndarray:
        pt_cam = self.proj_world_to_cam(pt_world)
        return self.pcalib.project_cam_to_image(pt_cam)

    def is_in_image(self, pt: np.ndarray) -> bool:
        return 0 <= pt[0] < self.pcalib.img_w and 0 <= pt[1] < self.pcalib.img_h

    # =========================================================================
    # Covisibility Graph (共視圖)
    # =========================================================================

    def add_covisible_kf(self, kf_id: int):
        if kf_id == self.kfid: return
        with self.cokfs_mutex:
            self.map_covkfs[kf_id] = self.map_covkfs.get(kf_id, 0) + 1

    def remove_covisible_kf(self, kf_id: int):
        with self.cokfs_mutex:
            if kf_id in self.map_covkfs:
                del self.map_covkfs[kf_id]

    def decrease_covisible_kf(self, kf_id: int):
        with self.cokfs_mutex:
            if kf_id in self.map_covkfs:
                self.map_covkfs[kf_id] -= 1
                if self.map_covkfs[kf_id] <= 0:
                    del self.map_covkfs[kf_id]

    def reset(self):
        self.id = -1
        self.kfid = 0
        self.img_time = 0.0

        with self.kps_mutex, self.grid_mutex, self.cokfs_mutex:
            self.map_kps.clear()
            self.vgridkps = [[] for _ in range(self.ngridcells)]
            self.nbkps = 0
            self.nb2dkps = 0
            self.nb3dkps = 0
            self.noccupcells = 0
            self.map_covkfs.clear()
            self.set_local_mapids.clear()

        with self.pose_mutex:
            self.T_w_c = gtsam.Pose3()
            self.T_c_w = gtsam.Pose3()

    def display_info(self):
        print("\n************************************")
        print(f"Frame #{self.id} (KF #{self.kfid}) info:")
        print(f"> Nb kps all (2d / 3d) : {self.nbkps} ({self.nb2dkps} / {self.nb3dkps})")
        print(f"> Nb covisible kfs : {len(self.map_covkfs)}")
        with self.pose_mutex:
            # 簡單打印平移部分
            print(f" twc : {self.T_w_c.translation()}")
        print("************************************\n")