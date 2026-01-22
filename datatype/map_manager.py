from collections import deque
from pickle import TRUE
from datatype.mappoint import MapPoint, MapPointStatus
import numpy as np
import cv2
import gtsam
import time

class MapManager:
    def __init__(self, config):
        self.config = config

        # 读取外参
        T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
        self.T_bc = np.asarray(T_bc_raw).reshape(4, 4)

        self.max_active_keyframes = self.config.get('window_size', 10)
        self.triangulation_max_depth = self.config.get('triangulation_max_depth', 100)
        self.triangulation_min_depth = self.config.get('triangulation_min_depth', 0.1)
        self.triangulation_max_reprojection_error = self.config.get('triangulation_max_reprojection_error', 3.0)
        self.optimization_max_reprojection_error = self.config.get('optimization_max_reprojection_error', 60.0)
        self.optimization_max_delete_reprojection_error = self.config.get('optimization_max_delete_reprojection_error', 1000.0)
        self.min_parallax_angle_deg = self.config.get('min_parallax_angle_deg', 5.0)

        self.cam_intrinsics = np.asarray(self.config.get('cam_intrinsics')).reshape(3, 3)

        # 1. 当前活跃窗口 (参与 Local BA)
        self.active_keyframes = {}   # {kf_id: KeyFrame}
        self.local_mappoints = {}    # {mp_id: MapPoint} - 窗口内可见的地图点

        # 2. 历史/全局地图 (不参与 Local BA，用于回环或纯存储)
        self.global_keyframes = {}   # {kf_id: KeyFrame}
        self.global_mappoints = {}   # {mp_id: MapPoint}

    def add_keyframe(self, kf):
        """
        添加新关键帧进入滑动窗口，并处理路标点关联
        """
        # 1. 加入活跃窗口
        self.active_keyframes[kf.get_id()] = kf

        # 2. 处理该帧观测到的特征点
        # 遍历当前帧的所有特征点
        for mp_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
            
            # 情况 A: 这是一个已知的活跃地图点
            if mp_id in self.local_mappoints:
                mp = self.local_mappoints[mp_id]
                mp.add_observation(kf.get_id(), pt_2d)
            
            # 情况 B: 这是一个已知的全局地图点 (这种情况在回环或重定位时发生，暂不展开，视作本地化)
            elif mp_id in self.global_mappoints:
                # TODO: 策略：如果老点被新帧看到，通常应该将其拉回 local map 参与优化
                pass 
            
            # 情况 C: 新产生的地图点 (Candidate)
            else:
                new_mp = MapPoint(mp_id, kf.get_id(), pt_2d)
                self.local_mappoints[mp_id] = new_mp
        
        # 3. 检查窗口大小，执行边缘化策略
        if len(self.active_keyframes) > self.max_active_keyframes:
            self.manage_sliding_window()

    def manage_sliding_window(self):
        """
        核心逻辑：滑动窗口管理
        将最老的关键帧移入 Global，并检查其观测的地图点是否需要随之移入 Global
        """
        # 1. 找到最老的活跃关键帧
        oldest_kf = min(self.active_keyframes.values(), key=lambda kf: kf.get_timestamp())
        oldest_kf_id = oldest_kf.get_id()

        print(f"[MapManager] Window full. Archiving KF {oldest_kf_id} to Global.")

        # 2. 将 KF 移入 Global 列表
        self.global_keyframes[oldest_kf_id] = oldest_kf
        del self.active_keyframes[oldest_kf_id]

        # 3. 检查该 KF 观测到的所有 MapPoints
        # 我们需要判断这些点是留在 Local 还是移入 Global
        # 获取该帧观测到的所有 MP ID (假设 KF 有这个方法获取所有特征ID)
        observed_mp_ids = oldest_kf.get_visual_feature_ids()
        
        mps_to_move_to_global = []
        mps_to_delete = []

        for mp_id in observed_mp_ids:
            if mp_id not in self.local_mappoints:
                continue
            
            mp = self.local_mappoints[mp_id]
            
            # --- 关键判断逻辑 ---
            # 检查该点是否还被窗口内 **其他** 活跃关键帧观测到
            is_still_active = False
            
            # 遍历该点的所有观测记录 {kf_id: obs}
            # mp.observations 存储了所有观测到该点的 kf_id
            for obs_kf_id in mp.get_observing_kf_ids():
                if obs_kf_id in self.active_keyframes:
                    is_still_active = True
                    break
            
            if is_still_active:
                # A. 仍然活跃：保留在 local_mappoints
                continue
            else:
                # B. 不再活跃：移出 local
                # 没有任何活跃帧能看到它了，追踪断了
                # 只有三角化成功的点才值得存入全局地图
                if mp.status == MapPointStatus.TRIANGULATED:
                    mps_to_move_to_global.append(mp_id)
                else:
                    # 还没三角化就滑出了窗口，说明初始化不好或那是外点，直接删除
                    mps_to_delete.append(mp_id)

        # 执行移动操作
        for mp_id in mps_to_move_to_global:
            self.global_mappoints[mp_id] = self.local_mappoints[mp_id]
            del self.local_mappoints[mp_id]
        
        # 执行删除操作
        for mp_id in mps_to_delete:
            del self.local_mappoints[mp_id]

        print(f"[MapManager] Moved {len(mps_to_move_to_global)} MPs to Global, Deleted {len(mps_to_delete)} candidates.")

    # --- 数据访问接口 (Getter) ---

    def get_keyframe(self, kf_id):
        """优先查找 active，其次查找 global"""
        if kf_id in self.active_keyframes:
            return self.active_keyframes[kf_id]
        return self.global_keyframes.get(kf_id)

    def get_map_point(self, mp_id):
        """优先查找 local，其次查找 global"""
        if mp_id in self.local_mappoints:
            return self.local_mappoints[mp_id]
        return self.global_mappoints.get(mp_id)

    def get_active_keyframes(self):
        # 返回列表
        return sorted(self.active_keyframes.values(), key=lambda kf: kf.get_id())

    def get_active_mappoints(self):
        # 仅返回 local 中已三角化的点用于前端追踪/优化
        return {mp.id: mp.position_3d for mp in self.local_mappoints.values() if mp.status == MapPointStatus.TRIANGULATED}

    def get_global_mappoints(self):
        # 仅返回 global 中已三角化的点用于前端追踪/优化(理论上这里应该都是三角化的点)
        return {mp.id: mp.position_3d for mp in self.global_mappoints.values() if mp.status == MapPointStatus.TRIANGULATED}

    def remove_observation_both_sides(self, mp_id, kf_id):
        """
        安全移除观测：同时更新 MapPoint 和 KeyFrame 的记录
        """
        # 1. 从 MapPoint 中移除对 KeyFrame 的引用
        mp = self.local_mappoints.get(mp_id)
        if mp:
            mp.remove_observation(kf_id)
        
        # 2. 从 KeyFrame 中移除对 MapPoint 的引用
        kf = self.get_keyframe(kf_id)
        if kf:
            kf.remove_features_by_ids([mp_id])

    # --- 辅助功能 ---
    def reset(self):
        print(f"[MapManager] Resetting all maps.")
        self.active_keyframes.clear()
        self.local_mappoints.clear()
        self.global_keyframes.clear()
        self.global_mappoints.clear()

    def check_mappoint_health(self, mappoint_id, candidate_position_3d=None):
        mp = self.local_mappoints.get(mappoint_id)
        if not mp:
            return False

        # 确定使用哪个3D位置
        mappoint_pos = None
        is_new_candidate = False
        
        # 优先使用传入的候选位置（通常是刚三角化完，还没写进mp.position_3d）
        if candidate_position_3d is not None:
            mappoint_pos = candidate_position_3d
            is_new_candidate = True
        elif mp.status == MapPointStatus.TRIANGULATED and mp.position_3d is not None:
            mappoint_pos = mp.position_3d # 这种情况应该不可能发生
        else:
            return False

        # 获取活跃窗口中的观测帧
        observing_kf_ids = mp.get_observing_kf_ids()
        witness_kfs = [self.active_keyframes[kf_id] for kf_id in observing_kf_ids if kf_id in self.active_keyframes]
        witness_kfs.sort(key=lambda x: x.get_id())

        # 至少需要两个观测帧
        min_obs = 2
        if len(witness_kfs) < min_obs:
            print(f"[MapManager] Mappoint {mp.id} rejected. Not enough obs ({len(witness_kfs)} < {min_obs})")
            return False
        
        # 获取两个关键帧的位姿
        first_kf = witness_kfs[0]
        last_kf = witness_kfs[-1] # 通常是当前帧
        T_c1_w = first_kf.get_T_c_w()
        T_c2_w = last_kf.get_T_c_w()
        T_w_c2 = last_kf.get_T_w_c()

        # 旋转到上一帧进行视差计算
        R_c1_w = T_c1_w[:3, :3]
        R_w_c2 = T_w_c2[:3, :3]
        R_c1_c2 = R_c1_w @ R_w_c2
        unpx1 = first_kf.get_feature_undistorted_position(mp.id)
        bv2 = last_kf.get_feature_bearing(mp.id)
        rot_bv = R_c1_c2 @ bv2
        rot_px = first_kf.camera.project_cam_to_image(rot_bv)
        if rot_px is None:
            return False # 投影到了相机背后
        parallax = np.linalg.norm(rot_px - unpx1)

        # 计算两个关键帧的深度 (相对于最后一帧，通常更关键)
        p_c1 = T_c1_w[:3, :3] @ mappoint_pos + T_c1_w[:3, 3]
        p_c2 = T_c2_w[:3, :3] @ mappoint_pos + T_c2_w[:3, 3]
        z_depth_first = p_c1[2]
        z_depth_last = p_c2[2]

        if z_depth_first < self.triangulation_min_depth or z_depth_last < self.triangulation_min_depth:
            print(f"[Health] Negative/Low depth. First: {z_depth_first:.2f}, Last: {z_depth_last:.2f}")
            if (parallax > 20.0) : 
                # TODO: 视差大，同时深度小，说明观测错误，移除最后一帧观测，同时在frame上移除
                # 视差小，有可能是三角化不稳定，保留观测
                mp.remove_observation(last_kf.get_id())
                self.remove_observation_both_sides(mp.id, last_kf.get_id())
            return False

        if z_depth_first > self.triangulation_max_depth or z_depth_last > self.triangulation_max_depth:
            print(f"[Health] Depth too large. First: {z_depth_first:.2f}, Last: {z_depth_last:.2f}")
            return False

        # 重投影误差检查
        check_list = [(first_kf, p_c1), (last_kf, p_c2)]
        for kf, p_c in check_list:
            uv_proj = kf.camera.project_cam_to_image(p_c) # 投影到像素平面（不考虑畸变）
            obs_uv = kf.get_feature_undistorted_position(mp.id) # 获取实际观测

            if uv_proj is None or obs_uv is None:
                return False

            error = np.linalg.norm(uv_proj - obs_uv)
            # 任意一个重投影误差过大，都认为该点不合格
            if error > self.triangulation_max_reprojection_error:
                print(f"[Health] Reprojection error too large. Error: {error:.2f}px")
                return False

        return True

    def check_mappoint_health_after_optimization(self, mappoint_id):
        mp = self.local_mappoints.get(mappoint_id)
        # 必须是已三角化的点才有3D位置
        if not mp or mp.position_3d is None:
            return False, True, True

        observing_kf_ids = [kf_id for kf_id in mp.get_observing_kf_ids() if kf_id in self.active_keyframes]
        
        # 观测帧数太少，被先验因子约束无法检查，直接返回True
        if len(observing_kf_ids) < 2:
            return True, True, True 

        # 检查全部KF
        kfs_to_check = [self.active_keyframes[kf_id] for kf_id in observing_kf_ids]

        # # 优化：只检查ID最小和最大的两个观测帧
        # first_kf_id = min(observing_kf_ids)
        # last_kf_id = max(observing_kf_ids)
        
        # # 将要检查的关键帧限制在这两个极端
        # kfs_to_check = [self.keyframes[first_kf_id]]
        # if first_kf_id != last_kf_id:
        #     kfs_to_check.append(self.keyframes[last_kf_id])

        reproj_error_total = 0.0
        for kf in kfs_to_check:
            T_w_c = kf.get_global_pose()
            T_c_w = np.linalg.inv(T_w_c)

            point_in_cam_homo = T_c_w @ np.append(mp.position_3d, 1.0)
            
            # 检查深度是否为正且在合理范围内
            depth = point_in_cam_homo[2]
            if depth <= self.min_depth or depth > self.max_depth:
                if depth < 0.0:
                    print(f"[MapManager] Mappoint {mp.id} has negative depth in KF {kf.get_id()}. Depth: {depth:.4f}m")
                    return False, False, True
                print(f"【Optimization Health Check】: mappoint {mp.id} failed depth check in KF {kf.get_id()}. Depth: {depth:.4f}m")
                return False, True, True

            # 检查重投影误差
            rvec, _ = cv2.Rodrigues(T_c_w[:3,:3])
            tvec = T_c_w[:3,3]
            reprojected_pt, _ = cv2.projectPoints(mp.position_3d.reshape(1,1,3), rvec, tvec, self.cam_intrinsics, None)
            reproj_error = np.linalg.norm(reprojected_pt.flatten() - mp.observations[kf.get_id()])
            reproj_error_total += reproj_error

        reproj_error_avg = reproj_error_total / len(kfs_to_check)
        if reproj_error_avg > self.optimization_max_reprojection_error:
            if reproj_error_avg > self.optimization_max_delete_reprojection_error:
                print(f"【Optimization Health Check】: mappoint {mp.id} failed reprojection is too large in KF {kf.get_id()}. Error: {reproj_error_avg:.2f}px")
                return False, True, False
            print(f"【Optimization Health Check】: mappoint {mp.id} failed reprojection in KF {kf.get_id()}. Error: {reproj_error_avg:.2f}px")
            return False, True, True

        return True, True, True