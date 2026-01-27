from enum import Enum
import numpy as np
import cv2

class MapPointStatus(Enum):
    CANDIDATE = 0  # 候选点，尚未被三角化
    TRIANGULATED = 1 # 已三角化，有3D位置

class MapPoint:
    def __init__(self, landmark_id, first_kf_id, first_pt_2d, descriptor=None):
        self.id = landmark_id
        
        # 初始状态为候选点，还没有3D位置
        self.status = MapPointStatus.CANDIDATE
        self.position_3d = None
        
        # 记录所有的观测 {kf_id: pt_2d_coords}
        self.observations = {first_kf_id: first_pt_2d}

        # 描述子管理
        self.descriptors = {}       # {kf_id: descriptor_np_array} 存储所有观测的描述子
        self.descriptor_dists = {}  # {kf_id: sum_distances} 存储每个描述子与其他所有描述子的距离之和
        self.best_descriptor = None # 最具代表性的描述子 (Median Descriptor)

        # 如果传入了初始描述子，则进行初始化
        if descriptor is not None:
            self.add_descriptor(first_kf_id, descriptor)

    def add_observation(self, kf_id, pt_2d, descriptor=None):
        self.observations[kf_id] = pt_2d
        if descriptor is not None:
            self.add_descriptor(kf_id, descriptor)

    def remove_observation(self, kf_id):
        if kf_id in self.observations:
            del self.observations[kf_id]

        if kf_id in self.descriptors:
            self.remove_descriptor(kf_id)

    def set_triangulated(self, position_3d):
        self.position_3d = position_3d
        self.status = MapPointStatus.TRIANGULATED

    def get_observation_count(self):
        return len(self.observations)

    def get_observing_kf_ids(self):
        return self.observations.keys()

    def get_observation(self, kf_id):
        return self.observations[kf_id]

    def get_point(self):
        return self.position_3d

    def is_ready_for_triangulation(self, keyframe_window, min_parallax):
        # 必须是候选点，且至少有2个观测
        if self.status != MapPointStatus.CANDIDATE or self.get_observation_count() < 2:
            return False, None, None

        # 找到第一个和最后一个观测它的、且仍在滑动窗口内的关键帧
        obs_ids = list(self.observations.keys())
        first_kf_id = min(obs_ids)
        last_kf_id = max(obs_ids)

        first_kf = next((kf for kf in keyframe_window if kf.get_id() == first_kf_id), None)
        last_kf = next((kf for kf in keyframe_window if kf.get_id() == last_kf_id), None)

        if first_kf is None or last_kf is None or first_kf_id == last_kf_id:
            return False, None, None

        # 检查视差
        pt1 = self.observations[first_kf_id]
        pt2 = self.observations[last_kf_id]
        parallax = np.linalg.norm(pt1 - pt2)

        if parallax > min_parallax:
            return True, first_kf, last_kf
        else:
            # print(f"[Trace l{self.id}]: FAILED triangulation check. Parallax: {parallax:.2f}px")
            return False, None, None

    # ==========================================================
    # 描述子增量维护
    # ==========================================================
    def add_descriptor(self, kf_id, desc):
        """
        添加描述子，并更新最具代表性的描述子 (best_descriptor)
        采用增量更新策略，避免每次全量计算 O(N^2)
        """
        if kf_id in self.descriptors:
            return

        self.descriptors[kf_id] = desc
        self.descriptor_dists[kf_id] = 0.0
        
        # 如果是第一个描述子，直接设为最佳
        if len(self.descriptors) == 1:
            self.best_descriptor = desc
            return

        current_desc_dist_sum = 0.0
        
        # 遍历现有的其他描述子，计算汉明距离
        for exist_kf_id, exist_desc in self.descriptors.items():
            if exist_kf_id == kf_id:
                continue
                
            # 计算汉明距离 (OpenCV Norm)
            dist = cv2.norm(desc, exist_desc, cv2.NORM_HAMMING)
            
            # 1. 更新旧描述子的距离和
            self.descriptor_dists[exist_kf_id] += dist
            
            # 2. 累加新描述子的距离和
            current_desc_dist_sum += dist

        self.descriptor_dists[kf_id] = current_desc_dist_sum

        # 3. 寻找新的最佳描述子 (距离之和最小的那个)
        # 优化：通常只需要比较"当前的最佳"和"新来的"谁更好，
        # 但为了严谨，这里还是遍历一次 descriptor_dists 找最小值 (O(N))
        min_dist_sum = float('inf')
        best_kf_id = -1
        
        for kid, dist_sum in self.descriptor_dists.items():
            if dist_sum < min_dist_sum:
                min_dist_sum = dist_sum
                best_kf_id = kid
        
        if best_kf_id != -1:
            self.best_descriptor = self.descriptors[best_kf_id]

    def remove_descriptor(self, kf_id):
        """
        移除描述子，并重新计算最具代表性的描述子
        """
        if kf_id not in self.descriptors:
            return

        removed_desc = self.descriptors[kf_id]
        del self.descriptors[kf_id]
        del self.descriptor_dists[kf_id]

        if len(self.descriptors) == 0:
            self.best_descriptor = None
            return

        # 1. 更新剩余描述子的距离和 (减去与被删除描述子的距离)
        for exist_kf_id, exist_desc in self.descriptors.items():
            dist = cv2.norm(removed_desc, exist_desc, cv2.NORM_HAMMING)
            self.descriptor_dists[exist_kf_id] -= dist

        # 2. 重新寻找最佳描述子
        min_dist_sum = float('inf')
        best_kf_id = -1
        
        for kid, dist_sum in self.descriptor_dists.items():
            if dist_sum < min_dist_sum:
                min_dist_sum = dist_sum
                best_kf_id = kid

        if best_kf_id != -1:
            self.best_descriptor = self.descriptors[best_kf_id]

    def compute_min_desc_dist(self, other_mp):
        """
        计算两个 MapPoint 之间的最小描述子距离。
        注意:这里做的是全量匹配 (All vs All)，非常严格。
        """
        if not self.descriptors or not other_mp.descriptors:
            return float('inf')

        min_dist = float('inf')

        # 遍历自己的所有描述子
        for my_desc in self.descriptors.values():
            # 遍历对方的所有描述子
            for other_desc in other_mp.descriptors.values():
                dist = cv2.norm(my_desc, other_desc, cv2.NORM_HAMMING)
                if dist < min_dist:
                    min_dist = dist
        
        return min_dist