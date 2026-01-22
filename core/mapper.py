import numpy as np
from datatype import mappoint
from datatype.mappoint import MapPointStatus
from utils.geometry import MultiViewGeometry

class Mapper:
    def __init__(self, config, map_manager):
        self.config = config
        self.map_manager = map_manager
        self.prev_keyframe = None
        self.cur_keyframe = None

    def triangulate(self, keyframe):
        """
        对当前关键帧进行三角化
        查找当前关键帧和观测到路标点最早的关键帧之间的特征点匹配关系进行三角化
        """
        self.cur_keyframe = keyframe
        
        # 获取当前关键帧的所有特征点ID
        cur_kf_feature_ids = keyframe.get_visual_feature_ids()
        if len(cur_kf_feature_ids) == 0:
            return
        
        # 用于批量三角化的数据容器
        mappoint_ids_to_triangulate = []
        first_kf_ids = []
        bvs_first = []
        bvs_cur = []

        # 遍历当前关键帧的所有特征点
        for mp_id in cur_kf_feature_ids:
            # 在 map_manager 中查找对应的 mappoint
            # 这里虽然会同时找local和global,但理论不可能找到global的点，因为追踪是连续的
            mappoint = self.map_manager.get_map_point(mp_id)
            if mappoint is None:
                print(f"[Mapper] Mappoint {mp_id} not found in map_manager")
                continue
            
            # 只处理 CANDIDATE 状态的路标点，不重复三角化
            if mappoint.status != MapPointStatus.CANDIDATE:
                print(f"[Mapper] Mappoint {mp_id} not in CANDIDATE status")
                continue

            # 获取所有观测该路标点的关键帧ID
            observing_kf_ids = list(mappoint.get_observing_kf_ids())
            if len(observing_kf_ids) < 2:
                # print(f"[Mapper] Mappoint {mp_id} observed by {observing_kf_ids}")
                continue
            
            # 找到最早的观测关键帧（ID最小）
            first_kf_id = min(observing_kf_ids)
            # print(f"[Mapper] first_kf_id: {first_kf_id}")
            first_kf = self.map_manager.get_keyframe(first_kf_id)
            if first_kf is None:
                print(f"[Mapper] First keyframe {first_kf_id} not found")
                continue
            
            # 确保最早关键帧不是当前关键帧
            if first_kf_id == keyframe.get_id():
                print(f"[Mapper] First keyframe {first_kf_id} is the current keyframe")
                continue
            
            # 获取两个关键帧的 bearing vectors
            bv_first = first_kf.get_feature_bearing(mp_id)
            bv_cur = keyframe.get_feature_bearing(mp_id)
            
            if bv_first is None or bv_cur is None:
                print(f"[Mapper] Bearing vector not found for mappoint {mp_id}")
                continue
            
            # 收集数据用于批量三角化
            mappoint_ids_to_triangulate.append(mp_id)
            first_kf_ids.append(first_kf_id)
            bvs_first.append(bv_first)
            bvs_cur.append(bv_cur)
        
        print(f"[Mapper] mappoints ready to triangulate: {len(mappoint_ids_to_triangulate)}/{len(cur_kf_feature_ids)}")

        if len(mappoint_ids_to_triangulate) == 0:
            print(f"[Mapper] No mappoints to triangulate")
            return
        
        # 按最早关键帧分组，因为不同点的最早关键帧可能不同
        n_triangulated = 0
        groups = {}  # {first_kf_id: [indices]}
        for i, first_kf_id in enumerate(first_kf_ids):
            if first_kf_id not in groups:
                groups[first_kf_id] = []
            groups[first_kf_id].append(i)
        
        # 对每个组进行批量三角化
        T_w_c_cur = keyframe.get_T_w_c()
        print(f"[Mapper] T_w_c_cur: {T_w_c_cur}")
        
        for first_kf_id, indices in groups.items():
            first_kf = self.map_manager.get_keyframe(first_kf_id)
            if first_kf is None:
                continue
            
            T_w_c_first = first_kf.get_T_w_c()
            
            # 提取该组的数据
            group_bvs_first = np.array([bvs_first[i] for i in indices])  # (M, 3)
            group_bvs_cur = np.array([bvs_cur[i] for i in indices])  # (M, 3)
            group_mp_ids = [mappoint_ids_to_triangulate[i] for i in indices]
            
            # 批量三角化(传入单位方向向量三角化，不需要K)
            success, points_3d = MultiViewGeometry.triangulate_points(
                T_w_c_first, T_w_c_cur, group_bvs_first, group_bvs_cur
            )
            
            if not success:
                continue
            
            # 对每个三角化点进行健康检查并更新状态
            for i, mp_id in enumerate(group_mp_ids):
                point_3d = points_3d[i]
                mappoint = self.map_manager.get_map_point(mp_id)
                
                if mappoint is None:
                    continue
                
                # 健康检查
                if self.map_manager.check_mappoint_health(mp_id, candidate_position_3d=point_3d):
                    # 通过健康检查，更新为已三角化状态
                    mappoint.set_triangulated(point_3d)
                    n_triangulated += 1
                    print(f"[Mapper] Mappoint {mp_id} triangulated successfully. Position: {point_3d}")
                else:
                    print(f"[Mapper] Mappoint {mp_id} failed health check after triangulation.")
        
        print(f"[Mapper] mappoints triangulated: {n_triangulated}/{len(mappoint_ids_to_triangulate)}")
        # 更新 prev_keyframe
        self.prev_keyframe = keyframe

    def check_initialization_quality(self, visual_init_ready):
        """
        检查初始化完成后的地图点质量
        如果初始化完成但地图点数量不足，返回 True 表示需要重置
        
        Args:
            visual_init_ready: 视觉初始化是否完成
            
        Returns:
            bool: True 表示需要重置，False 表示正常
        """
        if not visual_init_ready:
            # 初始化未完成，不检查
            return False
        
        # 1. 统计已三角化的地图点数量
        triangulated_mappoints = self.map_manager.get_active_mappoints()
        num_triangulated = len(triangulated_mappoints)
        
        print(f"[Mapper] Checking initialization quality: {num_triangulated} triangulated mappoints")
        
        # 条件1: 已三角化的地图点数量少于30
        if num_triangulated < 30:
            print(f"[Mapper] Initialization quality check failed: Only {num_triangulated} triangulated mappoints (< 30)")
            return True
        
        # 2. 统计前10个关键帧的3D点数量
        active_keyframes = self.map_manager.get_active_keyframes()
        if len(active_keyframes) == 0:
            return False
        
        # 获取前10个关键帧（按ID排序）
        first_10_keyframes = sorted(active_keyframes, key=lambda kf: kf.get_id())[:10]
        
        # 统计这10个关键帧中观测到的唯一已三角化地图点数量
        unique_3d_mappoints = set()
        for kf in first_10_keyframes:
            kf_feature_ids = kf.get_visual_feature_ids()
            # 统计该关键帧中观测到的已三角化地图点
            for mp_id in kf_feature_ids:
                mappoint = self.map_manager.get_map_point(mp_id)
                if mappoint is not None and mappoint.status == MapPointStatus.TRIANGULATED:
                    unique_3d_mappoints.add(mp_id)
        
        total_3d_points = len(unique_3d_mappoints)
        print(f"[Mapper] First 10 keyframes have {total_3d_points} unique 3D mappoints")
        
        # 条件2: 前10个关键帧的3D点少于10个
        if total_3d_points < 10:
            print(f"[Mapper] Initialization quality check failed: Only {total_3d_points} unique 3D mappoints in first 10 keyframes (< 10)")
            return True
        
        print(f"[Mapper] Initialization quality check passed: {num_triangulated} mappoints, {total_3d_points} 3D points in first 10 KFs")
        return False

    def map(self, keyframe):
        pass

    def reset(self):
        """
        重置 Mapper 状态
        """
        print(f"[Mapper] Resetting mapper state")
        self.prev_keyframe = None
        self.cur_keyframe = None
        print(f"[Mapper] Reset complete")