from datatype.mappoint import MapPoint, MapPointStatus
import numpy as np
import cv2
import gtsam
import time
import threading

class MapManager:
    def __init__(self, config):
        self.config = config

        # 线程锁
        # 必须使用 RLock (可重入锁)，因为有些内部函数可能会相互调用 (例如 manage_sliding_window 调用 get_keyframe)
        self.map_lock = threading.RLock()

        # 读取外参
        T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
        self.T_bc = np.asarray(T_bc_raw).reshape(4, 4)

        # 三角化参数
        self.triangulation_max_depth = self.config.get('triangulation_max_depth', 100)
        self.triangulation_min_depth = self.config.get('triangulation_min_depth', 0.1)
        self.triangulation_max_reprojection_error = self.config.get('triangulation_max_reprojection_error', 3.0)

        # 优化参数
        self.optimization_max_reprojection_error = self.config.get('optimization_max_reprojection_error', 60.0)
        self.optimization_max_delete_reprojection_error = self.config.get('optimization_max_delete_reprojection_error', 1000.0)
        self.min_parallax_angle_deg = self.config.get('min_parallax_angle_deg', 5.0)

        # 相机内参
        self.cam_intrinsics = np.asarray(self.config.get('cam_intrinsics')).reshape(3, 3)

        self.keyframes = {}   # {kf_id: KeyFrame} - 存所有活着的关键帧
        self.mappoints = {}   # {mp_id: MapPoint} - 存所有活着的地图点

    def add_keyframe(self, kf):
        """
        添加新关键帧进入滑动窗口，并处理路标点关联
        """
        with self.map_lock:
            # 1. 存入大地图
            self.keyframes[kf.get_id()] = kf
            
            # 2. 处理地图点
            for mp_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
                if mp_id in self.mappoints:
                    # 老点：增加观测
                    mp = self.mappoints[mp_id]
                    mp.add_observation(kf.get_id(), pt_2d)
                else:
                    # 新点：创建，这里是CANDIDATE状态
                    new_mp = MapPoint(mp_id, kf.get_id(), pt_2d)
                    self.mappoints[mp_id] = new_mp
            
            # TODO：关键帧剔除策略？

    def remove_observation_both_sides(self, mp_id, kf_id):
        """
        安全移除观测：同时更新 MapPoint 和 KeyFrame 的记录
        """
        with self.map_lock:
            # 1. 从 MapPoint 中移除对 KeyFrame 的引用
            mp = self.mappoints.get(mp_id)
            if mp:
                mp.remove_observation(kf_id)
            
            # 2. 从 KeyFrame 中移除对 MapPoint 的引用
            kf = self.get_keyframe(kf_id)
            if kf:
                kf.remove_features_by_ids([mp_id])

    def update_map_from_optimization(self, optimized_poses, optimized_points, outlier_observations=None):
        """
        线程安全地更新地图数据，并处理外点
        Args:
            optimized_poses: {kf_id: mat4x4}
            optimized_points: {mp_id: vec3}
            outlier_observations: list of (kf_id, mp_id) tuples
        """
        with self.map_lock:
            # 1. 剔除外点 (在更新位置之前做，防止错误观测拉偏后续逻辑)
            if outlier_observations:
                for kf_id, mp_id in outlier_observations:
                    # 同时从两端移除引用
                    self._remove_observation_internal(mp_id, kf_id)

            # 2. 更新位姿
            for kf_id, pose_matrix in optimized_poses.items():
                if kf_id in self.active_keyframes:
                    self.active_keyframes[kf_id].set_T_w_c(pose_matrix)
            
            # 3. 更新点
            for mp_id, pos_3d in optimized_points.items():
                if mp_id in self.mappoints:
                    self.local_mappoints[mp_id].set_triangulated(pos_3d)
            
            # 4. Map Point Culling (清理质量差的点)
            # TODO: 这里可以衔接check_mappoint_health_after_optimization
            # 策略：如果一个点在优化后（剔除外点后），观测数量 < 2，或者深度无效，则删除该点
            deleted_mps = 0
            
            # 只需要检查参与了优化的点 (optimized_points 的 key)
            for mp_id in optimized_points.keys():
                if mp_id not in self.mappoints:
                    continue
                
                mp = self.mappoints[mp_id]
                
                # 检查观测数量
                # 注意：mp.get_observing_kf_ids() 返回的是所有观测（包含 Global 的）
                n_obs = len(mp.get_observing_kf_ids())
                
                if n_obs < 2:
                    # 观测太少，认为是不可靠的点，删除
                    print(f"[MapManager] Culling MP {mp_id} (n_obs={n_obs})")
                    self._delete_mappoint(mp_id)
                    deleted_mps += 1
                    continue
                    
                # 还可以加一个简单的深度检查（防止优化后点跑到无穷远）
                # 这里略过，因为 Optimizer 里已经检查过正深度了
                
            print(f"[MapManager] Optimized: {len(optimized_poses)} Poses, {len(optimized_points)} Points. "
                  f"Outliers removed: {len(outlier_observations) if outlier_observations else 0}, "
                  f"Culled MPs: {deleted_mps}")

    # --- 辅助内部函数 (不加锁，供内部调用) ---
    def _remove_observation_internal(self, mp_id, kf_id):
        """内部调用，不加锁，移除双向引用"""
        # 1. KeyFrame 端移除
        kf = None
        if kf_id in self.keyframes:
            kf = self.keyframes[kf_id]
            
        if kf:
            kf.remove_features_by_ids([mp_id])
            
        # 2. MapPoint 端移除
        if mp_id in self.mappoints:
            self.mappoints[mp_id].remove_observation(kf_id)

    def _delete_mappoint(self, mp_id):
        """彻底删除一个地图点，移除它在所有帧上的引用"""
        mp = self.mappoints.get(mp_id)
        if not mp:
            return
            
        # 遍历该点所有的观测帧，把该点从那些帧的特征列表中移除
        # 这一步很重要，否则 Frame 会持有一个指向“已删除点”的 ID
        obs_frames = mp.get_observing_kf_ids()
        for kf_id in obs_frames:
            kf = self.keyframes.get(kf_id)
            
            if kf:
                kf.remove_features_by_ids([mp_id])
        
        # 从字典删除
        del self.mappoints[mp_id]

    # --- 数据访问接口 (Getter) ---
    def get_keyframe(self, kf_id):
        """优先查找 active，其次查找 global"""
        with self.map_lock:
            return self.keyframes.get(kf_id)

    def get_map_point(self, mp_id):
        """优先查找 local，其次查找 global"""
        with self.map_lock:
            return self.mappoints.get(mp_id)

    def get_triangulated_mappoints(self):
        with self.map_lock:
            return {mp.id: mp.position_3d for mp in self.mappoints.values() if mp.status == MapPointStatus.TRIANGULATED}

    def get_keyframes(self):
        with self.map_lock:
            return sorted(self.keyframes.values(), key=lambda kf: kf.get_id()) # 返回列表

    # --- 辅助功能 ---
    def reset(self):
        print(f"[MapManager] Resetting all maps.")
        self.keyframes.clear()
        self.mappoints.clear()

    def check_mappoint_health(self, mappoint_id, candidate_position_3d=None):
        with self.map_lock:
            mp = self.mappoints.get(mappoint_id)
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

            # 获取观测帧
            observing_kf_ids = mp.get_observing_kf_ids()
            witness_kfs = [self.keyframes[kf_id] for kf_id in observing_kf_ids if kf_id in self.keyframes]
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

    # 共视图相关
    def update_covisibility_graph(self, frame, last_kf_id):
        """
        [核心功能] 三角化后调用。
        1. 统计当前帧与所有邻居的共视权重。
        2. 双向更新：更新邻居对当前帧的权重，更新当前帧对邻居的权重。
        3. 扩充局部地图：将邻居看到但当前帧没看到的点加入 local_map_ids。
        """
        with self.map_lock: # 确保线程安全
            
            # 1. 初始化
            # map_cov_kfs: 字典 {kf_id: shared_count}
            # 这里的共视点是CANDIDATE或TRIANGULATED
            map_cov_kfs = {} 
            # set_local_map_ids: 集合 {mp_id}，用于收集邻居的地图点
            set_local_map_ids = set()
            
            # 获取当前帧所有特征点ID
            current_mp_ids = frame.get_visual_feature_ids()

            # ---------------------------------------------------------
            # Step 1: 遍历特征点，建立共视连接 (Identify Neighbors)
            # ---------------------------------------------------------
            for mp_id in current_mp_ids:
                # 获取地图点对象（这里计算共视分数时可以是CANDIDATE）
                map_point = self.get_map_point(mp_id)
                
                # 如果点无效 (可能被剔除)，跳过
                if map_point is None:
                    # 进行无效观测剔除
                    self._remove_observation_internal(mp_id, frame.get_id())
                    continue

                # 获取观测到该点的所有关键帧
                observing_kf_ids = map_point.get_observing_kf_ids()

                for kf_id in observing_kf_ids:
                    # 跳过自己
                    if kf_id != frame.get_id():
                        # 权重 +1
                        map_cov_kfs[kf_id] = map_cov_kfs.get(kf_id, 0) + 1

            # ---------------------------------------------------------
            # Step 2: 更新邻居状态 & 收集局部地图 (Update Neighbors)
            # ---------------------------------------------------------
            bad_kf_ids = set()
            
            for kf_id, cov_score in map_cov_kfs.items():
                pkf = self.get_keyframe(kf_id)
                
                if pkf is not None:
                    # A. 双向更新 (Update Neighbor's Graph)
                    # 告诉共视KF：我和你有 cov_score 这么多共视点
                    pkf.add_covisible_kf(frame.get_id(), cov_score)

                    # B. 局部地图扩充 (Local Map Expansion)
                    # 遍历共视KF看到的点，如果是当前帧没见过的 3D 点，加入候选集
                    neighbor_mp_ids = pkf.get_visual_feature_ids()
                    
                    for neighbor_mp_id in neighbor_mp_ids:
                        # 只有当前帧没观测到的点才叫"扩充"
                        if not frame.is_observing_feature(neighbor_mp_id):
                            # 进一步检查点是否有效且是3D点
                            neighbor_mp = self.get_map_point(neighbor_mp_id)
                            if neighbor_mp is not None and neighbor_mp.status == MapPointStatus.TRIANGULATED:
                                set_local_map_ids.add(neighbor_mp_id)
                else:
                    bad_kf_ids.add(kf_id)

            # 清理无效的关键帧
            for kf_id in bad_kf_ids:
                del map_cov_kfs[kf_id]
            
            # ---------------------------------------------------------
            # Step 3: 更新当前帧状态 (Update Current Frame)
            # ---------------------------------------------------------
            # A. 保存共视图
            frame.set_covisible_map(map_cov_kfs)

            # B. 局部地图融合策略
            # 继承上一个KF的局部地图
            baseline_ids = set()
            if last_kf_id is not None:
                last_kf = self.get_keyframe(last_kf_id)
                if last_kf:
                    baseline_ids = last_kf.get_local_map_ids()

            if len(baseline_ids) == 0:
                # 冷启动情况：直接使用新的
                frame.set_local_map_ids(set_local_map_ids)
            
            if len(set_local_map_ids) > 0.5 * len(baseline_ids):
                # 替换：说明新的共视关系带来的点更有价值
                frame.set_local_map_ids(set_local_map_ids)
            else:
                # 合并：补充进来
                merged_ids = baseline_ids.copy() 
                merged_ids.update(set_local_map_ids)
                frame.set_local_map_ids(merged_ids)

            print(f"[MapManager] Updated Covisibility: KF {frame.get_id()} local map size: {len(frame.get_local_map_ids())}")

    def get_best_covisibility_keyframes(self, target_kf_id, top_k=5):
        """
        [辅助查询] 获取共视程度最高的 K 个关键帧。
        通常用于 Local BA 或 Loop Closure 选帧。
        """
        target_kf = self.get_keyframe(target_kf_id)
        if not target_kf: return {}
        
        # 直接读取已经计算好的共视图 (O(1))，不用重新遍历特征点！
        # 这就是先执行 update_covisibility_graph 的好处
        cov_map = target_kf.get_covisible_map()
        
        # 排序
        sorted_kfs = sorted(cov_map.items(), key=lambda item: item[1], reverse=True)
        
        # 返回前 K 个
        return dict(sorted_kfs[:top_k])

    def merge_mappoints(self, prev_mpid, new_mpid, cur_frame):
        """
        [核心功能] 合并两个地图点。
        将 prev_mpid 的所有信息（观测、描述子）转移给 new_mpid 然后删除prev_mpid
        注意这里prev_mpid是新点,new_mpid才是历史点
        """
        with self.map_lock:
            # 1. 获取对象 & 基础检查
            prev_mp = self.mappoints.get(prev_mpid)
            new_mp = self.mappoints.get(new_mpid)

            if prev_mp is None:
                print(f"[MapManager] Merge skip: keep_mp {prev_mpid} not found")
                return
            if new_mp is None:
                print(f"[MapManager] Merge skip: remove_mp {new_mpid} not found")
                return
            
            # 只能合并进已三角化的点
            if new_mp.status != MapPointStatus.TRIANGULATED:
                print(f"[MapManager] Merge skip: keep_mp {new_mpid} is not TRIANGULATED")
                return

            # {kf_id: pt_2d}
            prev_kf_ids = list(prev_mp.observations.keys())
            
            # 3. 遍历旧点（前端观测的新点）的所有观测帧，将它们重定向到新点（历史点）
            for kf_id in prev_kf_ids:
                kf = self.keyframes.get(kf_id)
                if kf is not None:
                    # 更新关键帧内部的 ID 引用，将当前KF观测的新点ID替换成局部地图点的观测历史ID
                    if kf.replace_mappoint_id(prev_mpid, new_mpid):
                        obs_pt = prev_mp.observations[kf_id]
                        desc = prev_mp.descriptors.get[kf_id]
                        new_mp.add_observation(kf_id, obs_2d=obs_pt, descriptor=desc) # 添加当前KF的观测

                        # 共视图操作，为当前KF添加历史共视，为历史KF添加当前共视
                        new_kf_ids = new_mp.get_observing_kf_ids()
                        for new_kf_id in new_kf_ids:
                            if new_kf_id == kf_id: continue
                            new_kf = self.keyframes.get(new_kf_id)
                            if new_kf is not None:
                                kf.add_covisible_kf(new_kf_id, 1)
                                new_kf.add_covisible_kf(kf_id, 1)

            # 描述子转移
            for kf_id, desc in prev_mp.descriptors.items():
                if kf_id not in new_mp.descriptors:
                    new_mp.add_descriptor(kf_id, desc)

            if cur_frame is not None:
                if cur_frame.is_observing_feature(prev_mpid):
                    if cur_frame.replace_mappoint_id(prev_mpid, new_mpid):
                        pass
            
            # 清除多出来的当前帧观测的新点
            del self.mappoints[prev_mpid]
            print(f"[MapManager] Merged MP {prev_mpid} -> {new_mpid}. New obs count: {len(new_mp.observations)}")