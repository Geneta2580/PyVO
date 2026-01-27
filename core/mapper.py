from os import curdir
import numpy as np
import math
from datatype.mappoint import MapPointStatus
from utils.geometry import MultiViewGeometry

class Mapper:
    def __init__(self, config, map_manager):
        self.config = config
        self.map_manager = map_manager
        self.prev_keyframe = None
        self.cur_keyframe = None

        # 共视点匹配参数
        self.max_proj_dist = config.get('max_proj_dist', 10.0) # 最大投影距离
        self.max_desc_dist = config.get('max_desc_dist', 10.0) # 最大描述子距离

        # 局部地图点数量限制
        self.grid_cell_size = config.get('grid_cell_size', 35)
        width = self.config.get('image_width', 752)
        height = self.config.get('image_height', 480)
        n_w_cells = math.ceil(width / self.grid_cell_size)
        n_h_cells = math.ceil(height / self.grid_cell_size)
        self.max_kps = int(n_w_cells * n_h_cells)

        # 最大允许汉明距离阈值（如果两个特征点的差异超过了总位数的 X%则说明两者不是同一个点）
        self.dist_ratio = config.get('dist_ratio', 0.2) 

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
                # print(f"[Mapper] Mappoint {mp_id} not found in map_manager")
                continue
            
            # 只处理 CANDIDATE 状态的路标点，不重复三角化
            if mappoint.status != MapPointStatus.CANDIDATE:
                # print(f"[Mapper] Mappoint {mp_id} not in CANDIDATE status")
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
        triangulated_mappoints = self.map_manager.get_triangulated_mappoints()
        num_triangulated = len(triangulated_mappoints)
        
        print(f"[Mapper] Checking initialization quality: {num_triangulated} triangulated mappoints")
        
        # 条件1: 已三角化的地图点数量少于30
        if num_triangulated < 30:
            print(f"[Mapper] Initialization quality check failed: Only {num_triangulated} triangulated mappoints (< 30)")
            return True
        
        # 2. 统计前10个关键帧的3D点数量
        keyframes = self.map_manager.get_keyframes()
        if len(keyframes) == 0:
            return False
        
        # 获取前10个关键帧（按ID排序）
        first_10_keyframes = sorted(keyframes, key=lambda kf: kf.get_id())[:10]
        
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

    def match_to_local_map(self, frame):
        """
        [后端核心] 将局部地图点投影到当前帧，进行匹配和融合。
        """
        print(f"[Mapper] Refining matches for KF {frame.get_id()}...")

        # 1. 准备候选点集合
        max_local_mps = self.max_kps * 10
        candidate_mp_ids = frame.get_local_map_ids() # 这是一个 set

        # =========================================================
        # Part A: 候选点扩充 (Fallback Expansion)
        # 如果 candidate_mp_ids 数量不够，去共视图里再挖一挖
        # =========================================================
        if len(candidate_mp_ids) < max_local_mps:
            print(f"[Mapper] Candidate mappoints size: {len(candidate_mp_ids)} < {max_local_mps}, expanding from covisible map")
            with self.map_manager.map_lock:
                # 获取共视图
                cov_map = frame.get_covisible_map() # {kf_id: weight}

                # 获取最老的共视帧 ID (ID 最小)
                oldest_cov_kf_id = -1
                if len(cov_map) > 0:
                    # min(keys) 获取最小 ID
                    oldest_cov_kf_id = min(cov_map.keys())
                else:
                    # 如果没有共视帧(极端情况)，尝试回溯上一帧 ID
                    if frame.get_id() > 0:
                        oldest_cov_kf_id = frame.get_id() - 1
                
                # 时间回溯兜底 (kfid--)
                search_id = oldest_cov_kf_id
                target_kf = None
                
                # 尝试往回找，最多找比如 5 帧，或者一直找到 0
                while search_id >= 0:
                    target_kf = self.map_manager.get_keyframe(search_id)
                    if target_kf is not None:
                        break # 找到了
                    
                    search_id -= 1

                if target_kf is not None:
                    # 获取共视KF看到的点
                    inherited_ids = target_kf.get_local_map_ids()
                    if len(inherited_ids) == 0:
                         inherited_ids = target_kf.get_visual_feature_ids()

                    # 补充进候选集
                    for feature_id in inherited_ids:
                        # 只有当：1.有效 2.未被当前KF观测 3.不在集合中 时才添加
                        if (feature_id not in candidate_mp_ids) and (not frame.is_observing_feature(feature_id)):
                            mp = self.map_manager.get_map_point(feature_id)
                            # 只引入已经三角化稳定的点
                            if mp and mp.status == MapPointStatus.TRIANGULATED: 
                                candidate_mp_ids.add(feature_id)

        print(f"[Mapper] Search candidates size: {len(candidate_mp_ids)}")
        if len(candidate_mp_ids) == 0:
            return False

        # =========================================================
        # Part B: 投影及描述子匹配 (Projection & Descriptor Matching)
        # =========================================================
        # 输出[当前KF特征点，历史局部地图3d地图点]
        matches_found = self.match_to_map(frame, candidate_mp_ids)
        num_matches = len(matches_found)

        print(f"[Mapper] Match To Local Map found #{num_matches} matches")

        if num_matches == 0:
            return False

        # =========================================================
        # Part C: 融合与更新 (Merging)
        # =========================================================
        count_merged = 0

        # 将 matches_found 转换为 {prev_id: new_id} 的形式，方便处理
        # map_previd_newid 对应 C++ mergeMatches 的入参 map_kpids_lmids
        map_previd_newid = {}
        for feat_idx, new_lmid in matches_found.items():
            # 当前特征点原本的ID
            prev_lmid = frame.visual_feature_ids[feat_idx]
            
            # 只有当 ID 确实不同时才需要合并
            if prev_lmid != new_lmid:
                map_previd_newid[prev_lmid] = new_lmid

        with self.map_manager.map_lock:
            for prev_lmid, new_lmid in map_previd_newid.items():
                self.map_manager.merge_mappoints(prev_lmid, new_lmid, current_frame=frame)
                count_merged += 1

        print(f"[Mapper] Refinement result: Merged {count_merged} ghosts.")

    def match_to_map(self, frame, candidate_mp_ids):
        map_previd_newid = {} # 最终结果 {feature_id_in_frame: matched_local_mp_id}
        
        # 计算最大视场角
        vfov_tan = 0.5 * frame.camera.img_h / frame.camera.fy
        hfov_tan = 0.5 * frame.camera.img_w / frame.camera.fx

        max_tan_fov = max(hfov_tan, vfov_tan)
        max_rad_fov = np.arctan(max_tan_fov)

        view_th = np.cos(max_rad_fov)
        
        # 统计当前帧中已有的 3D 地图点数量
        n_tracked_3d = 0
        current_feature_ids = frame.get_visual_feature_ids()
        for fid in current_feature_ids:
            mp = self.map_manager.get_map_point(fid)
            if mp is not None and mp.status == MapPointStatus.TRIANGULATED:
                n_tracked_3d += 1

        # 动态调整搜索半径
        max_proj_dist = self.max_proj_dist
        if(n_tracked_3d < 30):
            max_proj_dist = self.max_proj_dist * 2

        map_kpids_near_dist = {} # {feat_idx: [(mp_id, dist)]}

        # 遍历候选局部地图3d点（当前KF未观测到的部分）
        for mp_id in candidate_mp_ids:
            # ---------------------------------------------------------
            # 检查候选局部3d点是否有效
            # ---------------------------------------------------------
            # 如果当前帧已经观测到了该点，则跳过
            if frame.is_observing_feature(mp_id):
                continue

            mp = self.map_manager.get_map_point(mp_id)
            if mp is None:
                continue
            
            if mp.status != MapPointStatus.TRIANGULATED and mp.descriptors is None:
                continue

            mp_pos = mp.get_point()
            proj_cam = frame.camera.project_world_to_cam(frame.get_T_w_c(), mp_pos)
            # 检查投影深度
            if proj_cam[2] < 0.1:
                continue

            # 检查视场角
            view_angle = proj_cam[2] / np.linalg.norm(proj_cam)
            if np.abs(view_angle) < view_th:
                continue

            # 检查畸变投影是否在图像内
            proj_img = frame.camera.project_cam_to_image_dist(proj_cam)
            if not frame.camera.is_in_image(proj_img):
                continue
            
            # 获取邻近投影点的特征点（作为和历史特征点匹配的候选）
            vnear_indices = frame.get_features_in_area(proj_img[0], proj_img[1])

            # ---------------------------------------------------------
            # 4. 遍历候选特征点 (Inner Loop: C++ for(const auto &kp : vnearkps))
            # ---------------------------------------------------------
            # 准备匹配变量
            min_dist = 32 * self.dist_ratio * 8.0 # 字节转比特（ORB描述子32个字节，256比特）
            best_id = -1
            sec_id = -1
            best_dist = min_dist
            sec_dist = min_dist

            for idx in vnear_indices:
                kp_id = frame.visual_feature_ids[idx]
                kp_px = frame.visual_features[idx].flatten()
                if kp_id < 0: continue

                # --- A. 像素距离检查 (Pixel Distance Check) ---
                px_dist = np.linalg.norm(proj_img - kp_px)
                if px_dist > max_proj_dist: continue
            
                # --- B. 数据关联严格检查 (Co-visibility Check) ---
                # 这里的near_mp是CANDIDATE或TRIANGULATED
                near_mp = self.map_manager.get_map_point(kp_id)

                if near_mp is None: 
                    self.map_manager.remove_observation_both_sides(kp_id, frame.get_id())
                    continue

                if near_mp.descriptors is None:
                    continue

                is_candidate = True

                # 这里相当于已知一个局部地图候选3d点，需要查找其在当前帧的匹配观测
                # 现在相当于找到了配对的可疑人员near_mp，但是他和原始输入查询对象mp不能同时被同一个KF观测到，否则就是两个点
                candidate_obs_kfs = set(mp.get_observing_kf_ids()) # 当前选中局部候选点的所有观测
                near_obs_kfs = near_mp.get_observing_kf_ids() # 当前KF潜在匹配点的所有观测
                
                # 这里当前候选（当前KF潜在的匹配点）不能和查询的（局部地图的mp）路标点同时在
                # 同一个KF上，这样说明两者物理上就不是一个点（这里的目标是合并匹配观测和局部地图点）
                for kf_id in candidate_obs_kfs:
                    if kf_id in near_obs_kfs:
                        is_candidate = False
                        break

                if not is_candidate: continue

                # --- C. 重投影一致性检查 (Co-projection Check) ---
                reproj_error = 0.0
                nb_co_kp = 0

                # 遍历可疑人员的所有观测KF，检查查询mp在这些KF上的重投影一致性（如果一致，说明可能是同一个点）
                for kf_id in near_obs_kfs:
                    co_kf = self.map_manager.get_keyframe(kf_id)
                    if co_kf is None: 
                        self.map_manager.remove_observation_both_sides(kp_id, kf_id)
                        print(f"[Mapper] KF {kf_id} not found for mappoint {kp_id}")
                        continue
                    
                    # 获取当前KF上潜在匹配对象在其所有观测KF上的所有2d像素观测坐标
                    co_kp_px = co_kf.get_feature_position(kp_id)
                    if co_kp_px is not None:
                        # 将局部地图点投影到像素匹配潜在KF的像素平面
                        T_w_c_near_kf = co_kf.get_T_w_c()
                        proj_candidate_in_near_kf = co_kf.camera.project_world_to_image_dist(T_w_c_near_kf, mp_pos)
                        if proj_candidate_in_near_kf is not None:
                            reproj_error += np.linalg.norm(co_kp_px - proj_candidate_in_near_kf)
                            nb_co_kp += 1
                    else:
                        self.map_manager.remove_observation_both_sides(kp_id, kf_id)
                        print(f"[Mapper] Co-KP {kp_id} not found for mappoint {mp_id}")
                        continue
                
                avg_reproj_error = reproj_error / nb_co_kp
                if (avg_reproj_error > max_proj_dist):
                    continue

                # --- D. 描述子匹配 ---
                dist = mp.compute_min_desc_dist(near_mp)

                if dist <= best_dist:
                    sec_dist = best_dist
                    sec_id = best_id
                    best_dist = dist
                    best_id = kp_id
                elif dist <= sec_dist:
                    sec_dist = dist
                    sec_id = kp_id

            # ---------------------------------------------------------
            # 5. 记录结果 
            # ---------------------------------------------------------
            # 最优匹配和次优匹配靠的太近，说明匹配失败（结果不唯一），直接舍弃
            if best_id != -1 and sec_id != -1:
                if 0.9 * sec_dist < best_dist:
                    best_id = -1
            
            if best_id < 0: continue

            # 将匹配结果存入暂存区，用于解决多对一冲突
            # 这里实际上是输入一个局部地图的3d候选点，输出一个当前帧的潜在匹配点
            # 但是这个查询是单向的，当前多个局部地图的3d候选点可能都声称同一个当前KF特征点与其匹配，因此需要解决多对一冲突
            if best_id not in map_kpids_near_dist:
                map_kpids_near_dist[best_id] = []
            map_kpids_near_dist[best_id].append((mp_id, best_dist))

        # ---------------------------------------------------------
        # 6. 解决多对一冲突 (Multi-to-One Conflict Resolution
        # ---------------------------------------------------------
        # 遍历一个当前匹配特征点可能的多个局部3d地图候选点，选择描述子距离最小的那个
        for kp_id, mp_list in map_kpids_near_dist.items():
            best_mp_id = -1
            min_dist = 1024.0

            for mp_id, dist in mp_list:
                if dist < min_dist:
                    min_dist = dist
                    best_mp_id = mp_id
            
            if best_mp_id >= 0:
                map_previd_newid[kp_id] = best_mp_id
        
        return map_previd_newid

    def reset(self):
        """
        重置 Mapper 状态
        """
        print(f"[Mapper] Resetting mapper state")
        self.prev_keyframe = None
        self.cur_keyframe = None
        print(f"[Mapper] Reset complete")