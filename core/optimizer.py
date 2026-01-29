import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
import time

class Optimizer:
    def __init__(self, config, map_manager):
        self.config = config
        self.map_manager = map_manager # 持有全局 MapManager 引用
        
        # 1. 噪声参数配置
        # 视觉观测噪声 (像素单位)
        sigma_px = 1.0
        self.visual_factor_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma_px)
        
        # 鲁棒核函数 (Huber)
        self.huber_k = 1.345
        self.robust_noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(self.huber_k),
            self.visual_factor_noise
        )

        # 锚点帧噪声 (用于固定锚点帧)
        self.pose_noise_fix = gtsam.noiseModel.Constrained.All(6)

        # 优化误差 Chi-square 阈值
        self.chi2_threshold = 5.9915 

        # 2. 相机内参 Setup
        cam_intrinsics = np.asarray(self.config.get('cam_intrinsics')).reshape(3, 3)
        fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
        cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
        s = 0.0
        self.K = gtsam.Cal3_S2(fx, fy, s, cx, cy)

        # 最小共视分数
        self.min_cov_score = self.config.get('min_cov_score', 50)

        # 是否应用二阶段优化
        self.apply_l2_optimization = self.config.get('apply_l2_optimization', True)

        # 3. 外参 (单目 VO 优化的是 Camera Pose，故 Body_P_Sensor 为 Identity)
        self.body_T_cam = gtsam.Pose3(np.eye(4)) 

    def optimize(self, new_kf):
        # =========================================================================
        # Step 1: 从 MapManager 获取数据 (线程安全地获取副本或引用)
        # =========================================================================
        cov_kfs_dict = new_kf.get_covisible_map()

        # 向共视图中添加自己
        cov_kfs_dict[new_kf.get_id()] = len(new_kf.get_visual_features())
        
        # 准备集合进行分类
        local_kfs = set()       # 待优化帧 (Variables)
        fixed_kfs = set()       # 固定帧 (Priors/Anchors)
        local_mps = set()       # 待优化地图点

        max_kf_id = max(cov_kfs_dict.keys())

        # 筛选出需要优化的共视KF
        for kf_id, score in cov_kfs_dict.items():
            kf = self.map_manager.get_keyframe(kf_id)
            if kf is None: continue
            
            # 优化条件：共视程度高 (>= min_cov_score)或者就是当前新帧本身
            if score >= self.min_cov_score or kf_id == new_kf.get_id():
                # 将这些KF观测到的点加入待优化列表
                local_kfs.add(kf)
                for mp_id in kf.get_visual_feature_ids():  
                    mp = self.map_manager.get_map_point(mp_id)
                    if mp is not None and not mp.is_bad():
                        local_mps.add(mp)
            else:
                fixed_kfs.add(kf) # 共视程度低 -> 作为固定锚点

        # 寻找二级共视KF
        # 如果一个 KF 观测到了 local_mps 中的点，但它不在 local_kfs 里，那么它应该作为 Fixed Frame 加入，以提供更多约束
        for mp in local_mps:
            obs_kf_ids = mp.get_observing_kf_ids()
            for obs_kf_id in obs_kf_ids:
                if obs_kf_id > max_kf_id: continue # 不添加未来KF

                kf = self.map_manager.get_keyframe(obs_kf_id)
                if kf is not None and kf not in local_kfs: # 不在local_kfs里，说明是二级共视KF
                    fixed_kfs.add(kf)

        # 固定帧检查
        # 如果固定帧不足2个，且待优化帧大于2个，则将最老的KF固定
        if len(fixed_kfs) < 2 and len(local_kfs) > 2:
            sorted_local = sorted(list(local_kfs), key=lambda x: x.get_id())
            oldest_kf = sorted_local[0]
            local_kfs.remove(oldest_kf)
            fixed_kfs.add(oldest_kf)

        print(f"【Optimizer】: =================================================")
        print(f"【Optimizer】: Starting BA")
        print(f"【Optimizer】: local_kfs: {len(local_kfs)}")
        print(f"【Optimizer】: fixed_kfs: {len(fixed_kfs)}")
        print(f"【Optimizer】: local_mps: {len(local_mps)}")
        print(f"【Optimizer】: =================================================")

        # =========================================================================
        # Step 2: 构建因子图 (Debug Version)
        # =========================================================================
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        
        added_poses = set()
        added_points = set()

        # -------------------------------------------------------------------------
        # 2.1 准备所有 Pose 变量
        # -------------------------------------------------------------------------
        all_window_kfs = {} 
        for kf in local_kfs.union(fixed_kfs):
            kf_id = kf.get_id()
            all_window_kfs[kf_id] = kf
            if kf_id not in added_poses:
                initial_estimate.insert(X(kf_id), gtsam.Pose3(kf.get_T_w_c()))
                added_poses.add(kf_id)
            if kf in fixed_kfs:
                graph.add(gtsam.PriorFactorPose3(X(kf_id), gtsam.Pose3(kf.get_T_w_c()), self.pose_noise_fix))

        # -------------------------------------------------------------------------
        # 2.2 添加 MapPoints 和 视觉因子
        # -------------------------------------------------------------------------
        # 详细统计丢弃原因
        stats = {
            'total_mps': 0,
            'accepted_mps': 0,
            'drop_not_in_window': 0, # 观测帧不在当前窗口(local+fixed)
            'drop_nan': 0,           # 数据无效
            'drop_depth': 0,         # 深度检查失败 (Z < 0)
            'drop_transform': 0,     # 坐标变换失败
            'reject_not_enough': 0,  # 有效观测 < 2
            'reject_low_parallax': 0 # 视差不足
        }

        for mp in local_mps:
            stats['total_mps'] += 1
            mp_id = mp.get_id()
            point_w = mp.get_point()

            valid_factors_buffer = [] 
            valid_observing_kfs = []

            obs_kf_ids = mp.get_observing_kf_ids()

            for kf_id in obs_kf_ids:
                # 检查 1: 帧是否存在于窗口
                if kf_id not in all_window_kfs:
                    stats['drop_not_in_window'] += 1
                    # print(f"[Debug] MP {mp_id} obs by KF {kf_id} DROPPED: Not in window")
                    continue

                kf = all_window_kfs[kf_id]
                uv_unpx = kf.get_feature_undistorted_position(mp_id)
                
                # 检查 2: 数据有效性
                if uv_unpx is None or not np.all(np.isfinite(uv_unpx)):
                    stats['drop_nan'] += 1
                    continue

                # 检查 3: 几何深度 (已放宽阈值)
                pose_w_c = gtsam.Pose3(kf.get_T_w_c())
                try:
                    point_c = pose_w_c.transformTo(point_w)
                    # 【修改】将阈值从 0.05 改为 0.001，防止单目尺度过小导致误杀
                    if point_c[2] < 0.3: 
                        stats['drop_depth'] += 1
                        # print(f"[Debug] MP {mp_id} in KF {kf_id} DROPPED: Depth {point_c[2]:.4f} < 0.001")
                        continue 
                except:
                    stats['drop_transform'] += 1
                    continue

                # 通过所有检查，暂存
                factor = gtsam.GenericProjectionFactorCal3_S2(
                    uv_unpx,
                    self.robust_noise_model,
                    X(kf_id),
                    L(mp_id),
                    self.K,
                    self.body_T_cam
                )
                valid_factors_buffer.append(factor)
                valid_observing_kfs.append(kf)

            # --- 准入考核 ---

            # 【考核 1】观测数量
            if len(valid_factors_buffer) < 2:
                stats['reject_not_enough'] += 1
                # print(f"[Debug] MP {mp_id} REJECTED: Valid factors {len(valid_factors_buffer)} < 2")
                continue 

            # # 【考核 2】视差检查
            # kf_first = valid_observing_kfs[0]
            # kf_last = valid_observing_kfs[-1]
            # cam1 = kf_first.get_T_w_c()[:3, 3]
            # cam2 = kf_last.get_T_w_c()[:3, 3]
            
            # baseline = np.linalg.norm(cam1 - cam2)
            # depth = np.linalg.norm(point_w - cam1)

            # # 视差阈值：0.5度
            # is_good_parallax = False
            # if depth > 1e-5: # 防止除0
            #     ratio = baseline / depth
            #     if ratio > 0.01: # 约 0.5度
            #         is_good_parallax = True
            #     else:
            #         # 也可以用角度兜底
            #         vec1 = point_w - cam1
            #         vec2 = point_w - cam2
            #         cos_theta = np.dot(vec1/np.linalg.norm(vec1), vec2/np.linalg.norm(vec2))
            #         angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            #         if angle > np.deg2rad(0.5):
            #             is_good_parallax = True
            
            # if not is_good_parallax:
            #     stats['reject_low_parallax'] += 1
            #     continue

            # --- 正式录用 ---
            stats['accepted_mps'] += 1
            if mp_id not in added_points:
                initial_estimate.insert(L(mp_id), point_w)
                added_points.add(mp_id)
            
            for f in valid_factors_buffer:
                graph.add(f)
        
        print(f"【Optimizer】: Graph Debug Stats: {stats}")
        print(f"【Optimizer】: Stage 1 Graph: {graph.size()} factors.")

        # =========================================================================
        # Step 3: 一阶段优化
        # =========================================================================
        try:
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(5)
            params.setRelativeErrorTol(1e-3)
            params.setAbsoluteErrorTol(1e-3)
            params.setVerbosityLM("SUMMARY") 

            t1 = time.time()
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
            result = optimizer.optimize()
            t2 = time.time()

            # params = gtsam.DoglegParams()
            # params.setDeltaInitial(1.0)
            # params.setMaxIterations(5)
            # params.setRelativeErrorTol(1e-3)
            # params.setAbsoluteErrorTol(1e-3)
            # params.setVerbosity("SUMMARY") 

            # t1 = time.time()
            # optimizer = gtsam.DoglegOptimizer(graph, initial_estimate, params)
            # result = optimizer.optimize()
            # t2 = time.time()

            print(f"【Optimizer】: Stage 1 Optimization took {(t2-t1)*1000:.2f} ms.")
            
        except Exception as e:
            print(f"【Optimizer】: Optimization Failed! {e}")
            return False

        # =========================================================================
        # Step 4: [外点检测 (Outlier Rejection)
        # =========================================================================
        bad_factors_indices = []
        outlier_observations = [] # 存储 (kf_id, mp_id)

        # 遍历因子图中的每一个因子
        for i in range(graph.size()):
            factor = graph.at(i)
            
            # 我们只关心投影因子 (GenericProjectionFactorCal3_S2)
            # 通过检查 error 是否过大
            # 注意：GTSAM 的 error() 返回的是 0.5 * r^T * Cov^-1 * r
            # Chi2 分布值通常是 2.0 * error()
            
            # 使用 Robust Kernel 时，error() 返回的是经过 Huber 加权后的 error
            # 如果误差很大，Huber 会把它压下来，但在判断外点时，我们通常看这个加权后的值是否依然很大，
            # 或者重新计算非加权的误差。这里简化处理，直接看优化后的 error。
            
            if len(factor.keys()) == 2: # 投影因子有两个 key: Pose, Point
                key1 = factor.keys()[0] # Pose Key (X)
                key2 = factor.keys()[1] # Point Key (L)
                
                # 确认 key 类型 (GTSAM Python 有点 tricky，我们通过 Symbol 判断)
                # 假设构建顺序是 X, L (通常 GTSAM 内部会排序，X 排在 L 前面)
                sym1 = gtsam.Symbol(key1)
                sym2 = gtsam.Symbol(key2)
                
                # 简单的 Check: 一个是 'x', 一个是 'l'
                if chr(sym1.chr()) == 'x' and chr(sym2.chr()) == 'l':
                    kf_id = sym1.index()
                    mp_id = sym2.index()
                elif chr(sym1.chr()) == 'l' and chr(sym2.chr()) == 'x':
                    mp_id = sym1.index()
                    kf_id = sym2.index()
                else:
                    continue # 可能是 Prior Factor，跳过

                # 计算误差
                error = factor.error(result) # 0.5 * (z - h(x))^2
                chi2 = 2.0 * error
                
                # 检查 Cheirality (点是否在相机后面)
                # 这一步比较耗时，我们可以先只看 chi2，如果 chi2 巨大通常也是 cheirality 错误
                # 或者手动变换一下检查深度
                pose = result.atPose3(X(kf_id))
                point = result.atPoint3(L(mp_id))
                point_in_cam = pose.transformTo(point)
                
                is_depth_positive = point_in_cam[2] > 0.1 # 0.1m 最小深度
                
                if chi2 > self.chi2_threshold or not is_depth_positive:
                    outlier_observations.append((kf_id, mp_id)) # 标记为外点
                    bad_factors_indices.append(i) # 记录因子索引
                    print(f"【Optimizer】: Outlier detected: KF {kf_id} - MP {mp_id} (Chi2: {chi2:.2f}, Depth: {point_in_cam[2]:.2f})")

        print(f"【Optimizer】: Found {len(outlier_observations)} outliers.")

        # =========================================================================
        # Step 5: 二阶段优化
        # =========================================================================
        new_graph = gtsam.NonlinearFactorGraph()
        new_estimate = gtsam.Values()
        second_stage_success = False
        
        # 将一阶段发现的外点转为集合，方便 O(1) 查找
        bad_obs_set = set(outlier_observations) # set of (kf_id, mp_id)

        if self.apply_l2_optimization and len(bad_obs_set) > 0:
            print(f"【Optimizer】: Rebuilding graph for L2 Refinement (removing {len(bad_obs_set)} outliers)...")

            # -----------------------------------------------------
            # 5.1 重建固定帧 (Priors)
            # -----------------------------------------------------
            added_poses_stage2 = set()
            for kf in fixed_kfs:
                kf_id = kf.get_id()
                if result.exists(X(kf_id)):
                    pose = result.atPose3(X(kf_id))
                else:
                    pose = gtsam.Pose3(kf.get_T_w_c())
                
                new_estimate.insert(X(kf_id), pose)
                new_graph.add(gtsam.PriorFactorPose3(X(kf_id), pose, self.pose_noise_fix))
                added_poses_stage2.add(kf_id)

            # -----------------------------------------------------
            # 5.2 重建优化帧 (Variables) - 使用一阶段优化后的结果作为初值
            # -----------------------------------------------------
            for kf in local_kfs:
                kf_id = kf.get_id()
                if result.exists(X(kf_id)):
                    new_estimate.insert(X(kf_id), result.atPose3(X(kf_id)))
                    added_poses_stage2.add(kf_id)
            
            # -----------------------------------------------------
            # 5.3 重建地图点和因子 (关键步骤)
            # -----------------------------------------------------
            active_mps_count = 0
            
            for mp in local_mps:
                mp_id = mp.get_id()
                
                # 如果点在一阶段就挂了（比如发散了被移除），跳过
                if not result.exists(L(mp_id)): continue

                obs_kf_ids = mp.get_observing_kf_ids()
                
                # 收集该点的所有有效观测 (非 Outlier)
                valid_factors_for_point = []
                
                for kf_id in obs_kf_ids:
                    # 1. 必须在当前窗口内
                    if kf_id not in added_poses_stage2: continue
                    # 2. 必须不是一阶段发现的外点
                    if (kf_id, mp_id) in bad_obs_set: continue

                    kf = self.map_manager.get_keyframe(kf_id)
                    uv_unpx = kf.get_feature_undistorted_position(mp_id)
                    if uv_unpx is None: continue
                    
                    # 创建因子：注意这里使用 visual_factor_noise (L2) 而不是 robust (Huber)
                    factor = gtsam.GenericProjectionFactorCal3_S2(
                        uv_unpx,
                        self.visual_factor_noise,
                        X(kf_id),
                        L(mp_id),
                        self.K,
                        self.body_T_cam
                    )
                    valid_factors_for_point.append(factor)

                # 只有当该点还有至少 2 个有效观测时，才加入二阶段优化
                if len(valid_factors_for_point) >= 2:
                    new_estimate.insert(L(mp_id), result.atPoint3(L(mp_id))) # 使用一阶段结果
                    for f in valid_factors_for_point:
                        new_graph.add(f)
                    active_mps_count += 1
            
            print(f"【Optimizer】: Stage 2 Graph: {new_graph.size()} factors, {active_mps_count} points.")
            # -----------------------------------------------------
            # 5.4 执行二阶段优化
            # -----------------------------------------------------
            try:
                params.setMaxIterations(10)
                params.setRelativeErrorTol(1e-3)
                params.setAbsoluteErrorTol(1e-3)
                params.setVerbosityLM("SUMMARY") 
                
                t3 = time.time()
                optimizer = gtsam.LevenbergMarquardtOptimizer(new_graph, new_estimate, params)
                result = optimizer.optimize()
                t4 = time.time()

                print(f"【Optimizer】: Stage 2 Optimization took {(t4-t3)*1000:.2f} ms.")

                second_stage_success = True
            except Exception as e:
                print(f"【Optimizer】: Second Stage Refinement Failed: {e}")

        # =========================================================================
        # Step 6: 再次检测外点
        # =========================================================================
        second_bad_factors_indices = []
        second_outlier_observations = [] # 存储 (kf_id, mp_id)
        if second_stage_success and new_graph.size() > 0:
            # 遍历因子图中的每一个因子
            for i in range(new_graph.size()):
                factor = new_graph.at(i)
                
                if len(factor.keys()) == 2: # 投影因子有两个 key: Pose, Point
                    key1 = factor.keys()[0] # Pose Key (X)
                    key2 = factor.keys()[1] # Point Key (L)
                    
                    # 确认 key 类型 (GTSAM Python 有点 tricky，我们通过 Symbol 判断)
                    # 假设构建顺序是 X, L (通常 GTSAM 内部会排序，X 排在 L 前面)
                    sym1 = gtsam.Symbol(key1)
                    sym2 = gtsam.Symbol(key2)
                    
                    # 简单的 Check: 一个是 'x', 一个是 'l'
                    if chr(sym1.chr()) == 'x' and chr(sym2.chr()) == 'l':
                        kf_id = sym1.index()
                        mp_id = sym2.index()
                    elif chr(sym1.chr()) == 'l' and chr(sym2.chr()) == 'x':
                        mp_id = sym1.index()
                        kf_id = sym2.index()
                    else:
                        continue # 可能是 Prior Factor，跳过

                    # 计算误差
                    error = factor.error(result) # 0.5 * (z - h(x))^2
                    chi2 = 2.0 * error
                    
                    # 检查 Cheirality (点是否在相机后面)
                    # 这一步比较耗时，我们可以先只看 chi2，如果 chi2 巨大通常也是 cheirality 错误
                    # 或者手动变换一下检查深度
                    pose = result.atPose3(X(kf_id))
                    point = result.atPoint3(L(mp_id))
                    point_in_cam = pose.transformTo(point)
                    
                    is_depth_positive = point_in_cam[2] > 0.1 # 0.1m 最小深度
                    
                    if chi2 > self.chi2_threshold or not is_depth_positive:
                        second_outlier_observations.append((kf_id, mp_id)) # 标记为外点
                        second_bad_factors_indices.append(i) # 记录因子索引
                        print(f"【Optimizer】: Second Stage: Outlier detected: KF {kf_id} - MP {mp_id} (Chi2: {chi2:.2f}, Depth: {point_in_cam[2]:.2f})")

            print(f"【Optimizer】: Second Stage: Found {len(second_outlier_observations)} outliers.")

        # =========================================================================
        # Step 7: 提取结果并回写 (包含外点剔除)
        # =========================================================================
        all_outliers = set(outlier_observations + second_outlier_observations)
        print(f"【Optimizer】: Total outliers: {len(all_outliers)}")

        # 更新优化帧位姿和地图点位置
        optimized_poses = {}
        optimized_points = {}

        for kf in local_kfs:
            kf_id = kf.get_id()
            if result.exists(X(kf_id)):
                optimized_poses[kf_id] = result.atPose3(X(kf_id)).matrix()

        for mp in local_mps:
            mp_id = mp.get_id()
            if result.exists(L(mp_id)):
                optimized_points[mp_id] = result.atPoint3(L(mp_id))

        # 调用 MapManager 回写 (传入 outliers)
        self.map_manager.update_map_from_optimization(optimized_poses, optimized_points, list(all_outliers))
        
        return True