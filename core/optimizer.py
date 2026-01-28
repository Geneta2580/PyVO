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
        self.min_cov_score = self.config.get('min_cov_score', 25)

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
        # Step 2: 构建因子图 (构建因子图和初始估计)
        # =========================================================================
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        
        # 已添加的状态变量
        added_poses = set()
        added_points = set()

        # 添加待优化帧的位姿初始值
        for kf in local_kfs:
            initial_estimate.insert(X(kf.get_id()), gtsam.Pose3(kf.get_T_w_c()))
            added_poses.add(kf.get_id())

        # 添加固定帧状态变量以及先验因子
        for kf in fixed_kfs:
            # 如果还没添加进 Values
            if kf.get_id() not in added_poses:
                initial_estimate.insert(X(kf.get_id()), gtsam.Pose3(kf.get_T_w_c()))
                added_poses.add(kf.get_id())
            
            # 添加强先验因子，使其固定不动
            graph.add(gtsam.PriorFactorPose3(
                X(kf.get_id()), 
                gtsam.Pose3(kf.get_T_w_c()), 
                self.pose_noise_fix
            ))

        # 添加待优化点的3D位置初始值及视觉因子
        for mp in local_mps:
            # 添加路标点初始状态变量
            if mp.get_id() not in added_points:
                initial_estimate.insert(L(mp.get_id()), mp.get_point())
                added_points.add(mp.get_id())

            # 遍历该点所有观测
            obs_kf_ids = mp.get_observing_kf_ids()
            
            for kf_id in obs_kf_ids:
                if kf_id not in added_poses: 
                    print(f"[SafeGuard] Skip: KF not in added_poses! KF {kf_id}")
                    continue

                kf = self.map_manager.get_keyframe(kf_id)
                if kf is None: 
                    print(f"[SafeGuard] Skip: KF is None! KF {kf_id}")
                    continue

                # 获取去畸变观测像素
                uv_unpx = kf.get_feature_undistorted_position(mp.get_id())
                if uv_unpx is None: 
                    print(f"[SafeGuard] Skip: Measurement is None! MP {mp.get_id()}")
                    continue

                if not np.all(np.isfinite(uv_unpx)):
                    print(f"[SafeGuard] Skip: Measurement contains NaN! MP {mp.get_id()}")
                    continue

                unpx_measured = uv_unpx

                # ============================= 安全检查 (防止畸变引起的超大残差) =============================
                if kf.get_id() in initial_estimate.keys(): # 也就是 added_poses
                    pose_w_c = initial_estimate.atPose3(X(kf_id))
                else:
                    pose_w_c = gtsam.Pose3(kf.get_T_w_c())
                    
                if mp.get_id() in initial_estimate.keys():
                    point_w = initial_estimate.atPoint3(L(mp.get_id()))
                else:
                    point_w = mp.get_point()

                if not np.all(np.isfinite(pose_w_c.matrix())):
                    print(f"[SafeGuard] Skip: Pose contains NaN! KF {kf_id}")
                    continue
                if not np.all(np.isfinite(point_w)):
                    print(f"[SafeGuard] Skip: Point contains NaN! MP {mp.get_id()}")
                    continue

                # 变换到相机系
                try:
                    point_c = pose_w_c.transformTo(point_w)
                except:
                    print(f"[SafeGuard] Skip: Transform to camera frame failed! KF {kf_id} - MP {mp.get_id()}")
                    continue
                
                if point_c[2] < 0.1:
                    print(f"[SafeGuard] Skip Factor: KF {kf_id} - MP {mp.get_id()} is behind camera (Z={point_c[2]:.2f})")
                    continue 
                # else:
                #     print(f"[SafeGuard] Accept Factor: KF {kf_id} - MP {mp.get_id()} is too far (Z={point_c[2]:.2f})")

                # 4. 检查重投影是否极其离谱 (可选，防止由畸变引起的超大残差)
                try:
                    # 手动投影一下
                    norm_xy = point_c[:2] / point_c[2]
                    pred_uv = self.K.uncalibrate(norm_xy)
                    
                    # 再次检查投影结果是否有效
                    if not np.all(np.isfinite(pred_uv)):
                        print(f"[SafeGuard] Skip: Projected Point is NaN! (Z={point_c[2]})")
                        continue

                    reproj_err = np.linalg.norm([pred_uv[0] - uv_unpx[0], pred_uv[1] - uv_unpx[1]])

                    # 如果初始残差 > 50 像素，说明这个匹配完全是错的，优化器拉不回来的
                    if reproj_err > 50.0:
                        print(f"[SafeGuard] Skip Factor: Large Initial Error ({reproj_err:.2f} px)")
                        continue
                    # else:
                    #     print(f"[SafeGuard] Accept Factor: Large Initial Error ({reproj_err:.2f} px)")

                except:
                    print(f"[SafeGuard] Skip: Reprojection check failed! KF {kf_id} - MP {mp.get_id()}")
                    continue
                # ============================= 安全检查 (防止畸变引起的超大残差) =============================

                # 添加视觉因子
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    unpx_measured,
                    self.robust_noise_model,
                    X(kf_id),
                    L(mp.get_id()),
                    self.K,
                    self.body_T_cam
                ))

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