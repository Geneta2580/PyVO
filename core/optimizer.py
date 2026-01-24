import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L

class Optimizer:
    def __init__(self, config, map_manager):
        self.config = config
        self.map_manager = map_manager # 持有全局 MapManager 引用
        
        # 1. 噪声参数配置
        # 视觉观测噪声 (像素单位)
        sigma_pix = 1.0
        self.visual_factor_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma_pix)
        
        # 鲁棒核函数 (Huber)
        self.huber_k = 1.345
        self.robust_noise_model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(self.huber_k),
            self.visual_factor_noise
        )

        # 先验噪声 (用于固定滑窗第一帧)
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001])
        )

        # 优化误差 Chi-square 阈值
        self.chi2_threshold = 5.9915 

        # 2. 相机内参 Setup
        cam_intrinsics = np.asarray(self.config.get('cam_intrinsics')).reshape(3, 3)
        fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
        cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
        s = 0.0
        self.K = gtsam.Cal3_S2(fx, fy, s, cx, cy)

        # 3. 外参 (单目 VO 优化的是 Camera Pose，故 Body_P_Sensor 为 Identity)
        self.body_T_cam = gtsam.Pose3(np.eye(4)) 

    def optimize(self, max_iterations=10):
        """
        执行局部光束法平差 (Local Bundle Adjustment)
        无需传入参数，直接从 self.map_manager 获取数据
        """
        # =========================================================================
        # Step 1: 从 MapManager 获取数据 (线程安全地获取副本或引用)
        # =========================================================================
        # 注意：这里获取的是此时此刻的快照
        keyframes = self.map_manager.get_active_keyframes()
        
        # 获取活跃点的 ID 和 3D 位置字典 {mp_id: np.array([x,y,z])}
        # 假设 MapManager.get_active_mappoints() 返回的是 {id: pos_3d}
        active_landmarks_dict = self.map_manager.get_active_mappoints()

        if len(keyframes) < 2:
            return False

        print(f"【Optimizer】: Starting BA with {len(keyframes)} KFs and {len(active_landmarks_dict)} LMs.")

        # =========================================================================
        # Step 2: 构建因子图 (构建因子图和初始估计)
        # =========================================================================
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        
        # 2.1 添加 KeyFrame 变量初始值
        for kf in keyframes:
            kf_id = kf.get_id()
            pose_matrix = kf.get_T_w_c() 
            initial_estimate.insert(X(kf_id), gtsam.Pose3(pose_matrix))

        # 2.2 添加 MapPoint 变量初始值
        # 我们只优化那些 "既在 active_landmarks_dict 中，又被 active keyframes 观测到" 的点
        points_added_to_graph = set()

        # 2.3 添加因子
        #   (A) 先验因子 (Fix Gauge): 固定窗口中的第一帧和第二帧
        first_kf = keyframes[0] 
        first_kf_id = first_kf.get_id()
        second_kf = keyframes[1]
        second_kf_id = second_kf.get_id()

        graph.add(gtsam.PriorFactorPose3(
            X(first_kf_id), 
            gtsam.Pose3(first_kf.get_T_w_c()), 
            self.prior_pose_noise
        ))

        graph.add(gtsam.PriorFactorPose3(
            X(second_kf_id), 
            gtsam.Pose3(second_kf.get_T_w_c()), 
            self.prior_pose_noise
        ))


        #   (B) 视觉重投影因子
        for kf in keyframes:
            kf_id = kf.get_id()
            
            # 获取该帧观测数据
            feat_ids = kf.get_visual_feature_ids()

            for mp_id in feat_ids:
                # 只有当该特征点对应一个已三角化的活跃路标时，才添加因子
                if mp_id in active_landmarks_dict:
                    
                    # 如果是第一次遇到这个点，添加它的初始值到 estimate
                    if mp_id not in points_added_to_graph:
                        initial_estimate.insert(L(mp_id), gtsam.Point3(active_landmarks_dict[mp_id]))
                        points_added_to_graph.add(mp_id)
                    
                    uv_unpx = kf.get_feature_undistorted_position(mp_id)
                    # 添加投影因子
                    measured = gtsam.Point2(uv_unpx[0], uv_unpx[1])
                    factor = gtsam.GenericProjectionFactorCal3_S2(
                        measured, 
                        self.robust_noise_model,
                        X(kf_id), 
                        L(mp_id), 
                        self.K, 
                        self.body_T_cam
                    )
                    graph.add(factor)



        # =========================================================================
        # [Debug] Pre-Optimization Sanity Check (优化前健全性检查)
        # =========================================================================
        # 目的：验证前端传入的 T_wc, Point3d 和 观测像素 是否在几何上自洽
        # 如果这里的误差很大，优化必然失败
        
        print(f"【Optimizer】: Running pre-optimization sanity check...")
        
        errors = []
        # 随机抽查 20 个点，而不是只查 1 个
        check_ids = list(points_added_to_graph)
        if len(check_ids) > 20:
            check_ids = np.random.choice(check_ids, 20, replace=False)
            
        for mp_id in check_ids:
            # 找到观测到该点的一帧 (为了方便，找最新的一帧)
            mp = self.map_manager.get_map_point(mp_id)
            if not mp: continue
            
            # 找一个在当前优化窗口内的观测帧
            obs_kf_id = None
            for kfid in mp.get_observing_kf_ids():
                if kfid in [kf.get_id() for kf in keyframes]:
                    obs_kf_id = kfid
                    break
            if obs_kf_id is None: continue
            
            # 开始投影
            check_kf = self.map_manager.get_keyframe(obs_kf_id)
            P_w = active_landmarks_dict[mp_id]
            T_wc = check_kf.get_T_w_c()
            obs_uv = check_kf.get_feature_undistorted_position(mp_id)
            
            # World -> Camera
            T_cw = np.linalg.inv(T_wc)
            P_c = T_cw[:3, :3] @ P_w + T_cw[:3, 3]
            
            if P_c[2] < 0.01: continue # 忽略这种明显错误的点
            
            # Project
            u_proj = self.config['cam_intrinsics'][0] * (P_c[0] / P_c[2]) + self.config['cam_intrinsics'][2]
            v_proj = self.config['cam_intrinsics'][4] * (P_c[1] / P_c[2]) + self.config['cam_intrinsics'][5]
            
            err = np.linalg.norm([u_proj - obs_uv[0], v_proj - obs_uv[1]])
            errors.append(err)

        if len(errors) > 0:
            median_error = np.median(errors)
            print(f"【Optimizer】: Sanity Check Median Error: {median_error:.2f} px (checked {len(errors)} points)")
            
            if median_error > 20.0:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"!!! CRITICAL: Median Error is too high ({median_error:.2f}) !!!")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # return False # 确实有问题，停止优化
            else:
                print(f"  > Sanity Check PASSED. (Outliers will be handled by Huber loss)")




        # =========================================================================
        # Step 3: 执行优化 (执行优化)
        # =========================================================================
        try:
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(max_iterations)
            params.setRelativeErrorTol(1e-5)
            params.setVerbosityLM("SUMMARY") 

            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
            result = optimizer.optimize()
            
        except Exception as e:
            print(f"【Optimizer】: Optimization Failed! {e}")
            return False

        # =========================================================================
        # Step 4: [新增] 外点检测 (Outlier Rejection)
        # =========================================================================
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
                    # 标记为外点
                    outlier_observations.append((kf_id, mp_id))
                    print(f"[Optimizer] Outlier detected: KF {kf_id} - MP {mp_id} (Chi2: {chi2:.2f}, Depth: {point_in_cam[2]:.2f})")

        print(f"【Optimizer】: Found {len(outlier_observations)} outliers.")

        # =========================================================================
        # Step 5: 提取结果并回写 (包含外点剔除)
        # =========================================================================
        optimized_poses = {}
        optimized_points = {}

        for kf in keyframes:
            kf_id = kf.get_id()
            if result.exists(X(kf_id)):
                optimized_poses[kf_id] = result.atPose3(X(kf_id)).matrix()

        for mp_id in points_added_to_graph:
            if result.exists(L(mp_id)):
                optimized_points[mp_id] = result.atPoint3(L(mp_id))

        # 调用 MapManager 回写 (传入 outliers)
        self.map_manager.update_map_from_optimization(optimized_poses, optimized_points, outlier_observations)
        
        return True