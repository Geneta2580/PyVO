import cv2
import numpy as np
import gtsam

class MultiViewGeometry:
    @staticmethod
    def compute_5pt_essential_matrix(bvs1, bvs2, nmaxiter, errth, do_optimize, fx, fy):
        assert len(bvs1) == len(bvs2), "bvs1 and bvs2 must have the same length"

        # 1. 数据预处理：Bearing Vector (x,y,z) -> Normalized Plane (x/z, y/z)
        eps = 1e-7
        pts1 = bvs1[:, :2] / (bvs1[:, 2:3] + eps)
        pts2 = bvs2[:, :2] / (bvs2[:, 2:3] + eps)

        confidence = 0.99
        if do_optimize:
            confidence = 0.999

        # 2. 计算归一化后的阈值
        # errth 是像素单位，OpenCV findEssentialMat 如果输入归一化坐标，
        # threshold 也必须是归一化单位 (tan theta)
        f_avg = (fx + fy) / 2.0
        norm_threshold = errth / f_avg

        # 3. 计算本质矩阵 E (RANSAC)
        # prob=0.999 对应高置信度，OpenCV 会自动根据 prob 计算需要的迭代次数
        E, mask_e = cv2.findEssentialMat(
            pts1, pts2,
            focal=1.0, pp=(0.0, 0.0), # 已经是归一化坐标，所以内参设为单位阵
            method=cv2.RANSAC,
            prob=confidence,
            threshold=norm_threshold
        )

        if E is None or E.shape != (3, 3):
            print(f"[MultiViewGeometry] Failed to compute Essential Matrix! E: {E}")
            return False, np.eye(3), np.zeros(3), []

        # 立即提取外点mask
        mask_e = mask_e.flatten()
        outliers_idx = np.where(mask_e == 0)[0].tolist()

        if len(outliers_idx) >= len(pts1) - 5: 
            return False, np.eye(3), np.zeros(3), []

        # 4. 恢复位姿 R, t (Cheirality Check)
        # recoverPose 会分解 E 矩阵，并三角化点云，选择使点都在相机前方的那个解
        # 注意：这里传入 mask_e，recoverPose 不更新mask（交给三角化检查以及PnP检查处理）
        _, R, t, _ = cv2.recoverPose(
            E, pts1, pts2,
            focal=1.0, pp=(0.0, 0.0),
            mask=mask_e
        )

        # R, t 是从 Frame 1 到 Frame 2 的变换: x2 = R * x1 + t  (R_21)
        R_cv = R
        t_cv = t.flatten()

        # (R_21) 转换为 (R_12)
        R_out = R_cv.T
        t_out = -R_cv.T @ t_cv

        return True, R_out, t_out, outliers_idx
    
    @staticmethod
    def p3p_ransac(
        bvs: np.ndarray,       # (N, 3) 归一化方向向量 [x, y, z]
        vwpts: np.ndarray,     # (N, 3) 世界坐标 [X, Y, Z]
        nmaxiter: int = 100,
        errth: float = 3.0,    # 像素误差阈值
        fx: float = 1.0, fy: float = 1.0,
        boptimize: bool = True
    ) :
        """
        使用 OpenCV solvePnPRansac 求解位姿
        返回: (success, Pose3_wc, outliers_idx)
        """
        if len(bvs) < 4:
            print(f"[MultiViewGeometry] P3P RANSAC Failed: Not enough points! len(bvs): {len(bvs)}")
            return False, np.zeros((4, 4)), []

        # 1. 数据准备
        # 将 Bearing Vectors 转换为归一化平面齐次坐标 (x/z, y/z)
        # 注意：这里假设内参 K 为单位阵，因为我们用的是归一化坐标
        eps = 1e-7
        z = bvs[:, 2:3]
        z[np.abs(z) < eps] = eps
        image_points = bvs[:, :2] / z # (N, 2)
        
        object_points = vwpts # (N, 3)

        # 2. 参数设置
        K = np.eye(3, dtype=np.float64)
        dist_coeffs = np.zeros(4) # 无畸变
        
        # 阈值归一化 (errth 是像素单位)
        focal_avg = (fx + fy) / 2.0
        reproj_threshold = errth / focal_avg

        # 3. P3P RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, K, dist_coeffs,
            iterationsCount=nmaxiter,
            reprojectionError=reproj_threshold,
            confidence=0.99,
            flags=cv2.SOLVEPNP_P3P
        )

        if not success or inliers is None or len(inliers) < 5:
            print(f"[MultiViewGeometry] P3P RANSAC Failed: Not enough inliers (< 5)!")
            return False, np.zeros((4, 4)), []

        # 4. 提取外点索引
        inliers = inliers.flatten()
        all_indices = np.arange(len(bvs))
        outliers_idx = np.setdiff1d(all_indices, inliers).tolist()

        # 5. 可选：非线性优化 (Refinement)
        if boptimize:
            # 仅使用内点进行优化
            # useExtrinsicGuess=True
            success, rvec, tvec = cv2.solvePnP(
                object_points[inliers], image_points[inliers], K, dist_coeffs,
                rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        # 6. 坐标系转换 (OpenCV -> GTSAM/World)
        # OpenCV rvec, tvec 是 T_cw (World -> Camera)
        # P_c = R * P_w + t
        R_cw, _ = cv2.Rodrigues(rvec)
        t_cw = tvec.flatten()
        
        # 转换为 T_wc (Camera -> World)
        # T_wc = T_cw.inverse()
        # R_wc = R_cw.T
        # t_wc = -R_cw.T * t_cw
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        
        pose_wc = np.eye(4)
        pose_wc[:3, :3] = R_wc
        pose_wc[:3, 3] = t_wc

        return True, pose_wc, outliers_idx

    @staticmethod
    def gtsam_pnp(
        vunkps: np.ndarray,    # (N, 2) 归一化平面坐标
        vwpts: np.ndarray,     # (N, 3) 3D 世界点
        initial_pose_wc: np.ndarray, # 初始位姿 T_wc
        nmaxiter: int = 10,
        chi2th: float = 5.9915,  # Chi-square 阈值
        use_robust: bool = True,
        fx: float = 1.0, fy: float = 1.0
    ):
        """
        使用 GTSAM 实现 PnP (Motion-only BA)
        替代 C++ 中的 ceresPnP
        通过对 3D 点添加强约束(Prior)来实现固定点、只优化位姿
        """
        if len(vunkps) != len(vwpts):
            print(f"[GTSAM PnP] vunkps and vwpts must have the same length")
            return False, np.zeros((4, 4)), []

        # 1. 构建因子图
        graph = gtsam.NonlinearFactorGraph()
        
        # 噪声模型 (Pixel noise)
        # 因为输入是归一化坐标，噪声 sigma 需要归一化：1.0 pixel / fx
        pixel_sigma = 1.0 / ((fx + fy) / 2.0)
        
        if use_robust:
            obs_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(1.345), # Huber k
                gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)
            )
        else:
            obs_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)

        # 3D点固定噪声
        point_fixed_noise = gtsam.noiseModel.Constrained.All(3)

        # 相机内参 (Cal3_S2)
        # 输入是归一化坐标，所以内参设为 (1, 1, 0, 0, 0)
        calib = gtsam.Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0)
        
        # 初始值
        initial_estimate = gtsam.Values()

        # 加入位姿初始值
        pose_key = gtsam.symbol('x', 0)
        initial_estimate.insert(pose_key, gtsam.Pose3(initial_pose_wc))
        
        for i in range(len(vunkps)):
            point_key = gtsam.symbol('l', i)

            # 观测数据
            measured = vunkps[i]
            point3d = vwpts[i]
            
            # 添加投影因子 (连接 Pose 和 Point)
            factor = gtsam.GenericProjectionFactorCal3_S2(
                measured, obs_noise, pose_key, point_key, calib
            )
            graph.add(factor)

            # 添加3D点固定因子
            graph.add(gtsam.PriorFactorPoint3(point_key, point3d, point_fixed_noise))

            # 将点的初始值加入estimate
            initial_estimate.insert(point_key, point3d)

        # 优化器配置
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(nmaxiter)
        params.setRelativeErrorTol(1e-5)
        # params.setVerbosityLM("SUMMARY") # Debug用
    
        try:
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
            result = optimizer.optimize()
            optimized_pose = result.atPose3(pose_key)
        except Exception as e:
            print(f"[GTSAM PnP] Optimization failed: {e}")
            return False, np.zeros((4, 4)), []

        # 4. 外点剔除 (根据 Chi2 误差)
        outliers_idx = []
        # 计算每个因子的误差      
        for i in range(len(vunkps)):
            # 或者，因为我们按顺序添加了 N 个 ProjectionFactor 和 N 个 PriorFactor
            # ProjectionFactor 的索引是 0, 2, 4 ... (偶数)
            factor_idx = i * 2 
            factor = graph.at(factor_idx)

            if not isinstance(factor, gtsam.GenericProjectionFactorCal3_S2): 
                print(f"[GTSAM PnP] Factor {i} is not a GenericProjectionFactorCal3_S2")
                continue

            # error 返回的是 0.5 * (z-h(x))^T * Cov^-1 * (z-h(x))
            # Chi2 error 通常是 2 * error
            err = factor.error(result) * 2.0 
            
            if err > chi2th:
                outliers_idx.append(i)

        return True, optimized_pose.matrix(), outliers_idx

    @staticmethod
    def triangulate_points(
        T_w_c1: np.ndarray,      # 第一个关键帧的位姿 T_wc (4x4)
        T_w_c2: np.ndarray,      # 第二个关键帧的位姿 T_wc (4x4)
        bvs1: np.ndarray,        # 第一个关键帧的 bearing vectors (N, 3)
        bvs2: np.ndarray,        # 第二个关键帧的 bearing vectors (N, 3)
    ):
        """
        使用两个关键帧的位姿和 bearing vectors 进行三角化
        返回: (success, points_3d)
        points_3d: (N, 3) 世界坐标系下的3D点
        """
        if len(bvs1) != len(bvs2) or len(bvs1) == 0:
            return False, np.empty((0, 3))

        # 1. 计算投影矩阵 P1, P2
        # P = K * [R|t] = K * T_cw (从世界到相机的变换)
        T_c1_w = np.linalg.inv(T_w_c1)
        T_c2_w = np.linalg.inv(T_w_c2)
        
        P1 = T_c1_w[:3, :]
        P2 = T_c2_w[:3, :]

        # 2. 将 bearing vectors 转换为归一化平面坐标 (用于三角化)
        # bearing vector [x, y, z] -> 归一化坐标 [x/z, y/z]
        eps = 1e-7
        z1 = bvs1[:, 2:3]
        z1[np.abs(z1) < eps] = eps
        pts1_norm = bvs1[:, :2] / z1  # (N, 2)

        z2 = bvs2[:, 2:3]
        z2[np.abs(z2) < eps] = eps
        pts2_norm = bvs2[:, :2] / z2  # (N, 2)

        # 3. 三角化
        # OpenCV triangulatePoints 需要 (2, N) 格式
        pts1_2d = pts1_norm.T  # (2, N)
        pts2_2d = pts2_norm.T  # (2, N)

        # 三角化
        points_4d = cv2.triangulatePoints(P1, P2, pts1_2d, pts2_2d)  # (4, N)

        # 4. 转换为3D点 (齐次坐标 -> 3D坐标)
        points_3d = (points_4d[:3, :] / (points_4d[3, :] + eps)).T  # (N, 3)

        return True, points_3d