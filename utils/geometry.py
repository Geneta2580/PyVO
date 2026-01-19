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

        # 4. 恢复位姿 R, t (Cheirality Check)
        # recoverPose 会分解 E 矩阵，并三角化点云，选择使点都在相机前方的那个解
        # 注意：这里传入 mask_e，recoverPose 会进一步更新 mask（剔除位于相机后方的点）
        inliers_count, R, t, mask_pose = cv2.recoverPose(
            E, pts1, pts2,
            focal=1.0, pp=(0.0, 0.0),
            mask=mask_e
        )

        if inliers_count < 5: # 5点法至少需要5个内点
            print(f"[MultiViewGeometry] Failed to compute Pose! inliers_count: {inliers_count}")
            return False, np.eye(3), np.zeros(3), []

        # 5. 提取外点索引
        # mask_pose: (N, 1) uint8, 0=outlier, 255=inlier
        mask_pose = mask_pose.flatten()
        outliers_idx = np.where(mask_pose == 0)[0].tolist()

        # R, t 是从 Frame 1 到 Frame 2 的变换: x2 = R * x1 + t
        return True, R, t.flatten(), outliers_idx
    
    @staticmethod
    def opencv_p3p_ransac(
        bvs: np.ndarray,       # (N, 3) 归一化方向向量 [x, y, z]
        vwpts: np.ndarray,     # (N, 3) 世界坐标 [X, Y, Z]
        nmaxiter: int = 100,
        errth: float = 1.0,    # 像素误差阈值
        fx: float = 1.0, fy: float = 1.0,
        boptimize: bool = True
    ) -> Tuple[bool, gtsam.Pose3, List[int]]:
        """
        使用 OpenCV solvePnPRansac 求解位姿
        返回: (success, Pose3_wc, outliers_idx)
        """
        if len(bvs) < 4:
            return False, gtsam.Pose3(), []

        # 1. 数据准备
        # 将 Bearing Vectors 转换为归一化平面坐标 (x/z, y/z)
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
            return False, gtsam.Pose3(), []

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
        
        pose_wc = gtsam.Pose3(gtsam.Rot3(R_wc), gtsam.Point3(t_wc))

        return True, pose_wc, outliers_idx

    @staticmethod
    def gtsam_pnp(
        vunkps: np.ndarray,    # (N, 2) 归一化平面坐标
        vwpts: np.ndarray,     # (N, 3) 3D 世界点
        initial_pose_wc: gtsam.Pose3, # 初始位姿 T_wc
        nmaxiter: int = 10,
        chi2th: float = 5.99,  # Chi-square 阈值
        buse_robust: bool = True,
        fx: float = 1.0, fy: float = 1.0
    ) -> Tuple[bool, gtsam.Pose3, List[int]]:
        """
        使用 GTSAM 实现 PnP (Motion-only BA)
        替代 C++ 中的 ceresPnP
        """
        if len(vunkps) != len(vwpts):
            return False, initial_pose_wc, []

        # 1. 构建因子图
        graph = gtsam.NonlinearFactorGraph()
        
        # 噪声模型 (Pixel noise)
        # 因为输入是归一化坐标，噪声 sigma 需要归一化：1.0 pixel / fx
        pixel_sigma = 1.0 / ((fx + fy) / 2.0)
        
        if buse_robust:
            # Huber 核函数 (Robust Cost Function)
            # GTSAM 的 robust noise model
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(1.345), # Huber k
                gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)
            )
        else:
            noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)

        # 相机内参 (Cal3_S2)
        # 输入是归一化坐标，所以内参设为 (1, 1, 0, 0, 0)
        # 也可以直接用 GenericProjectionFactor 配合真实 K，但这里为了匹配 C++ 逻辑
        # C++ 中是把去畸变后的点投影误差作为残差。
        calib = gtsam.Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0)
        
        # 待优化变量 Key
        pose_key = gtsam.symbol('x', 0)

        for i in range(len(vunkps)):
            # 注意：GenericProjectionFactor 也是 T_wc (Pose of camera in world)
            # 观测点 (u, v)
            measured = gtsam.Point2(vunkps[i][0], vunkps[i][1])
            # 3D 点 (常数)
            point3d = gtsam.Point3(vwpts[i])
            
            factor = gtsam.GenericProjectionFactorCal3_S2(
                measured, noise, pose_key, point3d, calib
            )
            graph.add(factor)

        # 2. 初始值
        initial_estimate = gtsam.Values()
        initial_estimate.insert(pose_key, initial_pose_wc)

        # 3. 优化器配置
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(nmaxiter)
        params.setRelativeErrorTol(1e-5)
        # params.setVerbosityLM("SUMMARY") # Debug用

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        
        try:
            result = optimizer.optimize()
            optimized_pose = result.atPose3(pose_key)
        except Exception as e:
            print(f"[GTSAM PnP] Optimization failed: {e}")
            return False, initial_pose_wc, []

        # 4. 外点剔除 (根据 Chi2 误差)
        outliers_idx = []
        # 计算每个因子的误差
        # graph.error(result) 返回总误差
        # 我们需要遍历因子计算单个误差
        
        for i in range(graph.size()):
            factor = graph.at(i)
            # error 返回的是 0.5 * (z-h(x))^T * Cov^-1 * (z-h(x))
            # Chi2 error 通常是 2 * error
            err = factor.error(result) * 2.0 
            
            # C++ 中 chi2th 是直接传进来的 (比如 5.99 对应 95% 2DOF)
            if err > chi2th:
                outliers_idx.append(i)

        return True, optimized_pose, outliers_idx