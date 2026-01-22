        valid_kps_ids = []
        valid_prev_kf_bvs = []  # 上一关键帧 Bearing Vectors
        valid_cur_bvs = [] # 当前帧 Bearing Vectors

        # 3. 视差检查 (Parallax Check)
        # 旋转补偿
        R_kf_prev_cur = prev_kf.get_T_c_w() @ self.cur_frame.get_T_w_c()
        
        avg_parallax = 0.0
        nb_parallax = 0

        # 获取当前帧所有特征点
        current_kps_ids = self.cur_frame.get_visual_feature_ids()
        
        for kp_id in current_kps_ids:
            # 查找上个关键帧一帧是否观测到了同一点
            prev_kf_bvs = prev_kf.get_feature_bearing(kp_id)
            prev_kf_unpx = prev_kf.get_feature_undistorted_position(kp_id)
            
            if prev_kf_bvs is None:
                continue

            # 收集数据 (用于后续 E 矩阵计算)
            # 注意：cv2.findEssentialMat 需要像素坐标或归一化坐标
            # 这里我们收集归一化平面坐标 (unpx)
            # kp.unpx 是 [x, y] 去畸变后的像素坐标，我们需要转为归一化坐标 x_n = (x-cx)/fx
            # 但 kp.bv 已经是归一化方向向量 [x, y, z]，直接用 x/z, y/z 即可
            
            # 存储相机归一化向量
            cur_kp_bv = self.cur_frame.get_feature_bearing(kp_id)
            valid_prev_kf_bvs.append(prev_kf_bvs)
            valid_cur_bvs.append(cur_kp_bv)
            valid_kps_ids.append(kp_id)

            # 计算旋转补偿后的视差
            # 将当前点旋转到上一帧坐标系，看与观测点的距离
            rot_bv = R_kf_prev_cur @ cur_kp_bv
            # 投影回上一帧的像素平面 (Project to Image)
            rot_px = prev_kf.camera.project_cam_to_image(rot_bv)
            
            dist = np.linalg.norm(rot_px - prev_kf_unpx) # unpx 是去畸变像素坐标
            avg_parallax += dist
            nb_parallax += 1

        # 计算平均视差
        avg_parallax /= nb_parallax

        # 如果视差太小，认为是对极几何退化 (Pure Rotation or Standstill)，跳过过滤
        if avg_parallax < 6.0:
            print(f"[VisualFrontEnd] Not enough parallax: {avg_parallax:.2f} px. Skipping.")
            return

        do_optimize = False
        # 单目情况下使用运动优化，在追踪情况差的时候使用
        if (self.map_manager.nbkfs > 2 and n_3d_prior < 30):
            print(f"[VisualFrontEnd] Using motion optimization for pose recovery n_3d_prior: {n_3d_prior}")
            do_optimize = True

        print(f"[VisualFrontEnd] 5-pt EssentialMatrix Ransac: {do_optimize}")
        print(f"[VisualFrontEnd] nb pts: {len(valid_kps_ids)}")
        print(f"[VisualFrontEnd] avg. parallax: {avg_parallax}")
        print(f"[VisualFrontEnd] nransac_iter_: {100}")
        print(f"[VisualFrontEnd] fransac_err_: {3.0}")
        print(f"[VisualFrontEnd] \n\n")

        # 计算本质矩阵 (RANSAC)
        success, R, t, outliers_idx = MultiViewGeometry.compute_5pt_essential_matrix(
            valid_prev_kf_bvs, valid_cur_bvs, 
            100,
            3.0,
            do_optimize,
            self.cur_frame.camera.fx,
            self.cur_frame.camera.fy,
        )

        print(f"[VisualFrontEnd] Epipolar nb outliers: {outliers_idx}")

        if not success:
            print(f"[VisualFrontEnd] No pose could be computed from 5-pt EssentialMatrix!")
            return

        if len(outliers_idx) > 0.5 * len(valid_kps_ids):
            print(f'Too many outliers, skipping as might be degenerate case')
            return

        # 剔除 Outliers - 批量删除，避免循环中多次重建索引
        if len(outliers_idx) > 0:
            ids_to_remove = np.array(valid_kps_ids)[outliers_idx]
            self.cur_frame.remove_features_by_ids(ids_to_remove)