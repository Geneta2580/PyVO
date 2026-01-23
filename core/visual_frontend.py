import time
import numpy as np
import cv2
import gtsam
from utils.geometry import MultiViewGeometry
from utils.visualization import visualize_optical_flow_tracking, visualize_epipolar_filtered_tracking
from datatype.mappoint import MapPointStatus
from utils.debug import Debugger

# 运动模型
class MotionModel:
    def __init__(self, config):
        self.config = config
        self.prev_time = -1
        self.prev_T_wc = None  # 存储为 gtsam.Pose3 对象
        # 存储速度矢量 (6x1: rotation + translation)
        self.log_rel_T_wc = np.zeros(6)
        self.traj_file = Debugger.initialize_trajectory_file("./output/trajectory_tum.txt")

    def apply_motion_model(self, T_wc_np, time):
        T_wc = gtsam.Pose3(T_wc_np)
        T_wc_pred = T_wc

        if self.prev_time > 0:
            dt = time - self.prev_time
            
            # 恒速模型预测：T_pred = T_curr * Expmap(velocity * dt)
            delta_pose = gtsam.Pose3.Expmap(self.log_rel_T_wc * dt)
            T_wc_pred = T_wc.compose(delta_pose)
            print(f"[MotionModel] apply_motion_model: {T_wc_pred.matrix()}")

        return T_wc_pred.matrix()

    def update_motion_model(self, T_wc_np, time):
        current_T_wc = gtsam.Pose3(T_wc_np)
        Debugger.log_pose_tum(self.traj_file, time, current_T_wc)
        if self.prev_time > 0:
            dt = time - self.prev_time
            if dt > 1e-6:
                # 计算相对变换：prev_T^{-1} * curr_T (从上一帧到当前帧的增量)
                print(f"[MotionModel] prev_T_wc: {self.prev_T_wc.matrix()}")
                print(f"[MotionModel] current_T_wc: {current_T_wc.matrix()}")
                relative_pose = self.prev_T_wc.between(current_T_wc)
                # v = Logmap(T_rel) / dt
                print(f"[MotionModel] Update relative_pose: {relative_pose.matrix()}")
                self.log_rel_T_wc = gtsam.Pose3.Logmap(relative_pose) / dt
                print(f"[MotionModel] log_rel_T_wc: {self.log_rel_T_wc}")

        # 更新历史状态
        self.prev_time = time
        self.prev_T_wc = current_T_wc


class VisualFrontend:
    def __init__(self, config, prev_frame, cur_frame, map_manager, feature_tracker, feature_extractor, mapper=None):
        self.config = config
        self.prev_frame = prev_frame
        self.cur_frame = cur_frame

        # 调用
        self.map_manager = map_manager
        self.feature_tracker = feature_tracker
        self.feature_extractor = feature_extractor
        self.mapper = mapper

        # 运动模型（恒速模型）
        self.motion_model = MotionModel(config)

        # 是否完成视觉初始化
        self.visual_init_ready = False

        # 初始化CLAHE参数
        tile_size = 50
        self.nbwtiles = self.cur_frame.camera.img_w / tile_size
        self.nbhtiles = self.cur_frame.camera.img_h / tile_size
        self.clahe = cv2.createCLAHE(clipLimit=config['clahe_value'], tileGridSize=(int(self.nbwtiles), int(self.nbhtiles)))

        # 预处理的灰度图
        self.prev_gray = None

        # 是否使用P3P
        self.do_p3p = False
        
        # 可视化选项
        self.visualize_optical_flow = config.get('visualize_optical_flow', False)
        
        # 存储追踪数据用于可视化
        self.tracking_prev_pts = None
        self.tracking_tracked_pts = None
        self.tracking_final_status = None
        self.tracking_kp_ids = None  # 保存特征点ID用于内点映射

        # 日志
        log_columns = [
            "id",
            "ref_kf_id",
            "visual_tracking_constant_vel_prior_valid_3d",
            "visual_tracking_3d_tracked_count",
            "visual_tracking_3d_tracked_ratio",
            "visual_tracking_2d_tracked_count",
            "visual_tracking_2d_tracked_ratio",
            "epipolar_filtering_avg_parallax",
            "epipolar_filtering_success",
            "epipolar_filtering_outliers_count",
            "epipolar_filtering_outliers_count_ratio",
            "p3p_ransac_success",
            "p3p_ransac_inliers_count",
            "p3p_ransac_inliers_count_ratio",
            "gtsam_pnp_inliers_count",
            "gtsam_pnp_inliers_count_ratio",
            "check_new_keyframe_is_kf_required",
            "check_new_keyframe_med_rot_parallax",
            "check_new_keyframe_n_frames_since_kf",
            "check_new_keyframe_n_visual_features_num",
        ] 
        self.logger = Debugger(config, file_prefix="visual_frontend", column_names=log_columns)

    def reset(self):
        """
        重置视觉前端状态，用于初始化失败时的重置
        """
        print(f"[VisualFrontend] Resetting visual frontend state")
        self.visual_init_ready = False
        self.prev_frame = None
        self.prev_gray = None
        self.do_p3p = False
        
        # 重置运动模型
        self.motion_model.prev_time = -1
        self.motion_model.prev_T_wc = None
        self.motion_model.log_rel_T_wc = np.zeros(6)
        
        print(f"[VisualFrontend] Reset complete")

    # def reset_frame(self):
    #     self.cur_frame.

    def visual_tracking(self, prev_frame, cur_frame, timestamp):
        print(f"[VisualFrontend] Mono tracking: {timestamp}")

        self.prev_frame = prev_frame
        self.cur_frame = cur_frame

        # 预处理图像
        curr_gray = self.preprocess_image(cur_frame.image)

        # 第一帧直接作为关键帧（ID为0或prev_frame为None都认为是第一帧）
        if self.prev_frame is None:
            print(f"[VisualFrontend] First frame, ready for keyframe creation...")
            # 第一帧作为关键帧，ref_kf_id 设置为自己的ID
            self.cur_frame.ref_kf_id = self.cur_frame.get_id()

            # 第一帧更新运动模型（单位阵，时间戳）
            self.motion_model.update_motion_model(self.cur_frame.get_T_w_c(), timestamp)

            self.prev_frame = self.cur_frame
            self.prev_gray = curr_gray
            return True, curr_gray

        # 设置参考关键帧：普通帧继承上一帧的参考关键帧
        if self.prev_frame is not None:
            self.cur_frame.ref_kf_id = self.prev_frame.ref_kf_id
        else:
            assert False, "[VisualFrontend] [Error]: Previous frame should not be None"

        # 预测新帧位姿
        # 应该基于上一帧的位姿进行预测，而不是新帧的初始位姿（单位矩阵）
        if self.prev_frame is not None:
            T_wc_prev = self.prev_frame.get_T_w_c()
        else:
            assert False, "[VisualFrontend] [Error]: Previous frame should not be None"
        
        T_wc_pred = self.motion_model.apply_motion_model(T_wc_prev, timestamp)
        self.cur_frame.set_T_w_c(T_wc_pred)

        # 追踪新帧
        n_tracked_valid_3d = self.KLT_tracking(self.prev_gray, curr_gray)

        # 对极约束去除外点（有必要则使用E矩阵更新位姿，否则直接使用运动模型预测的位姿）
        self.epipolar_filtering(n_tracked_valid_3d)

        # 检查视觉初始化准备是否完成
        if not self.visual_init_ready:
            # 经过光流追踪和对极约束去除外点后，特征点数量仍然较少，无法进行视觉初始化
            if(len(self.cur_frame.get_visual_feature_ids()) < 50):
                return False, curr_gray
            elif self.check_ready_for_init():
                print(f'[VisualFrontend] Ready for visual initialization!')
                self.visual_init_ready = True

                # 第二帧更新运动模型（初始位姿，时间戳）
                self.motion_model.update_motion_model(self.cur_frame.get_T_w_c(), timestamp)
                return True, curr_gray # 直接返回True，不进行后续处理
            else:
                print(f'[VisualFrontend] Not ready for visual initialization!')
                return False, curr_gray

        # 计算位姿
        self.compute_pose(n_tracked_valid_3d)

        # 更新运动模型
        self.motion_model.update_motion_model(self.cur_frame.get_T_w_c(), timestamp)

        # 检查是否为新关键帧（只判断，不创建）
        is_new_kf = self.check_new_keyframe(n_tracked_valid_3d)

        # 更新
        self.prev_frame = self.cur_frame
        self.prev_gray = curr_gray

        # 记录日志
        self.logger.finish_frame(timestamp)

        # 返回关键帧判断结果和灰度图（用于后续关键帧创建）
        return is_new_kf, curr_gray

    def preprocess_image(self, cur_image):
        print(f"[VisualFrontend] Preprocessing image")
        curr_gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)
        # 使用直方图均衡化增强图像对比度(注意opencv要求)
        clahe_curr_image = self.clahe.apply(curr_gray)
        curr_gray = clahe_curr_image

        return curr_gray
        
    def KLT_tracking(self, prev_gray, curr_gray):
        print(f"[VisualFrontend] KLT Tracking")
        # 追踪3d点2层/追踪2d点全层
        # 准备数据容器
        valid_3d_kp_ids = []
        valid_3d_kps_px = [] # 像素坐标
        valid_3d_priors = [] # 预测坐标
        
        valid_2d_kp_ids = []
        valid_2d_kps_px = []
        valid_2d_priors = []        

        prev_kps_ids = self.prev_frame.get_visual_feature_ids()
        prev_kps_features = self.prev_frame.get_visual_features()

        for kp_id, kp_px in zip(prev_kps_ids, prev_kps_features):
            map_point = self.map_manager.get_map_point(kp_id)
            flat_kp = kp_px.flatten() # (1, 2) -> (2,)
            if map_point is not None and map_point.status == MapPointStatus.TRIANGULATED:
                # 投影: World -> Image (Distorted) 这里的 T_cw是基于恒速模型预测的位姿
                proj_px = self.cur_frame.camera.project_world_to_image(self.cur_frame.get_T_w_c(), map_point.get_point())

                # 检查投影点是否在图像内（深度有可能为负，或者超出图像范围）
                if self.cur_frame.camera.is_in_image(proj_px):
                    valid_3d_kps_px.append(flat_kp)
                    valid_3d_priors.append(proj_px.flatten())
                    valid_3d_kp_ids.append(kp_id)
                    continue # 找到一个先验即可

                # 特征点不在图像内，加入到普通列表，稍后尝试全金字塔跟踪
                else:
                    valid_2d_kps_px.append(flat_kp)
                    valid_2d_priors.append(flat_kp) # 使用上一帧对应点的像素坐标作为预测坐标
                    valid_2d_kp_ids.append(kp_id)

            # 特征点不在地图中，加入到普通列表，稍后尝试全金字塔跟踪
            else:
                valid_2d_kps_px.append(flat_kp)
                valid_2d_priors.append(flat_kp)
                valid_2d_kp_ids.append(kp_id)

        print(f"[VisualFrontend KLT Tracking] Valid 3D points: {len(valid_3d_kp_ids)}")
        self.logger.log_flexible(self.cur_frame.timestamp, "visual_tracking_constant_vel_prior_valid_3d", len(valid_3d_kp_ids))

        # ---------------------------------------------------------
        # Step 1: 跟踪带有先验的 3D 点
        # ---------------------------------------------------------
        # 用于可视化的数据收集
        all_prev_pts = []
        all_tracked_pts = []
        all_final_status = []
        all_kp_ids = []  # 保存特征点ID用于后续内点映射
        nb_good_3d = 0 # 跟踪到的 3D 点数量
        
        if len(valid_3d_kp_ids) > 0:
            # 在传入追踪器之前，将 list 转为 numpy array
            np_3d_kps = np.array(valid_3d_kps_px, dtype=np.float32).reshape(-1, 2)
            np_3d_priors = np.array(valid_3d_priors, dtype=np.float32).reshape(-1, 2)

            # 3D 点通常只在较小的金字塔层级上跟踪(代码中 nbpyrlvl=1)，因为预测比较准
            pyramid_levels = 1
            tracked_pts, status = self.feature_tracker.fb_klt_tracking(
                prev_gray, curr_gray,
                np_3d_kps,
                np_3d_priors,
                pyramid_levels,
            )

            # 收集3D点追踪数据用于可视化
            all_prev_pts.extend(valid_3d_kps_px)
            all_tracked_pts.extend(tracked_pts)
            all_final_status.extend(status)
            all_kp_ids.extend(valid_3d_kp_ids)

            # 处理跟踪结果
            tracked_3d_list = []
            ids_3d_list = []
            ages_3d_list = []

            for i, is_good in enumerate(status):
                kp_id = valid_3d_kp_ids[i]
                if is_good:
                    # 添加跟踪成功点的的位置/id/age
                    # 从上一帧获取 age，如果上一帧也没有（新特征点），age 从 1 开始
                    prev_age = self.prev_frame.get_feature_age(kp_id) if self.prev_frame else None
                    update_age = (prev_age + 1) if prev_age is not None else 1

                    tracked_3d_list.append(tracked_pts[i])
                    ids_3d_list.append(kp_id)
                    ages_3d_list.append(update_age)
                    nb_good_3d += 1
                else:
                    # 跟踪失败：降级处理，加入到普通列表，稍后尝试全金字塔跟踪
                    valid_2d_kps_px.append(valid_3d_kps_px[i]) # 使用原始像素
                    valid_2d_priors.append(valid_3d_priors[i])
                    valid_2d_kp_ids.append(kp_id)

            if tracked_3d_list:
                self.cur_frame.add_visual_features(
                    np.array(tracked_3d_list), 
                    np.array(ids_3d_list), 
                    np.array(ages_3d_list)
                )

            print(f"[VisualFrontend KLT Tracking] Tracked {nb_good_3d} / {len(valid_3d_kps_px)} 3D points")
            self.logger.log_flexible(self.cur_frame.timestamp, "visual_tracking_3d_tracked_count", nb_good_3d)
            self.logger.log_flexible(self.cur_frame.timestamp, "visual_tracking_3d_tracked_ratio", nb_good_3d / len(valid_3d_kps_px))

            # 如果3D点跟踪数量不足，则使用P3P进行位姿优化，同时所有2D点都使用上一帧对应点的像素坐标作为预测坐标
            if nb_good_3d < 0.33 * len(valid_3d_kps_px):
                self.do_p3p = True
                valid_2d_priors = valid_2d_kps_px

        # ---------------------------------------------------------
        # Step 2: 跟踪未带有先验的 2D 点
        # ---------------------------------------------------------
        # 这里似乎使用3层金字塔跟踪
        nb_good_2d = 0 # 跟踪到的 2D 点数量
        if len(valid_2d_kps_px) > 0:
            np_2d_kps = np.array(valid_2d_kps_px, dtype=np.float32).reshape(-1, 2)
            np_2d_priors = np.array(valid_2d_priors, dtype=np.float32).reshape(-1, 2)
            tracked_pts, status = self.feature_tracker.fb_klt_tracking(
                prev_gray, curr_gray,
                np_2d_kps,
                np_2d_priors,
                3,
            )

            # 收集2D点追踪数据用于可视化
            all_prev_pts.extend(np_2d_kps)
            all_tracked_pts.extend(tracked_pts)
            all_final_status.extend(status)
            all_kp_ids.extend(valid_2d_kp_ids)

            tracked_2d_list = []
            ids_2d_list = []
            ages_2d_list = []
            for i, is_good in enumerate(status):
                kp_id = valid_2d_kp_ids[i]
                if is_good:
                    # 从上一帧获取 age，如果上一帧也没有（新特征点），age 从 1 开始
                    prev_age = self.prev_frame.get_feature_age(kp_id) if self.prev_frame else None
                    update_age = (prev_age + 1) if prev_age is not None else 1

                    tracked_2d_list.append(tracked_pts[i])
                    ids_2d_list.append(kp_id)
                    ages_2d_list.append(update_age)
                    nb_good_2d += 1
                else:
                    # 不添加跟踪失败点的的位置/id/age
                    continue
            
            if tracked_2d_list:
                self.cur_frame.add_visual_features(
                    np.array(tracked_2d_list), 
                    np.array(ids_2d_list), 
                    np.array(ages_2d_list)
                )

            print(f"[VisualFrontend KLT Tracking] no prior: {nb_good_2d} / {len(valid_2d_kps_px)} points.")
            self.logger.log_flexible(self.cur_frame.timestamp, "visual_tracking_2d_tracked_count", nb_good_2d)
            self.logger.log_flexible(self.cur_frame.timestamp, "visual_tracking_2d_tracked_ratio", nb_good_2d / len(valid_2d_kps_px))
        
        # 可视化光流追踪结果（追踪后，对极约束前）
        if self.visualize_optical_flow and len(all_prev_pts) > 0:
            all_prev_pts = np.array(all_prev_pts, dtype=np.float32)
            all_tracked_pts = np.array(all_tracked_pts, dtype=np.float32)
            all_final_status = np.array(all_final_status, dtype=bool)
            
            # 转换为BGR图像用于可视化
            curr_image_bgr = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR) if len(curr_gray.shape) == 2 else curr_gray
            
            vis_img = visualize_optical_flow_tracking(
                curr_image_bgr,
                all_prev_pts, all_tracked_pts, all_final_status,
                inliers_mask=None,
                window_name="KLT Tracking (Before Epipolar Filtering)",
                show_stats=True,
                frame_id=self.cur_frame.get_id(),
                kf_id=self.cur_frame.ref_kf_id
            )
            cv2.imshow("KLT Tracking", vis_img)
            cv2.waitKey(1)  # 非阻塞显示
            
            # 保存追踪数据用于对极约束后的可视化
            self.tracking_prev_pts = all_prev_pts
            self.tracking_tracked_pts = all_tracked_pts
            self.tracking_final_status = all_final_status
            self.tracking_kp_ids = all_kp_ids  # 保存ID用于内点映射
        
        return nb_good_3d
    
    def epipolar_filtering(self, n_tracked_valid_3d):
        """
        执行 2D-2D 对极几何剔除异常点
        """
        print("[VisualFrontEnd Epipolar Filtering] Starting Epipolar Filtering...")

        # 1. 获取参考关键帧 (Reference KeyFrame)
        # 使用当前帧的 ref_kf_id，因为当前帧已经绑定了参考关键帧
        ref_kf = self.map_manager.get_keyframe(self.cur_frame.ref_kf_id)

        if ref_kf is None:
            print(f"[VisualFrontEnd Epipolar Filtering] Error: Reference KeyFrame not found! ref_kf_id: {self.cur_frame.ref_kf_id}")
            return

        n_kps = len(ref_kf.get_visual_feature_ids())
        if n_kps < 8:
            print(f"[VisualFrontEnd Epipolar Filtering] Error: Not enough kps to compute Essential Matrix! n_kps: {n_kps} < 8")
            return

        # 2. 准备数据容器
        valid_kps_ids = []
        valid_prev_kf_bvs = []  # 上一关键帧 Bearing Vectors
        valid_cur_bvs = [] # 当前帧 Bearing Vectors

        # 3. 视差检查 (Parallax Check)
        # 旋转补偿
        R_ref_kf = ref_kf.get_T_c_w()[:3, :3]
        R_cur = self.cur_frame.get_T_w_c()[:3, :3]
        R_kf_prev_cur = R_ref_kf @ R_cur
        print(f'[VisualFrontEnd Epipolar Filtering] R_kf_prev_cur: {R_kf_prev_cur}')
        
        avg_parallax = 0.0
        nb_parallax = 0

        # 获取当前帧所有特征点
        current_kps_ids = self.cur_frame.get_visual_feature_ids()
        
        for kp_id in current_kps_ids:
            # 查找参考关键帧是否观测到了同一点
            ref_kf_bvs = ref_kf.get_feature_bearing(kp_id)
            ref_kf_unpx = ref_kf.get_feature_undistorted_position(kp_id)
            
            if ref_kf_bvs is None:
                print(f"[VisualFrontEnd Epipolar Filtering] Error: Reference KeyFrame not found! kp_id: {kp_id}")
                continue

            # 收集数据 (用于后续 E 矩阵计算)
            # 注意：cv2.findEssentialMat 需要像素坐标或归一化坐标
            # 这里我们收集归一化平面坐标 (unpx)
            # 但 kp.bv 已经是归一化方向向量 [x, y, z]，直接用 x/z, y/z 即可
            
            # 存储相机归一化向量
            cur_kp_bv = self.cur_frame.get_feature_bearing(kp_id)
            valid_prev_kf_bvs.append(ref_kf_bvs)
            valid_cur_bvs.append(cur_kp_bv)
            valid_kps_ids.append(kp_id)

            # 计算旋转补偿后的视差
            # 将当前点旋转到参考关键帧坐标系，看与观测点的距离
            rot_bv = R_kf_prev_cur @ cur_kp_bv
            # 投影回参考关键帧的像素平面 (Project to Image)
            rot_px = ref_kf.camera.project_cam_to_image(rot_bv)
            
            dist = np.linalg.norm(rot_px - ref_kf_unpx) # unpx 是去畸变像素坐标
            avg_parallax += dist
            nb_parallax += 1

        # 计算平均视差
        avg_parallax /= nb_parallax
        self.logger.log_flexible(self.cur_frame.timestamp, "epipolar_filtering_avg_parallax", avg_parallax)

        # 如果视差太小（2*3.0），认为是对极几何退化 (Pure Rotation or Standstill)，跳过过滤
        if avg_parallax < 6.0:
            print(f"[VisualFrontEnd Epipolar Filtering] Not enough parallax: {avg_parallax:.2f} px. Skipping.")
            return

        do_optimize = False
        # 单目情况下使用运动优化，在追踪情况差的时候使用
        if (len(self.map_manager.active_keyframes) > 2 and n_tracked_valid_3d < 30):
            print(f"[VisualFrontEnd Epipolar Filtering] Using motion optimization for pose recovery n_tracked_valid_3d: {n_tracked_valid_3d}")
            do_optimize = True

        print(f"[VisualFrontEnd Epipolar Filtering] 5-pt EssentialMatrix Ransac: {do_optimize}")
        print(f"[VisualFrontEnd Epipolar Filtering] nb pts: {len(valid_kps_ids)}")
        print(f"[VisualFrontEnd Epipolar Filtering] avg. parallax: {avg_parallax}")
        print(f"[VisualFrontEnd Epipolar Filtering] nransac_iter_: {100}")
        print(f"[VisualFrontEnd Epipolar Filtering] fransac_err_: {3.0}")
        print(f"[VisualFrontEnd Epipolar Filtering] \n\n")

        # 计算本质矩阵 (RANSAC)
        bvs_prev_np = np.array(valid_prev_kf_bvs, dtype=np.float32) # 列表转numpy数组
        bvs_cur_np = np.array(valid_cur_bvs, dtype=np.float32)
        success, R_ref_cur, t_ref_cur, outliers_idx = MultiViewGeometry.compute_5pt_essential_matrix(
            bvs_prev_np, bvs_cur_np, 
            100,
            3.0,
            do_optimize,
            self.cur_frame.camera.fx,
            self.cur_frame.camera.fy,
        )

        print(f'[VisualFrontEnd Epipolar Filtering] EssentialMatrix RANSAC Result: ')
        print(f'success: {success}')
        print(f'R: {R_ref_cur}')
        print(f't: {t_ref_cur}')
        print(f'outliers_idx: {len(outliers_idx)}')

        if not success:
            self.logger.log_flexible(self.cur_frame.timestamp, "epipolar_filtering_outliers_count", 0.0)
            self.logger.log_flexible(self.cur_frame.timestamp, "epipolar_filtering_outliers_count_ratio", 0.0)
            print(f"[VisualFrontEnd Epipolar Filtering] No pose could be computed from 5-pt EssentialMatrix!")
            return

        self.logger.log_flexible(self.cur_frame.timestamp, "epipolar_filtering_outliers_count", len(outliers_idx))
        self.logger.log_flexible(self.cur_frame.timestamp, "epipolar_filtering_outliers_count_ratio", len(outliers_idx) / len(valid_kps_ids))

        if len(outliers_idx) > 0.5 * len(valid_kps_ids):
            print(f'[VisualFrontEnd Epipolar Filtering] Too many outliers, skipping as might be degenerate case')
            return

        # 剔除 Outliers - 批量删除，避免循环中多次重建索引
        if len(outliers_idx) > 0:
            all_outlier_ids = np.array(valid_kps_ids)[outliers_idx]
            ids_to_remove = []
            ids_to_remove_3d = []
            
            # TODO：这里可能要删除已三角化的外点，先记录
            for kp_id in all_outlier_ids:
                map_point = self.map_manager.get_map_point(kp_id)
                if map_point is not None and map_point.status == MapPointStatus.TRIANGULATED:
                    # ids_to_remove.append(kp_id)
                    ids_to_remove_3d.append(kp_id)
                    continue
                else:
                    ids_to_remove.append(kp_id)
            
            if len(ids_to_remove) > 0:
                self.cur_frame.remove_features_by_ids(ids_to_remove)

        print(f'[VisualFrontEnd Epipolar Filtering] Epipolar outliers: {ids_to_remove_3d}')
        print(f'[VisualFrontEnd Epipolar Filtering] Epipolar nb outliers: {len(ids_to_remove_3d)}')
        
        # 可视化对极约束后的结果（包含内点信息）
        if self.visualize_optical_flow and self.tracking_prev_pts is not None and self.tracking_kp_ids is not None:
            # 构建内点掩码：根据特征点ID匹配
            if len(outliers_idx) > 0:
                outlier_ids = set(np.array(valid_kps_ids)[outliers_idx])
                # 创建内点掩码：对于追踪数据中的每个点，如果在对极约束中不是外点，则为内点
                inliers_mask = np.ones(len(self.tracking_final_status), dtype=bool)
                for i, kp_id in enumerate(self.tracking_kp_ids):
                    # 如果追踪失败，肯定不是内点
                    if not self.tracking_final_status[i]:
                        inliers_mask[i] = False
                    # 如果在对极约束的外点列表中，也不是内点
                    elif kp_id in outlier_ids:
                        inliers_mask[i] = False
            else:
                # 没有外点，所有追踪成功的都是内点
                inliers_mask = self.tracking_final_status.copy()
            
            # 获取图像用于可视化
            if self.cur_frame.image is not None:
                curr_image_bgr = self.cur_frame.image.copy()
            else:
                # 如果没有当前帧图像，使用灰度图
                curr_gray = self.preprocess_image(self.cur_frame.image) if self.cur_frame.image is not None else None
                if curr_gray is not None:
                    curr_image_bgr = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
                else:
                    return  # 无法可视化
            
            vis_img = visualize_optical_flow_tracking(
                curr_image_bgr,
                self.tracking_prev_pts, self.tracking_tracked_pts, self.tracking_final_status,
                inliers_mask=inliers_mask,
                window_name="KLT Tracking (After Epipolar Filtering)",
                show_stats=True,
                frame_id=self.cur_frame.get_id(),
                kf_id=self.cur_frame.ref_kf_id
            )
            cv2.imshow("KLT Tracking (With Inliers)", vis_img)
            
            # 显示对极约束过滤后的纯净追踪图像（只显示内点，绿色箭头）
            vis_img_clean = visualize_epipolar_filtered_tracking(
                curr_image_bgr,
                self.tracking_prev_pts, self.tracking_tracked_pts, self.tracking_final_status,
                inliers_mask=inliers_mask,
                window_name="Epipolar Filtered Tracking (Inliers Only)",
                show_stats=True,
                frame_id=self.cur_frame.get_id(),
                kf_id=self.cur_frame.ref_kf_id
            )
            cv2.imshow("Epipolar Filtered (Clean)", vis_img_clean)
            cv2.waitKey(1)  # 非阻塞显示

        # TODO：单目模式下的位姿恢复（这里可以直接考虑替换VGGT）
        # 如果是单目且跟踪点少，尝试用 E 分解出的 R, t 替换当前位姿
        if(do_optimize and len(self.map_manager.active_keyframes) > 2):
            T_ref_w = ref_kf.get_T_c_w()
            T_w_cur = self.cur_frame.get_T_w_c()
            T_ref_cur = T_ref_w @ T_w_cur

            # 从运动模型获取当前位姿预测的平移尺度
            scale = np.linalg.norm(T_ref_cur[:3, 3])

            # 归一化从本质矩阵中计算出的t并应用尺度
            # 相当于认为本质矩阵计算出的t的尺度和运动模型预测的尺度一致，只提供方向信息
            t_scaled = (t_ref_cur / np.linalg.norm(t_ref_cur)) * scale

            # 构建新的相对位姿 T_ref_cur_new (GTSAM Pose3)
            T_ref_cur_new = np.eye(4, dtype=np.float64)
            T_ref_cur_new[:3, :3] = R_ref_cur  # 填入本质矩阵分解的 R
            T_ref_cur_new[:3, 3] = t_scaled.flatten() # 填入修正尺度后的t

            T_w_ref_kf = ref_kf.get_T_w_c()
            T_w_c_new = T_w_ref_kf @ T_ref_cur_new

            self.cur_frame.set_T_w_c(T_w_c_new)
    
    def check_ready_for_init(self):
        """
        检查视觉初始化准备是否完成
        """
        # 计算视差 (不进行旋转补偿，因为此时没有可靠的旋转估计，或者假设为 Identity)
        avg_rot_parallax = self.compute_parallax(
            self.cur_frame.ref_kf_id,
            do_unrot = False
        )

        print(f"[VisualFrontEnd Init Check] Init current parallax: {avg_rot_parallax:.2f} px")

        if avg_rot_parallax <= self.config['init_parallax']:
            print(f" -> Not enough parallax (< {self.config['init_parallax']})")
            return False

        t_start = time.time()

        ref_kf = self.map_manager.get_keyframe(self.cur_frame.ref_kf_id)
        if ref_kf is None:
            print(f"[VisualFrontEnd Init Check] Error: Reference KeyFrame not found!")
            return False

        nb_kps = len(self.cur_frame.get_visual_feature_ids())
        if nb_kps < 8:
            print(f"[VisualFrontEnd Init Check] Error: Not enough kps to compute Essential Matrix! nb_kps: {nb_kps} < 8")
            return False

        # 准备数据计算本质矩阵 E
        valid_kps_ids = []
        valid_prev_kf_bvs = []  # 参考关键帧 Bearing Vectors
        valid_cur_bvs = [] # 当前帧 Bearing Vectors

        # 重新计算旋转补偿视差 (为了验证和筛选点)
        R_ref_kf = ref_kf.get_T_c_w()[:3, :3]
        R_cur = self.cur_frame.get_T_w_c()[:3, :3]
        R_kf_ref_cur = R_ref_kf @ R_cur

        # 遍历当前帧所有特征点
        current_kps_ids = self.cur_frame.get_visual_feature_ids()
        valid_dist = []

        for kp_id in current_kps_ids:
            cur_kp_bv = self.cur_frame.get_feature_bearing(kp_id)
            ref_kf_bvs = ref_kf.get_feature_bearing(kp_id)
            ref_kf_unpx = ref_kf.get_feature_undistorted_position(kp_id)
            if ref_kf_bvs is None:
                continue

            # 计算旋转补偿后的视差
            rot_bv = R_kf_ref_cur @ cur_kp_bv
            rot_px = ref_kf.camera.project_cam_to_image(rot_bv)

            # 视差距离
            dist = np.linalg.norm(rot_px - ref_kf_unpx)

            valid_prev_kf_bvs.append(ref_kf_bvs)
            valid_cur_bvs.append(cur_kp_bv)
            valid_kps_ids.append(kp_id)
            valid_dist.append(dist)

        if len(valid_prev_kf_bvs) < 8:
            print(f"[VisualFrontEnd Init Check] Error: Not enough ref KF kps to compute 5-pt Essential Matrix! valid_prev_kf_bvs: {len(valid_prev_kf_bvs)} < 8")
            return False

        if valid_dist:
            avg_rot_parallax = np.mean(valid_dist)
        else:
            avg_rot_parallax = 0.0
        
        if avg_rot_parallax < self.config['init_parallax']:
            print(f"[VisualFrontEnd Init Check] -> Not enough ROT-COMPENSATED parallax ({avg_rot_parallax:.2f} px)")
            return False

        # 计算 5点法本质矩阵
        print(f"[VisualFrontEnd Init Check] Computing 5-pt Essential Matrix ...")
        bvs_prev_np = np.array(valid_prev_kf_bvs, dtype=np.float32) # 列表转numpy数组
        bvs_cur_np = np.array(valid_cur_bvs, dtype=np.float32)
        success, R_ref_cur, t_ref_cur, outliers_idx = MultiViewGeometry.compute_5pt_essential_matrix(
            bvs_prev_np, bvs_cur_np, 
            100,
            3.0,
            do_optimize=True,
            fx=self.cur_frame.camera.fx,
            fy=self.cur_frame.camera.fy,
        )

        print(f"[VisualFrontEnd Init Check] Epipolar nb outliers: {outliers_idx}")
        print(f"[VisualFrontEnd Init Check] Epipolar nb inliers: {len(valid_kps_ids) - len(outliers_idx)}")

        if not success:
            print(f"[VisualFrontEnd Init Check] No pose could be computed from 5-pt EssentialMatrix!")
            return False

        # 剔除 Outliers
        if len(outliers_idx) > 0:
            ids_to_remove = np.array(valid_kps_ids)[outliers_idx]
            self.cur_frame.remove_features_by_ids(ids_to_remove)

        # 设置初始位姿 (人为设定尺度)
        # normalize t and apply scale
        # TODO: 调大试试？
        t_ref_cur_scaled = t_ref_cur / np.linalg.norm(t_ref_cur)
        t_ref_cur_scaled = t_ref_cur_scaled * 0.25 # Arbitrary scale for initialization (e.g. baseline)

        print(f"[VisualFrontEnd Init Check] Init translation: {t_ref_cur_scaled}")

        # 第一帧参考帧位姿应为单位阵
        pose_init = np.eye(4)
        pose_init[:3, :3] = R_ref_cur
        pose_init[:3, 3] = t_ref_cur_scaled
        self.cur_frame.set_T_w_c(pose_init)      

        print(f"[VisualFrontEnd Init Check] Initial second frame pose: {self.cur_frame.get_T_w_c()}")

        t_end = time.time()
        print(f"[VisualFrontEnd Init Check] Initialization took {(t_end - t_start)*1000:.1f} ms")

        return True

    def compute_parallax(self, kfid, do_unrot = True, median = False, do_2d_only = False):
        """
        计算当前帧和上一帧的平均视差
        """
        prev_kf = self.map_manager.get_keyframe(kfid)
        print(f'prev_kf: {prev_kf.id}')
        if prev_kf is None:
            print(f"[VisualFrontEnd] Error: Previous KeyFrame not found!")
            return 0.0

        # 计算相对旋转用于旋转补偿
        if do_unrot:
            R_kf_cur = prev_kf.get_T_c_w()[:3, :3] @ self.cur_frame.get_T_w_c()[:3, :3]
        else:
            R_kf_cur = np.eye(3)

        parallax_list = []
        current_kps_ids = self.cur_frame.get_visual_feature_ids()

        # 计算在上一个关键帧内可见的所有kps（pcurframe_->mapkps_？）
        for kp_id in current_kps_ids:
            # 过滤3d点（防止优化3d点影响视差判断，保持视差的纯粹性）
            mp = self.map_manager.get_map_point(kp_id)
            if do_2d_only and (mp.status == MapPointStatus.TRIANGULATED):
                continue

            prev_kf_kp = prev_kf.get_feature_position(kp_id)
            prev_kf_unpx = prev_kf.get_feature_undistorted_position(kp_id)
  
            cur_kf_unpx = self.cur_frame.get_feature_undistorted_position(kp_id)
            cur_kp_bv = self.cur_frame.get_feature_bearing(kp_id)
            unpx_prev = cur_kf_unpx

            if prev_kf_kp is None:
                continue

            if do_unrot:
                rot_bv = R_kf_cur @ cur_kp_bv
                unpx_prev = prev_kf.camera.project_cam_to_image(rot_bv) # 还是投影到上一帧的像素平面

            # 计算欧式距离
            dist = np.linalg.norm(unpx_prev - prev_kf_unpx)
            parallax_list.append(dist)

        if not parallax_list:
            return 0.0

        if median:
            return float(np.median(parallax_list))
        else:
            return float(np.mean(parallax_list))

    def compute_pose(self, n_tracked_valid_3d):
        if n_tracked_valid_3d < 4:
            print(f"[VisualFrontEnd] Not enough 3D kps for PnP")
            return

        # 1. 准备数据
        # 收集所有 3D 点及其对应的 2D 观测
        valid_bvs = []   # Bearing Vectors (用于 P3P)
        valid_wpts = []  # World Points
        valid_kps = []   # 2D Normalized Points (用于 GTSAM PnP)
        valid_kps_ids = [] # 地图点 ID (用于追踪 Outlier)
        
        current_kps_ids = self.cur_frame.get_visual_feature_ids()
        for kp_id in current_kps_ids:
            # 只使用三角化的点计算PnP
            map_point = self.map_manager.get_map_point(kp_id)
            if map_point is None or map_point.status != MapPointStatus.TRIANGULATED:
                # print(f"[VisualFrontEnd Compute Pose] Error: Map point {kp_id} is not valid!")
                continue
            
            current_kps_bvs = self.cur_frame.get_feature_bearing(kp_id)
            current_kps_unpx = self.cur_frame.get_feature_undistorted_position(kp_id)

            if current_kps_bvs is None or current_kps_unpx is None:
                print(f"[VisualFrontEnd Compute Pose] Error: Current kp {kp_id} is not valid!")
                continue

            valid_bvs.append(current_kps_bvs) # [x, y, z]
            # GTSAM PnP 需要归一化平面坐标
            # 归一化坐标: x_n = (u - cx)/fx，归一化坐标的话后续PnP解算使用K为单位阵
            norm_x = (current_kps_unpx[0] - self.cur_frame.camera.cx) / self.cur_frame.camera.fx
            norm_y = (current_kps_unpx[1] - self.cur_frame.camera.cy) / self.cur_frame.camera.fy
            valid_kps.append(np.array([norm_x, norm_y]))
            
            valid_wpts.append(map_point.get_point())
            valid_kps_ids.append(kp_id)

        np_bvs = np.array(valid_bvs)
        np_wpts = np.array(valid_wpts)
        np_kps = np.array(valid_kps)

        T_wc = self.cur_frame.get_T_w_c() # 当前预测位姿
        print(f"[VisualFrontEnd Compute Pose] Current predicted pose: {T_wc}")
        
        # 2. P3P RANSAC (如果需要，追踪3d点较少，追踪质量较差时使用)
        # TODO: 暂时设为True，应该为self.do_p3p
        if True:
            print(f"[VisualFrontEnd Compute Pose] Running P3P RANSAC on {len(np_bvs)} points...")

            success, p3p_pose, outliers_idx = MultiViewGeometry.p3p_ransac(
                np_bvs, np_wpts, 
                nmaxiter=100,
                errth=3.0,
                fx=self.cur_frame.camera.fx,
                fy=self.cur_frame.camera.fy,
                boptimize=False
            )
            
            # 检查 P3P 结果是否可用
            n_inliers = len(np_bvs) - len(outliers_idx)
            print(f"[VisualFrontEnd Compute Pose] P3P RANSAC success: {success}")
            print(f"[VisualFrontEnd Compute Pose] P3P RANSAC inliers: {n_inliers}")

            self.logger.log_flexible(self.cur_frame.timestamp, "p3p_ransac_inliers_count", n_inliers)
            self.logger.log_flexible(self.cur_frame.timestamp, "p3p_ransac_inliers_count_ratio", n_inliers / len(np_bvs))

            if not success or n_inliers < 5:
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(f'[VisualFrontEnd Compute Pose] P3P Failed or not enough inliers. Resetting.')
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                self.logger.log_flexible(self.cur_frame.timestamp, "p3p_ransac_inliers_count", 0)
                self.logger.log_flexible(self.cur_frame.timestamp, "p3p_ransac_inliers_count_ratio", 0.0)
                # 此时cur_frame.T_w_c，还是使用的运动模型预测位姿
                # TODO：reset？
                # self.reset_frame() 
                return

            # 更新位姿
            T_wc = p3p_pose
            self.cur_frame.set_T_w_c(T_wc)
            
            # 移除外点 (剔除数据，为接下来的 PnP 做准备)
            # 这里的逻辑是直接从 map_manager 移除观测，并从本地列表移除
            # 简单起见，我们重新构建 inlier 列表用于下一步 GTSAM 优化
            inlier_mask = np.ones(len(np_bvs), dtype=bool)
            inlier_mask[outliers_idx] = False
            
            np_kps = np_kps[inlier_mask]
            np_wpts = np_wpts[inlier_mask]
            # vkpids 也需要更新，用于最后移除
            valid_kps_ids = [valid_kps_ids[i] for i in range(len(valid_kps_ids)) if inlier_mask[i]]

        # 3. GTSAM PnP Optimization (Motion-only BA)
        success, optimized_pose, outliers_idx = MultiViewGeometry.gtsam_pnp(
            np_kps, np_wpts, T_wc,
            nmaxiter=5,
            chi2th=5.9915, # Chi2 阈值 (e.g., 5.99)
            use_robust=True,
            fx=self.cur_frame.camera.fx,
            fy=self.cur_frame.camera.fy
        )
        
        n_inliers = len(np_kps) - len(outliers_idx)
        
        print(f"[VisualFrontEnd Compute Pose] GTSAM PnP Outliers: {len(outliers_idx)} / {len(np_kps)}")

        self.logger.log_flexible(self.cur_frame.timestamp, "gtsam_pnp_inliers_count", n_inliers)
        self.logger.log_flexible(self.cur_frame.timestamp, "gtsam_pnp_inliers_count_ratio", n_inliers / len(np_kps))

        # 检查优化结果有效性
        if not success or n_inliers < 5 or len(outliers_idx) > 0.5 * len(np_kps):
            # 如果单目模式下 P3P 后优化依然挂了，重置
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print("[VisualFrontEnd Compute Pose] GTSAM PnP optimization failed after P3P. Resetting.")
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.logger.log_flexible(self.cur_frame.timestamp, "gtsam_pnp_inliers_count", 0)
            self.logger.log_flexible(self.cur_frame.timestamp, "gtsam_pnp_inliers_count_ratio", 0.0)
            # self.reset_frame()
            return

        # 4. 更新最终位姿
        self.cur_frame.set_T_w_c(optimized_pose)
        self.do_p3p = False # 成功，清除标记

        # 5. 移除外点：收集所有外点ID后统一删除
        if len(outliers_idx) > 0:
            outlier_ids = [valid_kps_ids[idx] for idx in outliers_idx]
            self.cur_frame.remove_features_by_ids(outlier_ids)

    def check_new_keyframe(self, n_tracked_valid_3d):
        # 获取参考关键帧 (Reference KF)
        ref_kf = self.map_manager.get_keyframe(self.cur_frame.ref_kf_id)
        if ref_kf is None:
            # 这种情况理论上不应发生，除非是第一帧
            return False

        # 计算平均/中值视差 (旋转补偿后)
        # do_unrot=True, median=True, do_2d_only=False
        med_rot_parallax = self.compute_parallax(
            self.cur_frame.ref_kf_id, do_unrot=True, median=True, do_2d_only=False
        )

        # 帧 ID 差 (当前帧与参考关键帧之间隔了多少普通帧)
        # 你的 Frame id 应该是全局递增的
        n_frames_since_kf = self.cur_frame.get_id() - ref_kf.get_id()
        
        # 统计数据
        # nb3dkps: 当前帧跟踪到的 3D 点数量
        # nbmaxkps: 最大特征点数配置
        nb_3d_kps = n_tracked_valid_3d
        visual_features_num = len(self.cur_frame.get_visual_feature_ids()) 
        n_max_kps = self.config['max_features_to_detect']
        
        # TODO: 暂时设为 False，后续对接后端时替换
        is_local_ba_running = False

        # [计算] 遍历参考帧的所有特征ID，检查它们是否对应有效的 3D 地图点
        n_ref_3d = 0
        ref_ids = ref_kf.get_visual_feature_ids()
        # 统计有多少个 id 在 map_manager 里能找到对应的 MapPoint
        for fid in ref_ids:
            mp = self.map_manager.get_map_point(fid)
            if mp is not None and mp.status == MapPointStatus.TRIANGULATED:
                n_ref_3d += 1
        
        # 防止除零或第一帧情况
        if n_ref_3d == 0: n_ref_3d = nb_3d_kps # 默认使用当前帧的 3D 点数

        # ---------------------------------------------------------------------
        # 条件 1: 跟踪极差 (特征点覆盖率极低)
        # ---------------------------------------------------------------------
        # 覆盖率 < 33% 且距离上一 KF 超过 5 帧 且 后端空闲
        if (visual_features_num < 0.33 * n_max_kps and 
            n_frames_since_kf >= 5 and 
            not is_local_ba_running):
            print(f"[VisualFrontEnd Check New Keyframe] Low occupancy ({visual_features_num}), inserting KF.")
            return True

        # ---------------------------------------------------------------------
        # 条件 2: 3D 点极少 (快跟丢了)
        # ---------------------------------------------------------------------
        # 3D 点 < 20 且 距离上一 KF 超过 2 帧
        if (nb_3d_kps < 20 and 
            n_frames_since_kf >= 2):
            print(f"[CheckKf] Low 3D kps ({nb_3d_kps}), inserting KF.")
            return True

        # ---------------------------------------------------------------------
        # 抑制条件: 如果 3D 点充足，且 (后端忙 OR 间隔太短)，则不插
        # ---------------------------------------------------------------------
        if (nb_3d_kps > 0.5 * n_max_kps and 
            (is_local_ba_running or n_frames_since_kf < 2)):
            return False

        # ---------------------------------------------------------------------
        # 条件 3: 视差或立体视觉间隔
        # ---------------------------------------------------------------------
        # 基础条件 cx: 视差达到阈值的一半 (或者立体模式下的帧间隔)
        min_parallax_threshold = self.config['init_parallax']
        cx = med_rot_parallax >= min_parallax_threshold / 2.0

        # ---------------------------------------------------------------------
        # 核心触发条件 (c0, c1, c2)
        # ---------------------------------------------------------------------
        # c0: 视差足够大 (主要条件)
        c0 = med_rot_parallax >= min_parallax_threshold
        
        # c1: 3D 点数显著减少 (相对于参考 KF 减少了 25% 以上)
        # 这通常意味着相机移动到了新区域，老点看不到了
        c1 = nb_3d_kps < 0.75 * n_ref_3d
        
        # c2: 覆盖率和 3D 点数都下降，且后端空闲
        c2 = (visual_features_num < 0.5 * n_max_kps and 
              nb_3d_kps < 0.85 * n_ref_3d and 
              not is_local_ba_running)

        # 最终判断: 满足任意核心条件 (c0/c1/c2) 且 满足基础视差条件 (cx)
        is_kf_required = (c0 or c1 or c2) and cx

        kf_flag = 1 if is_kf_required else 0
        self.logger.log_flexible(self.cur_frame.timestamp, "id", self.cur_frame.get_id())
        self.logger.log_flexible(self.cur_frame.timestamp, "ref_kf_id", self.cur_frame.ref_kf_id)
        self.logger.log_flexible(self.cur_frame.timestamp, "check_new_keyframe_is_kf_required", kf_flag)
        self.logger.log_flexible(self.cur_frame.timestamp, "check_new_keyframe_med_rot_parallax", med_rot_parallax)
        self.logger.log_flexible(self.cur_frame.timestamp, "check_new_keyframe_n_frames_since_kf", n_frames_since_kf)
        self.logger.log_flexible(self.cur_frame.timestamp, "check_new_keyframe_n_visual_features_num", visual_features_num)

        if is_kf_required:
            print(f"-------------------------------------------------------------------")
            print(f">>> Check Keyframe conditions met:")
            print(f"> Cur Frame ID: {self.cur_frame.get_id()} / Ref KF ID: {ref_kf.get_id()}")
            print(f"> Ref KF 3D kps: {n_ref_3d} / Cur 3D kps: {nb_3d_kps}")
            print(f"> Occupancy: {visual_features_num} / Median Parallax: {med_rot_parallax:.2f}")
            print(f"-------------------------------------------------------------------")

        return is_kf_required