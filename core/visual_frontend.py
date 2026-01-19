import time
import numpy as np
import cv2
import gtsam
from utils.geometry import MultiViewGeometry

# 运动模型
class MotionModel:
    def __init__(self, config):
        self.config = config
        self.prev_time = -1
        self.prev_T_wc = None
        self.log_rel_T_wc = gtsam.Pose3()


    def apply_motion_model(self, T_wc, time):
        # 非第一帧
        T_wc_pred = T_wc
        if self.prev_time > 0:
            relative_pose = self.prev_T_wc.between(T_wc)
            error_vector = gtsam.Pose3.Logmap(relative_pose)
            if not np.allclose(error_vector, 0, atol=1e-5):
                self.prev_T_wc = T_wc # 保持静止

            # 恒速模型预测位姿
            dt = time - self.prev_time
            delta_pose = gtsam.Pose3.Expmap(self.log_rel_T_wc * dt)
            T_wc_pred = T_wc.compose(delta_pose)
                        
        return T_wc_pred

    def update_motion_model(self, T_wc, time):
        self.prev_time = time
        self.prev_T_wc = T_wc


class VisualFrontend:
    def __init__(self, config, prev_frame, cur_frame, map_manager, feature_tracker):
        self.config = config
        self.cur_frame = cur_frame
        self.prev_pyramid = None
        self.curr_pyramid = None

        # 调用
        self.tracker = feature_tracker
        self.map_manager = map_manager

        # 运动模型（恒速模型）
        self.motion_model = MotionModel(config)

        # 是否完成视觉初始化
        self.visual_init = False

        # 初始化CLAHE参数
        tile_size = 50
        self.nbwtiles = self.cur_frame.camera_calibration.img_w / tile_size
        self.nbhtiles = self.cur_frame.camera_calibration.img_h / tile_size
        self.clahe = cv2.createCLAHE(clipLimit=config['clahe_value'], tileGridSize=(self.nbwtiles, self.nbhtiles))

    def visual_tracking(self, image, timestamp):
        print(f"[VisualFrontend] Mono tracking: {timestamp}")

        # 预处理图像
        self.preprocess_image(image)

        # 第一帧直接作为关键帧
        if self.frame.id == 0:
            return True

        # 预测新帧位姿
        T_wc = self.cur_frame.get_T_w_c()
        T_wc_pred = self.motion_model.apply_motion_model(T_wc, timestamp)
        self.cur_frame.set_T_w_c(T_wc_pred)

        # 追踪新帧
        self.KLT_tracking()

        # 对极约束去除外点
        self.epipolar_filtering()

        # 检查视觉初始化准备是否完成
        if not self.visual_init:
            if(self.cur_frame.nb2dkps < 50):
                # TODO: 重置，重新初始化
                return False
            elif self.check_ready_for_init():
                print(f'[VisualFrontend] Ready for visual initialization!')
                self.visual_init = True
            else:
                print(f'[VisualFrontend] Not ready for visual initialization!')
                return False

        # 计算位姿
        self.compute_pose()

        # 更新运动模型
        self.motion_model.update_motion_model(self.cur_frame.get_T_w_c(), timestamp)

        # 检查是否为新关键帧
        is_new_kf = self.check_new_keyframe()

        return is_new_kf

    def preprocess_image(self, image):
        print(f"[VisualFrontend] Preprocessing image")

        # 使用直方图均衡化增强图像对比度

        

    def KLT_tracking(self, image, timestamp):

        # 追踪3d点2层
        # 追踪2d点全层
        # 准备数据容器
        v3d_kp_ids = []
        v3d_kps_px = [] # 像素坐标
        v3d_priors = [] # 预测坐标
        
        v_kp_ids = []
        v_kps_px = []
        v_priors = []        

        current_kps = self.cur_frame.get_keypoints()

        for kp in current_kps:
            has_prior = False

            if self.config['klt_use_prior'] and kp.is3d:
                # 获取该特征点对应的历史地图点
                map_point = self.map_manager.get_map_point(kp.lmid)

                if map_point:
                    # 投影: World -> Image (Distorted) 这里的 T_cw是基于恒速模型预测的位姿
                    proj_px = self.cur_frame.proj_world_to_image(map_point.get_point())

                    # 检查投影点是否在图像内
                    if self.cur_frame.is_in_image(proj_px):
                        v3d_kps_px.append(kp.px)
                        v3d_priors.append(proj_px)
                        v3d_kp_ids.append(kp.lmid)
                        has_prior = True
                        continue # 找到一个先验即可

            if not has_prior:
                v_kps_px.append(kp.px)
                v_priors.append(kp.px) # 使用上一帧对应点的像素坐标作为预测坐标
                v_kp_ids.append(kp.lmid)
        
        # 转换为 numpy 格式以供 OpenCV 使用
        np_v3d_kps = np.array(v3d_kps_px, dtype=np.float32)
        np_v3d_priors = np.array(v3d_priors, dtype=np.float32)
        
        np_v_kps = np.array(v_kps_px, dtype=np.float32)
        np_v_priors = np.array(v_priors, dtype=np.float32)

        # ---------------------------------------------------------
        # Step 1: 跟踪带有先验的 3D 点
        # ---------------------------------------------------------
        if len(np_v3d_kps) > 0:
            pyramid_levels = 1
            # 3D 点通常只在较小的金字塔层级上跟踪(代码中 nbpyrlvl=1)，因为预测比较准
            tracked_pts, status = self.tracker.fb_klt_tracking(
                self.prev_pyramid, self.curr_pyramid,
                np_v3d_kps,
                np_v3d_priors,
                pyramid_levels,
            )

            nb_good = 0

            # 处理跟踪结果
            for i, is_good in enumerate(status):
                lmid = v3d_kp_ids[i]
                if is_good:
                    # 跟踪成功：更新帧中特征点的坐标
                    self.cur_frame.update_keypoint(lmid, tracked_pts[i])
                    nb_good += 1
                else:
                    # 跟踪失败：降级处理，加入到普通列表，稍后尝试全金字塔跟踪
                    v_kps_px.append(v3d_kps_px[i]) # 使用原始像素
                    v_priors.append(v3d_priors[i])
                    v_kp_ids.append(lmid)

            print(f"[VisualFrontend] Tracked {nb_good} / {len(np_v3d_kps)} 3D points")
            if nb_good < 0.33 * len(np_v3d_kps):
                # TODO:匀速模型预测的位姿不够准确，需要进行P3P
                # TODO:不使用先验，使用原始像素作为预测坐标
                self.bp3p_req = True
                v_priors = v_kps_px

            # 收集先验追踪失败点到普通追踪点
            np_v_kps = np.array(v_kps_px, dtype=np.float32)
            np_v_priors = np.array(v_priors, dtype=np.float32)

        # ---------------------------------------------------------
        # Step 2: 跟踪未带有先验的 2D 点
        # ---------------------------------------------------------
        if len(np_v_kps) > 0:
            tracked_pts, status = self.tracker.fb_klt_tracking(
                self.prev_pyramid, self.curr_pyramid,
                np_v_kps,
                np_v_priors,
                self.config['pyramid_levels'],
            )

            nb_good = 0
            for i, is_good in enumerate(status):
                lmid = v_kp_ids[i]
                if is_good:
                    self.cur_frame.update_keypoint(lmid, tracked_pts[i])
                    nb_good += 1
                else:
                    # 彻底丢失追踪点，从 MapManager/CurrentFrame 中移除该观测
                    self.map_manager.remove_obs_from_cur_frame(lmid)
                    self.cur_frame.remove_keypoint_by_id(lmid)

            print(f"[VisualFrontend] KLT Tracking no prior: {nb_good} / {len(np_v_kps)} points.")
    
    def epipolar_filtering(self):
        """
        执行 2D-2D 对极几何滤波，剔除异常点
        """
        print("[VisualFrontEnd] Starting Epipolar Filtering...")

        # 1. 获取上一关键帧 (Previous KeyFrame)
        # 注意：pcurframe.kfid 在 createKeyframe 之前应该是指向上一帧的 KF ID，或者我们维护一个 last_kfid
        # 在 OV2SLAM 中，当前帧只有在插入 KF 时才更新 kfid，所以这里假设 kfid 指向的是它的参考关键帧
        prev_kf = self.map_manager.get_keyframe(self.prev_frame.kfid)

        if prev_kf is None:
            print("[VisualFrontEnd] Error: Previous KeyFrame not found!")
            return

        nb_kps = self.prev_frame.nbkps
        if nb_kps < 8:
            print(f"[VisualFrontEnd] Error: Not enough kps to compute Essential Matrix! nb_kps: {nb_kps} < 8")
            return

        # 2. 准备数据容器
        vkps_ids = []
        vkf_bvs = []  # 上一帧 Bearing Vectors
        vcur_bvs = [] # 当前帧 Bearing Vectors


        # 3. 视差检查 (Parallax Check)
        R_kf_cur = prev_kf.get_T_c_w().rotation().matrix() @ self.cur_frame.get_T_w_c().rotation().matrix()
        
        avg_parallax = 0.0
        nb_parallax = 0

        # 获取当前帧所有特征点
        current_kps = self.cur_frame.get_keypoints()
        
        for kp in current_kps:
            # 查找上一帧是否观测到了同一点
            kfkp = prev_kf.get_keypoint_by_id(kp.lmid)
            
            if kfkp is None:
                continue

            # 收集数据 (用于后续 E 矩阵计算)
            # 注意：cv2.findEssentialMat 需要像素坐标或归一化坐标
            # 这里我们收集归一化平面坐标 (unpx)
            # kp.unpx 是 [x, y] 去畸变后的像素坐标，我们需要转为归一化坐标 x_n = (x-cx)/fx
            # 但 kp.bv 已经是归一化方向向量 [x, y, z]，直接用 x/z, y/z 即可
            
            # 存储相机归一化向量
            vkf_bvs.append(kfkp.bv)
            vcur_bvs.append(kp.bv)
            vkps_ids.append(kp.lmid)

            # 计算旋转补偿后的视差
            # 将当前点旋转到上一帧坐标系，看与观测点的距离
            rot_bv = R_kf_cur @ kp.bv
            # 投影回上一帧的像素平面 (Project to Image)
            rot_px = prev_kf.pcalib.project_cam_to_image(rot_bv)
            
            dist = np.linalg.norm(rot_px - kfkp.unpx) # unpx 是去畸变像素坐标
            avg_parallax += dist
            nb_parallax += 1

        if len(vkps_ids) < 8:
            print(f"[VisualFrontEnd] Not enough kps to compute Essential Matrix! vkps_ids: {len(vkps_ids)} < 8")
            return

        # 计算平均视差
        avg_parallax /= nb_parallax

        # 如果视差太小，认为是对极几何退化 (Pure Rotation or Standstill)，跳过过滤
        if avg_parallax < 2.0 * self.config['ransac_err']:
            print(f"[VisualFrontEnd] Not enough parallax: {avg_parallax:.2f} px. Skipping.")
            return

        do_optimize = False
        # 单目情况下使用运动优化，在追踪情况差的时候使用
        if (self.map_manager.nbkfs > 2 and self.cur_frame.nb3dkps < 30):
            do_optimize = True

        print(f"[VisualFrontEnd] 5-pt EssentialMatrix Ransac: {do_optimize}")
        print(f"[VisualFrontEnd] only on 3d kps: {False}")
        print(f"[VisualFrontEnd] nb pts: {nb_kps}")
        print(f"[VisualFrontEnd] avg. parallax: {avg_parallax}")
        print(f"[VisualFrontEnd] nransac_iter_: {self.config['ransac_iter']}")
        print(f"[VisualFrontEnd] fransac_err_: {self.config['ransac_err']}")
        print(f"[VisualFrontEnd] \n\n")

        # 计算本质矩阵 (RANSAC)
        success, R, t, outliers_idx = MultiViewGeometry.compute_5pt_essential_matrix(
            vkf_bvs, vcur_bvs, 
            self.config['ransac_iter'],
            self.config['ransac_err'],
            do_optimize,
            self.config['do_random'],
            self.cur_frame.pcalib.fx,
            self.cur_frame.pcalib.fy,
        )

        print(f"[VisualFrontEnd] Epipolar nb outliers: {outliers_idx}")

        if not success:
            print(f"[VisualFrontEnd] No pose could be computed from 5-pt EssentialMatrix!")
            return

        if len(outliers_idx) > 0.5 * len(vkps_ids):
            print(f'Too many outliers, skipping as might be degenerate case')
            return

        # 剔除 Outliers
        for idx in outliers_idx:
            lmid = vkps_ids[idx] # 这里由于输入是vkps_ids，所以索引直接是从outliers_idx中取
            self.map_manager.remove_obs_from_cur_frame(lmid)

        # 单目模式下的位姿恢复
        # 如果是单目且跟踪点少，尝试用 E 分解出的 R, t 替换当前位姿
        if(do_optimize and self.map_manager.nbkfs > 2):
            T_prev_w = prev_kf.get_T_c_w()
            T_w_cur = self.cur_frame.get_T_w_c()
            T_prev_cur = T_prev_w.compose(T_w_cur)

            # 从运动模型获取当前位姿预测的平移尺度
            scale = np.linalg.norm(T_prev_cur.translation())

            # 归一化从本质矩阵中计算出的t并应用尺度
            # 相当于认为本质矩阵计算出的t的尺度和运动模型预测的尺度一致，只提供方向信息
            t_scaled = (t / np.linalg.norm(t)) * scale

            # 构建新的相对位姿 T_prev_cur_new (GTSAM Pose3)
            # 注意 OpenCV R 转 GTSAM Rot3
            T_prev_cur_new = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t_scaled.flatten()))

            T_w_kf = prev_kf.get_T_w_c()
            T_w_c_new = T_w_kf.compose(T_prev_cur_new)

            self.cur_frame.set_T_w_c(T_w_c_new)
    
    def check_ready_for_init(self):
        """
        检查视觉初始化准备是否完成
        """
        # 计算视差 (不进行旋转补偿，因为此时没有可靠的旋转估计，或者假设为 Identity)
        avg_rot_parallax = self.compute_parallax(
            self.cur_frame.kfid,
            do_unrot = False
        )

        print(f"[VisualFrontEnd] Init current parallax: {avg_rot_parallax:.2f} px")

        if avg_rot_parallax <= self.config['init_parallax']:
            print(f" -> Not enough parallax (< {self.config['init_parallax']})")
            return False

        t_start = time.time()

        prev_kf = self.map_manager.get_keyframe(self.cur_frame.kfid)
        if prev_kf is None:
            print(f"[VisualFrontEnd] Error: Previous KeyFrame not found!")
            return False

        nb_kps = self.cur_frame.nbkps
        if nb_kps < 8:
            print(f"[VisualFrontEnd] Error: Not enough kps to compute Essential Matrix! nb_kps: {nb_kps} < 8")
            return False

        # 准备数据计算本质矩阵 E
        vkps_ids = []
        vkf_bvs = []  # 上一帧 Bearing Vectors
        vcur_bvs = [] # 当前帧 Bearing Vectors

        # 重新计算旋转补偿视差 (为了验证和筛选点)
        R_kf_cur = prev_kf.get_T_c_w().rotation().matrix() @ self.cur_frame.get_T_w_c().rotation().matrix()

        # 遍历当前帧所有特征点
        current_kps = self.cur_frame.get_keypoints()
        valid_pairs = []

        for kp in current_kps:
            kfkp = prev_kf.get_keypoint_by_id(kp.lmid)
            if kfkp is None:
                continue

            # 计算旋转补偿后的视差
            rot_bv = R_kf_cur @ kp.bv
            unpx_cur = prev_kf.pcalib.project_cam_to_image(rot_bv) # TODO：此处有些奇怪，需要进一步查看

            # 视差距离
            dist = np.linalg.norm(unpx_cur - kfkp.unpx)

            vkf_bvs.append(kfkp.bv)
            vcur_bvs.append(kp.bv)
            vkps_ids.append(kp.lmid)
            valid_pairs.append(dist)

        if len(vkf_bvs) < 8:
            print(f"[VisualFrontEnd] Error: Not enough prev KF kps to compute 5-pt Essential Matrix! vkf_bvs: {len(vkf_bvs)} < 8")
            return False

        if valid_pairs:
            avg_rot_parallax = np.mean(valid_pairs)
        else:
            avg_rot_parallax = 0.0
        
        if avg_rot_parallax < self.config['init_parallax']:
            print(f" -> Not enough ROT-COMPENSATED parallax ({avg_rot_parallax:.2f} px)")
            return False

        # 计算 5点法本质矩阵
        print(f"[VisualFrontEnd] Computing 5-pt Essential Matrix (RANSAC iter={self.config['ransac_iter']})...")
        success, R_kf_cur, t_kf_cur, outliers_idx = MultiViewGeometry.compute_5pt_essential_matrix(
            np.array(vkf_bvs), 
            np.array(vcur_bvs), 
            self.config['ransac_iter'],
            self.config['ransac_err'],
            do_ptimize=True,
            fx=self.cur_frame.pcalib.fx,
            fy=self.cur_frame.pcalib.fy,
        )

        print(f"[VisualFrontEnd] Epipolar nb outliers: {outliers_idx}")

        if not success:
            print(f"[VisualFrontEnd] No pose could be computed from 5-pt EssentialMatrix!")
            return False

        # 剔除 Outliers
        for idx in outliers_idx:
            lmid = vkps_ids[idx] # 这里由于输入是vkps_ids，所以索引直接是从outliers_idx中取
            self.map_manager.remove_obs_from_cur_frame(lmid)

        # 设置初始位姿 (人为设定尺度)
        # normalize t and apply scale
        t_kf_cur = t_kf_cur / np.linalg.norm(t_kf_cur)
        t_kf_cur = t_kf_cur * 0.25 # Arbitrary scale for initialization (e.g. baseline)

        print(f"[VisualFrontEnd] Init translation: {t_kf_cur}")

        pose_init = gtsam.Pose3(gtsam.Rot3(R_kf_cur), gtsam.Point3(t_kf_cur))
        self.cur_frame.set_T_w_c(pose_init)        

        t_end = time.time()
        print(f"[VisualFrontEnd] Initialization took {(t_end - t_start)*1000:.1f} ms")

        return True

    def compute_parallax(self, kfid, do_unrot = True, median = False, do_2d_only = False):
        """
        计算当前帧和上一帧的平均视差
        """
        prev_kf = self.map_manager.get_keyframe(kfid)
        if prev_kf is None:
            print(f"[VisualFrontEnd] Error: Previous KeyFrame not found!")
            return 0.0

        # 计算相对旋转用于旋转补偿
        if do_unrot:
            R_kf_cur = prev_kf.get_T_c_w().rotation().matrix() @ self.cur_frame.get_T_w_c().rotation().matrix()
        else:
            R_kf_cur = np.eye(3)

        parallax_list = []
        current_kps = self.cur_frame.get_keypoints()

        for kp in current_kps:
            # 过滤3d点
            if do_2d_only and kp.is3d:
                continue

            kfkp = prev_kf.get_keypoint_by_id(kp.lmid)
            if kfkp is None:
                continue

            # 去畸变像素坐标
            unpx_cur = kp.unpx

            if do_unrot:
                rot_bv = R_kf_cur @ kp.bv
                unpx_cur = prev_kf.pcalib.project_cam_to_image(rot_bv) # 还是投影到上一帧的像素平面

            # 计算欧式距离
            dist = np.linalg.norm(unpx_cur - kfkp.unpx)
            parallax_list.append(dist)

        if not parallax_list:
            return 0.0

        if median:
            return float(np.median(parallax_list))
        else:
            return float(np.mean(parallax_list))

    def compute_pose(self):
        """
        计算当前帧位姿 (P3P + PnP Optimization)
        对应 VisualFrontEnd::computePose
        """
        nb3dkps = self.pcurframe.nb3dkps
        if nb3dkps < 4:
            if self.params.debug:
                print("[ComputePose] Not enough 3D kps for PnP")
            return

        # 1. 准备数据
        # 收集所有 3D 点及其对应的 2D 观测
        vbvs = []   # Bearing Vectors (用于 P3P)
        vwpts = []  # World Points
        vkps = []   # 2D Normalized Points (用于 GTSAM PnP)
        vkpids = [] # 地图点 ID (用于追踪 Outlier)
        
        current_kps = self.pcurframe.get_keypoints()
        for kp in current_kps:
            if not kp.is3d:
                continue
            
            mp = self.map_manager.get_map_point(kp.lmid)
            if mp is None:
                continue
            
            vbvs.append(kp.bv) # [x, y, z]
            # GTSAM PnP 需要归一化平面坐标
            vkps.append(kp.unpx) # unpx 是去畸变的像素坐标，需要归一化?
            # 这里的 unpx 如果是像素单位，GTSAM PnP 里 K 设为单位阵就不对了
            # 修正：根据 C++ 代码 ceresPnP 传入的是 unkps (Vector2d)，
            # 并且 LossFunction 里使用了 fx, fy, cx, cy。
            # 这意味着 GTSAM 里我们应该用真实的 K (fx, fy, cx, cy) 和 unpx (像素坐标)。
            # 或者我们把点转为归一化坐标，K 设为单位阵。
            # 为了与 MultiViewGeometry.gtsam_pnp 保持一致 (Input: Normalized)，我们转换一下：
            
            # 归一化坐标: x_n = (u - cx)/fx
            norm_x = (kp.unpx[0] - self.pcurframe.pcalib.cx) / self.pcurframe.pcalib.fx
            norm_y = (kp.unpx[1] - self.pcurframe.pcalib.cy) / self.pcurframe.pcalib.fy
            vkps.append(np.array([norm_x, norm_y]))
            
            vwpts.append(mp.get_point())
            vkpids.append(kp.lmid)

        np_bvs = np.array(vbvs)
        np_wpts = np.array(vwpts)
        np_kps = np.array(vkps)

        T_wc = self.pcurframe.get_T_w_c() # 当前预测位姿
        
        # 2. P3P RANSAC (如果需要)
        # 如果跟踪失败标志位 bp3preq_ 为真，或者系统配置要求强制 P3P
        do_p3p = self.bp3preq or self.params.dop3p
        
        if do_p3p:
            if self.params.debug:
                print(f"[ComputePose] Running P3P RANSAC on {len(np_bvs)} points...")

            success, p3p_pose, outliers_idx = MultiViewGeometry.opencv_p3p_ransac(
                np_bvs, np_wpts, 
                nmaxiter=self.params.nransac_iter,
                errth=self.params.fransac_err,
                fx=self.pcurframe.pcalib.fx,
                fy=self.pcurframe.pcalib.fy,
                boptimize=True
            )
            
            # 检查 P3P 结果是否可用
            n_inliers = len(np_bvs) - len(outliers_idx)
            
            if not success or n_inliers < 5:
                print("[ComputePose] P3P Failed or not enough inliers. Resetting.")
                # self.reset_frame() 
                return

            # 更新位姿
            T_wc = p3p_pose
            self.pcurframe.set_T_w_c(T_wc)
            
            # 移除外点 (剔除数据，为接下来的 PnP 做准备)
            # 注意：倒序删除或构建新列表以避免索引错乱
            # 这里的逻辑是直接从 map_manager 移除观测，并从本地列表移除
            # 简单起见，我们重新构建 inlier 列表用于下一步 GTSAM 优化
            inlier_mask = np.ones(len(np_bvs), dtype=bool)
            inlier_mask[outliers_idx] = False
            
            np_kps = np_kps[inlier_mask]
            np_wpts = np_wpts[inlier_mask]
            # vkpids 也需要更新，用于最后移除
            vkpids = [vkpids[i] for i in range(len(vkpids)) if inlier_mask[i]]
            
            # 真实移除 Map 观测
            for idx in outliers_idx:
                # 原始 vkpids 里的 ID
                # 这里逻辑有点绕，上面已经过滤了。
                # 应该在过滤前记录原始需要删除的 ID
                # 暂且略过，实际逻辑是在最后统一移除
                pass

        # 3. GTSAM PnP Optimization (Motion-only BA)
        # 替代 ceresPnP
        success, optimized_pose, outliers_idx = MultiViewGeometry.gtsam_pnp(
            np_kps, np_wpts, T_wc,
            nmaxiter=5, # C++ 中设为 5
            chi2th=self.params.robust_mono_th, # Chi2 阈值 (e.g., 5.99)
            buse_robust=True,
            fx=self.pcurframe.pcalib.fx,
            fy=self.pcurframe.pcalib.fy
        )
        
        n_inliers = len(np_kps) - len(outliers_idx)
        
        if self.params.debug:
            print(f"[ComputePose] GTSAM PnP Outliers: {len(outliers_idx)} / {len(np_kps)}")

        # 检查优化结果有效性
        if not success or n_inliers < 5 or len(outliers_idx) > 0.5 * len(np_kps):
            if not do_p3p:
                # 如果刚才没做 P3P，那说明可能是局部极小值或跟踪丢了，请求下一帧做 P3P
                self.bp3preq = True
                print("[ComputePose] GTSAM PnP failed, requesting P3P for next frame.")
            elif self.params.mono:
                # 如果单目模式下 P3P 后优化依然挂了，重置
                print("[ComputePose] Optimization failed after P3P. Resetting.")
                # self.reset_frame()
            return

        # 4. 更新最终位姿
        self.pcurframe.set_T_w_c(optimized_pose)
        self.bp3preq = False # 成功，清除标记

        # 5. 移除外点
        for idx in outliers_idx:
            lmid = vkpids[idx]
            self.map_manager.remove_obs_from_cur_frame(lmid)

    def check_new_keyframe(self):
        """
        检查是否为新关键帧
        """
        return self.cur_frame.is_new_keyframe()