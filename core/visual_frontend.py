import numpy as np
import cv2
import gtsam

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
    def __init__(self, config, camera_calibration, cur_frame, feature_tracker):
        self.config = config
        self.camera_calibration = camera_calibration
        self.cur_frame = cur_frame
        self.pyramid = None
        self.motion_model = MotionModel(config)

        # 初始化CLAHE参数
        tile_size = 50
        self.nbwtiles = self.camera_calibration.img_w / tile_size
        self.nbhtiles = self.camera_calibration.img_h / tile_size
        self.clahe = cv2.createCLAHE(clipLimit=config['clahe_value'], tileGridSize=(self.nbwtiles, self.nbhtiles))

    def visual_tracking(self, image, timestamp):

        is_keyframe = self.mono_tracking(image, timestamp)

        return is_keyframe

    def mono_tracking(self, image, timestamp):
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

        return False

    def preprocess_image(self, image):
        print(f"[VisualFrontend] Preprocessing image")

        # 使用直方图均衡化增强图像对比度
        if self.config['use_clahe']:
            image = self.clahe.apply(image)
        else:
            image = image.copy()

        # 预先构筑金字塔用于光流追踪
        cv2.buildOpticalFlowPyramid(image, self.pyramid, (self.config['window_size'], self.config['pyramid_levels']))

    def KLT_tracking(self, image, timestamp):

        # 追踪3d点2层
        # 追踪2d点全层


       


        
