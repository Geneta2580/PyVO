import cv2
from datatype.camera import CameraCalibration
from datatype.frame import Frame
import queue
from core.feature_extractor import FeatureExtractor
from core.feature_tracker import FeatureTracker
from core.visual_frontend import VisualFrontend

class SLAMManager:
    def __init__(self, config):
        self.config = config
        
        # 全局图像管理
        self.image_stamps = []
        self.frame_id = 0

        # 初始化相机参数
        self.camera_calibration = CameraCalibration(config)
        
        # 初始化帧参数
        ncellsize = config['nmaxdist']
        self.cur_frame = Frame(self.camera_calibration, ncellsize)

        # 初始化视觉前端
        self.feature_extractor = FeatureExtractor(self.config)
        self.feature_tracker = FeatureTracker(self.config)
        self.visual_frontend = VisualFrontend(self.config, self.camera_calibration, self.cur_frame, self.feature_tracker)

        # 初始化其他线程
        self.image_queue = queue.Queue(maxsize=20)
        # ...

        self.is_running = True

    def start_all_threads(self):
        self.is_running = True
        print("[SLAMManager] All threads started.")

    def stop_all_threads(self):
        self.is_running = False
        print("[SLAMManager] All threads stopped.")

    def set_camera_calibration(self, camera_calibration: CameraCalibration):
        self.camera_calibration = camera_calibration

    def get_camera_calibration(self):
        return self.camera_calibration
        
    def process_image(self, timestamp, image, img_path):
        """
        处理每一帧图像
        Args:
            timestamp: 时间戳
            image: 图像数据 (numpy array)
            img_path: 图像路径
        """
        print(f"[SLAMManager] Processing image: {img_path} | Timestamp: {timestamp}")
        
        # 检查是否新帧
        if not self.is_new_frame(timestamp, img_path):
            return
        
        # 创建新帧
        self.frame_id += 1
        self.cur_frame.update_frame(self.frame_id, timestamp)

        # 进行视觉前端追踪
        is_keyframe = self.visual_frontend.visual_tracking(self.cur_frame, timestamp)
        if is_keyframe:
            
        cv2.imshow("Image", image)
        cv2.waitKey(1)

    def is_new_frame(self, timestamp, img_path):
        if timestamp not in self.image_stamps:
            self.image_stamps.append(timestamp)
            return True
        
        print(f"[SLAMManager] Image already processed: {img_path}")
        return False