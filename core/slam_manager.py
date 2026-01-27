import cv2
import numpy as np
from datatype.camera import CameraCalibration
from datatype.frame import Frame
from datatype.map_manager import MapManager
from datatype.mappoint import MapPointStatus
import queue
from core.feature_extractor import FeatureExtractor
from core.feature_tracker import FeatureTracker
from core.visual_frontend import VisualFrontend
from core.mapper import Mapper
from core.optimizer import Optimizer
from utils.viewer import Viewer

class SLAMManager:
    def __init__(self, config):
        self.config = config
        
        # 调试模式
        self.debug_single_frame = config.get('debug_single_frame', False)
        
        # 全局图像管理
        self.image_stamps = []
        self.frame_id = 0

        # 初始化相机参数
        self.global_camera = CameraCalibration(config)
        
        # 初始化帧参数
        self.prev_frame = None
        self.cur_frame = Frame(self.config, self.global_camera, 0, 0)
        self.last_kf_id = None

        # 初始化局部地图
        self.map_manager = MapManager(self.config)

        # 初始化视觉前端
        self.feature_tracker = FeatureTracker(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.mapper = Mapper(self.config, self.map_manager)
        self.visual_frontend = VisualFrontend(self.config, self.prev_frame, self.cur_frame, self.map_manager, 
                                              self.feature_tracker, self.feature_extractor, self.mapper)

        # 初始化其他线程
        # 优化器
        self.optimizer = Optimizer(self.config, self.map_manager)
        
        # 可视化
        if self.config.get('use_viewer', True):
            self.viewer = Viewer(self.config)
        else:
            self.viewer = None
        
        self.is_running = True

    def start_all_threads(self):
        self.is_running = True
        if self.viewer is not None:
            self.viewer.start()
            print("[SLAMManager] All threads started.")
        else:
            print("[SLAMManager] Viewer not initialized, skipping thread start.")

    def stop_all_threads(self):
        self.is_running = False
        if self.viewer is not None:
            self.viewer.stop()
            print("[SLAMManager] All threads stopped.")
        else:
            print("[SLAMManager] Viewer not initialized, skipping thread stop.")

    def process_image(self, timestamp, image, img_path):
        """
        处理每一帧图像
        Args:
            timestamp: 时间戳
            image: 图像数据 (numpy array)
            img_path: 图像路径
        """
        # 检查是否应该停止处理
        if not self.is_running:
            return
            
        print(f"[SLAMManager] Processing image: {img_path} | Timestamp: {timestamp}")
        
        # 检查是否新帧
        if not self.is_new_frame(timestamp, img_path):
            return

        # 创建新帧
        new_frame = Frame(self.config, self.global_camera, self.frame_id, timestamp)
        new_frame.image = image
        self.cur_frame = new_frame
        self.frame_id += 1

        # 进行视觉前端追踪（KLT追踪、PnP计算位姿）
        is_keyframe, curr_gray = self.visual_frontend.visual_tracking(self.prev_frame, self.cur_frame, timestamp)

        # 检查是否是初始化失败（返回 False 且 visual_init_ready 为 False）
        if not is_keyframe and not self.visual_frontend.visual_init_ready:
            # 初始化失败，重置所有状态
            print(f"[SLAMManager] Initialization failed, resetting all components...")
            # 在重置前，尝试更新viewer显示当前状态
            self._update_viewer_safe()
            self.reset()
            return

        # 如果是关键帧，进行关键帧创建和三角化
        if is_keyframe:
            self.cur_frame.is_keyframe = True
            self._create_keyframe(curr_gray)

        self.prev_frame = self.cur_frame
        
        # 更新viewer显示
        self._update_viewer_safe()
    
    def _update_viewer_safe(self):
        """
        安全地更新viewer，包含错误处理
        """
        if self.viewer is not None:
            if not self.viewer.is_alive():
                print("[SLAMManager] Warning: Viewer thread is not alive!")
                return
            
            try:
                # 获取当前帧位姿
                T_wc = self.cur_frame.get_T_w_c()
                
                # 获取局部地图点 (local_mappoints)
                local_mappoints_dict = self.map_manager.get_active_mappoints()
                local_points = np.array(list(local_mappoints_dict.values())) if len(local_mappoints_dict) > 0 else None
                
                # 获取全局地图点 (global_mappoints)
                global_mappoints_dict = self.map_manager.get_global_mappoints()
                global_points = np.array(list(global_mappoints_dict.values())) if len(global_mappoints_dict) > 0 else None
                
                # 更新viewer（包含数据有效性检查）
                self.viewer.update_data(
                    T_wc, 
                    local_points_3d=local_points,
                    global_points_3d=global_points
                )
            except Exception as e:
                print(f"[SLAMManager] Error updating viewer: {e}")
                import traceback
                traceback.print_exc()
        
        # 调试模式：单帧运行，等待用户输入
        if self.debug_single_frame:
            print(f"\n{'='*60}")
            print(f"[DEBUG] Frame {self.frame_id-1} processed. Press any key to continue to next frame...")
            print(f"{'='*60}\n")
            cv2.waitKey(0)
        else:
            # 非调试模式：短暂延迟以便查看
            cv2.waitKey(1)

    def is_new_frame(self, timestamp, img_path):
        if timestamp not in self.image_stamps:
            self.image_stamps.append(timestamp)
            return True
        
        print(f"[SLAMManager] Image already processed: {img_path}")
        return False

    def _create_keyframe(self, curr_gray):
        # 普通帧变成关键帧时，设置 ref_kf_id 为自己的ID
        # 这样后续普通帧会绑定到这个新的关键帧
        self.last_kf_id = self.cur_frame.ref_kf_id # 当前KF之前的最后一个KF ID
        self.cur_frame.ref_kf_id = self.cur_frame.get_id()
        
        # 在添加到 map_manager 之前，先补充提取特征点
        print(f"[SLAMManager] Creating keyframe {self.cur_frame.get_id()}, extracting features...")
        self.feature_extractor.extract_features(self.cur_frame, curr_gray)
        
        # 添加到 map_manager（会创建 CANDIDATE 状态的 landmark）
        self.map_manager.add_keyframe(self.cur_frame)
        print(f"[SLAMManager] Keyframe {self.cur_frame.get_id()} added to map_manager! ref_kf_id: {self.cur_frame.ref_kf_id}, features: {len(self.cur_frame.get_visual_feature_ids())}")
        
        if self.mapper is not None:
            # 三角化，会将成功三角化的地图点设置为 TRIANGULATED 状态
            self.mapper.triangulate(self.cur_frame)
            n_keyframes = len(self.map_manager.keyframes)

            # 设置共视图，并双向更新共视图
            self.map_manager.update_covisibility_graph(self.cur_frame, last_kf_id=self.last_kf_id)

            # 局部地图匹配
            self.mapper.match_to_local_map(self.cur_frame)

            # BA优化
            # if n_active_kfs > 3:
            #     self.optimizer.optimize()

            # 检查初始化质量（初始化完成后(至少两帧，防止重置死循环)，第一个滑窗满之前，检查地图点数量）
            if 2 < n_keyframes < 10:
                if self.mapper.check_initialization_quality(self.visual_frontend.visual_init_ready):
                    # 初始化质量不足，重置系统
                    print(f"[SLAMManager] Initialization quality insufficient, resetting system...")
                    self.reset()
                    self.prev_frame = None
                    return

    def update_viewer(self):
        if self.viewer is not None and self.viewer.is_alive():
            # 获取局部地图点 (local_mappoints)
            local_mappoints_dict = self.map_manager.get_active_mappoints()
            local_points = np.array(list(local_mappoints_dict.values())) if len(local_mappoints_dict) > 0 else None
            
            # 获取全局地图点 (global_mappoints)
            global_mappoints_dict = self.map_manager.get_global_mappoints()
            global_points = np.array(list(global_mappoints_dict.values())) if len(global_mappoints_dict) > 0 else None
            
            # 更新viewer
            self.viewer.update_data(
                self.cur_frame.get_T_w_c(), 
                local_points_3d=local_points,
                global_points_3d=global_points
            )

    def reset(self):
        """
        重置 SLAM 系统所有组件，清空所有状态
        用于初始化失败时的完全重置
        """
        print(f"[SLAMManager] ========== RESETTING SLAM SYSTEM ==========")
        
        # 重置地图管理器
        self.map_manager.reset()
        
        # 重置视觉前端
        self.visual_frontend.reset()
        
        # 重置 Mapper
        if self.mapper is not None:
            self.mapper.reset()
        
        # 重置特征提取器（重置特征ID计数器）
        self.feature_extractor.reset()
        
        # 重置帧状态
        self.prev_frame = None
        self.cur_frame = None
        
        print(f"[SLAMManager] ========== RESET COMPLETE ==========")