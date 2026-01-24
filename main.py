import argparse
import yaml
from pathlib import Path
import queue
import multiprocessing as mp

import cv2
from utils.dataloader import UnifiedDataloader
from core.slam_manager import SLAMManager

class DataloaderWrapper:
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def __iter__(self):
        return self
    
    def __next__(self):
        timestamp, event_type, data = next(self.dataloader)
        
        # 在main中进行图像读取
        if event_type == 'IMAGE':
            img_path = data
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"【Warning】: Failed to load image: {img_path}")
                # 跳过无效图像，继续下一个
                return self.__next__()
            
            return timestamp, event_type, (image, img_path)
        else:
            # IMU数据直接返回
            return timestamp, event_type, data

def main():
    # 1. 配置参数解析
    parser = argparse.ArgumentParser(description="PyVO: Visual Odometry For Python")

    # 配置文件路径
    parser.add_argument('--config', type=Path, default='config/kitti.yaml',
                        help="Path to the configuration file")
    args = parser.parse_args()

    # 加载配置文件
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            print("Configuration loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return

    # 2. 初始化模块和通信管道
    print("Initializing SLAM components and communication queues...")
    
    # 启动一个spawn主进程
    mp.set_start_method('spawn', force=True)

    # 初始化数据加载器
    dataloader_config = {'path': config['dataset_path'], 'dataset_type': config['dataset_type']}
    base_data_loader = UnifiedDataloader(dataloader_config)

    # 实例化所有模块
    slam_manager = SLAMManager(config)

    # 启动所有线程（包括viewer线程）
    slam_manager.start_all_threads()

    # 使用包装器，在main中进行图像读取
    data_loader = DataloaderWrapper(base_data_loader)

    # 检查是否启用调试模式
    debug_mode = config.get('debug_single_frame', False)
    if debug_mode:
        print("="*60)
        print("[DEBUG MODE] Single frame debug mode enabled.")
        print("Each frame will pause after processing. Press Enter to continue.")
        print("="*60)
    
    print("Starting SLAM processing...")
    
    try:
        # 遍历dataloader，处理每一帧图像
        frame_count = 0
        for timestamp, event_type, data in data_loader:
            if event_type == 'IMAGE':
                image, img_path = data
                
                # 在main中调用process_image处理每一帧图像
                slam_manager.process_image(timestamp, image, img_path)
                frame_count += 1
                
            elif event_type == 'IMU':
                # 处理IMU数据（如果需要）
                imu_values = data
                pass

    except KeyboardInterrupt:
        print("\n[Main Process] Caught KeyboardInterrupt, initiating shutdown...")
    finally:
        print("[Main Process] Shutting down all components...")
        # 停止所有线程（包括viewer线程）
        slam_manager.stop_all_threads()
        cv2.destroyAllWindows()
        print("[Main Process] SLAM system shut down.")

if __name__ == "__main__":
    main()