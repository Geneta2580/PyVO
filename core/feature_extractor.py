import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.min_dist = self.config.get('min_dist', 25)
        self.max_features_to_detect = self.config.get('max_features_to_detect', 200)
        self.next_feature_id = 0

    def extract_features(self, frame, gray_image):
        img_h, img_w = gray_image.shape

        # 第一帧处理（无历史特征）
        if frame.id == 0:
            new_features = self.detect_features(gray_image, self.max_features_to_detect, mask=None)
            if new_features is not None:
                self._add_new_features_to_frame(frame, new_features)
            return

        # 后续帧处理
        # 获取当前帧已有的特征点（通常是光流跟踪后的结果）
        current_features = frame.get_visual_features()
        current_ages = frame.visual_feature_ages

        # --- 核心修改：生成保留掩膜和占用掩膜 ---
        keep_mask, occupancy_mask = self.filter_features_by_age(
            current_features, current_ages, (img_h, img_w)
        )

        frame.remove_outliers_by_mask(keep_mask)

        # 补充新特征点
        # 计算还需要多少特征点
        current_num = np.sum(keep_mask) # 统计 True 的数量
        new_features_needed = self.max_features_to_detect - current_num

        if new_features_needed > 0:
            # 使用生成的 occupancy_mask (已包含了保留特征点的覆盖区域) 进行检测
            new_features = self.detect_features(gray_image, new_features_needed, occupancy_mask)
            if new_features is not None:
                self._add_new_features_to_frame(frame, new_features)

    def detect_features(self, image, max_corners, mask):
        """OpenCV 特征点提取封装"""
        if max_corners <= 0:
            return None
        
        features = cv2.goodFeaturesToTrack(
            image,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=self.min_dist,
            blockSize=7,
            mask=mask
        )
        return features

    def filter_features_by_age(self, features, feature_ages, image_shape):
        """
        根据 Age 优先级生成保留掩膜。
        不重新创建点列表，而是返回 keep_mask 和 occupancy_mask。
        """
        n_features = len(features)
        
        # 1. 初始化掩膜
        # keep_mask: True 表示保留，False 表示删除
        keep_mask = np.zeros(n_features, dtype=bool)
        # occupancy_mask: 用于 goodFeaturesToTrack，255 表示可检测区域
        occupancy_mask = np.full(image_shape, 255, dtype=np.uint8)

        if n_features == 0:
            return keep_mask, occupancy_mask

        # 2. 获取排序索引 (Age 降序)
        sorted_indices = np.argsort(feature_ages)[::-1]

        # 3. 按照优先级遍历并填充 Mask      
        pts = features[:, 0, :].astype(np.int32)
        
        h, w = image_shape

        for idx in sorted_indices:
            x, y = pts[idx]

            # 边界检查
            if 0 <= x < w and 0 <= y < h:
                # 如果该位置在 mask 中是 255 (未被占用)
                if occupancy_mask[y, x] == 255:
                    # 标记保留
                    keep_mask[idx] = True
                    # 在 mask 上画圆，占据位置
                    cv2.circle(occupancy_mask, (x, y), self.min_dist, 0, -1)
            else:
                # 超出边界的点，keep_mask 默认为 False，会被剔除
                pass

        return keep_mask, occupancy_mask

    def _add_new_features_to_frame(self, frame, new_features):
        """辅助函数：批量添加新特征"""
        num_new = len(new_features)
        new_ids = np.arange(self.next_feature_id, self.next_feature_id + num_new)
        self.next_feature_id += num_new
        
        # 新特征点初始 age 为 1
        new_ages = np.ones(num_new, dtype=int)
        
        frame.add_visual_features(new_features, new_ids, new_ages)

    def reset(self):
        """
        重置特征提取器状态（重置特征ID计数器）
        """
        print(f"[FeatureExtractor] Resetting feature ID counter")
        self.next_feature_id = 0
        print(f"[FeatureExtractor] Reset complete")