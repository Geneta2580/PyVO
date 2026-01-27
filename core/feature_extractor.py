from pyexpat import features
import cv2
import numpy as np
import math

class FeatureExtractor:
    def __init__(self, config):
        self.config = config

        self.max_features_to_detect = self.config.get('max_features_to_detect', 250)
        self.cell_size = self.config.get('grid_cell_size', 35)
        self.quality_level = self.config.get('max_quality_level', 0.001)
        
        width = self.config.get('image_width', 752)
        height = self.config.get('image_height', 480)
        n_w_cells = math.ceil(width / self.cell_size)
        n_h_cells = math.ceil(height / self.cell_size)
        self.max_kps = int(n_w_cells * n_h_cells)

        # TODO：这里似乎OV2SLAM写的很简单
        self.orb = cv2.ORB_create(
            nfeatures=self.max_features_to_detect * 2,
            scoreType=cv2.ORB_FAST_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        self.next_feature_id = 0

    def extract_features(self, frame, gray_image):
        img_h, img_w = gray_image.shape

        # 1.准备已有特征点（光流追踪后的结果）
        current_features = frame.get_visual_features()
        current_pts = []
        # print(f"[FeatureExtractor] current_features: {len(current_features)}")
        if len(current_features) > 0:
            current_pts = current_features[:, 0, :] # (N, 2)

            # 计算光流追踪的描述子
            tracked_descs, valid_mask = self._compute_descriptors(gray_image, current_features)
            
            # 如果部分点移到了图像边缘无法计算描述子，必须将其剔除
            # 否则它们会占用网格，但实际上已经不可用了
            if not np.all(valid_mask):
                frame.remove_outliers_by_mask(valid_mask)
                
                # 剔除后必须同步更新 current_pts，否则 mask 会错误地屏蔽掉空闲区域
                current_features = frame.get_visual_features()
                if len(current_features) > 0:
                    current_pts = current_features[:, 0, :]
                else:
                    current_pts = []
                
                # valid_mask 已经对齐了 tracked_descs，所以描述子列表是纯净的
                tracked_descs = tracked_descs 

            # 将更新后的描述子存入 Frame
            if len(frame.get_visual_features()) > 0:
                frame.descriptors = tracked_descs

        features_needed = self.max_kps - frame.get_n_occupied_cells()
        # print(f"[FeatureExtractor] features_needed: {features_needed}")
        if features_needed > 0:
            # 2.调用Single Scale Grid Detect提取新点
            new_pts = self.detect_single_scale(gray_image, current_pts)

            # 3.计算新点描述子并添加
            if len(new_pts) > 0:
                # 格式转换 list -> (N, 1, 2)
                new_features_np = np.array(new_pts, dtype=np.float32).reshape(-1, 1, 2)
                new_descs, valid_mask = self._compute_descriptors(gray_image, new_features_np)
                
                final_features = new_features_np[valid_mask]
                if len(final_features) > 0:
                    self._add_new_features_to_frame(frame, final_features, new_descs)

    def detect_single_scale(self, image, current_pts):
        """
        基于网格的单尺度特征检测
        包含：网格占用检查、每个网格最多取2点、动态阈值调整、亚像素优化。
        """
        h, w = image.shape
        cell_size = self.cell_size
        
        # 1. 网格初始化
        n_rows = int(h / cell_size)
        n_cols = int(w / cell_size)
        n_cells = n_rows * n_cols
        n_half_cell = int(cell_size / 4)
        
        # 2. 占用标记 (Occupancy)
        # voccupcells[row][col] = True/False
        occupied_cells = np.zeros((n_rows + 1, n_cols + 1), dtype=bool)
        
        # Mask: 用于防止特征点过近 (OpenCV minMaxLoc 支持 mask)
        mask = np.full((h, w), 255, dtype=np.uint8)
        
        if len(current_pts) > 0:
            pts_int = np.round(current_pts).astype(np.int32)
            for pt in pts_int:
                c, r = pt[0] // cell_size, pt[1] // cell_size
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    occupied_cells[r, c] = True
                
                # 在 Mask 上画黑圈，防止新点离老点太近
                cv2.circle(mask, tuple(pt), n_half_cell, 0, -1)
                
        # 3. 全局预处理 (优化点：Python 循环慢，先全局算 EigenMap)
        blurred_img = cv2.GaussianBlur(image, (3, 3), 0)
        eigen_map = cv2.cornerMinEigenVal(blurred_img, 3, 3)
        
        primary_detections = []   # 每个格子的第一优选
        secondary_detections = [] # 每个格子的第二优选 (备胎)
        nb_occupied_count = 0
        
        # 4. 遍历网格 (Grid Iteration)
        for r in range(n_rows):
            for c in range(n_cols):
                
                # 如果该格子已经被老点占了，跳过
                if occupied_cells[r, c]:
                    nb_occupied_count += 1
                    continue
                
                # 定义 ROI 坐标
                x0, y0 = c * cell_size, r * cell_size
                x1, y1 = min(x0 + cell_size, w), min(y0 + cell_size, h)
                
                # 获取 ROI 切片
                roi_eigen = eigen_map[y0:y1, x0:x1]
                roi_mask = mask[y0:y1, x0:x1]
                
                # --- 第一轮检测 (Primary) ---
                min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(roi_eigen, mask=roi_mask)
                
                # 检查质量阈值 (动态阈值 dmaxquality_)
                if max_v >= self.quality_level:
                    # max_loc 是 ROI 内的局部坐标，需转为全局
                    pt_global = (x0 + max_loc[0], y0 + max_loc[1])
                    
                    # 边界检查 (排除太靠边的点)
                    if 5 < pt_global[0] < w - 5 and 5 < pt_global[1] < h - 5:
                        primary_detections.append(pt_global)
                        
                        # 在 Mask 上把这个点扣掉，以便找第二个点
                        # 注意：需要修改 roi_mask，这会影响原 mask 数组吗？
                        # numpy 切片是视图，修改 roi_mask 会影响 mask，这正是我们要的
                        cv2.circle(roi_mask, max_loc, n_half_cell, 0, -1)
                        
                        # --- 第二轮检测 (Secondary) ---
                        min_v2, max_v2, min_loc2, max_loc2 = cv2.minMaxLoc(roi_eigen, mask=roi_mask)
                        
                        if max_v2 >= self.quality_level:
                            pt2_global = (x0 + max_loc2[0], y0 + max_loc2[1])
                            if 5 < pt2_global[0] < w - 5 and 5 < pt2_global[1] < h - 5:
                                secondary_detections.append(pt2_global)

        # 5. 结果聚合 (Aggregation)
        final_pts = []
        final_pts.extend(primary_detections)
        
        nb_kps = len(final_pts)
        
        # 如果第一轮检测完，点还是不够 (填不满网格)，就用第二轮的备胎来凑
        if nb_kps + nb_occupied_count < n_cells:
            nb_needed = n_cells - (nb_kps + nb_occupied_count)
            # 取前 nb_needed 个备胎
            final_pts.extend(secondary_detections[:nb_needed])
            
        # 6. 动态阈值调整 (Adaptive Thresholding)
        # 如果这次找到的点太少 (< 33% 空闲格子)，说明标准太高了，下次降低标准
        # 如果这次找到的点太多 (> 90% 空闲格子)，说明标准太低了，下次提高标准
        
        current_total = len(final_pts)
        free_cells = n_cells - nb_occupied_count
        
        if free_cells > 0:
            if current_total < 0.33 * free_cells:
                self.quality_level /= 2.0
                # print(f"[FeatExtract] Quality too high, reducing to {self.quality_level}")
            elif current_total > 0.90 * free_cells:
                self.quality_level *= 1.5
                # print(f"[FeatExtract] Quality too low, increasing to {self.quality_level}")
        
        # 7. 亚像素优化 (Sub-Pixel Refinement)
        if len(final_pts) > 0:
            pts_np = np.array(final_pts, dtype=np.float32)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            cv2.cornerSubPix(image, pts_np, (3, 3), (-1, -1), criteria)
            
            # 转回 list 或保持 numpy 都可以，这里返回 list 保持接口一致
            # print(f"[FeatureExtractor] final_pts: {len(final_pts)}")
            return pts_np.tolist()
            
        return []

    def _compute_descriptors(self, image, features_np):
        """
        计算特征点的描述子
        """
        # 注意: 传入 orb.compute 的点必须是 KeyPoint 对象
        pts_flat = features_np.reshape(-1, 2)

        # TODO:这里OV2SLAM里似乎没有指定size
        kps = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), _size=31.0) for p in pts_flat]
        kps_computed, descs = self.orb.compute(image, kps)

        if descs is None: return np.empty((0, 32), dtype=np.uint8), np.zeros(len(features_np), dtype=bool)
        
        # 简单对齐：假设顺序没变 (通常 compute 不会乱序，只会剔除)
        # 更严谨的做法是用坐标匹配，参考上一条回答
        valid_mask = np.zeros(len(features_np), dtype=bool)
        if len(kps_computed) == len(features_np):
             valid_mask[:] = True
        else:           
            # 将 computed 坐标转为 numpy 数组，方便广播计算
            computed_coords = np.array([kp.pt for kp in kps_computed], dtype=np.float32) # (M, 2)
            
            # 这里可能会稍微慢一点点，但是绝对准确 (O(N^2) 对于 N=250 来说是可以忽略的)
            # 遍历所有输入点
            for i, input_pt in enumerate(pts_flat):
                # 寻找 computed 中是否有距离极近的点 (< 0.01 像素)
                # 计算输入点到所有 computed 点的距离
                dists = np.linalg.norm(computed_coords - input_pt, axis=1)
                if np.min(dists) < 0.01: # 允许 0.01 像素的误差
                    valid_mask[i] = True
                    
        return descs, valid_mask

    def _add_new_features_to_frame(self, frame, new_features, new_descriptors):
        """辅助函数：批量添加新特征 (带描述子)"""
        num_new = len(new_features)
        new_ids = np.arange(self.next_feature_id, self.next_feature_id + num_new)
        self.next_feature_id += num_new
        
        # 新特征点初始 age 为 1
        new_ages = np.ones(num_new, dtype=int)
        
        # 传入描述子
        frame.add_visual_features(new_features, new_ids, new_ages, new_descriptors)

    def reset(self):
        """
        重置特征提取器状态（重置特征ID计数器）
        """
        print(f"[FeatureExtractor] Resetting feature ID counter")
        self.next_feature_id = 0
        print(f"[FeatureExtractor] Reset complete")