import numpy as np
import gtsam
import math

class Frame:
    def __init__(self, config, global_camera, frame_id, timestamp):
        self.config = config
        self.camera = global_camera
        self.id = frame_id
        self.timestamp = timestamp

        self.image = None
        self.T_w_c = np.eye(4)
        self.T_c_w = np.eye(4)

        # 形状保持一致：features (N, 1, 2), ids (N,), ages (N,)
        self.visual_features = np.empty((0, 1, 2), dtype=np.float32)
        self.visual_feature_ids = np.empty((0,), dtype=np.int64)
        self.visual_feature_ages = np.empty((0,), dtype=np.int64)
        self.visual_features_undistorted = np.empty((0, 1, 2), dtype=np.float32)
        self.visual_features_bvs = np.empty((0, 3), dtype=np.float32)

        # 辅助索引：{feature_id: array_index}
        # 用于 O(1) 查找和修改
        self._id_to_index = {}

        # 参考关键帧ID
        self.ref_kf_id = None

        # 描述子
        self.descriptors = np.empty((0, 32), dtype=np.uint8)

        # 共视图
        self.cov_map = {}
        self.local_map_ids = set()
        self._feature_id_set = set()

        # 网格查找加速
        self.grid_cell_size = self.config.get('grid_cell_size', 35) # 每个格子 20x20 像素
        self.grid_cols = math.ceil(self.camera.img_w / self.grid_cell_size)
        self.grid_rows = math.ceil(self.camera.img_h / self.grid_cell_size)
        
        # grid 是一个 list of list，存储的是特征点在 visual_features 数组中的 index
        # shape: [grid_cols][grid_rows] -> [idx1, idx2, ...]
        self.grid = [[[] for _ in range(self.grid_rows)] for _ in range(self.grid_cols)]

        # 占用网格计数
        self.n_occupied_cells = 0

        # 状态判断
        self.is_keyframe = False
        self.is_stationary = False

    # 写入类信息(write)
    def add_visual_features(self, visual_features, feature_ids, feature_ages, descriptors):
        """添加新特征点，并自动计算去畸变坐标和方向向量"""
        if len(visual_features) == 0: return
        
        # 维度修正
        if visual_features.ndim == 2:
            visual_features = visual_features.reshape(-1, 1, 2)

        undist_pts, bearings = self.camera.compute_geometric_attributes(visual_features)

        start_idx = len(self.visual_features) # 记录添加前的起始索引，用于后续更新 Grid

        # NumPy 拼接
        self.visual_features = np.concatenate([self.visual_features, visual_features])
        
        # 拼接新计算的属性
        self.visual_features_undistorted = np.concatenate([self.visual_features_undistorted, undist_pts])
        self.visual_features_bvs = np.concatenate([self.visual_features_bvs, bearings])
        
        self.visual_feature_ids = np.concatenate([self.visual_feature_ids, feature_ids])
        self.visual_feature_ages = np.concatenate([self.visual_feature_ages, feature_ages])
        
        if descriptors is not None:
            self.descriptors = np.concatenate([self.descriptors, descriptors])
        else:
            # 如果没有传入描述子（极少情况），填零或报错
            # print(f"[Frame] !!!Warning: No descriptors provided, filling with zeros!!!")
            empty_descs = np.zeros((len(visual_features), 32), dtype=np.uint8)
            self.descriptors = np.concatenate([self.descriptors, empty_descs])

        self._rebuild_index_map()
        self._assign_features_to_grid(start_idx, len(self.visual_features)) # 将新添加的点分配到网格中
        
    def set_visual_features(self, feature_ids, feature_features, feature_ages, descriptors=None):
        """全量覆盖"""
        self.visual_feature_ids = feature_ids
        self.visual_features = feature_features
        self.visual_feature_ages = feature_ages
        
        # 注意：这里假设传入的 feature_features 也是原始畸变坐标
        if len(feature_features) > 0:
            undist_pts, bearings = self.camera.compute_geometric_attributes(feature_features)
            self.visual_features_undistorted = undist_pts
            self.visual_features_bvs = bearings
        else:
            self.visual_features_undistorted = np.empty((0, 1, 2), dtype=np.float32)
            self.visual_features_bvs = np.empty((0, 3), dtype=np.float32)

        if descriptors is not None:
            self.descriptors = descriptors
        else:
            # print(f"[Frame] !!!Warning: No descriptors provided, filling with zeros!!!")
            self.descriptors = np.zeros((len(feature_features), 32), dtype=np.uint8)

        self._rebuild_index_map()
        self._reset_grid()
        self._assign_features_to_grid(0, len(self.visual_features))
        
    def set_T_w_c(self, T_w_c):
        self.T_w_c = T_w_c
        self.T_c_w = np.linalg.inv(T_w_c)

    # 读取类信息(read)
    def get_id(self):
        return self.id

    def get_timestamp(self):
        return self.timestamp

    def get_T_w_c(self):
        return self.T_w_c

    def get_T_c_w(self):
        return self.T_c_w

    def get_visual_features(self):
        return self.visual_features

    def get_visual_feature_ids(self):
        return self.visual_feature_ids

    def get_undistorted_features(self):
        return self.visual_features_undistorted

    def get_bearing_vectors(self):
        return self.visual_features_bvs

    def get_feature_position(self, feature_id):
        """通过 ID 获取坐标 (O(1))"""
        idx = self._id_to_index.get(feature_id)
        if idx is not None:
            # 返回引用，如果在这里修改，原数组也会变
            return self.visual_features[idx].flatten() 
        return None

    def get_feature_age(self, feature_id):
        """通过 ID 获取年龄 (O(1))"""
        idx = self._id_to_index.get(feature_id)
        if idx is not None:
            return self.visual_feature_ages[idx]
        return None

    def get_feature_bearing(self, feature_id):
        """O(1) 获取特征点的 Bearing Vector"""
        idx = self._id_to_index.get(feature_id)
        if idx is not None:
            return self.visual_features_bvs[idx]
        return None
        
    def get_feature_undistorted_position(self, feature_id):
        """O(1) 获取特征点的去畸变坐标"""
        idx = self._id_to_index.get(feature_id)
        if idx is not None:
            return self.visual_features_undistorted[idx].flatten()
        return None

    def get_is_stationary(self):
        return self.is_stationary

    def get_n_occupied_cells(self):
        return self.n_occupied_cells

    def get_features_in_area(self, x, y):
        """
        [C++ getSurroundingKeypoints 复刻版] 
        获取 (x,y) 所在网格及左上相邻网格内的所有特征点。
        这是一步"粗筛"，不进行精确半径检查。
        
        Args:
            x, y: 中心点像素坐标
        
        Returns:
            list[int]: 返回的是特征点在 self.visual_features 数组中的【下标/索引】(Array Indices)
                       范围是 0 到 len(features)-1。
                       你可以用这些 index 去访问 visual_feature_ids 或 visual_features。
        """
        indices = []
        
        # 1. 计算当前点所在的网格坐标
        # 对应 C++: floor(pt.y / ncellsize_)
        c_kp = int(x / self.grid_cell_size)
        r_kp = int(y / self.grid_cell_size)

        # 2. 遍历 2x2 邻域 (当前格 + 左/上/左上)
        for c in range(c_kp - 1, c_kp + 1):
            for r in range(r_kp - 1, r_kp + 1):
                
                # 3. 边界检查
                if 0 <= c < self.grid_cols and 0 <= r < self.grid_rows:
                    
                    # 4. 收集索引
                    # self.grid[c][r] 里面存的就是 int 类型的数组下标
                    cell_indices = self.grid[c][r]
                    if cell_indices:
                        indices.extend(cell_indices)
                        
        return indices

    # ==========================================
    # 性能优化方法：O(1) 更新
    # ==========================================
    def update_feature_position(self, feature_id, new_position):
        """通过 ID 更新坐标 (O(1))，并同步更新几何属性"""
        idx = self._id_to_index.get(feature_id)
        if idx is not None:
            # 1. 更新原始坐标
            self.visual_features[idx] = new_position.reshape(1, 2)
            
            # 2. 计算新的几何属性
            input_pt = new_position.reshape(1, 1, 2)
            undist, bv = self.camera.compute_geometric_attributes(input_pt)
            
            # 3. 更新数组
            self.visual_features_undistorted[idx] = undist
            self.visual_features_bvs[idx] = bv
            return True
        return False

    # ==========================================
    # 性能优化方法：批量删除
    # ==========================================
    def remove_features_by_ids(self, ids_to_remove):
        if len(ids_to_remove) == 0: return

        remove_mask = np.isin(self.visual_feature_ids, ids_to_remove)
        keep_mask = ~remove_mask
        
        self.visual_features = self.visual_features[keep_mask]      
        self.visual_feature_ids = self.visual_feature_ids[keep_mask]
        self.visual_feature_ages = self.visual_feature_ages[keep_mask]
        self.visual_features_undistorted = self.visual_features_undistorted[keep_mask]
        self.visual_features_bvs = self.visual_features_bvs[keep_mask]
        if len(self.descriptors) == len(keep_mask):
             self.descriptors = self.descriptors[keep_mask]
        
        self._rebuild_index_map()
        self._reset_grid()
        self._assign_features_to_grid(0, len(self.visual_features))

    def remove_outliers_by_mask(self, keep_mask):
        """
        直接通过布尔掩膜删除 (常用于 RANSAC 后)
        keep_mask: 长度等于当前特征点数量的 bool 数组
        """
        self.visual_features = self.visual_features[keep_mask]
        self.visual_feature_ids = self.visual_feature_ids[keep_mask]
        self.visual_feature_ages = self.visual_feature_ages[keep_mask]
        self.visual_features_undistorted = self.visual_features_undistorted[keep_mask]
        self.visual_features_bvs = self.visual_features_bvs[keep_mask]
        if len(self.descriptors) == len(keep_mask):
             self.descriptors = self.descriptors[keep_mask]
        self._rebuild_index_map()
        self._reset_grid()
        self._assign_features_to_grid(0, len(self.visual_features))

    # ==========================================
    # 内部工具
    # ==========================================
    def _rebuild_index_map(self):
        """
        重建 ID 到 Index 的映射表。
        虽然这是 O(N)，但比每次删除后的 O(N^2) 级联效应要好得多。
        只在涉及数组长度变化的操作后调用一次。
        """
        # 字典推导式，速度很快
        self._id_to_index = {
            fid: idx for idx, fid in enumerate(self.visual_feature_ids)
        }

    def _reset_grid(self):
        """清空网格"""
        self.grid = [[[] for _ in range(self.grid_rows)] for _ in range(self.grid_cols)]
        self.n_occupied_cells = 0

    def _assign_features_to_grid(self, start_idx, end_idx):
        """将指定范围内的特征点索引填入网格"""
        for i in range(start_idx, end_idx):
            # 取出像素坐标
            kp = self.visual_features[i].flatten() # (x, y)
            
            # 计算网格坐标
            c = int(kp[0] / self.grid_cell_size)
            r = int(kp[1] / self.grid_cell_size)
            
            # 边界检查
            if 0 <= c < self.grid_cols and 0 <= r < self.grid_rows:
                if len(self.grid[c][r]) == 0:
                    self.n_occupied_cells += 1

                self.grid[c][r].append(i) # 存入 index

    # ==========================================
    # 共视图与局部地图接口 (新增)
    # ==========================================
    def add_covisible_kf(self, kf_id, weight=1):
        """
        双向更新时被调用：添加或更新共视关键帧权重。
        注意：这里必须是【累加】逻辑，而不是覆盖。
        
        Args:
            kf_id: 邻居关键帧 ID
            weight: 增加的权重 (默认为 1)
        """
        # 1. 防止自环
        if kf_id == self.id:
            return

        # 2. 累加权重 (Accumulate Weight)
        # 对于新生成的KF，get 返回 0，然后 + weight
        # 对于已存在的KF，取旧值 + weight
        self.cov_map[kf_id] = self.cov_map.get(kf_id, 0) + weight

    def remove_covisible_kf(self, kf_id):
        """从共视图中删除对应 ID 的关键帧及其分数"""
        if kf_id in self.cov_map:
            del self.cov_map[kf_id]

    def set_covisible_map(self, cov_map):
        """设置完整的共视表 (覆盖)"""
        self.cov_map = cov_map

    def get_covisible_map(self):
        """获取共视表"""
        return self.cov_map

    def set_local_map_ids(self, ids_set):
        """设置局部地图点集合"""
        self.local_map_ids = ids_set

    def get_local_map_ids(self):
        return self.local_map_ids

    def is_observing_feature(self, mp_id):
        """检查当前帧是否观测到了指定 ID 的地图点 (O(1))"""
        # 懒加载 set，节省内存和计算
        if not self._feature_id_set or len(self._feature_id_set) != len(self.visual_feature_ids):
             self._feature_id_set = set(self.visual_feature_ids)
        return mp_id in self._feature_id_set

    def replace_mappoint_id(self, old_id, new_id):
        """
        [MapManager 专用] 将帧内引用的特征点 ID 从 old_id 修改为 new_id。
        用于地图点合并时更新历史关键帧的引用。
        Returns:
            bool: 是否成功找到并替换
        """
        # 1. 检查是否存在 old_id
        # 使用辅助索引快速查找
        idx = self._id_to_index.get(old_id)
        
        if idx is None:
            return False
            
        # 2. 修改 ID 数组
        self.visual_feature_ids[idx] = new_id
        
        # 3. 更新辅助索引
        del self._id_to_index[old_id]
        self._id_to_index[new_id] = idx
        
        # 4. 更新 set 缓存 (如果有)
        if self._feature_id_set:
            if old_id in self._feature_id_set:
                self._feature_id_set.remove(old_id)
                self._feature_id_set.add(new_id)
                
        return True