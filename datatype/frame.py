import numpy as np
import gtsam

class Frame:
    def __init__(self, global_camera, frame_id, timestamp):
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

        # 状态判断
        self.is_keyframe = False
        self.is_stationary = False

    # 写入类信息(write)
    def add_visual_features(self, visual_features, feature_ids, feature_ages):
        """添加新特征点，并自动计算去畸变坐标和方向向量"""
        if len(visual_features) == 0: return
        
        # 维度修正
        if visual_features.ndim == 2:
            visual_features = visual_features.reshape(-1, 1, 2)

        undist_pts, bearings = self.camera.compute_geometric_attributes(visual_features)

        # NumPy 拼接
        self.visual_features = np.concatenate([self.visual_features, visual_features])
        
        # 拼接新计算的属性
        self.visual_features_undistorted = np.concatenate([self.visual_features_undistorted, undist_pts])
        self.visual_features_bvs = np.concatenate([self.visual_features_bvs, bearings])
        
        self.visual_feature_ids = np.concatenate([self.visual_feature_ids, feature_ids])
        self.visual_feature_ages = np.concatenate([self.visual_feature_ages, feature_ages])
        
        self._rebuild_index_map()
        
    def set_visual_features(self, feature_ids, feature_features, feature_ages):
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

        self._rebuild_index_map()
        
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
        
        self._rebuild_index_map()

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
        self._rebuild_index_map()

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