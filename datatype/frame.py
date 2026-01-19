import numpy as np
import gtsam

class Frame:
    def __init__(self, global_camera, kf_id, timestamp):
        self.camera = global_camera
        self.id = kf_id
        self.timestamp = timestamp

        self.image = None
        self.T_w_c = np.eye(4)

        self.visual_features = None
        self.visual_feature_ids = None

        self.is_keyframe = False
        self.is_stationary = False

    # 写入类信息(write)
    def update_frame(self, kf_id, timestamp):
        self.id = kf_id
        self.timestamp = timestamp

    def add_visual_features(self, visual_features, feature_ids):
        self.visual_features = visual_features
        self.visual_feature_ids = feature_ids
        
    def set_T_w_c(self, T_w_c):
        self.T_w_c = T_w_c

    # 读取类信息(read)
    def get_id(self):
        return self.id

    def get_timestamp(self):
        return self.timestamp

    def get_T_w_c(self):
        return self.T_w_c

    def get_visual_features(self):
        return self.visual_features

    def get_visual_feature_ids(self):
        return self.visual_feature_ids

    def get_is_stationary(self):
        return self.is_stationary