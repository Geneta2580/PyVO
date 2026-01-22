from typing import Any


import cv2
import numpy as np

class FeatureTracker:
    def __init__(self, config):
        self.klt_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self.max_fb_dist = config['max_fb_dist']
        self.klt_err = config.get('klt_err', 30.0)  # KLT误差阈值

    def in_border(self, pts, img_shape, border_size=1.0):
        h, w = img_shape[:2]
        x = pts[:, 0]
        y = pts[:, 1]
        
        return (x >= border_size) & (x < w - border_size) & \
               (y >= border_size) & (y < h - border_size)

    def fb_klt_tracking(self, prev_gray, curr_gray, prev_pts, prior_pts, pyramid_levels):
        """
        Forward-Backward KLT Tracking with Priors
        
        Args:
            prev_img: 上一帧灰度图
            cur_img: 当前帧灰度图
            prev_pts: 上一帧的关键点坐标 (N, 2) float32
            prior_pts: 当前帧关键点的先验/预测位置 (N, 2) float32
            win_size: KLT 窗口大小 (int)
            nb_pyr_lvl: 金字塔层数
            max_err: 允许的最大光流误差
            max_fb_dist: 前后向追踪的一致性阈值
            
        Returns:
            tracked_pts: 追踪后的点 (N, 2)
            status: 布尔掩码 (N,), True 表示追踪成功
        """
        # 确保输入是 float32 类型的 numpy 数组
        # (N, 1, 2)
        prev_pts = np.atleast_2d(prev_pts).astype(np.float32)
        prior_pts = np.atleast_2d(prior_pts).astype(np.float32)

        if len(prev_pts) == 0:
            print(f"[FeatureTracker] No previous points to track")
            return prior_pts, np.array([], dtype=bool)

        # 参数配置
        lk_params = dict(
            winSize=(9, 9),
            maxLevel=pyramid_levels,
            criteria=self.klt_criteria
        )
        
        # 检查输入数据有效性
        if np.any(np.isnan(prev_pts)) or np.any(np.isinf(prev_pts)):
            print(f"[FeatureTracker] ERROR: prev_pts contains NaN or Inf!")
            return prior_pts, np.array([], dtype=bool)
        if np.any(np.isnan(prior_pts)) or np.any(np.isinf(prior_pts)):
            print(f"[FeatureTracker] ERROR: prior_pts contains NaN or Inf!")
            return prior_pts, np.array([], dtype=bool)
        
        # 1. Forward Tracking (Prev -> Cur)
        # 使用 prior_pts 作为初始猜测 (cv2.OPTFLOW_USE_INITIAL_FLOW)
        # 确保输入形状为 (N, 1, 2) 供 OpenCV 使用
        # cur_pts (N, 1, 2)
        # status_forward (N, 1)
        cur_pts, status_forward, err_forward = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, prior_pts, 
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW + cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            **lk_params
        )
        
        # 2. Backward Tracking (Cur -> Prev)
        pts_back, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, cur_pts, prev_pts,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW + cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            **lk_params
        )
        
        # 尺寸变换，确保形状为 (N, 2)
        prev_pts_reshape = prev_pts.reshape(-1, 2)
        cur_pts_reshape = cur_pts.reshape(-1, 2)
        pts_back_reshape = pts_back.reshape(-1, 2)

        # 检查追踪点是否在图像边界内
        in_border_mask = self.in_border(cur_pts_reshape, curr_gray.shape)

        # Forward-Backward 一致性检查
        fb_dist = np.linalg.norm(prev_pts_reshape - pts_back_reshape, axis=1)
        fb_mask = fb_dist < self.max_fb_dist

        # 综合所有条件前，确保所有 mask 都是一维的 (N,)
        status_forward = status_forward.flatten().astype(bool)
        status_backward = status_backward.flatten().astype(bool)
        err_forward = err_forward.flatten()

        # 综合所有条件
        final_status = (
            status_forward & 
            status_backward & 
            (err_forward <= self.klt_err) & 
            in_border_mask & 
            fb_mask
        )

        return cur_pts, final_status