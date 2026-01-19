import cv2
import numpy as np

class FeatureTracker:
    def __init__(self, config):
        self.nmax_iters = config['nmax_iters']
        self.win_size = config['window_size']
        self.klt_err = config['klt_err']
        self.klt_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.nmax_iters, self.klt_err)
        self.max_fb_dist = config['max_fb_dist']

    def in_border(self, pts, img_shape, border_size=1.0):
        """
        检查点是否在图像边界内 (向量化实现)
        Args:
            pts: (N, 2) array
            img_shape: (h, w)
        """
        h, w = img_shape[:2]
        x = pts[:, 0]
        y = pts[:, 1]
        
        return (x >= border_size) & (x < w - border_size) & \
               (y >= border_size) & (y < h - border_size)

    def fb_klt_tracking(self, prev_pyramid, curr_pyramid, prev_pts, prior_pts, pyramid_levels):
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
        prev_pts = np.atleast_2d(prev_pts).astype(np.float32)
        prior_pts = np.atleast_2d(prior_pts).astype(np.float32)

        if len(prev_pts) == 0:
            return prior_pts, np.array([], dtype=bool)

        # 参数配置
        lk_params = dict(
            winSize=(self.win_size, self.win_size),
            maxLevel=pyramid_levels,
            criteria=self.klt_criteria
        )

        # -----------------------------------------------------------
        # 1. Forward Tracking (Prev -> Cur)
        # -----------------------------------------------------------
        # 使用 prior_pts 作为初始猜测 (cv2.OPTFLOW_USE_INITIAL_FLOW)
        cur_pts, status_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(
            prev_pyramid, curr_pyramid, prev_pts, prior_pts, 
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW + cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            **lk_params
        )

        # -----------------------------------------------------------
        # 2. Backward Tracking (Cur -> Prev)
        # -----------------------------------------------------------
        # 从 cur_pts 反向追踪回 prev_img
        # 注意：C++源码中反向追踪也使用了 USE_INITIAL_FLOW，并将 vkps(prev_pts) 作为目标猜测
        # 这有助于提高反向追踪的稳定性
        pts_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_pyramid, prev_pyramid, cur_pts, prev_pts,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW + cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            **lk_params
        )

        # -----------------------------------------------------------
        # 3. 验证与剔除 (Vectorized Validation)
        # -----------------------------------------------------------
        # 展平状态数组
        status_fwd = status_fwd.flatten().astype(bool)
        status_bwd = status_bwd.flatten().astype(bool)
        err_fwd = err_fwd.flatten()

        # 条件 1: OpenCV 追踪状态必须成功
        # 条件 2: 光流误差小于阈值
        # 条件 3: 追踪到的点必须在图像边界内
        in_border_mask = self.in_border(cur_pts, cur_img.shape)
        
        # 条件 4: Forward-Backward 一致性检查
        fb_dist = np.linalg.norm(prev_pts - pts_back, axis=1)
        fb_mask = fb_dist < self.max_fb_dist

        # 综合所有条件
        final_status = (
            status_fwd & 
            status_bwd & 
            (err_fwd <= self.klt_err) & 
            in_border_mask & 
            fb_mask
        )

        return cur_pts, final_status