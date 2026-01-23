"""
光流追踪可视化工具

该模块提供了用于可视化光流追踪结果的函数，包括：
- 追踪前后的特征点
- 追踪成功/失败的状态
- 内点统计信息
- 追踪成功率

使用示例：
    # 在visual_frontend的KLT_tracking方法中使用
    from utils.visualization import visualize_optical_flow_tracking
    
    # 追踪后
    tracked_pts, final_status = feature_tracker.fb_klt_tracking(...)
    
    # 可视化（追踪后，对极约束前）
    vis_img = visualize_optical_flow_tracking(
        curr_image,
        prev_pts, tracked_pts, final_status
    )
    cv2.imshow("Tracking", vis_img)
    
    # 对极约束后，获取内点信息
    # ... epipolar_filtering ...
    # 假设outliers_idx是外点索引
    inliers_mask = np.ones(len(final_status), dtype=bool)
    inliers_mask[outliers_idx] = False
    
    # 再次可视化（包含内点信息）
    vis_img = visualize_optical_flow_tracking(
        curr_image,
        prev_pts, tracked_pts, final_status,
        inliers_mask=inliers_mask
    )
    cv2.imshow("Tracking with Inliers", vis_img)
"""

import cv2
import numpy as np


def visualize_optical_flow_tracking(
    curr_image,
    prev_pts, 
    tracked_pts, 
    final_status,
    inliers_mask=None,
    window_name="Optical Flow Tracking",
    show_stats=True,
    frame_id=None,
    kf_id=None
):
    """
    可视化光流追踪结果
    
    Args:
        curr_image: 当前帧图像 (BGR格式)
        prev_pts: 上一帧特征点坐标 (N, 2) 或 (N, 1, 2)
        tracked_pts: 追踪后的特征点坐标 (N, 2) 或 (N, 1, 2)
        final_status: 追踪状态布尔数组 (N,)，True表示追踪成功
        inliers_mask: 内点掩码 (N,)，True表示经过对极约束过滤后的内点，可选
        window_name: 窗口名称
        show_stats: 是否在图像上显示统计信息
        frame_id: 当前帧ID，可选
        kf_id: 关键帧ID（参考关键帧ID），可选
    
    Returns:
        vis_image: 可视化图像
    """
    # 输入验证
    if prev_pts is None or len(prev_pts) == 0:
        print("[Visualization] Warning: No previous points provided")
        return curr_image.copy()
    
    if tracked_pts is None or len(tracked_pts) == 0:
        print("[Visualization] Warning: No tracked points provided")
        return curr_image.copy()
    
    # 确保输入格式正确
    if prev_pts.ndim == 3:
        prev_pts = prev_pts.reshape(-1, 2)
    if tracked_pts.ndim == 3:
        tracked_pts = tracked_pts.reshape(-1, 2)
    
    # 确保长度一致
    if len(prev_pts) != len(tracked_pts):
        print(f"[Visualization] Warning: Point count mismatch: prev={len(prev_pts)}, tracked={len(tracked_pts)}")
        min_len = min(len(prev_pts), len(tracked_pts))
        prev_pts = prev_pts[:min_len]
        tracked_pts = tracked_pts[:min_len]
        final_status = final_status[:min_len]
        if inliers_mask is not None:
            inliers_mask = inliers_mask[:min_len]
    
    if len(final_status) != len(prev_pts):
        print(f"[Visualization] Warning: Status length mismatch, truncating to match points")
        final_status = final_status[:len(prev_pts)]
        if inliers_mask is not None and len(inliers_mask) != len(prev_pts):
            inliers_mask = inliers_mask[:len(prev_pts)]
    
    # 转换为整数坐标用于绘制
    prev_pts_int = prev_pts.astype(np.int32)
    tracked_pts_int = tracked_pts.astype(np.int32)
    
    # 处理灰度图：转换为BGR格式
    if len(curr_image.shape) == 2:
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2BGR)
    
    # 创建可视化图像：左侧显示统计信息，右侧显示当前帧图像
    h, w = curr_image.shape[:2]
    stats_panel_width = 300  # 左侧统计信息面板宽度
    vis_image = np.zeros((h, w + stats_panel_width, 3), dtype=np.uint8)
    
    # 右侧放置当前帧图像
    vis_image[:, stats_panel_width:] = curr_image
    
    # 坐标不需要偏移（因为只在当前帧图像上绘制）
    tracked_pts_vis = tracked_pts_int.copy()
    prev_pts_vis = prev_pts_int.copy()
    
    # 统计信息
    total_points = len(final_status)
    tracked_success = np.sum(final_status)
    tracked_failed = total_points - tracked_success
    success_ratio = tracked_success / total_points if total_points > 0 else 0.0
    
    # 内点统计
    if inliers_mask is not None:
        n_inliers = np.sum(inliers_mask & final_status)  # 既是追踪成功又是内点
        inlier_ratio = n_inliers / total_points if total_points > 0 else 0.0
    else:
        n_inliers = tracked_success  # 如果没有提供内点信息，假设所有追踪成功的都是内点
        inlier_ratio = success_ratio
    
    # 在当前帧图像上绘制特征点和箭头
    for i in range(total_points):
        # 调整坐标：加上统计面板的偏移量
        prev_pt_vis = (prev_pts_vis[i][0] + stats_panel_width, prev_pts_vis[i][1])
        tracked_pt_vis = (tracked_pts_vis[i][0] + stats_panel_width, tracked_pts_vis[i][1])
        
        if final_status[i]:
            # 追踪成功的点：根据是否为内点选择颜色
            if inliers_mask is not None and inliers_mask[i]:
                # 内点：绿色
                color = (0, 255, 0)
                thickness = 2
            else:
                # 追踪成功但不是内点（如果提供了内点信息）：黄色
                if inliers_mask is not None:
                    color = (0, 255, 255)
                    thickness = 1
                else:
                    # 没有内点信息，所有追踪成功的都显示为绿色
                    color = (0, 255, 0)
                    thickness = 2
            
            # 在当前帧图像上绘制上一帧特征点（箭头起点）
            cv2.circle(vis_image, prev_pt_vis, 3, color, -1)
            
            # 绘制箭头：从上一帧特征点位置指向当前帧追踪到的特征点位置
            cv2.arrowedLine(vis_image, prev_pt_vis, tracked_pt_vis, color, thickness, 
                          tipLength=0.3, line_type=cv2.LINE_AA)
            
            # 在当前帧图像上绘制终点（当前帧追踪到的特征点）
            cv2.circle(vis_image, tracked_pt_vis, 3, color, -1)
        else:
            # 追踪失败的点：在当前帧图像上显示上一帧的特征点（红色）
            cv2.circle(vis_image, prev_pt_vis, 3, (0, 0, 255), -1)
    
    # 在左侧统计面板上绘制统计信息
    if show_stats:
        # 创建统计信息面板背景
        panel_bg = np.zeros((h, stats_panel_width, 3), dtype=np.uint8)
        panel_bg.fill(40)  # 深灰色背景
        
        # 绘制面板边框
        cv2.rectangle(panel_bg, (0, 0), (stats_panel_width - 1, h - 1), (100, 100, 100), 2)
        
        # 将面板放置到左侧
        vis_image[:, :stats_panel_width] = panel_bg
        
        # 文本参数
        text_x = 15
        text_y = 30
        line_height = 28
        font_scale = 0.6
        font_thickness = 2
        
        # 统计信息文本
        stats_text = [
            "=== Tracking Stats ===",
            "",
        ]
        
        # 添加帧ID信息
        if frame_id is not None:
            stats_text.append(f"Frame ID: {frame_id}")
        if kf_id is not None:
            stats_text.append(f"Ref KF ID: {kf_id}")
        if frame_id is not None or kf_id is not None:
            stats_text.append("")  # 空行分隔
        
        # 添加追踪统计信息
        stats_text.extend([
            f"Total Points: {total_points}",
            f"Tracked Success: {tracked_success}",
            f"Success Rate: {success_ratio*100:.1f}%",
            f"Tracked Failed: {tracked_failed}",
            "",
            f"Inliers: {n_inliers}",
            f"Inlier Rate: {inlier_ratio*100:.1f}%"
        ])
        
        # 绘制统计信息文本
        for i, text in enumerate(stats_text):
            if text:  # 跳过空行
                # 标题行使用不同颜色
                if "===" in text:
                    color = (200, 200, 255)
                    font_scale_title = 0.65
                else:
                    color = (255, 255, 255)
                    font_scale_title = font_scale
                
                cv2.putText(vis_image, text, (text_x, text_y + i * line_height),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, color, font_thickness)
    
    return vis_image


def visualize_optical_flow_from_frames(
    prev_frame,
    curr_frame,
    prev_pts,
    tracked_pts,
    final_status,
    inliers_mask=None,
    window_name="Optical Flow Tracking"
):
    """
    从Frame对象可视化光流追踪结果（便捷函数）
    
    Args:
        prev_frame: 上一帧Frame对象
        curr_frame: 当前帧Frame对象
        prev_pts: 追踪前的特征点坐标
        tracked_pts: 追踪后的特征点坐标
        final_status: 追踪状态
        inliers_mask: 内点掩码（可选）
        window_name: 窗口名称
    
    Returns:
        vis_image: 可视化图像
    """
    curr_image = curr_frame.image if curr_frame.image is not None else np.zeros(
        (curr_frame.camera.img_h, curr_frame.camera.img_w, 3), dtype=np.uint8
    )
    
    return visualize_optical_flow_tracking(
        curr_image, prev_pts, tracked_pts, final_status,
        inliers_mask, window_name, show_stats=True
    )


def visualize_epipolar_filtered_tracking(
    curr_image,
    prev_pts,
    tracked_pts,
    final_status,
    inliers_mask,
    window_name="Epipolar Filtered Tracking (Inliers Only)",
    show_stats=True,
    frame_id=None,
    kf_id=None
):
    """
    可视化对极约束过滤后的纯净追踪图像（只显示内点）
    
    与 visualize_optical_flow_tracking 的区别：
    - 只显示经过对极约束过滤后的内点（绿色箭头）
    - 不显示追踪成功但不是内点的点（黄色箭头）
    - 不显示追踪失败的点（红色点）
    
    Args:
        curr_image: 当前帧图像 (BGR格式)
        prev_pts: 上一帧特征点坐标 (N, 2) 或 (N, 1, 2)
        tracked_pts: 追踪后的特征点坐标 (N, 2) 或 (N, 1, 2)
        final_status: 追踪状态布尔数组 (N,)，True表示追踪成功
        inliers_mask: 内点掩码 (N,)，True表示经过对极约束过滤后的内点，必须提供
        window_name: 窗口名称
        show_stats: 是否在图像上显示统计信息
        frame_id: 当前帧ID，可选
        kf_id: 关键帧ID（参考关键帧ID），可选
    
    Returns:
        vis_image: 可视化图像（只包含内点）
    """
    # 输入验证
    if prev_pts is None or len(prev_pts) == 0:
        print("[Visualization] Warning: No previous points provided")
        return curr_image.copy()
    
    if tracked_pts is None or len(tracked_pts) == 0:
        print("[Visualization] Warning: No tracked points provided")
        return curr_image.copy()
    
    if inliers_mask is None:
        print("[Visualization] Warning: inliers_mask is required for epipolar filtered visualization")
        return curr_image.copy()
    
    # 确保输入格式正确
    if prev_pts.ndim == 3:
        prev_pts = prev_pts.reshape(-1, 2)
    if tracked_pts.ndim == 3:
        tracked_pts = tracked_pts.reshape(-1, 2)
    
    # 确保长度一致
    if len(prev_pts) != len(tracked_pts):
        print(f"[Visualization] Warning: Point count mismatch: prev={len(prev_pts)}, tracked={len(tracked_pts)}")
        min_len = min(len(prev_pts), len(tracked_pts))
        prev_pts = prev_pts[:min_len]
        tracked_pts = tracked_pts[:min_len]
        final_status = final_status[:min_len]
        inliers_mask = inliers_mask[:min_len]
    
    if len(final_status) != len(prev_pts):
        print(f"[Visualization] Warning: Status length mismatch, truncating to match points")
        final_status = final_status[:len(prev_pts)]
        inliers_mask = inliers_mask[:len(prev_pts)]
    
    # 转换为整数坐标用于绘制
    prev_pts_int = prev_pts.astype(np.int32)
    tracked_pts_int = tracked_pts.astype(np.int32)
    
    # 处理灰度图：转换为BGR格式
    if len(curr_image.shape) == 2:
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2BGR)
    
    # 创建可视化图像：左侧显示统计信息，右侧显示当前帧图像
    h, w = curr_image.shape[:2]
    stats_panel_width = 300  # 左侧统计信息面板宽度
    vis_image = np.zeros((h, w + stats_panel_width, 3), dtype=np.uint8)
    
    # 右侧放置当前帧图像
    vis_image[:, stats_panel_width:] = curr_image
    
    # 坐标不需要偏移（因为只在当前帧图像上绘制）
    tracked_pts_vis = tracked_pts_int.copy()
    prev_pts_vis = prev_pts_int.copy()
    
    # 统计信息：只统计内点
    total_points = len(final_status)
    # 内点 = 追踪成功 AND 对极约束内点
    inliers_only = inliers_mask & final_status
    n_inliers = np.sum(inliers_only)
    inlier_ratio = n_inliers / total_points if total_points > 0 else 0.0
    
    # 在当前帧图像上绘制特征点和箭头（只绘制内点）
    for i in range(total_points):
        # 只绘制内点（追踪成功且是对极约束内点）
        if not inliers_only[i]:
            continue
        
        # 调整坐标：加上统计面板的偏移量
        prev_pt_vis = (prev_pts_vis[i][0] + stats_panel_width, prev_pts_vis[i][1])
        tracked_pt_vis = (tracked_pts_vis[i][0] + stats_panel_width, tracked_pts_vis[i][1])
        
        # 内点：绿色
        color = (0, 255, 0)
        thickness = 2
        
        # 在当前帧图像上绘制上一帧特征点（箭头起点）
        cv2.circle(vis_image, prev_pt_vis, 3, color, -1)
        
        # 绘制箭头：从上一帧特征点位置指向当前帧追踪到的特征点位置
        cv2.arrowedLine(vis_image, prev_pt_vis, tracked_pt_vis, color, thickness, 
                      tipLength=0.3, line_type=cv2.LINE_AA)
        
        # 在当前帧图像上绘制终点（当前帧追踪到的特征点）
        cv2.circle(vis_image, tracked_pt_vis, 3, color, -1)
    
    # 在左侧统计面板上绘制统计信息
    if show_stats:
        # 创建统计信息面板背景
        panel_bg = np.zeros((h, stats_panel_width, 3), dtype=np.uint8)
        panel_bg.fill(40)  # 深灰色背景
        
        # 绘制面板边框
        cv2.rectangle(panel_bg, (0, 0), (stats_panel_width - 1, h - 1), (100, 100, 100), 2)
        
        # 将面板放置到左侧
        vis_image[:, :stats_panel_width] = panel_bg
        
        # 文本参数
        text_x = 15
        text_y = 30
        line_height = 28
        font_scale = 0.6
        font_thickness = 2
        
        # 统计信息文本
        stats_text = [
            "=== Epipolar Filtered ===",
            "=== (Inliers Only) ===",
            "",
        ]
        
        # 添加帧ID信息
        if frame_id is not None:
            stats_text.append(f"Frame ID: {frame_id}")
        if kf_id is not None:
            stats_text.append(f"Ref KF ID: {kf_id}")
        if frame_id is not None or kf_id is not None:
            stats_text.append("")  # 空行分隔
        
        # 添加内点统计信息
        stats_text.extend([
            f"Total Points: {total_points}",
            f"Inliers Only: {n_inliers}",
            f"Inlier Rate: {inlier_ratio*100:.1f}%",
            "",
            "Note: Only showing",
            "epipolar inliers",
            "(green arrows)"
        ])
        
        # 绘制统计信息文本
        for i, text in enumerate(stats_text):
            if text:  # 跳过空行
                # 标题行使用不同颜色
                if "===" in text:
                    color = (200, 200, 255)
                    font_scale_title = 0.65
                else:
                    color = (255, 255, 255)
                    font_scale_title = font_scale
                
                cv2.putText(vis_image, text, (text_x, text_y + i * line_height),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, color, font_thickness)
    
    return vis_image
