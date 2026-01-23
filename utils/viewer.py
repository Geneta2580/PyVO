import open3d as o3d
import numpy as np
import threading
import queue
import time

class Viewer(threading.Thread):
    def __init__(self, config):
        threading.Thread.__init__(self)
        self.config = config
        self.queue = queue.Queue(maxsize=10)
        self.is_running = True
        
        # Open3D 几何对象
        self.vis = None
        self.current_cam = o3d.geometry.LineSet()  # 当前相机视锥
        self.trajectory = o3d.geometry.LineSet()   # 轨迹线
        self.local_points = o3d.geometry.PointCloud() # 局部地图点
        self.global_points = o3d.geometry.PointCloud() # 全局地图点
        
        # 轨迹数据
        self.traj_points = []
        
        # 历史相机位姿数据
        self.history_camera_poses = [] 
        self.history_cameras = [] 
        
        # 相机视锥大小参数
        self.cam_scale = 0.5
        
        # 历史相机显示配置
        self.max_history_cameras = config.get('viewer_max_history_cameras', 100)
        self.show_history_cameras = config.get('viewer_show_history_cameras', True)
        
        # 点云颜色配置
        self.local_color = [0, 1, 0]  # 绿色
        self.global_color = [0, 0, 1]  # 蓝色
        
        # 初始显示范围
        self.initial_view_range = config.get('viewer_initial_range', 100.0)

        print("[Viewer] Viewer initialized.")

    def run(self):
        # 初始化窗口
        try:
            self.vis = o3d.visualization.Visualizer()
            if not self.vis.create_window(window_name="VO Debugger", width=1024, height=768, visible=True):
                print("[Viewer] ERROR: Failed to create window!")
                return
            
            render_option = self.vis.get_render_option()
            render_option.background_color = np.asarray([1.0, 1.0, 1.0])
            render_option.point_size = 3.0  #稍微调大一点点云
            
            # 添加坐标轴
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            self.vis.add_geometry(coord_frame)
            
            # 添加几何体到场景
            # 注意：初始添加时我们允许 reset_bounding_box=True (默认)，以便初始化视角
            self.vis.add_geometry(self.current_cam)
            self.vis.add_geometry(self.trajectory)
            self.vis.add_geometry(self.local_points)
            self.vis.add_geometry(self.global_points)
            
            # 创建虚拟bounding box来扩展场景范围
            half_range = self.initial_view_range / 2.0
            bbox_points = np.array([
                [-half_range, -half_range, -half_range],
                [half_range, half_range, half_range],
            ])
            virtual_bbox = o3d.geometry.PointCloud()
            virtual_bbox.points = o3d.utility.Vector3dVector(bbox_points)
            virtual_bbox.paint_uniform_color([1.0, 1.0, 1.0]) 
            self.vis.add_geometry(virtual_bbox)
            self.virtual_bbox = virtual_bbox
            
            # --- 初始视角设置 (只执行一次) ---
            ctr = self.vis.get_view_control()
            ctr.set_constant_z_far(self.initial_view_range * 20)
            ctr.set_constant_z_near(0.0001)
            
            # 设置初始视角位置
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, -1, 0])      # 根据你的坐标系定义
            ctr.set_front([0, 0, -1])   # 根据你的坐标系定义
            
            # 计算 Zoom
            zoom_factor = 200.0 / self.initial_view_range
            zoom_value = max(0.01, min(5.0, zoom_factor))
            ctr.set_zoom(zoom_value)
            
            print(f"[Viewer] Initial view setup complete.")
            
            # 初始渲染
            self.vis.poll_events()
            self.vis.update_renderer()

        except Exception as e:
            print(f"[Viewer] ERROR during initialization: {e}")
            import traceback
            traceback.print_exc()
            return

        print("[Viewer] Loop started. You can now control the camera manually.")
        
        while self.is_running:
            try:
                # 获取数据
                data = self.queue.get(timeout=0.05)
                # 更新几何体
                self.update_geometry(data)
            except queue.Empty:
                pass
            
            # 渲染循环
            if not self.vis.poll_events():
                print("[Viewer] Window closed.")
                break
            
            self.vis.update_renderer()
            time.sleep(0.01) 
            
        self.vis.destroy_window()

    def update_data(self, T_wc, local_points_3d=None, global_points_3d=None):
        # (保持不变)
        if not self._is_valid_pose(T_wc): return
        
        local_points_valid = self._is_valid_points(local_points_3d)
        global_points_valid = self._is_valid_points(global_points_3d)
        
        if self.queue.full():
            try: self.queue.get_nowait()
            except queue.Empty: pass
        
        snapshot = {
            'T_wc': np.copy(T_wc),
            'local_points': np.copy(local_points_3d) if local_points_valid else None,
            'global_points': np.copy(global_points_3d) if global_points_valid else None
        }
        self.queue.put(snapshot)
    
    def _is_valid_pose(self, T_wc):
        # (保持不变)
        if T_wc is None or not isinstance(T_wc, np.ndarray) or T_wc.shape != (4, 4): return False
        if np.any(np.isnan(T_wc)) or np.any(np.isinf(T_wc)): return False
        return True
    
    def _is_valid_points(self, points):
        # (保持不变)
        if points is None or not isinstance(points, np.ndarray): return False
        if len(points.shape) != 2 or points.shape[1] != 3: return False
        if len(points) == 0: return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)): return False
        return True

    # 删除 _save_view_parameters 和 _restore_view_parameters 方法，不再需要

    def update_geometry(self, data):
        if 'T_wc' not in data: return
        T_wc = data['T_wc']
        local_points = data.get('local_points', None)
        global_points = data.get('global_points', None)
        
        try:
            # 1. 轨迹更新
            traj_point = T_wc[:3, 3]
            is_new_pose = (len(self.traj_points) == 0 or 
                          not np.allclose(self.traj_points[-1], traj_point, atol=1e-4))
            
            if is_new_pose:
                self.history_camera_poses.append(T_wc.copy())
                self.traj_points.append(traj_point.copy())
                
                # 增量添加历史相机
                if self.show_history_cameras:
                    self._add_history_camera(T_wc)
                
                # 更新轨迹线
                if len(self.traj_points) > 1:
                    self.trajectory.points = o3d.utility.Vector3dVector(np.array(self.traj_points))
                    lines = [[i, i+1] for i in range(len(self.traj_points)-1)]
                    self.trajectory.lines = o3d.utility.Vector2iVector(lines)
                    self.trajectory.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
                    self.vis.update_geometry(self.trajectory)
            
            # 2. 更新当前相机 (update_geometry 不会重置视角)
            self.draw_camera(T_wc)

            # 3. 更新局部地图点
            if local_points is not None:
                self.local_points.points = o3d.utility.Vector3dVector(local_points)
                self.local_points.paint_uniform_color(self.local_color)
                self.vis.update_geometry(self.local_points)
            
            # 4. 更新全局地图点
            if global_points is not None:
                self.global_points.points = o3d.utility.Vector3dVector(global_points)
                self.global_points.paint_uniform_color(self.global_color)
                self.vis.update_geometry(self.global_points)
            
            # 关键：不要在这里调用任何重置视角的代码
            
        except Exception as e:
            print(f"[Viewer] Error updating geometry: {e}")
            
    def _create_camera_geometry(self, T_wc, color=[1, 0, 0], scale=None):
        if scale is None: scale = self.cam_scale
        R, t = T_wc[:3, :3], T_wc[:3, 3]
        w, h, z = 1.0 * scale, 0.75 * scale, 1.0 * scale
        pts_c = np.array([[0, 0, 0], [-w, -h, z], [w, -h, z], [w, h, z], [-w, h, z]])
        pts_w = (R @ pts_c.T).T + t
        cam_geometry = o3d.geometry.LineSet()
        cam_geometry.points = o3d.utility.Vector3dVector(pts_w)
        lines = [[0,1], [0,2], [0,3], [0,4], [1,2], [2,3], [3,4], [4,1]]
        cam_geometry.lines = o3d.utility.Vector2iVector(lines)
        cam_geometry.paint_uniform_color(color)
        return cam_geometry
    
    def _add_history_camera(self, T_wc):
        history_cam = self._create_camera_geometry(
            T_wc, color=[0.5, 0.5, 0.5], scale=self.cam_scale * 0.5
        )
        
        # 关键修改：reset_bounding_box=False
        # 这样添加新物体时，视口不会自动缩放或重置
        self.vis.add_geometry(history_cam, reset_bounding_box=False)
        self.history_cameras.append(history_cam)
        
        if len(self.history_cameras) > self.max_history_cameras:
            old_cam = self.history_cameras.pop(0)
            # 移除几何体时也建议设为 False，不过移除通常影响较小
            self.vis.remove_geometry(old_cam, reset_bounding_box=False)
            if len(self.history_camera_poses) > self.max_history_cameras:
                self.history_camera_poses.pop(0)
    
    def draw_camera(self, T_wc):
        # 逻辑不变，仅更新数据
        R, t = T_wc[:3, :3], T_wc[:3, 3]
        w, h, z = 1.0 * self.cam_scale, 0.75 * self.cam_scale, 1.0 * self.cam_scale
        pts_c = np.array([[0, 0, 0], [-w, -h, z], [w, -h, z], [w, h, z], [-w, h, z]])
        pts_w = (R @ pts_c.T).T + t
        
        self.current_cam.points = o3d.utility.Vector3dVector(pts_w)
        # 必须重新设置 line indices，否则有些版本可能会出错
        lines = [[0,1], [0,2], [0,3], [0,4], [1,2], [2,3], [3,4], [4,1]]
        self.current_cam.lines = o3d.utility.Vector2iVector(lines)
        self.current_cam.paint_uniform_color([1, 0, 0])
        
        self.vis.update_geometry(self.current_cam)

    def stop(self):
        self.is_running = False
        self.join()