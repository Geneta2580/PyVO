import numpy as np
import cv2
import threading
from enum import Enum

class CameraModel(str, Enum):
    PINHOLE = "pinhole"
    FISHEYE = "fisheye"
    
    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Unsupported camera model: {value}. Choose between: pinhole / fisheye")

class CameraCalibration:
    def __init__(self, config):
        self.config = config
        self.camera_model = config['camera_model']
        self.fx = config['cam_intrinsics'][0]
        self.fy = config['cam_intrinsics'][4]
        self.cx = config['cam_intrinsics'][2]
        self.cy = config['cam_intrinsics'][5]
        self.k1 = config['distortion_coefficients'][0]
        self.k2 = config['distortion_coefficients'][1]
        self.p1 = config['distortion_coefficients'][2]
        self.p2 = config['distortion_coefficients'][3]
        self.alpha = config['alpha']
        self.img_w = int(config['image_width'])
        self.img_h = int(config['image_height'])
        self.img_size = (self.img_w, self.img_h)

        print(f"\n Setting up camera, model selected : {self.camera_model}")
        self.model = CameraModel.from_string(self.camera_model)
        
        if self.model == CameraModel.PINHOLE:
            print("\nPinhole Camera Model created\n")
        elif self.model == CameraModel.FISHEYE:
            print("\nFisheye Camera Model selected")

        # 内参矩阵 K (3x3)
        self.K = np.array([
            [self.fx, 0., self.cx],
            [0., self.fy, self.cy],
            [0., 0., 1.]
        ], dtype=np.float64)

        # 畸变系数 D (4x1)
        self.D = np.array([self.k1, self.k2, self.p1, self.p2], dtype=np.float64)

        # 逆内参
        self.iK = np.linalg.inv(self.K)
        
        # 缓存逆内参的具体数值，方便快速访问
        self.ifx = self.iK[0, 0]
        self.ify = self.iK[1, 1]
        self.icx = self.iK[0, 2]
        self.icy = self.iK[1, 2]

        # ROI Mask 初始化
        nborder = 5
        self.roi_rect = (nborder, nborder, self.img_w - 2 * nborder, self.img_h - 2 * nborder) # (x, y, w, h)
        self.roi_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        
        # 设置 ROI 区域为 255
        x, y, w, h = self.roi_rect
        self.roi_mask[y:y+h, x:x+w] = 255

        # 映射表 (Undistortion maps)
        self.undist_map_x = None
        self.undist_map_y = None

        print(f'\n Camera Calibration initialized\n')
        print(f'Camera intrinsics: {self.K}')
        print(f'Camera distortion coefficients: {self.D}\n')

    def is_in_image(self, pt_2d):
        if pt_2d is None:
            return False
        x, y = pt_2d
        return x >= 0 and x < self.img_w and y >= 0 and y < self.img_h

    def _update_roi_mask(self, roi):
        nborder = 5
        x, y, w, h = roi
        
        # 缩小 ROI 以避免边界伪影
        x += nborder
        y += nborder
        w -= nborder
        h -= nborder
        
        self.roi_rect = (x, y, w, h)
        
        self.roi_mask.fill(0)
        # 注意 numpy 索引是 [y:y+h, x:x+w]
        if w > 0 and h > 0:
            self.roi_mask[y:y+h, x:x+w] = 255

    def project_world_to_cam(self, T_wc, pt_3d_w):
        T_cw = np.linalg.inv(T_wc)
        pt_3d_w_hom = np.append(pt_3d_w, 1.0)
        pt_3d_c_hom = T_cw @ pt_3d_w_hom
        pt_3d_c = pt_3d_c_hom[:3]
        return pt_3d_c

    def project_world_to_image(self, T_wc, pt_3d_w):
        T_cw = np.linalg.inv(T_wc)

        pt_3d_w_hom = np.append(pt_3d_w, 1.0)
        pt_3d_c_hom = T_cw @ pt_3d_w_hom
        pt_3d_c = pt_3d_c_hom[:3]

        # 检查深度
        if pt_3d_c[2] <= 0:
            print(f"[Camera] project_world_to_image: Depth is negative")
            return None
        
        pt_2d = self.project_cam_to_image(pt_3d_c)
        return pt_2d

    def project_world_to_image_dist(self, T_wc, pt_3d_w):
        """
        将世界坐标点投影到原始(畸变)图像平面
        """
        T_cw = np.linalg.inv(T_wc)

        pt_3d_w_hom = np.append(pt_3d_w, 1.0)
        pt_3d_c_hom = T_cw @ pt_3d_w_hom
        pt_3d_c = pt_3d_c_hom[:3]

        # 检查深度
        if pt_3d_c[2] <= 0:
            print(f"[Camera] project_world_to_image_dist: Depth is negative")
            return None

        pt_2d = self.project_cam_to_image_dist(pt_3d_c)
        return pt_2d

    def project_cam_to_image(self, pt_3d):
        """
        3D点投影到图像 (不考虑畸变，使用当前 K)
        pt_3d: numpy array shape (3,)
        returns: numpy array shape (2,) [u, v]
        """
        # z = pt_3d[2]
        # if abs(z) < 1e-6: return ... 
        
        invz = 1.0 / pt_3d[2]
        u = self.fx * pt_3d[0] * invz + self.cx
        v = self.fy * pt_3d[1] * invz + self.cy
        return np.array([u, v], dtype=np.float32)

    def project_cam_to_image_dist(self, pt_3d):
        """
        3D点投影到图像 (考虑畸变 D)
        相当于 C++ 中的 projectCamToImageDist
        """
        # 如果没有畸变系数，直接线性投影
        if np.all(self.D == 0):
            return self.project_cam_to_image(pt_3d)

        # OpenCV projectPoints 需要 shape (N, 3) 或 (N, 1, 3)
        # rvec, tvec 设为 0，因为 pt_3d 已经在相机坐标系下
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        pts = pt_3d.reshape(1, 1, 3).astype(np.float64)

        if self.model == CameraModel.PINHOLE:
            img_pts, _ = cv2.projectPoints(pts, rvec, tvec, self.K, self.D)
            return img_pts.flatten() # [u, v]
        
        elif self.model == CameraModel.FISHEYE:
            # fisheye.projectPoints 需要单独处理
            # 注意：cv2.fisheye.projectPoints 在某些版本可能行为不同，
            # 这里模仿 C++ 逻辑：先透视除法得到 normalized，再 distort
            
            # 手动归一化 (x/z, y/z)
            x = pt_3d[0] / pt_3d[2]
            y = pt_3d[1] / pt_3d[2]
            normalized_pt = np.array([[[x, y]]], dtype=np.float64)
            
            distorted_pt = cv2.fisheye.distortPoints(normalized_pt, self.K, self.D)
            return distorted_pt.flatten()
        
        return np.zeros(2)

    def undistort_image_point(self, pt_2d):
        """
        将畸变图像上的点去畸变
        pt_2d: [u, v]
        """
        if np.all(self.D == 0):
            return pt_2d

        src_pt = pt_2d.reshape(1, 1, 2).astype(np.float64)
        
        if self.model == CameraModel.PINHOLE:
            # undistortPoints 返回的是归一化平面坐标 (x,y)，除非指定了 P (这里用 self.K 作为 P)
            dst_pt = cv2.undistortPoints(src_pt, self.K, self.D, P=self.K)
        elif self.model == CameraModel.FISHEYE:
            dst_pt = cv2.fisheye.undistortPoints(src_pt, self.K, self.D, P=self.K)
        else:
            return pt_2d
        
        return dst_pt.flatten()

    def compute_geometric_attributes(self, raw_pixels):
        """
        核心新方法：批量计算特征点的几何属性
        Input: 
            raw_pixels: (N, 1, 2) 原始畸变图像上的坐标
        Output:
            undistorted_pixels: (N, 1, 2) 去畸变后的像素坐标 (基于新的 self.K)
            bearing_vectors: (N, 3) 单位方向向量 (x, y, z)
        """
        if len(raw_pixels) == 0:
            return np.empty((0, 1, 2)), np.empty((0, 3))

        # 确保输入是 float 类型
        src_pts = raw_pixels.astype(np.float64)

        # 1. 计算去畸变后的像素坐标 (用于显示或光流一致性)
        # 使用 raw_K/raw_D 作为输入， self.K (即 new_K) 作为投影矩阵 P
        if self.model == CameraModel.PINHOLE:
            undist_px = cv2.undistortPoints(src_pts, self.K, self.D, P=self.K)
        elif self.model == CameraModel.FISHEYE:
            undist_px = cv2.fisheye.undistortPoints(src_pts, self.K, self.D, P=self.K)
        
        # 2. 计算归一化坐标 (用于计算 Bearing Vectors)
        # P=None (或 np.eye(3)) 会返回归一化平面坐标 (x_n, y_n)
        if self.model == CameraModel.PINHOLE:
            norm_pts = cv2.undistortPoints(src_pts, self.K, self.D, P=None)
        elif self.model == CameraModel.FISHEYE:
            norm_pts = cv2.fisheye.undistortPoints(src_pts, self.K, self.D, P=None)
        
        # 3. 将归一化坐标转换为单位 Bearing Vectors
        # norm_pts shape is (N, 1, 2) -> flatten to (N, 2)
        xy = norm_pts.reshape(-1, 2)
        
        # 添加 z=1
        # (N, 3)
        xyz = np.hstack((xy, np.ones((xy.shape[0], 1))))
        
        # 归一化为单位向量
        norms = np.linalg.norm(xyz, axis=1, keepdims=True)
        bearing_vectors = xyz / norms

        return undist_px.astype(np.float32), bearing_vectors.astype(np.float32)

    def update_intrinsic(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([
            [self.fx, 0., self.cx],
            [0., self.fy, self.cy],
            [0., 0., 1.]
        ], dtype=np.float64)
        self.iK = np.linalg.inv(self.K)