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

        # 互斥锁
        self.intrinsic_mutex = threading.Lock()

        # 设置去畸变映射表
        self.set_undist_map()

    def set_undist_map(self):
        """
        计算去畸变映射表 (单目)
        """
        print("\n[CameraCalibration] Computing the undistortion mapping!")
        
        with self.intrinsic_mutex:
            new_K = None
            img_size_cv = self.img_size

            if self.model == CameraModel.PINHOLE:
                # cv2.getOptimalNewCameraMatrix 返回 newCameraMatrix, validPixROI
                new_K, valid_roi = cv2.getOptimalNewCameraMatrix(
                    self.K, self.D, img_size_cv, self.alpha, img_size_cv
                )
                # valid_roi 是 (x, y, w, h)
                self.roi_rect = valid_roi 
                
                self.undist_map_x, self.undist_map_y = cv2.initUndistortRectifyMap(
                    self.K, self.D, None, new_K, img_size_cv, cv2.CV_32FC1
                )

            elif self.model == CameraModel.FISHEYE:
                # OpenCV python binding for fisheye estimateNewCameraMatrix is slightly different usually
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.K, self.D, img_size_cv, np.eye(3),balance=self.alpha
                )
                
                self.undist_map_x, self.undist_map_y = cv2.fisheye.initUndistortRectifyMap(
                    self.K, self.D, np.eye(3), new_K, img_size_cv, cv2.CV_32FC1
                )

            # 更新类成员变量为去畸变后的参数
            self.K = new_K
            self.D = np.zeros(4, dtype=np.float64) # 去畸变后畸变系数为0
            self.k1 = self.k2 = self.p1 = self.p2 = 0.0

            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]

            self.iK = np.linalg.inv(self.K)
            self.ifx = self.iK[0, 0]
            self.ify = self.iK[1, 1]
            self.icx = self.iK[0, 2]
            self.icy = self.iK[1, 2]

            # 更新 ROI mask
            self._update_roi_mask(self.roi_rect)

            print("\n[CameraCalibration] Undist Camera Calibration set as : \n")
            print(f" K = \n{self.K}")
            print(f" D = {self.D.flatten()}")
            print(f" ROI = {self.roi_rect}")


    def _update_roi_mask(self, roi: tuple):
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

    def rectify_image(self, img: np.ndarray) -> np.ndarray:
        """
        对图像进行去畸变
        """
        with self.intrinsic_mutex:
            if self.undist_map_x is not None and self.undist_map_y is not None:
                return cv2.remap(img, self.undist_map_x, self.undist_map_y, cv2.INTER_LINEAR)
            else:
                return img.copy()

    def project_cam_to_image(self, pt_3d: np.ndarray) -> np.ndarray:
        """
        3D点投影到图像 (不考虑畸变，使用当前 K)
        pt_3d: numpy array shape (3,)
        returns: numpy array shape (2,) [u, v]
        """
        with self.intrinsic_mutex:
            # z = pt_3d[2]
            # if abs(z) < 1e-6: return ... 
            
            invz = 1.0 / pt_3d[2]
            u = self.fx * pt_3d[0] * invz + self.cx
            v = self.fy * pt_3d[1] * invz + self.cy
            return np.array([u, v], dtype=np.float32)

    def project_cam_to_image_dist(self, pt_3d: np.ndarray) -> np.ndarray:
        """
        3D点投影到图像 (考虑畸变 D)
        相当于 C++ 中的 projectCamToImageDist
        """
        with self.intrinsic_mutex:
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

    def undistort_image_point(self, pt_2d: np.ndarray) -> np.ndarray:
        """
        将畸变图像上的点去畸变
        pt_2d: [u, v]
        """
        with self.intrinsic_mutex:
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

    def update_intrinsic(self, fx, fy, cx, cy):
        with self.intrinsic_mutex:
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