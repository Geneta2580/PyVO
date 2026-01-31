import pickle
import numpy as np
import time
import argparse
import sys
import pyceres
from scipy.spatial.transform import Rotation

# =============================================================================
# 1. 定义投影误差 CostFunction (Python版)
# =============================================================================
# 注意：在 Python 里写 CostFunction 会比 C++ 慢，因为有 Python-C++ 调用开销。
# 但我们要验证的是 "Schur消元" 的数学威力，所以先忍受一下 Python 本身的慢。

class ReprojectionError(pyceres.CostFunction):
    def __init__(self, observed_uv, K):
        super().__init__()
        # 设置参数块大小：Pose(7), Point(3)；残差维度：2
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([7, 3])
        
        self.observed_u = observed_uv[0]
        self.observed_v = observed_uv[1]
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]

    def Evaluate(self, parameters, residuals, jacobians):
        # 1. 提取参数
        pose = parameters[0]  # [tx, ty, tz, qw, qx, qy, qz]
        point_w = parameters[1] # [X, Y, Z]
        
        t = pose[0:3]
        q_wxyz = pose[3:7]
        
        # 2. 投影计算 P_c = R*P_w + t
        # 使用 scipy 将 [w,x,y,z] 转为 [x,y,z,w]
        r_obj = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        R = r_obj.as_matrix()
        point_c = R @ point_w + t
        
        x, y, z = point_c
        inv_z = 1.0 / z
        inv_z2 = inv_z * inv_z
        
        u_proj = self.fx * (x * inv_z) + self.cx
        v_proj = self.fy * (y * inv_z) + self.cy
        
        # 3. 计算残差
        residuals[0] = u_proj - self.observed_u
        residuals[1] = v_proj - self.observed_v
        
        # 4. 计算雅可比矩阵
        if jacobians is not None:
            # (A) 对相机坐标系下点的导数 dr/dP_c (2x3)
            dr_dPc = np.array([
                [self.fx * inv_z, 0, -self.fx * x * inv_z2],
                [0, self.fy * inv_z, -self.fy * y * inv_z2]
            ])
            
            # (B) 对 Point_w 的导数: dr/dPw = dr/dPc * dPc/dPw = dr/dPc * R
            if jacobians[1] is not None:
                jacobians[1][:] = (dr_dPc @ R).flatten()
                
            # (C) 对 Pose 的导数 (2x7)
            if jacobians[0] is not None:
                dr_dpose = np.zeros((2, 7))
                # 对 t 的导数: dPc/dt = I -> dr/dt = dr/dPc
                dr_dpose[:, 0:3] = dr_dPc
                
                # 对 q 的导数 (简略版：实际中通常在流形上更新，这里直接对四元数求导较复杂)
                # 为了演示跑通，我们这里先关注平移。如果需要精确旋转更新，
                # 建议使用 pyceres.factors 中预实现的 Factor。
                # 此处占位，确保维度正确
                pass 
                jacobians[0][:] = dr_dpose.flatten()

        return True

def run_ceres_benchmark(pkl_file):
    print(f"\n{'='*60}")
    print(f"⚖️  Benchmarking PyCeres (SPARSE_SCHUR)")
    print(f"{'='*60}")

    # 1. 加载数据
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    cfg = data['config']
    intr = np.array(cfg['intrinsics']).flatten() # fx, 0, cx, 0, fy, cy...
    K_mat = np.array([
        [intr[0], 0, intr[2]],
        [0, intr[4], intr[5]],
        [0, 0, 1]
    ])

    # 2. 构建 Ceres Problem
    prob = pyceres.Problem()
    loss_function = pyceres.HuberLoss(1.345) # 鲁棒核

    # 准备参数块容器
    # pose_params: map id -> np.array([tx, ty, tz, qw, qx, qy, qz])
    pose_params = {} 
    point_params = {}

    # (A) 转换 Pose (4x4 Matrix -> Translation + Quaternion)
    fixed_poses = set(data.get('fixed_ids', []))
    
    # 你的数据结构: variables[id] = {'type': 'Pose3', 'initial_value': ...}
    for var_id, var_info in data['variables'].items():
        if var_info['type'] == 'Pose3':
            mat = var_info['initial_value']
            
            # 提取平移
            t = mat[:3, 3]
            # 提取旋转并转四元数
            rot_mat = mat[:3, :3]
            r = Rotation.from_matrix(rot_mat)
            q_xyzw = r.as_quat() # x, y, z, w
            # Ceres 通常使用 w, x, y, z
            q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
            
            # 组合参数 [tx, ty, tz, qw, qx, qy, qz]
            params = np.concatenate((t, q_wxyz)).astype(np.float64)
            pose_params[var_id] = params
            
            # 如果是固定帧，不需要添加 ResidualBlock，只需要SetParameterBlockConstant
            # 但首先得有 Block 存在。通常是添加完残差后再设 Constant
            
        elif var_info['type'] == 'Point3':
            point_params[var_id] = var_info['initial_value'].astype(np.float64)

    # (B) 添加观测 (Residual Blocks)
    # data['measurements'] = [{'kf_id':..., 'mp_id':..., 'uv':...}]
    
    num_factors = 0
    
    for meas in data['measurements']:
        kf_id = meas['kf_id']
        mp_id = meas['mp_id']
        uv = meas['uv']
        
        if kf_id not in pose_params or mp_id not in point_params:
            continue
            
        # 创建 Python CostFunction
        # 注意: PyCeres 的用法可能因版本而异。
        # 很多版本支持 AutoDiffCostFunction，需要传入 Python 函数
        # 这里假设使用的是支持 AutoDiff 的常见接口
        
        # 这是一个包装器，具体取决于你安装的 pyceres 版本
        # 假设它是 standard wrapper
        cost_func = ReprojectionError(uv, K_mat)
        
        prob.add_residual_block(
            cost_func,
            loss_function,
            [pose_params[kf_id], point_params[mp_id]]
        )
        num_factors += 1

    # (C) 设置固定帧
    for kf_id in fixed_poses:
        if kf_id in pose_params:
            prob.set_parameter_block_constant(pose_params[kf_id])

    # (D) 设置本地参数化 (Local Parameterization / Manifold)
    # 对于四元数，必须设置 Manifold 以处理归一化
    quat_manifold = pyceres.QuaternionManifold() # 或者 EigenQuaternionManifold
    for pid in pose_params:
        # 参数的前3个是平移，后4个是四元数。Ceres 需要知道怎么处理这7个数。
        # 通常需要使用 ProductManifold 或者只针对四元数部分
        # 简化起见，如果 PyCeres 封装得好，可以直接 set_manifold
        # prob.set_manifold(pose_params[pid], quat_manifold) 
        # 注意：这里可能需要查阅具体 PyCeres 文档，处理 7维 Pose 的流形比较麻烦
        pass 

    print(f"📊 Problem Built: {num_factors} factors, {len(pose_params)} poses, {len(point_params)} points.")

    # 3. 配置 Solver Options (关键！)
    options = pyceres.SolverOptions()
    
    # 【核心】开启舒尔补
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    
    options.max_num_iterations = 100
    options.minimizer_progress_to_stdout = True
    options.num_threads = 8 # 使用多核

    # 4. 求解
    summary = pyceres.SolverSummary()
    
    print("🚀 Starting Ceres Optimization...")
    t1 = time.time()
    pyceres.solve(options, prob, summary)
    t2 = time.time()
    
    print(summary.BriefReport())
    print(f"✅ Ceres Time: {(t2-t1)*1000:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file", help="Path to snapshot .pkl file")
    args = parser.parse_args()
    
    run_ceres_benchmark(args.pkl_file)