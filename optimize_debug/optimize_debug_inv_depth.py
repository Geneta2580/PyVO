import pickle
import numpy as np
import gtsam
import time
import argparse
import sys

# 尝试导入 unstable (逆深度因子通常在这里)
try:
    import gtsam_unstable
except ImportError:
    print("❌ Error: gtsam_unstable module not found. Inverse Depth Factors require it.")
    sys.exit(1)

from gtsam.symbol_shorthand import X, L

def world_to_inv_depth_vector5(point_w, anchor_pose_wc):
    """
    将世界坐标系下的 3D 点转换为相对于锚点帧的逆深度 Vector5
    Vector5 格式通常为: [theta, phi, rho, u, v] (gtsam_unstable 约定)
    """
    # 1. 转换到锚点相机坐标系
    # T_cw = T_wc.inverse()
    # P_c = T_cw * P_w
    anchor_pose_cw = anchor_pose_wc.inverse()
    p_c = anchor_pose_cw.transformFrom(point_w) # Point3
    
    x, y, z = p_c[0], p_c[1], p_c[2]
    
    # 2. 计算球坐标 (theta, phi) 和 逆深度 (rho)
    # 假设 Z 是光轴
    if z < 1e-3: return None # 或者是背后点

    # 具体参数化取决于 Variant3 的定义，通常是：
    # theta:方位角 (azimuth), phi:仰角 (elevation)
    # x = z * tan(theta) -> theta = atan2(x, z)
    # y = z * tan(phi) / cos(theta) -> phi = atan2(y, sqrt(x^2 + z^2))
    
    theta = np.arctan2(x, z)
    phi = np.arctan2(y, np.sqrt(x*x + z*z))
    rho = 1.0 / z
    
    # 这里我们不需要计算 u, v，因为因子初始化时只需要 theta, phi, rho
    # 但是 Variant3 的 Variable 可能包含锚点像素坐标作为状态的一部分（取决于具体实现）
    # 在 Python wrapper 中，通常只需要初始化前 3 维，或者给 5 维 (后两维补0或补uv)
    # 为了安全，我们返回 5 维向量，后两位填 0 (因为 uv 是作为观测值存在 factor 里的，而不是变量)
    
    return np.array([theta, phi, rho])

def run_inv_depth_benchmark(pkl_file):
    print(f"\n{'='*60}")
    print(f"🧪 Benchmarking Inverse Depth (InvDepthFactorVariant3a/b)")
    print(f"{'='*60}")

    # 1. 加载数据
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # 恢复配置
    cfg = data.get('config', data.get('meta', {}).get('config', {}))
    intr_raw = np.array(cfg.get('intrinsics', [700, 700, 0, 320, 240, 0, 0, 0])).flatten()
    # 适配内参格式
    if len(intr_raw) == 9: fx, fy, cx, cy = intr_raw[0], intr_raw[4], intr_raw[2], intr_raw[5]
    else: fx, fy, cx, cy = intr_raw[0], intr_raw[1], intr_raw[2], intr_raw[3]
    K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
    
    # 噪声模型 (必须是 Robust)
    visual_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(2, 1.0) # sigma=1.0
    )
    
    pose_noise_fix = gtsam.noiseModel.Constrained.All(6)
    body_T_cam = gtsam.Pose3(np.eye(4)) # 假设已经是相机坐标系，或者 T_bc 为 Identity

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    added_poses = {} # id -> Pose3
    fixed_ids = set(data.get('fixed_ids', []))

    # (A) 恢复 Pose
    print("📥 Loading Poses...")
    variables = data.get('variables', {})
    measurements = data.get('measurements', data.get('factors', []))
    
    for var_id, var_info in variables.items():
        if var_info['type'] == 'Pose3':
            pose = gtsam.Pose3(np.array(var_info['initial_value']))
            initial_estimate.insert(X(var_id), pose)
            added_poses[var_id] = pose
            
            if var_info.get('is_fixed', False) or var_id in fixed_ids:
                graph.add(gtsam.PriorFactorPose3(X(var_id), pose, pose_noise_fix))

    # (B) 转换 Point -> Inverse Depth 并构建因子
    print("🔄 Converting Points to Inverse Depth & Building Graph...")
    
    # 整理观测
    mp_obs_map = {}
    for meas in measurements:
        mid, kid, uv = meas['mp_id'], meas['kf_id'], meas['uv']
        if mid not in mp_obs_map: mp_obs_map[mid] = []
        mp_obs_map[mid].append((kid, np.array(uv))) # (kf_id, uv)

    points_added = 0
    factors_added = 0
    
    for var_id, var_info in variables.items():
        if var_info['type'] == 'Point3':
            mp_id = var_id
            obs_list = mp_obs_map.get(mp_id, [])
            
            # 过滤无效点
            valid_obs = [o for o in obs_list if o[0] in added_poses]
            if len(valid_obs) < 2: continue
            
            # 1. 确定锚点帧 (Anchor Frame) - 通常取第一次观测的帧
            # 按关键帧 ID 排序，确保因果一致性
            valid_obs.sort(key=lambda x: x[0]) 
            anchor_kf_id, anchor_uv = valid_obs[0]
            anchor_pose_wc = added_poses[anchor_kf_id] # 假设 body_T_cam = I，如果是 VIO 需要转换
            
            # 2. 初始化变量 (Vector5)
            pt_w = np.array(var_info['initial_value'])
            inv_vec5 = world_to_inv_depth_vector5(pt_w, anchor_pose_wc)
            
            if inv_vec5 is None: continue # 转换失败（如 Z<0）
            
            initial_estimate.insert(L(mp_id), inv_vec5)
            points_added += 1
            
            # 3. 添加因子
            # 3.1 锚点因子 (Variant3a) - 处理锚点帧的观测
            # 参数: (PoseKey, LandmarkKey, pixel, K, noise, body_T_cam)
            f_3a = gtsam_unstable.InvDepthFactorVariant3a(
                X(anchor_kf_id), L(mp_id), 
                anchor_uv, K, visual_noise, body_T_cam
            )
            graph.add(f_3a)
            factors_added += 1
            
            # 3.2 其他帧因子 (Variant3b)
            for kf_id, uv in valid_obs[1:]:
                # 参数: (AnchorPoseKey, CurrentPoseKey, LandmarkKey, pixel, K, noise, body_T_cam)
                f_3b = gtsam_unstable.InvDepthFactorVariant3b(
                    X(anchor_kf_id), X(kf_id), L(mp_id), 
                    uv, K, visual_noise, body_T_cam
                )
                graph.add(f_3b)
                factors_added += 1

    print(f"📊 Graph Stats: {factors_added} factors, {len(added_poses)} poses, {points_added} inv-depth points")

    # (C) 优化
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(5)
    params.setRelativeErrorTol(1e-3)
    
    print("🚀 Starting Optimization (LM)...")
    try:
        t1 = time.time()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
        t2 = time.time()
        
        duration = (t2 - t1) * 1000
        print(f"✅ Optimization Successful!")
        print(f"⏱️  Time: {duration:.2f} ms")
        print(f"📉 Initial Error: {graph.error(initial_estimate):.2f}")
        print(f"📉 Final Error:   {graph.error(result):.2f}")
        
        # 简单检查 Point 2463 (如果存在)
        suspect_id = 2463
        if result.exists(L(suspect_id)):
            vec = result.atVector(L(suspect_id))
            rho = vec[2]
            print(f"\n🧐 Suspect Point {suspect_id} Analysis:")
            print(f"   Optimized Rho (InvDepth): {rho:.6f}")
            print(f"   Equivalent Depth: {1.0/rho if abs(rho)>1e-6 else 'Infinity'} m")
            if abs(rho) < 1e-4:
                print("   ✅ Point successfully converged to 'Infinity' (Valid State)!")
            else:
                print("   ℹ️ Point has a finite depth.")

    except Exception as e:
        print(f"❌ Optimization Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file", help="Path to snapshot .pkl file")
    args = parser.parse_args()
    
    run_inv_depth_benchmark(args.pkl_file)