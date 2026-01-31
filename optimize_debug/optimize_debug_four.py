import pickle
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
import time
import argparse
import sys
import matplotlib.pyplot as plt
import scipy.sparse as sp

def replay_optimization(pkl_file, use_ordering=False, use_dogleg=False, use_smart=False):
    print(f"\n{'='*60}")
    print(f"🔍 Replaying: {pkl_file}")
    print(f"⚙️  Config: Ordering={use_ordering}, Dogleg={use_dogleg}, SmartFactor={use_smart}")
    print(f"{'='*60}")

    # 1. 加载数据
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # -------------------------------------------------------------------------
    # 2. 恢复配置 (Intrinsics, Noise)
    # -------------------------------------------------------------------------
    cfg = data['config']
    # 处理内参：通常 config 里可能是 list，需要转 numpy
    intr_raw = np.array(cfg['intrinsics']).flatten()
    fx, fy = intr_raw[0], intr_raw[4]
    cx, cy = intr_raw[2], intr_raw[5]
    K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
    
    # 噪声模型
    huber_k = cfg.get('huber_k', 1.345)
    sigma_px = cfg.get('sigma_px', 1.0)
    
    # 视觉噪声
    visual_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(huber_k),
        gtsam.noiseModel.Isotropic.Sigma(2, sigma_px)
    )
    # SmartFactor 需要纯 Isotropic 噪声
    smart_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma_px)
    
    # 固定帧噪声
    pose_noise_fix = gtsam.noiseModel.Isotropic.Precision(6, 1e12)
    
    body_T_cam = gtsam.Pose3(np.eye(4))

    # -------------------------------------------------------------------------
    # 3. 构建 Graph 和 Values
    # -------------------------------------------------------------------------
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    added_poses = set()
    added_points = set()
    fixed_kfs = set()

    # (A) 解析 Variables (Pose & Point)
    # 你的数据结构: variables[id] = {'type': '...', 'initial_value': ..., 'is_fixed': ...}
    
    # 先把 Point 数据分离出来，方便 SmartFactor 处理
    point_data_map = {} 
    
    for var_id, var_info in data['variables'].items():
        if var_info['type'] == 'Pose3':
            # 添加 Pose 初值
            pose_mat = var_info['initial_value']
            initial_estimate.insert(X(var_id), gtsam.Pose3(pose_mat))
            added_poses.add(var_id)
            
            # 记录是否为固定帧
            if var_info.get('is_fixed', False):
                fixed_kfs.add(var_id)
                # 添加 Prior
                graph.add(gtsam.PriorFactorPose3(
                    X(var_id), gtsam.Pose3(pose_mat), pose_noise_fix
                ))
                
        elif var_info['type'] == 'Point3':
            point_data_map[var_id] = var_info['initial_value']

    # (B) 解析 Measurements 并添加因子
    # 你的数据结构: measurements = [{'kf_id':..., 'mp_id':..., 'uv':...}, ...]

    # 预处理：按 mp_id 归类观测，方便 SmartFactor
    mp_observations = {}
    for meas in data['measurements']:
        mid = meas['mp_id']
        kid = meas['kf_id']
        uv = meas['uv']
        if mid not in mp_observations: mp_observations[mid] = []
        mp_observations[mid].append((kid, uv))

    # 开始添加视觉因子
    for mp_id, point_w in point_data_map.items():
        obs_list = mp_observations.get(mp_id, [])
        if len(obs_list) < 2: continue # 复现时忽略单次观测

        if use_smart:
            # === Smart Factor 模式 ===
            smart_params = gtsam.SmartProjectionParams()
            # smart_params.setRankTolerance(1e-9)
            
            factor = gtsam.SmartProjectionPoseFactorCal3_S2(smart_noise, K, body_T_cam, smart_params)
            for kid, uv in obs_list:
                if kid in added_poses: # 确保观测帧在图里
                    factor.add(uv, X(kid))
            
            graph.add(factor)
            # Smart 模式不需要 insert L(mp_id) 到 initial_estimate
            
        else:
            # === Standard Factor 模式 ===

            # 如果是那个导致 Dogleg 崩溃的点，直接跳过
            if mp_id == 2463: 
                print(f"🔪 Manually removing bad point {mp_id} for testing...")
                continue

            initial_estimate.insert(L(mp_id), point_w)
            added_points.add(mp_id)
            
            for kid, uv in obs_list:
                if kid in added_poses:
                    graph.add(gtsam.GenericProjectionFactorCal3_S2(
                        uv, visual_noise, X(kid), L(mp_id), K, body_T_cam
                    ))

    print(f"📊 Graph Stats: {graph.size()} factors, {len(added_poses)} poses, {len(added_points)} points.")

    # -------------------------------------------------------------------------
    # 4. 配置优化器
    # -------------------------------------------------------------------------
    if use_dogleg:
        params = gtsam.DoglegParams()
        params.setDeltaInitial(1.0)
    else:
        params = gtsam.LevenbergMarquardtParams()
    
    # 通用参数 (宽松一点以模拟实时性)
    params.setMaxIterations(5)
    params.setRelativeErrorTol(1e-3)
    params.setAbsoluteErrorTol(1e-3)

    # Ordering 配置 (仅 LM + Standard Factor)
    if use_ordering and not use_smart and not use_dogleg:
        print("⚡ Applying Ordering: Points First -> Poses Last")
        ordering = gtsam.Ordering()
        # 1. 先消元 Points
        for mid in added_points: ordering.push_back(L(mid))
        # 2. 后消元 Poses
        for kid in added_poses: ordering.push_back(X(kid))
        params.setOrdering(ordering)

    # -------------------------------------------------------------------------
    # 5. 执行优化
    # -------------------------------------------------------------------------
    print("🚀 Starting Optimization...")
    try:
        t_start = time.time()
        
        if use_dogleg:
            optimizer = gtsam.DoglegOptimizer(graph, initial_estimate, params)
        else:
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
            
        result = optimizer.optimize()
        t_end = time.time()
        
        duration_ms = (t_end - t_start) * 1000
        print(f"✅ Success! Time: {duration_ms:.2f} ms")
        print(f"📉 Initial Error: {graph.error(initial_estimate):.2f}")
        print(f"📉 Final Error:   {graph.error(result):.2f}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file", help="Path to snapshot .pkl file")
    args = parser.parse_args()

    # 依次运行 4 种方案进行对比
    
    print("\n[Test 1: Default LM (Baseline)] - 你现在的慢速配置")
    replay_optimization(args.pkl_file, use_ordering=False, use_dogleg=False)

    print("\n[Test 2: LM + Ordering] - 模拟 Ceres Sparse Schur")
    replay_optimization(args.pkl_file, use_ordering=True, use_dogleg=False)

    print("\n[Test 3: Dogleg] - 更强的鲁棒性")
    replay_optimization(args.pkl_file, use_ordering=False, use_dogleg=True)
    
    print("\n[Test 4: SmartFactor] - 隐式消元")
    replay_optimization(args.pkl_file, use_smart=True)