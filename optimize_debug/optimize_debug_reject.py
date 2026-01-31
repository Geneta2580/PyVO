import pickle
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
import time
import argparse

import pickle
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
import matplotlib.pyplot as plt
import scipy.linalg

def run_verification(pkl_file, apply_fix=False):
    print(f"\n{'='*60}")
    print(f"🧪 Mode: {'WITH FIX (Parallax Check)' if apply_fix else 'NO FIX (Baseline)'}")
    print(f"{'='*60}")

    # 1. 加载数据
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # ---------------------------------------------------------
    # 适配你的数据结构 (无 meta)
    # ---------------------------------------------------------
    cfg = data['config']
    
    # 恢复内参
    intr_raw = np.array(cfg['intrinsics']).reshape(3,3)
    fx, fy = intr_raw[0,0], intr_raw[1,1]
    cx, cy = intr_raw[0,2], intr_raw[1,2]
    K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
    
    # 恢复噪声模型
    huber_k = cfg.get('huber_k', 1.345)
    sigma_px = cfg.get('sigma_px', 1.0)
    
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(huber_k),
        gtsam.noiseModel.Isotropic.Sigma(2, sigma_px)
    )
    pose_fix_noise = gtsam.noiseModel.Constrained.All(6)
    body_T_cam = gtsam.Pose3(np.eye(4))

    # ---------------------------------------------------------
    # 2. 准备图和初值
    # ---------------------------------------------------------
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    added_poses = set()
    added_points = set()
    
    # 缓存 Pose 对象用于计算基线
    kf_poses_cache = {} 

    # (A) 恢复 Poses & Priors
    # 遍历 variables 字典，提取 type='Pose3' 的项
    for var_id, var_info in data['variables'].items():
        if var_info['type'] == 'Pose3':
            pose_mat = var_info['initial_value']
            pose = gtsam.Pose3(pose_mat)
            
            initial.insert(X(var_id), pose)
            kf_poses_cache[var_id] = pose
            added_poses.add(var_id)
            
            # 检查 is_fixed 标记
            if var_info.get('is_fixed', False):
                graph.add(gtsam.PriorFactorPose3(X(var_id), pose, pose_fix_noise))

    # (B) 整理观测数据
    # data['measurements'] 是 list，转为 mp_id -> [(kf, uv), ...]
    mp_obs_map = {}
    for meas in data['measurements']:
        mid = meas['mp_id']
        kid = meas['kf_id']
        uv = meas['uv']
        if mid not in mp_obs_map: mp_obs_map[mid] = []
        mp_obs_map[mid].append((kid, uv))

    # ---------------------------------------------------------
    # 3. 添加 Points (核心验证逻辑)
    # ---------------------------------------------------------
    rejected_count = 0
    suspect_id = 2463 # 重点关注对象 (如果ID变了请修改这里)

    # 遍历 variables 字典，提取 type='Point3' 的项
    for var_id, var_info in data['variables'].items():
        if var_info['type'] != 'Point3': continue
        
        mid = var_id
        pt_w = var_info['initial_value']
        obs = mp_obs_map.get(mid, [])

        # 忽略观测不足的点
        if len(obs) < 2: continue

        should_add = True
        
        # === 核心验证逻辑 ===
        if apply_fix:
            kf1_id = obs[0][0]
            kf2_id = obs[-1][0]
            
            # 确保观测帧在 Pose 列表里 (防止数据不一致)
            if kf1_id in kf_poses_cache and kf2_id in kf_poses_cache:
                p1 = kf_poses_cache[kf1_id].translation()
                p2 = kf_poses_cache[kf2_id].translation()
                
                # 1. 计算基线 (Baseline)
                baseline = np.linalg.norm(p1 - p2)
                
                # 2. 计算深度 (Depth)
                dist = np.linalg.norm(pt_w - p1)
                
                # --- 判据 A: 绝对基线过短 (例如悬停) ---
                if baseline < 0.1: # 5cm
                    print(f"🔪 [KILL] Suspect {mid} removed! Baseline too short: {baseline:.4f}m")
                    should_add = False
                
                # --- 判据 B: 视差比例过小 (针对远点) ---
                elif dist > 1e-4:
                    ratio = baseline / dist
                    if ratio < 0.1: # 1% 视差
                        print(f"🔪 [KILL] Suspect {mid} removed! Ratio too small: {ratio:.6f} (B={baseline:.3f}, D={dist:.1f})")
                        should_add = False

                    elif dist > 100:
                        print(f"🔪 [KILL] Suspect {mid} removed! Depth too large: {dist:.1f}m")
                        should_add = False
                    
        
        # === 添加到图 ===
        if should_add:
            initial.insert(L(mid), pt_w)
            added_points.add(mid)
            
            for kid, uv in obs:
                if kid in added_poses:
                    graph.add(gtsam.GenericProjectionFactorCal3_S2(
                        uv, noise, X(kid), L(mid), K, body_T_cam
                    ))
        else:
            rejected_count += 1

    print(f"📊 Stats: {len(added_poses)} Poses, {len(added_points)} Points. (Rejected {rejected_count} bad points)")

    # ---------------------------------------------------------
    # 4. 执行优化
    # ---------------------------------------------------------
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(5)
    params.setRelativeErrorTol(1e-3)
    params.setAbsoluteErrorTol(1e-3)
    params.setVerbosityLM("SUMMARY")
    
    try:
        t1 = time.time()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()
        t2 = time.time()
        print(f"✅ Optimization Time: {(t2-t1)*1000:.2f} ms")
        print(f"📉 Final Error: {graph.error(result):.2f}")
        
    except Exception as e:
        print(f"❌ Optimization Failed/Slow: {e}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file", help="Path to snapshot .pkl file")
    args = parser.parse_args()
    
    # 1. 先跑一次不带修复的 (还原现场 - 应该很慢)
    run_verification(args.pkl_file, apply_fix=False)
    
    # 2. 再跑一次带修复的 (验证疗效 - 应该很快)
    run_verification(args.pkl_file, apply_fix=True)