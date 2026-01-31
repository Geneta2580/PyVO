import pickle
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
import matplotlib.pyplot as plt
import argparse
import sys

def visualize_hessian_and_diagnose(pkl_file):
    print(f"🏥 Analyzing Hessian for: {pkl_file}")

    # =========================================================================
    # 1. 加载数据 & 构建图
    # =========================================================================
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load pickle file: {e}")
        return

    # 1.1 恢复配置
    cfg = data.get('config', data.get('meta', {}).get('config', {}))
    # 处理内参 (兼容 list 或 numpy)
    intr_raw = np.array(cfg.get('intrinsics', [700, 700, 0, 320, 240, 0, 0, 0])).flatten()
    # fx, fy, s, cx, cy...
    if len(intr_raw) >= 5:
        # 假设格式: fx, 0, cx, 0, fy, cy... 或者直接 fx, fy, s, cx, cy
        # 这里使用标准 Cal3_S2 构造: fx, fy, s, cx, cy
        # 注意：你需要根据你的实际存储格式调整索引。通常: fx=0, fy=4, cx=2, cy=5 (3x3矩阵展平)
        if len(intr_raw) == 9:
             K = gtsam.Cal3_S2(intr_raw[0], intr_raw[4], 0.0, intr_raw[2], intr_raw[5])
        else:
             K = gtsam.Cal3_S2(intr_raw[0], intr_raw[1], 0.0, intr_raw[2], intr_raw[3])
    else:
        K = gtsam.Cal3_S2(500, 500, 0, 320, 240)

    visual_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    )
    pose_noise_fix = gtsam.noiseModel.Isotropic.Sigma(6, 1e-5)
    body_T_cam = gtsam.Pose3(np.eye(4))

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    print("📥 Rebuilding Graph from Snapshot...")
    
    # 1.2 恢复变量 (Pose & Point)
    # 为了兼容不同版本的保存格式，先做一次预处理
    variables = data.get('variables', {})
    measurements = data.get('measurements', data.get('factors', []))
    fixed_ids = set(data.get('fixed_ids', data.get('priors', {}).keys()))

    # 加载 Poses
    added_poses = set()
    for var_id, var_info in variables.items():
        if var_info['type'] == 'Pose3':
            # 确保是 numpy 矩阵
            mat = np.array(var_info['initial_value'])
            pose = gtsam.Pose3(mat)
            initial_estimate.insert(X(var_id), pose)
            added_poses.add(var_id)
            
            # 检查是否固定 (Prior)
            is_fixed = var_info.get('is_fixed', False) or (var_id in fixed_ids)
            if is_fixed:
                graph.add(gtsam.PriorFactorPose3(X(var_id), pose, pose_noise_fix))

    # 加载 Points & Factors
    # 先整理观测: mp_id -> [(kf_id, uv), ...]
    mp_obs_map = {}
    for meas in measurements:
        mid = meas['mp_id']
        kid = meas['kf_id']
        uv = np.array(meas['uv'])
        if mid not in mp_obs_map: mp_obs_map[mid] = []
        mp_obs_map[mid].append((kid, uv))

    points_added = 0
    for var_id, var_info in variables.items():
        if var_info['type'] == 'Point3':
            mp_id = var_id
            obs_list = mp_obs_map.get(mp_id, [])
            
            # 忽略观测不足的点
            if len(obs_list) < 2: continue
            
            pt_val = np.array(var_info['initial_value'])
            initial_estimate.insert(L(mp_id), pt_val)
            points_added += 1
            
            for kf_id, uv in obs_list:
                if kf_id in added_poses:
                    graph.add(gtsam.GenericProjectionFactorCal3_S2(
                        uv, visual_noise, X(kf_id), L(mp_id), K, body_T_cam))

    print(f"📊 Graph Stats: {graph.size()} factors, {len(added_poses)} poses, {points_added} points")

    # =========================================================================
    # 2. 线性化 & 获取 Hessian
    # =========================================================================
    print("🔄 Linearizing system...")
    try:
        gaussian_graph = graph.linearize(initial_estimate)
        
        # 关键步骤：构建 Ordering
        # Hessian 的行列顺序完全由 Ordering 决定
        variable_index = gtsam.VariableIndex(graph)
        ordering = gtsam.Ordering.Colamd(variable_index)
        
        print("🧮 Computing Dense Hessian (this may take memory)...")
        hessian_dense = gaussian_graph.augmentedHessian(ordering)
        
        # 去掉最后一行一列 (Error/RHS vector)
        H = hessian_dense[:-1, :-1]
        
    except Exception as e:
        print(f"❌ Linearization or Hessian computation failed: {e}")
        return

    # =========================================================================
    # 3. 特征值分析 (Diagnosis)
    # =========================================================================
    print("🔢 Computing Eigenvalues...")
    # eigh 用于对称矩阵，返回的 eigenvalues 默认是从小到大排序的
    vals, vecs = np.linalg.eigh(H)
    
    min_eig = vals[0]
    max_eig = vals[-1]
    # 防止除以零
    cond_num = max_eig / (min_eig if min_eig > 1e-15 else 1e-15)

    print(f"\n{'='*50}")
    print(f"📉 Min Eigenvalue:   {min_eig:.6e}")
    print(f"📈 Max Eigenvalue:   {max_eig:.6e}")
    print(f"💀 Condition Number: {cond_num:.6e}")
    print(f"{'='*50}")

    if min_eig < 1e-6:
        print("❌ SYSTEM IS SINGULAR / ILL-CONDITIONED!")
        print("   This confirms the presence of degenerate variables.")
    else:
        print("✅ System seems invertible.")

    # =========================================================================
    # 4. 可视化 (Visualization)
    # =========================================================================
    print("🎨 Generating plots...")
    
    # 图 1: 特征值谱
    plt.figure(figsize=(10, 5))
    plt.semilogy(vals, 'b.', markersize=2)
    plt.title(f"Eigenvalue Spectrum (Min: {min_eig:.1e}, Cond: {cond_num:.1e})")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig("eigenvalue_spectrum.png")
    print("   -> Saved 'eigenvalue_spectrum.png'")

    # 图 2: Hessian 热力图 (对数刻度)
    plt.figure(figsize=(10, 10))
    # 使用 log10 显示，因为数值跨度极大
    H_vis = np.log10(np.abs(H) + 1e-15) 
    plt.imshow(H_vis, cmap='viridis', interpolation='none')
    plt.colorbar(label='log10(|H_ij|)')
    plt.title(f"Hessian Heatmap (Size: {H.shape[0]}x{H.shape[1]})")
    plt.tight_layout()
    plt.savefig("hessian_heatmap.png")
    print("   -> Saved 'hessian_heatmap.png'")

    # =========================================================================
    # 5. 🕵️‍♂️ 罪魁祸首定位 (Singularity Identification)
    # =========================================================================
    # 只有当系统奇异时才进行
    if min_eig < 1e-4:
        print("\n🔍 Investigating Null Space (Singularity Source)...")
        
        # 获取对应最小特征值的特征向量
        # vecs 的列对应 vals
        zero_eigenvector = np.abs(vecs[:, 0])
        
        # 找到能量最大的 5 个索引 (这些索引对应 Hessian 矩阵的行/列)
        bad_indices = np.argsort(zero_eigenvector)[::-1][:5]
        
        print("⚠️ Variables causing the singularity (sorted by impact):")
        
        # 构建从 Hessian Index 到 GTSAM Variable 的映射表
        # 必须按照 Ordering 的顺序遍历
        key_map = [] # list of (start_idx, end_idx, variable_string, key_obj)
        current_idx = 0
        
        n_vars = ordering.size()
        for i in range(n_vars):
            key = ordering.at(i) # 使用 .at(i) 获取第 i 个变量的 Key
            
            sym = gtsam.Symbol(key)
            char = sym.chr()
            
            # 确定维度
            dim = 0
            var_type = "Unknown"
            if char == ord('x'): # Pose
                dim = 6
                var_type = "Pose"
            elif char == ord('l'): # Landmark
                dim = 3
                var_type = "Point"
            
            if dim > 0:
                key_map.append({
                    'start': current_idx,
                    'end': current_idx + dim,
                    'name': f"{var_type} {sym.index()} ({chr(char)}{sym.index()})",
                    'key': key
                })
                current_idx += dim
        
        # 查表
        for rank, bad_idx in enumerate(bad_indices):
            impact = zero_eigenvector[bad_idx]
            found_var = None
            
            for entry in key_map:
                if entry['start'] <= bad_idx < entry['end']:
                    found_var = entry
                    break
            
            if found_var:
                # 如果是 Point，尝试打印它的坐标信息
                extra_info = ""
                if "Point" in found_var['name']:
                    try:
                        pt = initial_estimate.atPoint3(found_var['key'])
                        extra_info = f" Pos: [{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}]"
                    except: pass
                    
                print(f"   {rank+1}. Matrix Index {bad_idx:<4} | Impact: {impact:.4f} | {found_var['name']}{extra_info}")
            else:
                print(f"   {rank+1}. Matrix Index {bad_idx:<4} | Impact: {impact:.4f} | Unknown Variable?")

        print("\n💡 Diagnosis:")
        print("   - High impact means this variable can move freely without increasing error.")
        print("   - If it's a Point with Large Z, it confirms the 'Point at Infinity' issue.")
        print("   - If it's a Pose, check PriorFactors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze GTSAM Hessian for Singularities")
    parser.add_argument("pkl_file", help="Path to the debug snapshot .pkl file")
    args = parser.parse_args()
    
    visualize_hessian_and_diagnose(args.pkl_file)