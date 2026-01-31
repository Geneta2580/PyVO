import pickle
import numpy as np
import gtsam

# 1. 加载数据
with open("debug_snapshot_1.04s_8526factors.pkl", "rb") as f:
    data = pickle.load(f)

# 2. 找到 l2463 (注意：Symobol l2463 对应的 index 就是 2463)
bad_id = 2463

if bad_id in data['variables']:
    print(f"💀 Found Suspect: Point {bad_id}")
    pt_val = data['variables'][bad_id]['initial_value']
    print(f"   Initial Pos: {pt_val}")
    
    # 3. 找观测帧
    obs_kfs = []
    for meas in data['measurements']:
        if meas['mp_id'] == bad_id:
            obs_kfs.append(meas['kf_id'])
    
    print(f"   Observed by KFs: {obs_kfs}")
    
    # 4. 计算视差 (基线)
    if len(obs_kfs) >= 2:
        kf1 = data['variables'][obs_kfs[0]]['initial_value'] # Pose Matrix
        kf2 = data['variables'][obs_kfs[-1]]['initial_value']
        
        pos1 = kf1[:3, 3]
        pos2 = kf2[:3, 3]
        
        baseline = np.linalg.norm(pos1 - pos2)
        dist = np.linalg.norm(pt_val - pos1)
        
        print(f"   Baseline: {baseline:.6f} m")
        print(f"   Depth:    {dist:.6f} m")
        print(f"   Ratio:    {baseline/dist if dist>0 else 'inf'}")
    else:
        print("   ❌ Only 1 observation! (Should have been filtered!)")
else:
    print("Point not found in variables?")