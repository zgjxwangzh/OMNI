import numpy as np
import os

os.chdir('/Users/condenast/Downloads/omni_29dof_v260705/retargeted')

# 列出所有跳高文件
files = [f for f in os.listdir('.') if f.startswith('跳高') and f.endswith('.npz') and '_v' not in f]
files.sort()
print("跳高文件:", files)

# 加载跳高 06 (索引 5)
f = files[5]  # 跳高 06_chr00.npz
print(f"加载文件：{f}")
data = np.load(f)
ja = data['joint_angles']
names = list(data['joint_names'])

print("\n=== v3 跳高 06 关键关节范围 ===")
for name in ['hip_pitch_l_joint', 'hip_roll_l_joint', 'hip_yaw_l_joint', 
             'knee_pitch_l_joint', 'hip_pitch_r_joint', 'hip_roll_r_joint', 'hip_yaw_r_joint']:
    idx = names.index(name)
    min_val = ja[:, idx].min()
    max_val = ja[:, idx].max()
    print(f"{name}: {min_val:.2f} ~ {max_val:.2f} (range: {max_val-min_val:.2f})")

print("\n根位置高度:", data['root_positions'][:, 2].min(), "~", data['root_positions'][:, 2].max())
