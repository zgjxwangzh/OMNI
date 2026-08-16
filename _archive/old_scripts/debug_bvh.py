"""调试 BVH 解析：通道数、offset、第一帧值、FK 链"""
import sys
import numpy as np
sys.path.insert(0, '/Users/condenast/Downloads/omni_29dof_v260705')
from bvh_retarget import BVHParser

p = BVHParser('/Users/condenast/Downloads/omni_29dof_v260705/第一组 跳高 翻箱/跳高06_chr00.bvh')

print("\n=== 各关节通道数 ===")
total = 0
for j in p.joints:
    ch = p.joint_channels.get(j, [])
    total += len(ch)
    if j in ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Spine', 'LeftArm', 'LeftShoulder']:
        print(f"  {j:<15} {len(ch)} ch: {ch}")
print(f"  总通道数: {total} (实际 {p.frames.shape[1]})")

print("\n=== 关键 offset ===")
for j in ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase']:
    print(f"  {j:<15} {p.joint_offsets.get(j)}")

jd = p.get_joint_data(0)
print("\n=== 第一帧关键值 ===")
for j in ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot']:
    print(f"  {j:<15} {jd[j]['values']}")

g = p.get_global_transforms(jd)
print("\n=== 第一帧全局位置 (BVH cm) ===")
for j in ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase']:
    print(f"  {j:<15} {np.round(g[j][1], 1)}")
