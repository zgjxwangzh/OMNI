#!/usr/bin/env python3
"""FK 自洽性验证: NPZ joint_pos(BFS 序假设) → URDF FK → 与 NPZ body_pos_w 对比。
同时验证 MJCF 关节序 vs NPZ 列序的映射。
"""
import sys, os
sys.path.insert(0, "liuzq_jump_train_v11/scripts")
import numpy as np
import modify_npz as M

URDF = "liuzq_jump_train_v11/omni_29dof_v260705/assets/omni_29dof_nohead_noshoe/urdf/omni_29dof_nohead_noshoe_merged_modify_feet.urdf"
NPZ = "liuzq_jump_train_v11/motion/jump_high_firstjump_50fps.npz"

joints, children_map, body_index = M.parse_urdf(URDF)
print(f"URDF joints (in file order): {len(joints)}")
for i, j in enumerate(joints):
    print(f"  {i:2d} {j['name']:28s} parent={j['parent']:2d} child={j['child']:2d}")

d = np.load(NPZ)
jp, bp, bq = d['joint_pos'], d['body_pos_w'], d['body_quat_w']
T = jp.shape[0]

def fk_err(jp_frames):
    errs = np.zeros((jp_frames.shape[0], 30))
    for t in range(jp_frames.shape[0]):
        bpf, bqq = M.compute_fk(jp_frames[t], joints, children_map)
        # FK 结果以原点为根 → 变换到 NPZ root pose
        rq = bq[t, 0]
        bpf = np.array([bp[t, 0] + M.quat_rotate(rq, bpf[i]) for i in range(30)])
        bqq = np.array([M.quat_mul(rq, bqq[i]) for i in range(30)])
        errs[t] = np.linalg.norm(bpf - bp[t], axis=-1)
    return errs

# 全帧 FK
errs = fk_err(jp)
print(f"\n全帧 FK: mean={errs.mean():.4f}m max={errs.max():.4f}m")
print("逐 body 平均误差 (>1cm):")
for b in range(30):
    e = errs[:, b].mean()
    if e > 0.01:
        print(f"  [{b:2d}] {M.BODY_NAMES[b]:28s} {e:.4f}m")
