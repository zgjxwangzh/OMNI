#!/usr/bin/env python3
"""判别 NPZ joint_pos 列序: 假设A(modify_npz BFS含neck/head) vs 假设B(robot BFS含ankle_roll)。
两种映射各做全帧 FK, 对比 body_pos_w 误差; 再用 lzq normalizer mean 交叉验证。
"""
import sys
sys.path.insert(0, "liuzq_jump_train_v11/scripts")
import numpy as np
import modify_npz as M

URDF = "liuzq_jump_train_v11/omni_29dof_v260705/assets/omni_29dof_nohead_noshoe/urdf/omni_29dof_nohead_noshoe_merged_modify_feet.urdf"
NPZ = "training_data/jump_high_firstjump_50fps.npz"

joints, children_map, _ = M.parse_urdf(URDF)
d = np.load(NPZ)
jp, bp, bq = d['joint_pos'], d['body_pos_w'], d['body_quat_w']
T = jp.shape[0]

# modify_npz 的 FK 按 NPZ_JOINT_NAMES 名字索引角度。
# 构造假设: 给每个 URDF 关节名一个角度来源列(或 None=0)。
# URDF 29 关节 (无 neck/head):
URDF_JOINTS = [j['name'] for j in joints]

# 假设 A: col i = NPZ_JOINT_NAMES[i] (shoulder_yaw@17/18, 无ankle_roll, 27/28=neck/head)
A_MAP = {name: i for i, name in enumerate(M.NPZ_JOINT_NAMES)}  # name → col
# 假设 B: col = robot BFS 序 (ankle_roll@17/18, shoulder_yaw@19/20, ..., wrist_roll@27/28)
BFS_ROBOT = [
    "hip_pitch_l_joint", "hip_pitch_r_joint", "waist_yaw_joint",
    "hip_roll_l_joint", "hip_roll_r_joint", "waist_roll_joint",
    "hip_yaw_l_joint", "hip_yaw_r_joint", "waist_pitch_joint",
    "knee_pitch_l_joint", "knee_pitch_r_joint",
    "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",
    "ankle_pitch_l_joint", "ankle_pitch_r_joint",
    "shoulder_roll_l_joint", "shoulder_roll_r_joint",
    "ankle_roll_l_joint", "ankle_roll_r_joint",
    "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_l_joint", "elbow_pitch_r_joint",
    "elbow_yaw_l_joint", "elbow_yaw_r_joint",
    "wrist_pitch_l_joint", "wrist_pitch_r_joint",
    "wrist_roll_l_joint", "wrist_roll_r_joint",
]
B_MAP = {name: i for i, name in enumerate(BFS_ROBOT)}


def fk_with_map(col_map, frames):
    """col_map: joint_name → NPZ col. 覆盖 modify_npz FK 的角度取法。"""
    errs = np.zeros((frames.shape[0], 30))
    # 构造每帧的角度向量(按 NPZ_JOINT_NAMES 索引的29维)
    for t in range(frames.shape[0]):
        ang29 = np.zeros(29)
        for name, col in col_map.items():
            if name in M.NPZ_JOINT_NAMES:
                ang29[M.NPZ_JOINT_NAMES.index(name)] = frames[t, col]
        bpf, bqq = M.compute_fk(ang29, joints, children_map)
        rq = bq[t, 0]
        bpf = np.array([bp[t, 0] + M.quat_rotate(rq, bpf[i]) for i in range(30)])
        errs[t] = np.linalg.norm(bpf - bp[t], axis=-1)
    return errs


for label, m in [("假设A (shoulder_yaw@17/18, 无ankle_roll)", A_MAP),
                 ("假设B (ankle_roll@17/18, robot BFS)", B_MAP)]:
    errs = fk_with_map(m, jp)
    per = errs.mean(axis=0)
    print(f"\n[{label}]")
    print(f"  FK 误差: mean={errs.mean():.4f}m max={errs.max():.4f}m")
    worst = np.argsort(-per)[:6]
    for b in worst:
        print(f"    [{b:2d}] {M.BODY_NAMES[b]:26s} {per[b]:.4f}m")
