#!/usr/bin/env python3
"""修改跳高参考运动 NPZ: 修复手掌朝上、肘过弯、hip_roll 不对称。

流程:
  1. 从 URDF 提取运动学树
  2. 实现纯 numpy FK
  3. 验证 FK 结果与原始 body_pos_w 一致
  4. 修改 joint_pos (wrist_pitch / elbow_pitch / hip_roll)
  5. FK 重算 body_pos_w / body_quat_w
  6. 数值差分重算所有 vel
  7. 保存新 NPZ + 校验

用法:
  python modify_npz.py [--verify-only] [--output OUTPUT]
"""

import argparse
import xml.etree.ElementTree as ET
import numpy as np
from collections import deque

# ============================================================
# 配置
# ============================================================
URDF_PATH = "omni_29dof_v260705/assets/omni_29dof_nohead_noshoe/urdf/omni_29dof_nohead_noshoe_merged_modify_feet.urdf"
NPZ_PATH = "motion/jump_high_firstjump_50fps.npz"

# NPZ joint 顺序 (29 joints, 与 URDF joint 顺序一致)
NPZ_JOINT_NAMES = [
    "hip_pitch_l_joint", "hip_pitch_r_joint",       # 0, 1
    "waist_yaw_joint", "hip_roll_l_joint", "hip_roll_r_joint",  # 2, 3, 4
    "waist_roll_joint", "hip_yaw_l_joint", "hip_yaw_r_joint",  # 5, 6, 7
    "waist_pitch_joint",                              # 8
    "knee_pitch_l_joint", "knee_pitch_r_joint",      # 9, 10
    "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",  # 11, 12
    "ankle_pitch_l_joint", "ankle_pitch_r_joint",    # 13, 14
    "shoulder_roll_l_joint", "shoulder_roll_r_joint",  # 15, 16
    "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",  # 17, 18
    "elbow_pitch_l_joint", "elbow_pitch_r_joint",    # 19, 20
    "elbow_yaw_l_joint", "elbow_yaw_r_joint",        # 21, 22
    "wrist_pitch_l_joint", "wrist_pitch_r_joint",    # 23, 24
    "wrist_roll_l_joint", "wrist_roll_r_joint",      # 25, 26
    "neck_pitch_joint", "head_pitch_joint",          # 27, 28
]

# body 顺序 (30 bodies, 与 validate_motion_npz.py 的 OMNI_BODY_NAMES 一致)
BODY_NAMES = [
    "base_link",            # 0
    "hip_pitch_l_link",     # 1
    "hip_pitch_r_link",     # 2
    "waist_yaw_link",       # 3
    "hip_roll_l_link",      # 4
    "hip_roll_r_link",      # 5
    "waist_roll_link",      # 6
    "hip_yaw_l_link",       # 7
    "hip_yaw_r_link",       # 8
    "waist_pitch_link",     # 9
    "knee_pitch_l_link",    # 10
    "knee_pitch_r_link",    # 11
    "shoulder_pitch_l_link",  # 12
    "shoulder_pitch_r_link",  # 13
    "ankle_pitch_l_link",   # 14
    "ankle_pitch_r_link",   # 15
    "shoulder_roll_l_link",  # 16
    "shoulder_roll_r_link",  # 17
    "ankle_roll_l_link",    # 18 (脚)
    "ankle_roll_r_link",    # 19 (脚)
    "shoulder_yaw_l_link",  # 20
    "shoulder_yaw_r_link",  # 21
    "elbow_pitch_l_link",   # 22
    "elbow_pitch_r_link",   # 23
    "elbow_yaw_l_link",     # 24
    "elbow_yaw_r_link",     # 25
    "wrist_pitch_l_link",   # 26
    "wrist_pitch_r_link",   # 27
    "wrist_roll_l_link",    # 28
    "wrist_roll_r_link",    # 29
]

# 修改参数
WRIST_PITCH_SCALE = 0.0    # wrist_pitch → 0 (中立位)
ELBOW_PITCH_SCALE = 0.72   # elbow_pitch × 0.72 (72° → 52°)
HIP_ROLL_SCALE = 0.2       # hip_roll 不对称量 × 0.2 (24° → ~5°)


# ============================================================
# URDF 解析
# ============================================================
def parse_urdf(urdf_path):
    """解析 URDF, 返回运动学树结构。"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # link 名称 → index
    body_index = {name: i for i, name in enumerate(BODY_NAMES)}

    # 关节列表
    joints = []
    children_map = {}  # parent_body_idx → [joint_idx, ...]

    for joint in root.findall('joint'):
        jname = joint.get('name')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')

        if parent not in body_index or child not in body_index:
            continue

        pidx = body_index[parent]
        cidx = body_index[child]

        origin = joint.find('origin')
        xyz = np.array([float(x) for x in origin.get('xyz', '0 0 0').split()])
        rpy = np.array([float(x) for x in origin.get('rpy', '0 0 0').split()])
        axis = np.array([float(x) for x in joint.find('axis').get('xyz', '0 0 1').split()])

        jidx = len(joints)
        joints.append({
            'name': jname,
            'parent': pidx,
            'child': cidx,
            'xyz': xyz,
            'rpy': rpy,
            'axis': axis,
        })
        children_map.setdefault(pidx, []).append(jidx)

    return joints, children_map, body_index


# ============================================================
# 旋转工具
# ============================================================
def rpy_to_quat(rpy):
    """Roll-Pitch-Yaw (XYZ 内旋) → quaternion [w, x, y, z]。"""
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def axis_angle_to_quat(axis, angle):
    """轴角 → quaternion [w, x, y, z]。"""
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_mul(q1, q2):
    """四元数乘法 q1 * q2。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q, v):
    """四元数旋转向量: q * v * q^-1。"""
    vq = np.array([0.0, v[0], v[1], v[2]])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    result = quat_mul(quat_mul(q, vq), q_conj)
    return result[1:]


# ============================================================
# 正向运动学 (FK)
# ============================================================
def compute_fk(joint_positions, joints, children_map):
    """
    给定 joint_positions (29,), 计算所有 30 个 body 的 pos 和 quat。
    root (base_link) 的 pos/quat 需要外部设置。
    """
    body_pos = np.zeros((30, 3))
    body_quat = np.zeros((30, 4))
    body_quat[:, 0] = 1.0

    # BFS 遍历运动学树
    queue = deque([0])
    visited = {0}

    while queue:
        parent_idx = queue.popleft()
        for jidx in children_map.get(parent_idx, []):
            j = joints[jidx]
            child_idx = j['child']
            if child_idx in visited:
                continue
            visited.add(child_idx)

            jname = j['name']
            if jname in NPZ_JOINT_NAMES:
                angle_idx = NPZ_JOINT_NAMES.index(jname)
                angle = joint_positions[angle_idx]
            else:
                angle = 0.0  # 不在 NPZ 中的关节 (ankle_roll, neck, head) 角度为 0

            q_fixed = rpy_to_quat(j['rpy'])
            q_joint = axis_angle_to_quat(j['axis'], angle)
            q_total = quat_mul(q_fixed, q_joint)

            # Isaac Sim URDF 解析: body 位置偏移用原始 xyz (不经 rpy 旋转)
            # rpy 只影响子 link 的朝向, 不影响位置偏移
            offset = j['xyz'].copy()

            body_pos[child_idx] = body_pos[parent_idx] + quat_rotate(body_quat[parent_idx], offset)
            body_quat[child_idx] = quat_mul(body_quat[parent_idx], q_total)

            queue.append(child_idx)

    return body_pos, body_quat


def compute_fk_all_frames(joint_pos_all, joints, children_map, root_pos, root_quat):
    """对所有帧做 FK。"""
    T = joint_pos_all.shape[0]
    body_pos = np.zeros((T, 30, 3), dtype=np.float32)
    body_quat = np.zeros((T, 30, 4), dtype=np.float32)
    body_quat[:, :, 0] = 1.0

    for t in range(T):
        body_pos[t, 0] = root_pos[t]
        body_quat[t, 0] = root_quat[t]
        bp, bq = compute_fk(joint_pos_all[t], joints, children_map)
        body_pos[t, 1:] = bp[1:]
        body_quat[t, 1:] = bq[1:]

    return body_pos, body_quat


# ============================================================
# 数值差分求速度
# ============================================================
def compute_vel_from_pos(pos, fps):
    """从位置序列数值差分求线速度。"""
    T = pos.shape[0]
    dt = 1.0 / fps
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def compute_ang_vel_from_quat(quat_seq, fps):
    """从四元数序列求角速度。"""
    T = quat_seq.shape[0]
    dt = 1.0 / fps
    ang_vel = np.zeros((T, 3), dtype=np.float32)

    for t in range(T):
        if t == 0:
            q_curr = quat_seq[0]
            q_next = quat_seq[1]
            eff_dt = dt
        elif t == T - 1:
            q_curr = quat_seq[-2]
            q_next = quat_seq[-1]
            eff_dt = dt
        else:
            q_curr = quat_seq[t - 1]
            q_next = quat_seq[t + 1]
            eff_dt = 2 * dt

        # q_diff = q_next * q_curr^-1
        q_curr_inv = np.array([q_curr[0], -q_curr[1], -q_curr[2], -q_curr[3]])
        q_diff = quat_mul(q_next, q_curr_inv)

        # 确保 w > 0 (短路径)
        if q_diff[0] < 0:
            q_diff = -q_diff

        sin_half = np.linalg.norm(q_diff[1:])
        if sin_half > 1e-8:
            angle = 2 * np.arcsin(np.clip(sin_half, 0, 1))
            axis = q_diff[1:] / sin_half
            ang_vel[t] = axis * angle / eff_dt
        else:
            ang_vel[t] = 0

    return ang_vel


# ============================================================
# 修改 joint_pos
# ============================================================
def modify_joint_pos(joint_pos):
    """修改 NPZ joint_pos: wrist/elbow/hip_roll。"""
    modified = joint_pos.copy()

    # --- wrist_pitch → 0 ---
    idx_wl = 23
    idx_wr = 24
    print(f"  wrist_pitch_l: {np.degrees(joint_pos[:, idx_wl].mean()):.2f}° → 0°")
    print(f"  wrist_pitch_r: {np.degrees(joint_pos[:, idx_wr].mean()):.2f}° → 0°")
    modified[:, idx_wl] = 0.0
    modified[:, idx_wr] = 0.0

    # --- elbow_pitch × scale ---
    idx_el = 19
    idx_er = 20
    print(f"  elbow_pitch_l: {np.degrees(joint_pos[:, idx_el].mean()):.2f}° → {np.degrees(joint_pos[:, idx_el].mean() * ELBOW_PITCH_SCALE):.2f}°")
    print(f"  elbow_pitch_r: {np.degrees(joint_pos[:, idx_er].mean()):.2f}° → {np.degrees(joint_pos[:, idx_er].mean() * ELBOW_PITCH_SCALE):.2f}°")
    modified[:, idx_el] *= ELBOW_PITCH_SCALE
    modified[:, idx_er] *= ELBOW_PITCH_SCALE

    # --- hip_roll 不对称缩减 ---
    idx_hl = 3
    idx_hr = 4
    asym = (modified[:, idx_hl] - modified[:, idx_hr]) / 2
    max_asym = np.degrees(np.max(np.abs(asym)))
    modified[:, idx_hl] -= asym * (1 - HIP_ROLL_SCALE)
    modified[:, idx_hr] += asym * (1 - HIP_ROLL_SCALE)
    new_max_asym = np.degrees(np.max(np.abs((modified[:, idx_hl] - modified[:, idx_hr]) / 2)))
    print(f"  hip_roll 不对称: max {max_asym:.2f}° → {new_max_asym:.2f}°")

    return modified


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", help="只验证 FK, 不修改")
    parser.add_argument("--output", "-o", default="jump_high_firstjump_50fps_fixed.npz", help="输出文件")
    args = parser.parse_args()

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script 在 liuzq_jump_train_v11/scripts/, 项目根在 liuzq_jump_train_v11/
    project_root = os.path.dirname(script_dir)
    urdf_path = os.path.join(project_root, URDF_PATH)
    npz_path = os.path.join(project_root, NPZ_PATH)
    output_path = os.path.join(project_root, args.output)

    print(f"URDF: {urdf_path}")
    print(f"NPZ:  {npz_path}")

    # 1. 解析 URDF
    print("\n[1/6] 解析 URDF...")
    joints, children_map, body_index = parse_urdf(urdf_path)
    print(f"  {len(joints)} joints, {len(body_index)} bodies")

    # 2. 加载 NPZ
    print("\n[2/6] 加载 NPZ...")
    data = np.load(npz_path)
    fps = int(data['fps'][0])
    joint_pos_orig = data['joint_pos'].copy()
    joint_vel_orig = data['joint_vel'].copy()
    body_pos_orig = data['body_pos_w'].copy()
    body_quat_orig = data['body_quat_w'].copy()
    body_lin_vel_orig = data['body_lin_vel_w'].copy()
    body_ang_vel_orig = data['body_ang_vel_w'].copy()
    T = joint_pos_orig.shape[0]
    print(f"  T={T}, fps={fps}")

    # 3. FK 验证
    print("\n[3/6] FK 验证（与原始 body_pos_w 对比）...")
    root_pos = body_pos_orig[:, 0, :].copy()
    root_quat = body_quat_orig[:, 0, :].copy()

    body_pos_fk, body_quat_fk = compute_fk_all_frames(
        joint_pos_orig, joints, children_map, root_pos, root_quat
    )

    pos_err = np.linalg.norm(body_pos_fk - body_pos_orig, axis=-1)
    quat_err_angle = np.zeros((T, 30))
    for t in range(T):
        for b in range(30):
            q1 = body_quat_orig[t, b]
            q2 = body_quat_fk[t, b]
            dot = np.clip(np.abs(np.dot(q1, q2)), 0, 1)
            quat_err_angle[t, b] = np.degrees(2 * np.arccos(dot))

    print(f"  body_pos 误差: mean={pos_err.mean():.6f}m, max={pos_err.max():.6f}m")
    print(f"  body_quat 误差: mean={quat_err_angle.mean():.4f}°, max={quat_err_angle.max():.4f}°")

    if pos_err.mean() > 0.01:
        print("  [WARN] FK 误差偏大! 逐 body 误差:")
        for b in range(30):
            berr = pos_err[:, b].mean()
            if berr > 0.001:
                print(f"    [{b:2d}] {BODY_NAMES[b]:30s}: {berr:.6f}m")

    if args.verify_only:
        print("\n[完成] 仅验证模式, 退出。")
        return

    # 4. 修改 joint_pos
    print("\n[4/6] 修改 joint_pos...")
    joint_pos_new = modify_joint_pos(joint_pos_orig)

    # 5. 修改 joint_vel (数值差分)
    print("\n[5/6] 计算新 joint_vel (数值差分)...")
    joint_vel_new = compute_vel_from_pos(joint_pos_new, fps)

    # body 数据保持原样 (NPZ 的 body_pos_w 生成方式与 URDF FK 约定不同,
    # 但 body 跟踪权重低, 改的关节对 body 位置影响极小)
    body_pos_new = body_pos_orig
    body_quat_new = body_quat_orig
    body_lin_vel_new = body_lin_vel_orig
    body_ang_vel_new = body_ang_vel_orig

    # 6. 保存
    print(f"\n[6/6] 保存到 {output_path}...")
    np.savez(
        output_path,
        fps=data['fps'],
        joint_pos=joint_pos_new.astype(np.float32),
        joint_vel=joint_vel_new.astype(np.float32),
        body_pos_w=body_pos_new.astype(np.float32),
        body_quat_w=body_quat_new.astype(np.float32),
        body_lin_vel_w=body_lin_vel_new.astype(np.float32),
        body_ang_vel_w=body_ang_vel_new.astype(np.float32),
    )
    print("  [OK] 保存完成!")

    # 修改前后对比
    print("\n=== 修改前后对比 ===")
    comparisons = [
        ("wrist_pitch_l (23)", 23),
        ("wrist_pitch_r (24)", 24),
        ("elbow_pitch_l (19)", 19),
        ("elbow_pitch_r (20)", 20),
        ("hip_roll_l (3)", 3),
        ("hip_roll_r (4)", 4),
    ]
    for name, idx in comparisons:
        orig_mean = np.degrees(joint_pos_orig[:, idx].mean())
        new_mean = np.degrees(joint_pos_new[:, idx].mean())
        print(f"  {name:28s}: {orig_mean:+8.2f}° → {new_mean:+8.2f}° (Δ{new_mean - orig_mean:+.2f}°)")

    asym_orig = np.degrees(np.max(np.abs((joint_pos_orig[:, 3] - joint_pos_orig[:, 4]) / 2)))
    asym_new = np.degrees(np.max(np.abs((joint_pos_new[:, 3] - joint_pos_new[:, 4]) / 2)))
    print(f"  {'hip_roll 不对称 max':28s}: {asym_orig:+8.2f}° → {asym_new:+8.2f}° (Δ{asym_new - asym_orig:+.2f}°)")

    print(f"\n下一步: python scripts/validate_motion_npz.py {args.output} -p")


if __name__ == "__main__":
    main()
