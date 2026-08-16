#!/usr/bin/env python3
"""
将 BVH 重定向的 retargeted .npz 转换为 high_dynamic 框架所需的 .npz 格式。

输入（bvh_retarget.py 输出）：
    joint_angles   : (T, 29)  motor order
    root_positions : (T, 3)   URDF 世界系 (Z-up)
    root_rotations : (T, 4)   四元数 (w,x,y,z)，yaw-only（归一化后）
    joint_names    : (29,)    motor order 关节名

输出（high_dynamic 框架要求）：
    joint_pos  : (T, 29)      policy order 关节角度
    joint_vel  : (T, 29)      policy order 关节角速度（中心差分）
    body_quat_w: (T, 1, 4)    base_link 四元数 (w,x,y,z)

关节顺序映射：
    retarget NPZ = motor order（env-omni31.yaml 去掉 head）
    high_dynamic = policy order（high_dynamic.yaml 的 policy_joint_names）
    映射关系由 MOTOR_TO_POLICY_IDX 定义

使用方法：
    python convert_to_highdynamic.py --input retargeted/跳高06_chr00_v5.npz \
                                     --output motion_data/

    # 指定帧率（默认 30）
    python convert_to_highdynamic.py --input retargeted/跳高06_chr00_v5.npz \
                                     --output motion_data/ --fps 50
"""
import argparse
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 关节顺序映射
# ─────────────────────────────────────────────────────────────
# retarget NPZ 的 OMNI_JOINTS（= motor order = env-omni31.yaml 去掉 head）：
#   [0:6]  左腿  hip_pitch/roll/yaw, knee_pitch, ankle_pitch/roll
#   [6:12] 右腿  同上
#   [12:15] 腰    waist_yaw/roll/pitch
#   [15:22] 左臂  shoulder_pitch/roll/yaw, elbow_pitch/yaw, wrist_pitch/roll
#   [22:29] 右臂  同上
#
# high_dynamic policy order（policy_joint_names）：
#   hip_pitch L/R, waist_yaw, hip_roll L/R, waist_roll,
#   hip_yaw L/R, waist_pitch, knee_pitch L/R,
#   shoulder_pitch L/R, ankle_pitch L/R,
#   shoulder_roll L/R, ankle_roll L/R,
#   shoulder_yaw L/R, elbow_pitch L/R, elbow_yaw L/R,
#   wrist_pitch L/R, wrist_roll L/R

MOTOR_TO_POLICY_IDX = np.array([
    0, 6, 12,   # hip_pitch L/R, waist_yaw
    1, 7, 13,   # hip_roll  L/R, waist_roll
    2, 8, 14,   # hip_yaw   L/R, waist_pitch
    3, 9,       # knee_pitch L/R
    15, 22,     # shoulder_pitch L/R
    4, 10,      # ankle_pitch L/R
    16, 23,     # shoulder_roll L/R
    5, 11,      # ankle_roll L/R
    17, 24,     # shoulder_yaw L/R
    18, 25,     # elbow_pitch L/R
    19, 26,     # elbow_yaw L/R
    20, 27,     # wrist_pitch L/R
    21, 28,     # wrist_roll L/R
])

# 验证：policy order 应有 29 个元素，覆盖 0-28 每个恰好一次
assert len(MOTOR_TO_POLICY_IDX) == 29
assert sorted(MOTOR_TO_POLICY_IDX.tolist()) == list(range(29))


def compute_velocities(positions, dt):
    """中心差分计算速度"""
    N = len(positions)
    vel = np.zeros_like(positions)
    if N < 2:
        return vel
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    vel[0] = (positions[1] - positions[0]) / dt
    vel[-1] = (positions[-1] - positions[-2]) / dt
    return vel


def convert_npz(input_path, output_path, fps=30.0):
    """将 retargeted NPZ 转换为 high_dynamic NPZ"""
    data = np.load(input_path)

    joint_angles = data['joint_angles']          # (T, 29) motor order
    root_positions = data['root_positions']       # (T, 3)
    root_rotations = data.get('root_rotations')   # (T, 4) or None
    joint_names = list(data['joint_names'])

    T, n_joints = joint_angles.shape
    assert n_joints == 29, f"Expected 29 joints, got {n_joints}"
    dt = 1.0 / fps

    print(f"输入: {input_path}")
    print(f"  帧数: {T}, 关节数: {n_joints}, 帧率: {fps} fps")
    print(f"  关节名: {joint_names[:3]}... (前3个)")

    # ── 1. 关节角度：motor order → policy order ──
    joint_pos = joint_angles[:, MOTOR_TO_POLICY_IDX]  # (T, 29)
    print(f"\n  joint_pos 范围: [{joint_pos.min():.3f}, {joint_pos.max():.3f}] rad")

    # ── 2. 关节速度：先差分再重排 ──
    joint_vel_motor = compute_velocities(joint_angles, dt)  # (T, 29) motor order
    joint_vel = joint_vel_motor[:, MOTOR_TO_POLICY_IDX]     # (T, 29) policy order
    print(f"  joint_vel 范围: [{joint_vel.min():.3f}, {joint_vel.max():.3f}] rad/s")

    # ── 3. body 四元数 ──
    if root_rotations is not None and len(root_rotations) == T:
        body_quat_w = np.array(root_rotations, dtype=np.float32)  # (T, 4)
        # 确保四元数连续性（消除符号跳变）
        for i in range(1, T):
            if np.dot(body_quat_w[i], body_quat_w[i - 1]) < 0:
                body_quat_w[i] = -body_quat_w[i]
        body_quat_w = body_quat_w.reshape(T, 1, 4)  # (T, 1, 4) - 1 body
        print(f"  body_quat_w: 使用 retarget 根旋转（yaw-only 归一化）")
        # 检查四元数范数
        norms = np.linalg.norm(body_quat_w.reshape(T, 4), axis=1)
        print(f"  body_quat_w 范数: [{norms.min():.6f}, {norms.max():.6f}]")
    else:
        print(f"  body_quat_w: 无根旋转数据，使用单位四元数")
        body_quat_w = np.zeros((T, 1, 4), dtype=np.float32)
        body_quat_w[:, 0, 0] = 1.0  # w=1

    # ── 4. 保存 ──
    np.savez(
        output_path,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_quat_w=body_quat_w,
    )

    print(f"\n输出: {output_path}")
    print(f"  joint_pos  : {joint_pos.shape}")
    print(f"  joint_vel  : {joint_vel.shape}")
    print(f"  body_quat_w: {body_quat_w.shape}")

    # ── 5. 基本验证 ──
    print("\n── 数据验证 ──")
    # 检查关节限位（policy order 下的限位）
    motor_limits = {
        "hip_pitch_l_joint": (-2.6864, 2.6864),
        "hip_roll_l_joint": (-0.52, 2.96),
        "hip_yaw_l_joint": (-2.75, 2.75),
        "knee_pitch_l_joint": (0, 2.87),
        "ankle_pitch_l_joint": (-0.87, 0.52),
        "ankle_roll_l_joint": (-0.26, 0.26),
        "hip_pitch_r_joint": (-2.6864, 2.6864),
        "hip_roll_r_joint": (-2.96, 0.52),
        "hip_yaw_r_joint": (-2.75, 2.75),
        "knee_pitch_r_joint": (0, 2.87),
        "ankle_pitch_r_joint": (-0.87, 0.52),
        "ankle_roll_r_joint": (-0.26, 0.26),
        "waist_yaw_joint": (-2.7, 2.7),
        "waist_roll_joint": (-0.52, 0.52),
        "waist_pitch_joint": (-0.52, 0.52),
        "shoulder_pitch_l_joint": (-3.14, 2.7),
        "shoulder_roll_l_joint": (-0.52, 2.355),
        "shoulder_yaw_l_joint": (-2.61, 2.61),
        "elbow_pitch_l_joint": (-2.61, 0.52),
        "elbow_yaw_l_joint": (-2.09, 2.09),
        "wrist_pitch_l_joint": (-1.57, 1.57),
        "wrist_roll_l_joint": (-1.57, 1.57),
        "shoulder_pitch_r_joint": (-3.14, 2.7),
        "shoulder_roll_r_joint": (-2.355, 0.52),
        "shoulder_yaw_r_joint": (-2.61, 2.61),
        "elbow_pitch_r_joint": (-2.61, 0.52),
        "elbow_yaw_r_joint": (-2.09, 2.09),
        "wrist_pitch_r_joint": (-1.57, 1.57),
        "wrist_roll_r_joint": (-1.57, 1.57),
    }
    # motor order 限位列表
    motor_joint_names = [
        "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
        "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
        "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
        "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
        "elbow_pitch_l_joint", "elbow_yaw_l_joint",
        "wrist_pitch_l_joint", "wrist_roll_l_joint",
        "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
        "elbow_pitch_r_joint", "elbow_yaw_r_joint",
        "wrist_pitch_r_joint", "wrist_roll_r_joint",
    ]
    motor_limits_arr = np.array([motor_limits[n] for n in motor_joint_names])
    policy_limits = motor_limits_arr[MOTOR_TO_POLICY_IDX]

    violations = 0
    for j in range(29):
        lo, hi = policy_limits[j]
        jmin, jmax = joint_pos[:, j].min(), joint_pos[:, j].max()
        if jmin < lo - 0.01 or jmax > hi + 0.01:
            violations += 1
            print(f"  ⚠ 关节 {j} 超限: [{jmin:.3f}, {jmax:.3f}] vs 限位 [{lo:.3f}, {hi:.3f}]")

    if violations == 0:
        print(f"  ✓ 所有 29 个关节均在限位范围内")
    else:
        print(f"  ⚠ {violations} 个关节超限（可能需要裁剪）")

    # root 位置统计
    print(f"\n── Root 位置统计 ──")
    print(f"  X: [{root_positions[:, 0].min():.3f}, {root_positions[:, 0].max():.3f}] m")
    print(f"  Y: [{root_positions[:, 1].min():.3f}, {root_positions[:, 1].max():.3f}] m")
    print(f"  Z: [{root_positions[:, 2].min():.3f}, {root_positions[:, 2].max():.3f}] m")

    return joint_pos, joint_vel, body_quat_w


def main():
    parser = argparse.ArgumentParser(
        description="retargeted NPZ → high_dynamic NPZ 转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python convert_to_highdynamic.py --input retargeted/跳高06_chr00_v5.npz --output motion_data/
  python convert_to_highdynamic.py --input retargeted/跳高06_chr00_v5.npz --output motion_data/ --fps 50
""")
    parser.add_argument("--input", required=True, help="输入 .npz 文件")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--fps", type=float, default=30.0, help="帧率 (默认 30)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在 {input_path}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + "_highdynamic.npz")

    convert_npz(str(input_path), str(output_path), fps=args.fps)


if __name__ == "__main__":
    main()
