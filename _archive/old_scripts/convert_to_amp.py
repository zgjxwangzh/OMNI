#!/usr/bin/env python3
"""
将 BVH 重定向的 .npz 文件转换为 TienKung-Lab 的 JSON 动作数据格式

TienKung-Lab 有两种动作数据：
  1. motion_visualization/  - 用于播放验证:
     [root_pos(3), root_euler(3), dof_pos(N), root_linvel(3), root_angvel(3), dof_vel(N)]
     注意：环境 visualize_motion 读 [3:6] 为 XYZ 欧拉角(弧度)，不是四元数！
     dof 顺序按环境切片: 左腿(6)→右腿(6)→左臂→右臂→腰
  2. motion_amp_expert/     - 用于AMP训练:  由 play_amp_animation.py 从 visualization 自动生成

使用方法（在 AutoDL 上运行）：

步骤 1：转换 .npz → motion_visualization JSON
    python convert_to_amp.py --input retargeted/ --output legged_lab/envs/omni/datasets/motion_visualization/

步骤 2：用 play_amp_animation.py 生成 expert 数据（会自动处理格式转换）
    # 先修改 walk_cfg.py 中 amp_motion_files_display 指向新生成的文件
    python legged_lab/scripts/play_amp_animation.py --task=omni_walk --num_envs=1 \
        --save_path legged_lab/envs/omni/datasets/motion_amp_expert/上楼梯01.txt

步骤 3：修改 walk_cfg.py 中 amp_motion_files 指向新 expert 文件，开始训练
    python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=4096 \
        --load_run=2026-07-28_23-43-16 --checkpoint=model_4900.pt --logger=tensorboard

也可以直接用 --expert 模式输出 motion_amp_expert 格式（但建议走完整流程）。
"""
import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path


# 环境 visualize_motion 的 dof 切片顺序（omni 环境与官方 tienkung_env.py 一致，
# 52 列布局，只映射 20 个关节；腰/肘yaw/腕不读，可视化时保持 0）
# [6:12]左腿 [12:18]右腿 [18:22]左臂 [22:26]右臂 [26:29]linvel [32:52]速度
VISUAL_JOINT_ORDER = [
    # 左腿 (6)
    "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    # 右腿 (6)
    "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    # 左臂 (4)
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint", "elbow_pitch_l_joint",
    # 右臂 (4)
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint", "elbow_pitch_r_joint",
]


def load_npz(npz_path):
    """加载重定向后的 npz 文件"""
    data = np.load(npz_path)
    return {
        'joint_angles': data['joint_angles'],       # (N, 29)
        'joint_names': list(data['joint_names']),
        'root_positions': data['root_positions'],    # (N, 3)
        'foot_contact': data.get('foot_contact'),    # (N, 2) or None
        'root_rotations': data.get('root_rotations'),  # (N, 4) or None
    }


def quat_multiply(q1, q2):
    """四元数乘法 (w,x,y,z)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def euler_to_quat(roll, pitch, yaw):
    """欧拉角 → 四元数 (w,x,y,z)"""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def quat_to_mat(q):
    """四元数 (w,x,y,z) → 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def mat_to_euler_xyz(R):
    """旋转矩阵 → XYZ 欧拉角 (scipy Rotation.from_euler('XYZ') 约定: R = Rz@Ry@Rx)"""
    b = -np.arcsin(np.clip(R[2, 0], -1, 1))
    a = np.arctan2(R[2, 1], R[2, 2])
    c = np.arctan2(R[1, 0], R[0, 0])
    return np.array([a, b, c])


def compute_velocities(positions, dt):
    """从位置序列计算速度（中心差分）"""
    N = len(positions)
    vel = np.zeros_like(positions)
    if N < 3:
        return vel
    # 中心差分
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    vel[0] = (positions[1] - positions[0]) / dt
    vel[-1] = (positions[-1] - positions[-2]) / dt
    return vel


def npz_to_visualization_json(npz_path, output_path, fps=30.0):
    """将 .npz 转换为 motion_visualization JSON 格式

    格式: [root_pos(3), root_euler(3), dof_pos(N), root_linvel(3), root_angvel(3), dof_vel(N)]
    环境 visualize_motion 读 [3:6] 为 XYZ 欧拉角，dof 按 VISUAL_JOINT_ORDER 切片
    """
    data = load_npz(npz_path)
    joint_angles = data['joint_angles']     # (N, 29)
    root_positions = data['root_positions']  # (N, 3)
    n_frames, n_joints = joint_angles.shape

    # npz 关节顺序 → 环境切片顺序（只取环境映射的 20 个关节）
    name_to_col = {nm: i for i, nm in enumerate(data['joint_names'])}
    order_idx = [name_to_col[nm] for nm in VISUAL_JOINT_ORDER]
    dof_pos = joint_angles[:, order_idx]
    n_dof = dof_pos.shape[1]

    dt = 1.0 / fps

    # 根位置速度
    root_lin_vel = compute_velocities(root_positions, dt)

    # 根旋转 → XYZ 欧拉角
    if data.get('root_rotations') is not None and len(data['root_rotations']) == n_frames:
        root_quats = np.array(data['root_rotations'], dtype=float)
        for i in range(1, n_frames):
            if np.dot(root_quats[i], root_quats[i - 1]) < 0:
                root_quats[i] = -root_quats[i]
        print("  根旋转: 使用 retarget 保存的骨盆姿态")
    else:
        root_quats = None
        print("  根旋转: 无，用单位姿态")

    root_euler = np.zeros((n_frames, 3))
    for i in range(n_frames):
        if root_quats is not None:
            root_euler[i] = mat_to_euler_xyz(quat_to_mat(root_quats[i]))

    # 根角速度：欧拉角差分（环境实际不使用，占位）
    root_ang_vel = compute_velocities(root_euler, dt)

    # 关节速度（同样按环境顺序）
    dof_vel = compute_velocities(dof_pos, dt)

    # 组装帧数据
    frames = []
    for i in range(n_frames):
        frame = []
        frame.extend(root_positions[i].tolist())   # root_pos (3)
        frame.extend(root_euler[i].tolist())       # root_euler (3)
        frame.extend(dof_pos[i].tolist())          # dof_pos (29)
        frame.extend(root_lin_vel[i].tolist())     # root_lin_vel (3)
        frame.extend(root_ang_vel[i].tolist())     # root_ang_vel (3)
        frame.extend(dof_vel[i].tolist())          # dof_vel (29)
        frames.append(frame)

    n_cols = 3 + 3 + n_dof + 3 + 3 + n_dof
    print(f"  帧数: {n_frames}, 每帧: {n_cols} 列")
    print(f"        = root_pos(3) + root_euler(3) + dof_pos({n_dof})")
    print(f"          + root_linvel(3) + root_angvel(3) + dof_vel({n_dof})")

    # 写入 JSON
    motion_data = {
        "LoopMode": "Wrap",
        "FrameDuration": round(dt, 6),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 1.0,
        "Frames": frames,
    }

    with open(output_path, 'w') as f:
        # 手动写出和 walk.txt 一致的格式
        f.write('{\n')
        f.write(f'"LoopMode": "{motion_data["LoopMode"]}",\n')
        f.write(f'"FrameDuration": {motion_data["FrameDuration"]},\n')
        f.write(f'"EnableCycleOffsetPosition": {str(motion_data["EnableCycleOffsetPosition"]).lower()},\n')
        f.write(f'"EnableCycleOffsetRotation": {str(motion_data["EnableCycleOffsetRotation"]).lower()},\n')
        f.write(f'"MotionWeight": {motion_data["MotionWeight"]},\n')
        f.write('\n"Frames":\n[\n')
        for i, frame in enumerate(frames):
            line = '  [' + ', '.join(f'{v:.18e}' for v in frame) + ']'
            if i < len(frames) - 1:
                line += ','
            f.write(line + '\n')
        f.write(']\n}\n')

    print(f"  输出: {output_path}")


def npz_to_amp_expert_json(npz_path, output_path, fps=30.0):
    """将 .npz 直接转换为 motion_amp_expert JSON 格式

    格式: [dof_pos(N), dof_vel(N), end_effector_pos(M)]
    注意：end_effector_pos 需要从仿真中获取，这里设为零占位
    """
    data = load_npz(npz_path)
    joint_angles = data['joint_angles']     # (N, 29)
    n_frames, n_joints = joint_angles.shape

    dt = 1.0 / fps
    dof_vel = compute_velocities(joint_angles, dt)

    # 估算末端执行器位置（左右脚，各 3 分量）
    # 简化：用 root_y + leg_length 估算
    root_positions = data['root_positions']
    foot_l_pos = np.zeros((n_frames, 3))
    foot_r_pos = np.zeros((n_frames, 3))
    for i in range(n_frames):
        # 粗略估算：脚在根下方，高度由膝盖角度决定
        hip_y = root_positions[i, 1]
        knee_l = joint_angles[i, 3]  # knee_pitch_l_joint
        knee_r = joint_angles[i, 9]  # knee_pitch_r_joint
        leg_len = 0.4 + 0.4 * np.cos(knee_l)  # 大腿+小腿投影
        foot_l_pos[i] = [root_positions[i, 0], hip_y - leg_len, root_positions[i, 2]]
        leg_len_r = 0.4 + 0.4 * np.cos(knee_r)
        foot_r_pos[i] = [root_positions[i, 0], hip_y - leg_len_r, root_positions[i, 2]]

    frames = []
    for i in range(n_frames):
        frame = []
        # dof_pos (29)
        frame.extend(joint_angles[i].tolist())
        # dof_vel (29)
        frame.extend(dof_vel[i].tolist())
        # end_effector_pos: 左右脚 (6)
        frame.extend(foot_l_pos[i].tolist())
        frame.extend(foot_r_pos[i].tolist())
        frames.append(frame)

    n_ee = 6
    n_cols = n_joints + n_joints + n_ee
    print(f"  帧数: {n_frames}, 每帧: {n_cols} 列")
    print(f"        = dof_pos({n_joints}) + dof_vel({n_joints}) + ee_pos({n_ee})")

    motion_data = {
        "LoopMode": "Wrap",
        "FrameDuration": round(dt, 6),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 1.0,
        "Frames": frames,
    }

    with open(output_path, 'w') as f:
        # 手动写出和 walk.txt 一致的格式
        f.write('{\n')
        f.write(f'"LoopMode": "{motion_data["LoopMode"]}",\n')
        f.write(f'"FrameDuration": {motion_data["FrameDuration"]},\n')
        f.write(f'"EnableCycleOffsetPosition": {str(motion_data["EnableCycleOffsetPosition"]).lower()},\n')
        f.write(f'"EnableCycleOffsetRotation": {str(motion_data["EnableCycleOffsetRotation"]).lower()},\n')
        f.write(f'"MotionWeight": {motion_data["MotionWeight"]},\n')
        f.write('\n"Frames":\n[\n')
        for i, frame in enumerate(frames):
            line = '  [' + ', '.join(f'{v:.18e}' for v in frame) + ']'
            if i < len(frames) - 1:
                line += ','
            f.write(line + '\n')
        f.write(']\n}\n')

    print(f"  输出: {output_path}")


def convert_directory(input_dir, output_dir, mode='visualization', fps=30.0):
    """批量转换目录下的所有 .npz"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_path.glob("*.npz"))
    print(f"找到 {len(npz_files)} 个 .npz 文件\n")

    convert_fn = npz_to_visualization_json if mode == 'visualization' else npz_to_amp_expert_json

    for npz_file in npz_files:
        print(f"转换: {npz_file.name}")
        txt_name = npz_file.stem + '.txt'
        convert_fn(str(npz_file), str(output_path / txt_name), fps=fps)
        print()

    print(f"{'='*50}")
    print(f"完成: {len(npz_files)} 个文件 → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=".npz → TienKung-Lab JSON 动作数据转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成 motion_visualization 格式（推荐，用于 play_amp_animation 验证）
  python convert_to_amp.py --input retargeted/ --output envs/omni/datasets/motion_visualization/

  # 直接生成 motion_amp_expert 格式（快捷但跳过验证）
  python convert_to_amp.py --input retargeted/ --output envs/omni/datasets/motion_amp_expert/ --expert

  # 只转换单个文件
  python convert_to_amp.py --input retargeted/上楼梯01_chr00.npz --output ./
""")
    parser.add_argument("--input", required=True, help=".npz 文件或目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--expert", action="store_true",
                        help="直接输出 motion_amp_expert 格式（默认输出 motion_visualization）")
    parser.add_argument("--fps", type=float, default=30.0, help="帧率 (默认 30)")
    args = parser.parse_args()

    mode = 'expert' if args.expert else 'visualization'
    print(f"模式: {'motion_amp_expert (快捷)' if mode == 'expert' else 'motion_visualization (推荐)'}")
    print(f"帧率: {args.fps} fps")
    print()

    input_path = Path(args.input)
    if input_path.is_dir():
        convert_directory(str(input_path), args.output, mode=mode, fps=args.fps)
    elif input_path.suffix == '.npz':
        output_file = Path(args.output) / (input_path.stem + '.txt')
        Path(args.output).mkdir(parents=True, exist_ok=True)
        if mode == 'expert':
            npz_to_amp_expert_json(str(input_path), str(output_file), fps=args.fps)
        else:
            npz_to_visualization_json(str(input_path), str(output_file), fps=args.fps)
    else:
        print(f"错误: 不支持的输入格式 {input_path.suffix}")


if __name__ == "__main__":
    main()
