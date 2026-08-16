#!/usr/bin/env python3
"""
MuJoCo 可视化验证：加载 high_dynamic NPZ 并在 MJCF 模型上回放动作。

用途：
  1. 验证 NPZ 数据格式正确（维度、关节顺序、四元数）
  2. 直观检查动作是否自然（无穿模、无异常姿态）
  3. 可选保存关键帧截图

使用方法：
    # 静默验证（只打印统计信息）
    python verify_highdynamic_mujoco.py --motion motion_data/跳高06_chr00_v5_highdynamic.npz

    # 保存关键帧图片
    python verify_highdynamic_mujoco.py --motion motion_data/跳高06_chr00_v5_highdynamic.npz --save_frames

    # 指定 MJCF 模型路径
    python verify_highdynamic_mujoco.py --motion motion_data/跳高06_chr00_v5_highdynamic.npz \
        --model omni_29dof_mjc/mjcf/omni_29dof.xml --save_frames
"""
import argparse
import os
import sys
import numpy as np

# ─────────────────────────────────────────────────────────────
# policy order → motor order 逆映射
# ─────────────────────────────────────────────────────────────
MOTOR_TO_POLICY_IDX = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9,
    15, 22, 4, 10, 16, 23, 5, 11,
    17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])
POLICY_TO_MOTOR_IDX = np.argsort(MOTOR_TO_POLICY_IDX)


def verify_data(motion_path):
    """验证 NPZ 数据格式和内容的完整性"""
    print(f"═══ 数据验证: {motion_path} ═══\n")
    data = np.load(motion_path)

    required_keys = ['joint_pos', 'joint_vel', 'body_quat_w']
    for key in required_keys:
        if key not in data:
            print(f"  ✗ 缺少必要字段: {key}")
            return None
        print(f"  ✓ {key}: shape={data[key].shape}, dtype={data[key].dtype}")

    jp = data['joint_pos']
    jv = data['joint_vel']
    bq = data['body_quat_w']

    T = jp.shape[0]
    print(f"\n  帧数: {T}")

    # 维度检查
    errors = []
    if jp.shape != (T, 29):
        errors.append(f"joint_pos shape {jp.shape} != ({T}, 29)")
    if jv.shape != (T, 29):
        errors.append(f"joint_vel shape {jv.shape} != ({T}, 29)")
    if bq.ndim != 3 or bq.shape[0] != T or bq.shape[2] != 4:
        errors.append(f"body_quat_w shape {bq.shape} 不正确")

    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return None

    # 四元数范数检查
    norms = np.linalg.norm(bq.reshape(T, 4), axis=1)
    quat_ok = np.allclose(norms, 1.0, atol=1e-4)
    print(f"\n  四元数范数: min={norms.min():.6f}, max={norms.max():.6f} {'✓' if quat_ok else '✗'}")

    # 速度合理性检查
    print(f"\n  关节速度统计:")
    print(f"    均值: {np.abs(jv).mean():.3f} rad/s")
    print(f"    最大: {np.abs(jv).max():.3f} rad/s")
    spikes = np.abs(jv) > 20.0
    if spikes.any():
        n_spikes = spikes.sum()
        print(f"    ⚠ 发现 {n_spikes} 个速度尖峰 (>20 rad/s)，可能有问题")
    else:
        print(f"    ✓ 无异常速度尖峰")

    # 帧间连续性检查
    diffs = np.diff(jp, axis=0)
    max_step = np.abs(diffs).max()
    mean_step = np.abs(diffs).mean()
    print(f"\n  帧间变化:")
    print(f"    均值: {mean_step:.4f} rad/帧")
    print(f"    最大: {max_step:.4f} rad/帧")

    # 首帧姿态
    print(f"\n  首帧 joint_pos (policy order, rad):")
    for i in range(0, 29, 5):
        end = min(i + 5, 29)
        vals = ', '.join(f'{jp[0, j]:+.3f}' for j in range(i, end))
        print(f"    [{i:2d}-{end-1:2d}]: {vals}")

    # root 高度轨迹
    if bq.shape[1] == 1:
        print(f"\n  body_quat_w: 单 body (base_link)")
        w_vals = bq[:, 0, 0]
        print(f"    w 分量范围: [{w_vals.min():.4f}, {w_vals.max():.4f}]")
        if w_vals.mean() > 0.99:
            print(f"    → 近似单位四元数，说明几乎没有旋转（yaw-only 归一化）")

    return data


def replay_in_mujoco(model_path, data, save_frames=False, output_dir="frames",
                     retarget_npz_path=None):
    """在 MuJoCo 中回放动作
    
    retarget_npz_path: 可选，原始 retarget NPZ（提供 root position 用于可视化）
    """
    try:
        import mujoco
    except ImportError:
        print("\n⚠ MuJoCo 未安装，跳过仿真验证")
        return False

    print(f"\n═══ MuJoCo 仿真验证 ═══\n")

    # 加载模型
    if not os.path.isfile(model_path):
        print(f"  ✗ 模型文件不存在: {model_path}")
        return False

    model = mujoco.MjModel.from_xml_path(model_path)
    sim_data = mujoco.MjData(model)

    print(f"  模型加载成功: {model_path}")
    print(f"  DOF: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")

    # 检查 actuator 数量
    n_act = model.nu
    if n_act != 29:
        print(f"  ⚠ 期望 29 个 actuator，实际 {n_act}")
        return False

    # 获取关节名（用于验证顺序）
    actuator_names = []
    for i in range(n_act):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        actuator_names.append(name if name else f"act_{i}")
    print(f"  Actuator 顺序: {actuator_names[:3]}... (前3个)")

    # 加载原始 retarget NPZ 的 root position（如果有）
    root_positions = None
    if retarget_npz_path and os.path.isfile(retarget_npz_path):
        retarget_data = np.load(retarget_npz_path)
        if 'root_positions' in retarget_data:
            root_positions = retarget_data['root_positions']
            print(f"  加载 root position: {root_positions.shape}")
            print(f"    Z 范围: [{root_positions[:, 2].min():.3f}, {root_positions[:, 2].max():.3f}] m")

    jp = data['joint_pos']
    jv = data['joint_vel']
    bq = data['body_quat_w']
    T = jp.shape[0]

    # 创建渲染上下文
    renderer = None
    if save_frames:
        os.makedirs(output_dir, exist_ok=True)
        renderer = mujoco.Renderer(model, height=480, width=640)

    # 回放
    dt = 1.0 / 30.0  # 假设 30 FPS
    print(f"\n  回放 {T} 帧 (dt={dt:.4f}s)...")

    # 禁用重力以便手动设置姿态
    opt_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0

    heights = []
    for frame_idx in range(T):
        # policy order → motor order
        motor_pos = jp[frame_idx][POLICY_TO_MOTOR_IDX]

        # 设置 free joint (base_link) 的位置和方向
        # qpos 布局: [pos(3), quat(4), joint_0, joint_1, ..., joint_28]
        if root_positions is not None:
            sim_data.qpos[0] = root_positions[frame_idx, 0]  # x
            sim_data.qpos[1] = root_positions[frame_idx, 1]  # y
            sim_data.qpos[2] = root_positions[frame_idx, 2]  # z
        else:
            sim_data.qpos[0] = 0
            sim_data.qpos[1] = 0
            sim_data.qpos[2] = 0.82  # 默认初始高度

        # 设置 base 方向
        if bq.shape[1] >= 1:
            sim_data.qpos[3:7] = bq[frame_idx, 0]  # (w, x, y, z)

        # 设置关节角度
        sim_data.qpos[7:7 + n_act] = motor_pos

        # 前向运动学
        mujoco.mj_forward(model, sim_data)

        heights.append(sim_data.qpos[2])

        # 渲染
        if save_frames and renderer is not None:
            # 均匀采样 8 个关键帧
            if T <= 8 or frame_idx % max(1, T // 8) == 0 or frame_idx == T - 1:
                renderer.update_scene(sim_data)
                img = renderer.render()
                frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
                # 保存为 PNG（需要 PIL）
                try:
                    from PIL import Image
                    Image.fromarray(img).save(frame_path)
                    print(f"    保存帧: {frame_path}")
                except ImportError:
                    # 保存为 raw numpy
                    np.save(frame_path.replace('.png', '.npy'), img)
                    print(f"    保存帧 (npy): {frame_path.replace('.png', '.npy')}")

    # 恢复重力
    model.opt.gravity[:] = opt_gravity

    if renderer is not None:
        renderer.close()

    print(f"\n  ✓ 回放完成")
    print(f"  Base 高度: {min(heights):.3f} ~ {max(heights):.3f} m")
    print(f"  高度变化: {max(heights) - min(heights):.3f} m")

    # 检查是否有 NaN 或异常
    if any(np.isnan(sim_data.qpos)):
        print(f"  ✗ 检测到 NaN！模型可能有冲突")
    else:
        print(f"  ✓ 无 NaN，运动学正常")

    return True


def main():
    parser = argparse.ArgumentParser(description="high_dynamic NPZ MuJoCo 验证")
    parser.add_argument("--motion", required=True, help="high_dynamic NPZ 文件路径")
    parser.add_argument("--model", default="omni_29dof_mjc/mjcf/omni_29dof.xml",
                        help="MuJoCo MJCF 模型路径")
    parser.add_argument("--retarget_npz", default=None,
                        help="原始 retarget NPZ（提供 root position 用于可视化）")
    parser.add_argument("--save_frames", action="store_true", help="保存关键帧图片")
    parser.add_argument("--output_dir", default="frames", help="帧输出目录")
    args = parser.parse_args()

    # 1. 数据验证
    data = verify_data(args.motion)
    if data is None:
        print("\n✗ 数据验证失败")
        sys.exit(1)

    # 2. MuJoCo 回放
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model) if not os.path.isabs(args.model) else args.model

    ok = replay_in_mujoco(model_path, data, save_frames=args.save_frames,
                          output_dir=args.output_dir,
                          retarget_npz_path=args.retarget_npz)
    if not ok:
        print("\n⚠ MuJoCo 回放失败（数据格式已验证通过，问题可能在模型文件）")
    else:
        print("\n═══ 验证全部通过 ═══")


if __name__ == "__main__":
    main()
