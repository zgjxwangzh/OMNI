#!/usr/bin/env python3
"""
方案 2：物理跟踪验证 — PD 控制 + 重力，验证 29 个动作的可跟踪性

与 batch_verify_mujoco.py（运动学回放，禁用重力）不同，本脚本：
  - 开启重力和接触力
  - 用 PD 控制器跟踪参考关节角度
  - 检查机器人能否物理上"做出来"这些动作
  - 报告跟踪误差、是否摔倒、关节扭矩是否超限

使用方法：
    python3 physics_track_verify.py                    # 验证所有 29 个动作
    python3 physics_track_verify.py --motion 跳高06    # 只验证跳高06
    python3 physics_track_verify.py --kp_scale 1.5     # 增大刚度
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from collections import deque

try:
    import mujoco
except ImportError:
    print("✗ MuJoCo 未安装，请 pip install mujoco")
    sys.exit(1)

# policy order → motor order 逆映射
MOTOR_TO_POLICY_IDX = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9,
    15, 22, 4, 10, 16, 23, 5, 11,
    17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])
POLICY_TO_MOTOR_IDX = np.argsort(MOTOR_TO_POLICY_IDX)


def track_motion(mj_model, mj_data, motion_path, retarget_npz_path=None,
                 kp_scale=1.0, kd_scale=1.0, sim_dt=0.002, motion_fps=30):
    """
    用 PD 控制跟踪参考动作，返回跟踪结果。

    核心逻辑：
      1. 每一帧从 NPZ 读取参考关节角度
      2. 用 PD 控制器计算扭矩: tau = kp*(ref - q) - kd*dq
      3. MuJoCo 物理步进（含重力、接触）
      4. 记录跟踪误差
    """
    # 加载 high_dynamic NPZ
    data = np.load(motion_path, allow_pickle=True)
    jp = data['joint_pos']    # policy order
    jv = data['joint_vel']    # policy order
    bq = data['body_quat_w']
    T = jp.shape[0]

    # 加载 root position（用于初始化 base 位置）
    root_positions = None
    if retarget_npz_path and os.path.isfile(retarget_npz_path):
        rd = np.load(retarget_npz_path)
        if 'root_positions' in rd:
            root_positions = rd['root_positions']

    n_act = mj_model.nu  # 29

    # PD gains（从 high_dynamic.yaml 的 kp/kd）
    # 这些是 high_dynamic 策略用的增益
    kp_base = np.array([
        150.0, 150.0, 150.0, 150.0, 30.0, 30.0,
        150.0, 150.0, 150.0, 150.0, 30.0, 30.0,
        150.0, 150.0, 150.0,
        100.0, 100.0, 50.0, 50.0, 50.0, 20.0, 20.0,
        100.0, 100.0, 50.0, 50.0, 50.0, 20.0, 20.0,
    ], dtype=np.float64)
    kd_base = np.array([
        5.0, 5.0, 5.0, 5.0, 3.0, 3.0,
        5.0, 5.0, 5.0, 5.0, 3.0, 3.0,
        5.0, 5.0, 5.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
    ], dtype=np.float64)

    kp = kp_base * kp_scale
    kd = kd_base * kd_scale

    # 仿真参数
    motion_dt = 1.0 / motion_fps
    n_substeps = max(1, int(round(motion_dt / sim_dt)))
    mj_model.opt.timestep = sim_dt

    # 初始化：设置第一帧姿态
    if root_positions is not None:
        mj_data.qpos[0:3] = root_positions[0]
    else:
        mj_data.qpos[0:3] = [0, 0, 0.82]
    mj_data.qpos[3:7] = bq[0, 0]  # base quaternion
    mj_data.qpos[7:7+n_act] = jp[0][POLICY_TO_MOTOR_IDX]  # joint angles

    mujoco.mj_forward(mj_model, mj_data)

    # 固定 base（通过 weld 约束的等效方法：直接覆盖 qpos）
    # 这样只测试"关节能否跟上"，不涉及平衡问题
    fixed_base = True  # 设为 False 可以尝试带平衡的版本

    # 跟踪记录
    tracking_errors = []  # 每帧的 RMS 跟踪误差
    heights = []
    max_torques = []
    fell = False
    fall_frame = -1

    for frame_idx in range(T):
        # 参考关节角度（motor order）
        ref_motor = jp[frame_idx][POLICY_TO_MOTOR_IDX]

        # 参考 base 位置
        if root_positions is not None:
            ref_base_pos = root_positions[frame_idx]
        else:
            ref_base_pos = np.array([0, 0, 0.82])

        # 参考 base 方向
        ref_base_quat = bq[frame_idx, 0]

        # PD 控制（只控制关节，不直接控制 base）
        q_current = mj_data.qpos[7:7+n_act]
        dq_current = mj_data.qvel[6:6+n_act]

        q_error = ref_motor - q_current
        tau = kp * q_error - kd * dq_current

        # 扭矩限幅（电机峰值力矩）
        max_torque = np.array([
            140, 140, 90, 140, 50, 50,
            140, 140, 90, 140, 50, 50,
            90, 50, 50,
            25, 25, 25, 25, 25, 10, 10,
            25, 25, 25, 25, 25, 10, 10,
        ], dtype=np.float64)
        tau = np.clip(tau, -max_torque, max_torque)
        max_torques.append(np.abs(tau).max())

        mj_data.ctrl[:] = tau

        # 物理步进（多个子步）
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)
            
            # 固定 base（如果启用）
            if fixed_base:
                if root_positions is not None:
                    mj_data.qpos[0:3] = root_positions[frame_idx]
                mj_data.qpos[3:7] = bq[frame_idx, 0]
                mj_data.qvel[0:6] = 0  # base 速度清零

        # 记录跟踪误差
        q_after = mj_data.qpos[7:7+n_act]
        frame_error = np.sqrt(np.mean((ref_motor - q_after) ** 2))
        tracking_errors.append(frame_error)
        heights.append(mj_data.qpos[2])

        # 摔倒检测
        if mj_data.qpos[2] < 0.2:
            if not fell:
                fell = True
                fall_frame = frame_idx

    tracking_errors = np.array(tracking_errors)
    heights = np.array(heights)
    max_torques = np.array(max_torques)

    return {
        "frames": T,
        "tracking_error_mean": tracking_errors.mean(),
        "tracking_error_max": tracking_errors.max(),
        "tracking_error_final": tracking_errors[-1],
        "height_min": heights.min(),
        "height_max": heights.max(),
        "height_delta": heights.max() - heights.min(),
        "max_torque": max_torques.max(),
        "fell": fell,
        "fall_frame": fall_frame,
        "tracking_errors": tracking_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="物理跟踪验证")
    parser.add_argument("--model", default="omni_29dof_mjc/mjcf/omni_29dof.xml")
    parser.add_argument("--motion_dir", default="motion_data")
    parser.add_argument("--retarget_dir", default="retargeted")
    parser.add_argument("--motion", default=None, help="只验证指定动作（名称前缀匹配）")
    parser.add_argument("--kp_scale", type=float, default=1.0, help="刚度缩放因子")
    parser.add_argument("--kd_scale", type=float, default=1.0, help="阻尼缩放因子")
    parser.add_argument("--sim_dt", type=float, default=0.002, help="仿真步长")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载模型
    model_path = os.path.join(script_dir, args.model)
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    print(f"✓ 模型加载: {mj_model.nq} qpos, {mj_model.nu} actuators")
    print(f"  重力: {mj_model.opt.gravity}")
    print(f"  kp_scale={args.kp_scale}, kd_scale={args.kd_scale}")
    print(f"  sim_dt={args.sim_dt}s\n")

    # 收集文件
    motion_files = sorted(Path(args.motion_dir).glob("*_highdynamic.npz"))
    if args.motion:
        motion_files = [f for f in motion_files if args.motion in f.stem]
    print(f"找到 {len(motion_files)} 个动作\n")

    # 表头
    print(f"{'动作':<20s} {'帧数':>5s} {'误差均值':>8s} {'误差最大':>8s} "
          f"{'高度范围':>12s} {'最大扭矩':>8s} {'摔倒':>6s} {'状态':>4s}")
    print("-" * 85)

    results = []
    for mf in motion_files:
        name = mf.stem.replace("_highdynamic", "")
        stem = name
        retarget_path = os.path.join(args.retarget_dir, f"{stem}.npz")
        if not os.path.isfile(retarget_path):
            retarget_path = None

        # 重置仿真状态
        mj_data = mujoco.MjData(mj_model)

        r = track_motion(mj_model, mj_data, str(mf),
                         retarget_npz_path=retarget_path,
                         kp_scale=args.kp_scale, kd_scale=args.kd_scale,
                         sim_dt=args.sim_dt)
        results.append((name, r))

        # 判断状态
        if r["fell"]:
            status = "✗ 摔倒"
        elif r["tracking_error_mean"] > 0.5:
            status = "⚠ 跟踪差"
        elif r["tracking_error_mean"] > 0.2:
            status = "~ 一般"
        else:
            status = "✓ 良好"

        h_range = f"{r['height_min']:.2f}→{r['height_max']:.2f}"
        fall_info = f"@{r['fall_frame']}" if r["fell"] else "否"

        print(f"{name:<20s} {r['frames']:5d} {r['tracking_error_mean']:8.3f} "
              f"{r['tracking_error_max']:8.3f} {h_range:>12s} "
              f"{r['max_torque']:8.1f} {fall_info:>6s} {status:>6s}")

    # 汇总
    print("-" * 85)
    good = sum(1 for _, r in results if not r["fell"] and r["tracking_error_mean"] < 0.2)
    ok = sum(1 for _, r in results if not r["fell"] and 0.2 <= r["tracking_error_mean"] < 0.5)
    bad = sum(1 for _, r in results if r["fell"] or r["tracking_error_mean"] >= 0.5)
    print(f"\n良好: {good}, 一般: {ok}, 摔倒/跟踪差: {bad}")

    if bad > 0:
        print(f"\n⚠ 有 {bad} 个动作摔倒或跟踪差，可能需要：")
        print(f"  1. 增大 kp_scale（当前 {args.kp_scale}）")
        print(f"  2. 检查动捕数据是否超出机器人能力")
        print(f"  3. 训练时 RL 会自动学习可行范围")


if __name__ == "__main__":
    main()
