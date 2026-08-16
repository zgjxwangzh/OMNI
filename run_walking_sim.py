#!/usr/bin/env python3
"""
方案 1：走路策略 MuJoCo 物理仿真

用 SDK 自带的走路 ONNX（loco_mode）在 MuJoCo 中做物理仿真：
  - 开启重力、接触力、PD 控制
  - 用 MXPolicy 的逻辑构建 obs → ONNX 推理 → PD 控制
  - 验证整个部署管线能跑通

使用方法：
    python3 run_walking_sim.py
    python3 run_walking_sim.py --steps 2000    # 跑 2000 步
    python3 run_walking_sim.py --device cuda   # 用 GPU 推理
"""
import argparse
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="走路策略 MuJoCo 物理仿真")
    parser.add_argument("--model", default="omni_29dof_mjc/mjcf/omni_29dof.xml")
    parser.add_argument("--onnx", default="omni_rl_sdk/policy/loco_mode/model/omni_7dof_63k_2file.onnx")
    parser.add_argument("--config", default="omni_rl_sdk/policy/loco_mode/config/LocoMode.yaml")
    parser.add_argument("--steps", type=int, default=4000, help="仿真步数 (默认 4000 = 10s@400Hz)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--headless", action="store_true", default=True, help="无头模式（默认）")
    args = parser.parse_args()

    try:
        import mujoco
    except ImportError:
        print("✗ MuJoCo 未安装，请 pip install mujoco")
        sys.exit(1)

    try:
        import onnxruntime as ort
    except ImportError:
        print("✗ onnxruntime 未安装，请 pip install onnxruntime-gpu")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── 1. 加载 MuJoCo 模型 ──
    model_path = os.path.join(script_dir, args.model)
    if not os.path.isfile(model_path):
        print(f"✗ 模型文件不存在: {model_path}")
        sys.exit(1)

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    print(f"✓ MuJoCo 模型加载: {mj_model.nq} qpos, {mj_model.nu} actuators")
    print(f"  重力: {mj_model.opt.gravity}")

    # ── 2. 加载 ONNX 模型 ──
    onnx_path = os.path.join(script_dir, args.onnx)
    if not os.path.isfile(onnx_path):
        print(f"✗ ONNX 文件不存在: {onnx_path}")
        print("  请确认 omni_rl_sdk 目录已上传")
        sys.exit(1)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" \
        else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    actual_provider = session.get_providers()[0]
    print(f"✓ ONNX 模型加载: {onnx_path}")
    print(f"  provider: {actual_provider}")

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    print(f"  input: {input_name} {input_shape}")
    print(f"  output: {output_name} {session.get_outputs()[0].shape}")

    # ── 3. 加载配置 ──
    import yaml
    config_path = os.path.join(script_dir, args.config)
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # 解析关键参数
    num_obs = cfg['observation']['num_obs']          # 90
    history_length = cfg['observation']['history_length']  # 10
    decimation = cfg.get('decimation', 4)
    action_scale = cfg['action']['scale']
    default_pos = np.array(cfg['dof']['default_pos'], dtype=np.float32)
    mujoco_to_isaac_idx = np.array(cfg['dof']['mujoco_to_isaac_idx'], dtype=np.int32)
    num_actions = cfg['dof']['num_actions']           # 29
    num_actions_policy = len(mujoco_to_isaac_idx)     # 25

    # obs 缩放
    obs_scale_ang_vel = cfg['observation'].get('obs_scale_ang_vel', 0.25)
    obs_scale_gravity = cfg['observation'].get('obs_scale_gravity', 1.0)
    obs_scale_dof_pos = cfg['observation'].get('obs_scale_dof_pos', 1.0)
    obs_scale_dof_vel = cfg['observation'].get('obs_scale_dof_vel', 0.05)
    obs_scale_command = np.array(cfg['observation'].get('obs_scale_command', [1.0, 1.0, 1.0]), dtype=np.float32)
    max_cmd = np.array(cfg.get('max_cmd', [1.0, 1.0, 0.6]), dtype=np.float32)

    # gait 参数
    phase_ratio = np.array(cfg.get('phase_ratio', [0.4, 0.4]), dtype=np.float32)
    phase_offset = np.array(cfg.get('phase_offset', [0.4, 0.9]), dtype=np.float32)
    gait_cycle = cfg.get('gait_cycle', 0.9)
    cmd_deadzone = cfg.get('cmd_deadzone', 0.05)

    # PD gains
    kp = np.array(cfg['dof']['kp'], dtype=np.float32)
    kd = np.array(cfg['dof']['kd'], dtype=np.float32)

    print(f"\n✓ 配置加载:")
    print(f"  obs={num_obs}, history={history_length}, decimation={decimation}")
    print(f"  action_scale={action_scale}, policy_joints={num_actions_policy}")

    # ── 4. 初始化状态 ──
    # 设置初始姿态
    mj_data.qpos[0:3] = [0, 0, 0.82]  # base position
    mj_data.qpos[3:7] = [1, 0, 0, 0]  # base quaternion (w,x,y,z)
    mj_data.qpos[7:7+29] = default_pos  # joint angles

    # 前向运动学
    mujoco.mj_forward(mj_model, mj_data)

    # History buffer
    history = []
    for _ in range(history_length):
        history.append(np.zeros(num_obs, dtype=np.float32))

    # 内部状态
    last_action_isaac = np.zeros(num_actions_policy, dtype=np.float32)
    gait_phase = np.array([0.0, 0.0], dtype=np.float32)
    episode_length = 0
    phase_ratio_cur = np.array([0.0, 0.0], dtype=np.float32)
    walk_mask = 0

    # 速度指令：向前走
    commands = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # vx, vy, yaw_rate

    # ── 5. 仿真循环 ──
    dt = mj_model.opt.timestep
    control_dt = dt * decimation
    print(f"\n═══ 开始仿真 ═══")
    print(f"  dt={dt:.4f}s, control_dt={control_dt:.4f}s")
    print(f"  步数: {args.steps} ({args.steps * dt:.1f}s)")
    print(f"  速度指令: vx={commands[0]} m/s\n")

    heights = []
    base_positions = []
    step = 0

    for sim_step in range(args.steps):
        # 每 decimation 步推理一次
        if sim_step % decimation == 0:
            # 构建 obs（与 MXPolicy.get_observation 一致）
            q = mj_data.qpos[7:7+num_actions].astype(np.float32)
            dq = mj_data.qvel[6:6+num_actions].astype(np.float32)

            # gravity orientation
            quat = mj_data.qpos[3:7].astype(np.float32)
            qw, qx, qy, qz = quat
            gravity_ori = np.array([
                2 * (-qz * qx + qw * qy),
                -2 * (qz * qy + qw * qx),
                1 - 2 * (qw * qw + qz * qz),
            ], dtype=np.float32)

            # ang_vel
            ang_vel = mj_data.qvel[3:6].astype(np.float32)

            # dof pos/vel in policy order
            motor_pos = (q - default_pos)[mujoco_to_isaac_idx]
            motor_vel = dq[mujoco_to_isaac_idx]

            # gait phase
            t = episode_length * control_dt / gait_cycle
            gait_phase[0] = (t + phase_offset[0]) % 1.0
            gait_phase[1] = (t + phase_offset[1]) % 1.0
            if walk_mask > 0:
                episode_length += 1

            # 拼装 obs
            obs = np.concatenate([
                ang_vel * obs_scale_ang_vel,
                gravity_ori * obs_scale_gravity,
                commands * obs_scale_command * max_cmd,
                motor_pos * obs_scale_dof_pos,
                motor_vel * obs_scale_dof_vel,
                last_action_isaac,
                np.sin(2 * np.pi * gait_phase),
                np.cos(2 * np.pi * gait_phase),
                phase_ratio_cur,
            ]).astype(np.float32)

            history.append(obs.copy())
            if len(history) > history_length:
                history.pop(0)

            obs_flat = np.concatenate(history).reshape(1, -1)

            # ONNX 推理
            outputs = session.run([output_name], {input_name: obs_flat.astype(np.float32)})
            action_policy = outputs[0].squeeze().astype(np.float32)[:num_actions_policy]

            # EMA smooth
            beta = cfg['action'].get('beta', 1.0)
            if beta < 1.0:
                action_policy = (1.0 - beta) * last_action_isaac + beta * action_policy
            last_action_isaac = action_policy.copy()

            # clip
            action_clip = cfg['action'].get('clip')
            if action_clip:
                action_policy = np.clip(action_policy, -action_clip, action_clip)

            # scale
            action_policy = action_policy * action_scale

            # 映射回 29 维 motor 顺序
            action_motor = np.zeros(num_actions, dtype=np.float32)
            for i in range(num_actions_policy):
                action_motor[mujoco_to_isaac_idx[i]] = action_policy[i]

            # 目标位置 = default + action
            target_pos = default_pos + action_motor

        # PD 控制
        q_current = mj_data.qpos[7:7+num_actions].astype(np.float64)
        dq_current = mj_data.qvel[6:6+num_actions].astype(np.float64)
        q_error = target_pos - q_current

        tau = kp * q_error - kd * dq_current
        mj_data.ctrl[:] = tau

        # 步进仿真
        mujoco.mj_step(mj_model, mj_data)

        # 记录
        heights.append(mj_data.qpos[2])
        base_positions.append(mj_data.qpos[0:3].copy())
        step = sim_step

        # 每 400 步打印一次状态
        if (sim_step + 1) % 400 == 0:
            t_now = (sim_step + 1) * dt
            h = mj_data.qpos[2]
            pos = mj_data.qpos[0:3]
            print(f"  t={t_now:.1f}s  h={h:.3f}m  pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    # ── 6. 结果统计 ──
    print(f"\n═══ 仿真结果 ═══")
    print(f"  总步数: {step + 1}")
    print(f"  仿真时间: {(step + 1) * dt:.1f}s")
    print(f"  初始高度: {heights[0]:.3f}m")
    print(f"  最终高度: {heights[-1]:.3f}m")
    print(f"  最低高度: {min(heights):.3f}m")
    print(f"  最高高度: {max(heights):.3f}m")
    print(f"  X 位移: {base_positions[-1][0] - base_positions[0][0]:.3f}m")
    print(f"  Y 位移: {base_positions[-1][1] - base_positions[0][1]:.3f}m")

    # 检查是否摔倒
    if min(heights) < 0.3:
        print(f"\n  ⚠ 机器人摔倒了！最低高度 {min(heights):.3f}m < 0.3m")
    elif max(heights) > 2.0:
        print(f"\n  ⚠ 机器人飞起来了！最高高度 {max(heights):.3f}m > 2.0m")
    else:
        print(f"\n  ✓ 机器人保持直立，走路仿真正常")

    # NaN 检查
    if np.any(np.isnan(mj_data.qpos)):
        print(f"  ✗ 检测到 NaN！")
    else:
        print(f"  ✓ 无 NaN")


if __name__ == "__main__":
    main()
