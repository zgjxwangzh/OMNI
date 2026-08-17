#!/usr/bin/env python3
"""
ONNX → MuJoCo 部署验证脚本（omni_rl_sdk 方式）

与 deploy_onnx_mujoco.py 同类型的独立验证脚本，但：
- 使用 omni_rl_sdk 编译好的 C++ SDK（env_interface_py.so）驱动 MuJoCo
- 使用 31-DOF 模型 + dcmotor 执行器 + YAML 配置（当前能成功跑的方式）
- 从命令行指定 ONNX / NPZ，快速验证策略能否驱动参考动作

用法：
    cd omni_rl_sdk
    python3 deploy_verify.py --onnx policy/high_dynamic/config/model/policy.onnx \
                             --motion policy/high_dynamic/config/motion/jump_high_firstjump_50fps.npz

    # 无 GUI、指定步数
    python3 deploy_verify.py --onnx model.onnx --motion ref.npz --no_gui --steps 2000
"""
import argparse
import os
import sys
import time

import numpy as np

# SDK 根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="ONNX → MuJoCo 部署验证（omni_rl_sdk 方式）")
    parser.add_argument("--onnx", required=True, help="ONNX 模型文件路径")
    parser.add_argument("--motion", required=True, help="参考动作 NPZ 文件路径")
    parser.add_argument("--steps", type=int, default=4000, help="总控制步数（默认 4000 步 ≈ 10 秒）")
    parser.add_argument("--no_gui", action="store_true", help="不显示可视化（默认显示）")
    args = parser.parse_args()

    if not os.path.isfile(args.onnx):
        print(f"✗ ONNX 文件不存在: {args.onnx}")
        sys.exit(1)
    if not os.path.isfile(args.motion):
        print(f"✗ NPZ 文件不存在: {args.motion}")
        sys.exit(1)

    print("=" * 60)
    print("  ONNX → MuJoCo 部署验证")
    print("=" * 60)
    print(f"  ONNX:  {args.onnx}")
    print(f"  NPZ:   {args.motion}")

    # ---------- 1. 加载环境（C++ SDK）----------
    print("\n[1] 初始化 MuJoCo 环境（C++ SDK）...")
    from deploy_omni_sim_real import create_optimized_interface
    from deploy_omni_sim_real.OmniCtrlCommand import OmniStateAndCmd, OmniPolicyOutput

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "deploy_omni_sim_real/config/env-omni31.yaml")
    env = create_optimized_interface(config_path)
    print("  ✓ 环境初始化成功")

    # ---------- 2. 加载策略（ONNX + NPZ）----------
    print("\n[2] 加载策略（HighDynamicPolicy）...")
    from policy.high_dynamic.high_dynamic_policy import HighDynamicPolicy

    hd_config = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "policy/high_dynamic/config/high_dynamic.yaml")
    motion_abs = os.path.abspath(args.motion)
    policy = HighDynamicPolicy(
        hd_config,
        device="cpu",
        model_path=os.path.abspath(args.onnx),
        motions=[{"name": "verify", "motion_path": motion_abs}],
    )
    print(f"  ✓ 策略加载成功（history={policy.history_length}, num_obs={policy.num_obs}, "
          f"action_scale={policy.action_scale_policy[0]:.3f}）")

    # ---------- 3. 状态封装 ----------
    state_cmd = OmniStateAndCmd(31)
    policy_output = OmniPolicyOutput(31)
    rd = env.get_robot_data()
    state_cmd.update_from_robot_data(rd, env.get_remote_control_data())
    policy_output.sync_from_robot_data(rd)

    # 预热
    policy.select_motion(0)
    policy.reset()
    policy.warmup_from_state(state_cmd)

    # ---------- 4. 主循环 ----------
    print(f"\n[3] 开始仿真（{args.steps} 步）...")
    control_dt = 0.0025
    decimation = policy.motion_play_max_frames and 1 or 1  # 占位，下面用 8
    decimation = 8  # 与 high_dynamic.yaml 一致（50Hz 策略）

    step_counter = 0
    heights = []
    action_log = []
    target = policy.default_dof_pos[:29].copy()

    t0 = time.time()
    for step in range(args.steps):
        rd = env.get_robot_data()
        state_cmd.update_from_robot_data(rd, env.get_remote_control_data())

        if step_counter % decimation == 0:
            obs = policy.get_observation(state_cmd)
            target = policy.get_action(obs)
            action_log.append(target.copy())

        step_counter += 1

        # 写入目标（前 29 个身体关节）
        policy_output.actions[:29] = target
        policy_output.kps[:29] = policy.kps
        policy_output.kds[:29] = policy.kds

        rd = policy_output.flush_to_robot_data(rd)
        env.set_motor_ctrl(rd)

        # 记录 base 高度（用 IMU 四元数无法直接得高度，用关节数据间接判断）
        heights.append(np.mean(np.abs(state_cmd.q[:12])))  # 腿部平均角度作为姿态指标

        if not args.no_gui:
            time.sleep(max(0, control_dt - (time.time() - t0)))
            t0 = time.time()

    # ---------- 5. 汇总 ----------
    print("\n" + "=" * 60)
    print("  验证结果")
    print("=" * 60)
    print(f"  总步数: {step_counter}")
    print(f"  推理次数: {len(action_log)}")

    if len(action_log) > 0:
        acts = np.array(action_log)
        print(f"  目标关节角范围: [{acts.min():.3f}, {acts.max():.3f}] rad")
        print(f"  目标关节角均值: {acts.mean():.3f} rad")
        # 检查是否发散
        if np.abs(acts).max() > 10.0:
            print("  ⚠ 目标关节角异常大（可能发散）")
        else:
            print("  ✓ 目标关节角在合理范围（未发散）")

    print("\n  验证完成。")
    print("  提示：可视化窗口已打开（如果没加 --no_gui），观察机器人是否正常执行动作。")


if __name__ == "__main__":
    main()
