#!/usr/bin/env python3
"""
Reference Tracking 训练入口
============================

使用 rsl_rl PPO 训练 OMNI 29-DOF 的 reference tracking 策略。

使用方法：
    # 通过 Isaac Lab 启动（AutoDL）
    isaaclab.sh -p training/train.py --headless

    # 指定参数
    isaaclab.sh -p training/train.py --headless --num_envs 2048 --max_iter 10000

    # 恢复训练
    isaaclab.sh -p training/train.py --headless --resume logs/ref_tracking/model_5000.pt

    # TensorBoard 查看训练曲线
    tensorboard --logdir logs/ref_tracking
"""

import argparse
import os
import sys
import torch
import numpy as np

# ─────────────────────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Reference Tracking Training")
    parser.add_argument("--headless", action="store_true", help="无头模式（必须）")
    parser.add_argument("--num_envs", type=int, default=2048, help="并行环境数")
    parser.add_argument("--max_iter", type=int, default=20000, help="最大训练迭代数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--experiment", type=str, default="ref_tracking_jump", help="实验名称")
    parser.add_argument("--log_dir", type=str, default="logs/ref_tracking", help="日志目录")
    parser.add_argument("--save_interval", type=int, default=500, help="保存间隔（迭代数）")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--motion", type=str, default=None, help="指定训练动作（文件名前缀匹配）")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 0. 环境检查 ──
    print("=" * 60)
    print("  OMNI 29-DOF Reference Tracking Training")
    print("=" * 60)

    # 检查 Isaac Lab
    try:
        import isaaclab
        print(f"  ✓ Isaac Lab: {isaaclab.__version__}")
    except ImportError:
        print("  ✗ Isaac Lab 未安装！请确认在 Isaac Lab conda 环境中运行")
        print("    使用: isaaclab.sh -p training/train.py")
        sys.exit(1)

    # 检查 GPU
    if not torch.cuda.is_available():
        print("  ✗ CUDA 不可用！请检查 nvidia-smi")
        sys.exit(1)
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")

    # 检查 rsl_rl
    try:
        from rsl_rl.runners import OnPolicyRunner
        print(f"  ✓ rsl_rl: OnPolicyRunner 可用")
    except ImportError:
        print("  ✗ rsl_rl 未安装！")
        print("    安装: pip install rsl_rl")
        sys.exit(1)

    # ── 1. 创建环境 ──
    from isaaclab.envs import DirectRLEnv
    from training.env_reference import ReferenceTrackingEnv, ReferenceTrackingEnvCfg
    from training.train_config import PPO_CONFIG, TRAINING_CONFIG

    env_cfg = ReferenceTrackingEnvCfg()
    env_cfg.sim_device = args.device
    env_cfg.seed = args.seed

    # 如果指定了动作，过滤 motion_files
    if args.motion:
        motion_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motion_data")
        all_files = sorted([
            os.path.join(motion_dir, f)
            for f in os.listdir(motion_dir)
            if f.endswith("_highdynamic.npz") and args.motion in f
        ])
        if not all_files:
            print(f"  ✗ 未找到匹配 '{args.motion}' 的动作文件")
            sys.exit(1)
        env_cfg.motion_files = all_files
        print(f"  训练动作: {[os.path.basename(f) for f in all_files]}")

    print(f"\n  创建环境: num_envs={args.num_envs}...")
    env = ReferenceTrackingEnv(cfg=env_cfg)
    print(f"  ✓ 环境创建成功: obs={env._num_obs}, act={env._num_actions}")

    # ── 2. 配置 PPO ──
    ppo_cfg = PPO_CONFIG.copy()
    ppo_cfg["learning_rate"] = args.lr

    # rsl_rl 的 Runner 配置
    runner_cfg = {
        "algorithm": {
            "class_name": "PPO",
            **ppo_cfg,
        },
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": ppo_cfg["actor_hidden_dims"],
            "critic_hidden_dims": ppo_cfg["critic_hidden_dims"],
            "activation": ppo_cfg["activation"],
        },
        "runner": {
            "class_name": "OnPolicyRunner",
            "max_iterations": args.max_iter,
            "save_interval": args.save_interval,
            "experiment_name": args.experiment,
            "run_name": "",
        },
    }

    # ── 3. 创建 Runner ──
    log_dir = os.path.join(args.log_dir, args.experiment)
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(
        env=env,
        train_cfg=runner_cfg,
        log_dir=log_dir,
        device=args.device,
    )

    # ── 4. 恢复训练（如果指定）──
    if args.resume:
        print(f"\n  恢复训练: {args.resume}")
        runner.load(args.resume)

    # ── 5. 开始训练 ──
    print(f"\n{'=' * 60}")
    print(f"  开始训练！")
    print(f"  实验名: {args.experiment}")
    print(f"  迭代数: {args.max_iter}")
    print(f"  并行环境: {args.num_envs}")
    print(f"  学习率: {args.lr}")
    print(f"  日志目录: {log_dir}")
    print(f"{'=' * 60}\n")

    runner.learn(num_learning_iterations=args.max_iter, init_at_random_ep_len=True)

    print(f"\n  ✓ 训练完成！")
    print(f"  模型保存在: {log_dir}")
    print(f"  导出 ONNX: python training/export_onnx.py --checkpoint {log_dir}/model_{args.max_iter}.pt")


if __name__ == "__main__":
    main()
