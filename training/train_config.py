"""
Isaac Lab 环境注册 & rsl_rl PPO 训练配置
==========================================

两部分配置：
1. 环境配置（ReferenceTrackingEnvCfg）→ 注册到 Isaac Lab
2. rsl_rl PPO 超参 → 训练算法参数

使用方法：
    # 通过 isaaclab.sh 启动
    isaaclab.sh -p training/train.py --task omni_high_dynamic --num_envs 2048
"""

# ═══════════════════════════════════════════════════════════════
# Part 1: Isaac Lab 环境注册
# ═══════════════════════════════════════════════════════════════

from isaaclab.envs import DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils import configclass
from training.env_reference import ReferenceTrackingEnvCfg

# ── 跳高专用配置（可快速验证）──
@configclass
class JumpHighDynamicEnvCfg(ReferenceTrackingEnvCfg):
    """跳高动作的训练配置（只加载跳高 NPZ）"""
    motion_files: list[str] = []  # 空 = 加载全部；也可指定具体文件路径


# ── 全动作配置 ──
@configclass
class AllMotionsEnvCfg(ReferenceTrackingEnvCfg):
    """加载全部 29 个动作的训练配置"""
    pass


# ═══════════════════════════════════════════════════════════════
# Part 2: rsl_rl PPO 训练超参
# ═══════════════════════════════════════════════════════════════
# 以下字典用于配置 rsl_rl 的 PPO 算法
# 与 rsl_rl.modules.ActorCritic 和 rsl_rl.algorithms.PPO 对应

PPO_CONFIG = {
    # ── 网络结构 ──
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",              # 激活函数

    # ── PPO 核心参数 ──
    "learning_rate": 1e-4,            # 学习率
    "gamma": 0.99,                    # 折扣因子
    "lam": 0.95,                      # GAE lambda
    "entropy_coef": 0.005,            # 熵正则系数（鼓励探索）
    "value_loss_coef": 0.5,           # value loss 系数
    "max_grad_norm": 1.0,             # 梯度裁剪
    "num_learning_epochs": 5,         # 每次更新的 epoch 数
    "num_steps_per_env": 24,          # 每个环境每次收集的步数
    "minibatch_size": None,           # None = 自动计算
    "batch_size": None,               # None = num_envs * num_steps_per_env

    # ── 噪声/探索 ──
    "noise_std": 1.0,                 # 初始动作噪声标准差

    # ── 其他 ──
    "clip_param": 0.2,                # PPO clip 参数
    "schedule": "adaptive",           # 学习率调度：fixed / adaptive
    "desired_kl": 0.01,               # adaptive schedule 的目标 KL
}

# ═══════════════════════════════════════════════════════════════
# Part 3: 训练运行配置
# ═══════════════════════════════════════════════════════════════

TRAINING_CONFIG = {
    # ── 运行参数 ──
    "task": "omni_high_dynamic",
    "experiment_name": "ref_tracking_jump",
    "seed": 42,

    # ── 环境参数 ──
    "num_envs": 2048,                 # 并行环境数（GPU 并行）
    "max_iterations": 20000,          # 最大训练迭代数

    # ── 保存 ──
    "save_interval": 500,             # 每 N 次迭代保存一次 checkpoint
    "log_dir": "logs/ref_tracking",   # 日志目录

    # ── 设备 ──
    "device": "cuda:0",
    "sim_device": "cuda:0",

    # ── 恢复训练 ──
    "resume": False,
    "resume_path": None,              # checkpoint 路径

    # ── 记录 ──
    "enable_tensorboard": True,
}
