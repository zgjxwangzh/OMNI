# Copyright (c) 2026, The jump_high project developers.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner configuration for the Omni 29-DOF high-jump task.

镜像 100m_sprint 的配置(与该机器上已成功训练同款机器人的方案一致)。
使用 rsl-rl `policy` 风格配置。

2026-08-12 V9 迁移(服务器 Isaac Lab 2.2.x / rsl-rl 2.3.3):
- 删 `actor_obs_normalization`/`critic_obs_normalization`(2.3.0 才有, 2.2.x 传了 TypeError),
  改用 runner 级 `empirical_normalization=True`(rsl-rl 2.3.3 支持, 等价替代 per-actor/critic 归一化)。
- 加 `obs_groups`(2.3+/main 的 rsl_rl_compat 迁移需要, 2.2.x 忽略)。
- 其余超参/网络 [512,256,128] 原样保留。

2026-08-11: 随 50Hz 控制缩放(xMimic LOW_FREQ_SCALE=0.5 同款: dex_evt
rsl_rl_ppo_cfg.py:35-44 已核实)。num_steps 24→12(保 0.24s 真实时间),
gamma 0.99→0.99²、lam 0.95→0.95²(按折半频率折算折扣)。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class OmniJumpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for high-jumping the Omni 29-DOF humanoid."""

    num_steps_per_env = 12
    max_iterations = 20000
    save_interval = 100
    experiment_name = "omni_jump"
    empirical_normalization = True
    clip_actions = 100
    # 新版 isaaclab_rl(>=2.3) 用 obs_groups 决定 actor/critic 观测组; 2.2 忽略该字段
    obs_groups: dict = {"actor": ["policy"], "critic": ["critic"]}

    def __post_init__(self):
        super().__post_init__()

        # 50Hz(原 100Hz 的一半): 12 步 @0.02s = 0.24s, 真实时间不变(xMimic LOW_FREQ_SCALE=0.5)
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.8,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )

        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.002,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.9801,  # 0.99²: 频率折半, 每步对应更长真实时间, 折扣按真实时间折算
            lam=0.9025,  # 0.95²
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
