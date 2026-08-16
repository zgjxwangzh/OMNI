# Copyright (c) 2025-2026, The Omni Lab Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# RSL-RL 兼容层：在 Isaac Lab 各版本间统一生成 rsl_rl 所需的 runner 配置 dict。
#
# 背景：rsl-rl >= 4.0 的 OnPolicyRunner / PPO 只认 `actor` / `critic` 两个模型配置，
#       不再接受旧的 `policy` 字段。
#   - Isaac Lab main（> 2.3.2，含本地开发分支）自带了 handle_deprecated_rsl_rl_cfg，
#     会自动迁移并校验（policy -> actor/critic，且填充 distribution_cfg 等）。
#   - Isaac Lab 官方 2.3.x（如服务器 /opt/isaaclab）没有该函数，配置里的 `policy`
#     不会被迁移，直接传给 rsl-rl 4.0 会在 PPO.construct_algorithm 处 KeyError。
# 本模块对两者都安全：main 走内置迁移，2.3.x 走 dict 级手动迁移。

import importlib.metadata as metadata

from packaging import version

# 本工程环境有 "policy" 与 "critic" 两个观测组（见 omni_jump_tasks_v7/jump_env_cfg.py）。
# 仅在配置里没写 obs_groups 时用作兜底；jump_env_cfg 里已显式设置，不会被覆盖。
_DEFAULT_OBS_GROUPS = {"actor": ["policy"], "critic": ["critic"]}


def build_runner_cfg(agent_cfg, installed_version=None) -> dict:
    """构建可直接传给 rsl_rl ``OnPolicyRunner`` 的配置 dict。

    Args:
        agent_cfg: Isaac Lab 的 RSL-RL runner 配置对象（``RslRlBaseRunnerCfg`` 子类）。
        installed_version: rsl-rl-lib 版本字符串；为 ``None`` 时自动探测。

    Returns:
        兼容已安装 rsl-rl 版本的配置 dict。
    """
    if installed_version is None:
        installed_version = metadata.version("rsl-rl-lib")

    # 1) Isaac Lab main：使用内置迁移（会原地修改 agent_cfg）。
    migrated_by_main = False
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        migrated_by_main = True
    except ImportError:
        # Isaac Lab 2.3.x：没有 handle_deprecated_rsl_rl_cfg，走下面的 dict 级迁移。
        pass

    cfg = agent_cfg.to_dict()

    # 2) Isaac Lab 2.3.x 路径：policy -> actor/critic（rsl-rl >= 4.0）。
    if not migrated_by_main and version.parse(installed_version) >= version.parse("4.0.0"):
        if isinstance(cfg.get("policy"), dict) and len(cfg["policy"]):
            _migrate_policy_to_actor_critic(cfg)

    # 3) obs_groups 必须是 dict，否则 rsl-rl 4.x 的 resolve_obs_groups 无法处理。
    if not isinstance(cfg.get("obs_groups"), dict) or len(cfg["obs_groups"]) == 0:
        cfg["obs_groups"] = _DEFAULT_OBS_GROUPS

    return cfg


def _migrate_policy_to_actor_critic(cfg: dict) -> None:
    """把旧式 ``policy`` 配置转成 rsl-rl >= 4.0 需要的 ``actor`` / ``critic``。

    字段映射与 Isaac Lab main 上 handle_deprecated_rsl_rl_cfg 的迁移逻辑一致，
    仅作用在 to_dict() 得到的 dict 上（因为 Isaac Lab 2.3.x 的 RslRlOnPolicyRunnerCfg
    本身没有 actor / critic 字段，无法在配置对象层面迁移）。
    """
    policy = cfg.pop("policy")
    cfg["actor"] = {
        "class_name": "MLPModel",
        "hidden_dims": policy.get("actor_hidden_dims"),
        "activation": policy.get("activation"),
        "obs_normalization": policy.get("actor_obs_normalization"),
        "stochastic": True,
        "init_noise_std": policy.get("init_noise_std"),
        "noise_std_type": policy.get("noise_std_type"),
        "state_dependent_std": policy.get("state_dependent_std"),
    }
    cfg["critic"] = {
        "class_name": "MLPModel",
        "hidden_dims": policy.get("critic_hidden_dims"),
        "activation": policy.get("activation"),
        "obs_normalization": policy.get("critic_obs_normalization"),
        "stochastic": False,
    }
