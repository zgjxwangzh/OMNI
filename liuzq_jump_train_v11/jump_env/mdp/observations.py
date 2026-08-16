"""跳高环境自定义观测函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def base_lin_vel_dropout(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg | None" = None,
    dropout_prob: float = 0.2,
) -> torch.Tensor:
    """机体系线速度的 dropout 版(替换 `isaaclab.envs.mdp.base_lin_vel`)。

    目的: 真机部署若无法提供机体系线速度(无 IMU 速度估计/里程计, 填 0),
    训练阶段就必须让网络见过"线速度不可用"的情况, 否则部署时 obs 分布漂移、
    策略直接失效。做法是**回合级 dropout**: 每个环境每个回合以 `dropout_prob`
    概率把整回合的 `root_lin_vel_b` 恒清零, 其余回合保留真值——网络必须学会
    在"线速度无信息"时仍能站稳/跳好, 等价于用其他 obs(角速度、关节速度、
    相位、参考命令)兜底。

    mask 存在 `env.extras` 里, 回合开始(`episode_length_buf == 0`)重新掷骰子;
    即使 env reset 重建 extras(mask 变 None)也会自动重新初始化。
    ObsTerm 自带的 noise 仍会叠加(缺失时输出 ±noise 而非严格 0), 这比部署
    填 0 更严苛、更鲁棒。

    返回值和原 `base_lin_vel` 同 shape (N, 3)。
    """
    # 注意: 与 isaaclab 原版 base_lin_vel 一致, 默认绑 "robot" 实体。
    # 不能用 env.primary_asset_name —— load_managers 阶段该属性还没设置。
    asset = env.scene[asset_cfg.name] if asset_cfg is not None else env.scene["robot"]
    vel = asset.data.root_lin_vel_b  # (N, 3) 机体系线速度

    key = "_base_lin_vel_drop_mask"
    mask = env.extras.get(key)
    if mask is None or mask.shape[0] != env.num_envs:
        mask = (torch.rand(env.num_envs, device=vel.device) > dropout_prob).float()
    # 回合开始重新掷骰子
    re_roll = (torch.rand(env.num_envs, device=vel.device) > dropout_prob).float()
    mask = torch.where(env.episode_length_buf == 0, re_roll, mask)
    env.extras[key] = mask

    return vel * mask.unsqueeze(-1)
