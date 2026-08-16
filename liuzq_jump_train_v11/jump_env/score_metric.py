"""跳高"成绩代理"指标(纯 torch, 无 isaaclab 依赖, 可独立单测)。

比赛规则(世界人形机器人运动会 1.3.7)成绩定义:
  **成绩 = 最高点时身体任何部位最低点到地面的垂直距离**。

口径与 `scripts/build_first_jump.py:404-415` 参考验证一致:
  apex = 回合内 base_link 高度最大的时刻;
  成绩 = apex 时刻所有 body 中最低 link 的高度。
参考第一跳: 成绩 0.464m, 跳高(apex-站立)= 0.18m。

`JumpScoreTracker` 维护每 env 的回合累计状态:
  - `update`: 每个物理步推进"最高点", 并在创新高时记录当时的最低 link 高度;
  - `log`: 回合结束时汇总为标量 dict(注入 `extras["log"]`, 进 tensorboard);
  - `reset`: 回合结束清空, 进入新回合。
"""

from __future__ import annotations

import torch


class JumpScoreTracker:
    """每 env 追踪最高点成绩: apex root 高度 + apex 时刻最低 link 高度。"""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.apex_root_h = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.score_at_apex = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.stand_root_h = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, root_h: torch.Tensor, min_link_h: torch.Tensor):
        """推进累计。root_h/min_link_h: (N,) 当前步各 env 的 base_link 高度与最低 link 高度。"""
        # 回合首个 step: 记录站立基准高度(带姿态扰动后的实际站立高度)
        fresh = ~self.active
        self.stand_root_h = torch.where(fresh, root_h, self.stand_root_h)
        self.active.fill_(True)
        # 推进最高点: 仅在该 env 的 base_link 创新高时刷新"最高点成绩"
        new_apex = root_h > self.apex_root_h
        self.apex_root_h = torch.maximum(self.apex_root_h, root_h)
        self.score_at_apex = torch.where(new_apex, min_link_h, self.score_at_apex)

    def log(self, env_ids) -> dict[str, torch.Tensor]:
        """汇总已结束 env 的回合成绩(标量, PPO 在 tensorboard 里按步平均)。"""
        return {
            "Jump/score_apex_lowest_link": torch.mean(self.score_at_apex[env_ids]),
            "Jump/apex_root_h": torch.mean(self.apex_root_h[env_ids]),
            "Jump/jump_height": torch.mean(self.apex_root_h[env_ids] - self.stand_root_h[env_ids]),
        }

    def reset(self, env_ids):
        """清空已结束 env 的累计量, 进入新回合。"""
        self.apex_root_h[env_ids] = 0.0
        self.score_at_apex[env_ids] = 0.0
        self.stand_root_h[env_ids] = 0.0
        self.active[env_ids] = False
