"""Omni 原地跳高环境子类: 每回合追踪并输出"成绩代理"指标。

比赛规则(世界人形机器人运动会 1.3.7)成绩定义:
  **成绩 = 最高点时身体任何部位最低点到地面的垂直距离**。

本类只做**日志扩展**, 不改变奖励/观测/终止/命令任何训练信号, 不影响策略与断点:
  - 每个 step 用当前 base_link 高度推进"回合最高点", 并在创新高时记录当时的
    最低 link 高度(即最高点时刻的成绩);
  - 回合结束(终止/超时)reset 时, 把成绩与相关量注入 `extras["log"]`,
    经 rsl_rl 的 `logger.process_env_step` 写入 tensorboard(标量名见下);
  - reset 后清空累计量, 进入新回合。

tensorboard 标量:
  - `Jump/score_apex_lowest_link`: 成绩(最高点身体最低 link 高度)
  - `Jump/apex_root_h`:            回合内 base_link 最高点高度
  - `Jump/jump_height`:            apex_root_h - 站立基准
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnvCfg

from .mdp.score_metric import JumpScoreTracker


class OmniJumpEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv 子类, 在每回合结束输出跳高成绩代理指标。"""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        self._score_tracker = JumpScoreTracker(cfg.scene.num_envs, cfg.sim.device)
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def step(self, action):
        self._update_metrics()
        return super().step(action)

    def _update_metrics(self):
        """用当前 body 状态推进最高点/成绩累计。"""
        robot = self.scene["robot"]
        root_h = robot.data.root_pos_w[:, 2]
        min_link_h = robot.data.body_pos_w[:, :, 2].min(dim=1).values
        self._score_tracker.update(root_h, min_link_h)

    def _reset_idx(self, env_ids):
        self._update_metrics()
        super()._reset_idx(env_ids)
        self.extras["log"].update(self._score_tracker.log(env_ids))
        self._score_tracker.reset(env_ids)
