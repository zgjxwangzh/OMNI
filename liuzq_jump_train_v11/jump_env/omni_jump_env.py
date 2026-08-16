"""Omni 原地跳高环境子类: 每回合追踪并输出"成绩代理"指标。

比赛规则(世界人形机器人运动会 1.3.7)成绩定义:
  **成绩 = 最高点时身体任何部位最低点到地面的垂直距离**。

口径与 `scripts/build_first_jump.py:404-415` 参考验证完全一致:
  apex = 回合内 base_link(root) 高度最大的时刻;
  成绩 = apex 时刻 `body_pos_w[:, :, 2].min()`(所有 body 中最低 link 的高度)。
参考第一跳成绩 = 0.464m, 跳高(apex - 站立)= 0.18m。
具体累计逻辑见 `jump_env/score_metric.py` 的 `JumpScoreTracker`(纯 torch, 已单测)。

本类只做**日志扩展**, 不改变奖励/观测/终止/命令任何训练信号, 不影响策略与断点:
  - 每个 step 用当前 base_link 高度推进"回合最高点", 并在创新高时记录当时的
    最低 link 高度(即最高点时刻的成绩);
  - 回合结束(终止/超时)reset 时, 把成绩与相关量注入 `extras["log"]`,
    经 rsl_rl 的 `logger.process_env_step` 写入 tensorboard(标量名见下);
  - reset 后清空累计量, 进入新回合。

tensorboard 标量(键含 "/" 会原样登记):
  - `Jump/score_apex_lowest_link`: 成绩(最高点身体最低 link 高度, 参考 0.464m)
  - `Jump/apex_root_h`:            回合内 base_link 最高点高度
  - `Jump/jump_height`:            apex_root_h - 站立基准(参考 0.18m)

训练时盯 `Jump/score_apex_lowest_link` 即可: 往 0.46m 方向涨 = 在真跳;
贴着 ~0.03m(脚高)不动 = 策略在钻空子(蹲着不动刷跟踪 / 不蓄力), 立刻可见。
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnvCfg

from jump_env.score_metric import JumpScoreTracker


class OmniJumpEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv 子类, 在每回合结束输出跳高成绩代理指标。"""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        # 先建 tracker 再 super: 父类构造不会触发 reset, 但防御后续 reset 路径;
        # num_envs/device 直接从 cfg 读(此时父类属性尚未建立)。
        self._score_tracker = JumpScoreTracker(cfg.scene.num_envs, cfg.sim.device)
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def step(self, action):
        # 本 step 内部会先做终止检查再 reset; 在进入 super 之前用当前状态
        # (上一步末)推进一次累计, 保证每个物理步都被计入。
        self._update_metrics()
        return super().step(action)

    def _update_metrics(self):
        """用当前 body 状态推进最高点/成绩累计(无副作用, 每 step 调用)。"""
        robot = self.scene["robot"]
        root_h = robot.data.root_pos_w[:, 2]
        min_link_h = robot.data.body_pos_w[:, :, 2].min(dim=1).values
        self._score_tracker.update(root_h, min_link_h)

    def _reset_idx(self, env_ids):
        # 终止/超时步: 先用终止前(当前)状态补最后一次累计, 覆盖终止步本身
        self._update_metrics()
        # 先跑父类: 它会重建 extras["log"], 之后再把成绩追加进去
        super()._reset_idx(env_ids)
        # 注入成绩。注意: 首次 reset(含初始化)也注入 —— rsl_rl 的 logger 按
        # 首个条目 extras 的键集合登记 scalar, 若首轮缺 Jump/* 键, 曲线永远不出现。
        # 首次 reset 时 tracker 仅记录了站立态, 故该点为站立基线(成绩≈脚高 0.03m),
        # 属有效数据。
        self.extras["log"].update(self._score_tracker.log(env_ids))
        # 清空已 reset 环境的累计量, 进入新回合
        self._score_tracker.reset(env_ids)
