"""V7 终止函数移植到 jump_high(xMimic) 框架的纯 torch 实现。

2026-08-13 V11: 把 my_omni_jump_train_v7/omni_jump_tasks_v7/mdp/terminations.py 的
bad_anchor_ori / fell / out_of_bounds 移植过来(jump_high 的 vendored mdp 没有这些,
isaaclab 内置 mdp 只有 time_out / root_height_below_minimum)。

运行时零 isaaclab 依赖(纯 torch + TYPE_CHECKING import), 便于无 sim 单测。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """用四元数(wxyz)对向量做逆旋转。与 isaaclab quat_apply_inverse 公式一致。

    quat_apply_inverse(q, v) = v - w*t + xyz×t, 其中 t = 2 * xyz × v, xyz = q[1:]。
    """
    xyz = quat[..., 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return vec - quat[..., 0:1] * t + xyz.cross(t, dim=-1)


def bad_anchor_ori(
    env: "ManagerBasedRLEnv", asset_cfg: "SceneEntityCfg", command_name: str, threshold: float
) -> torch.Tensor:
    """基座倾角相对参考过大(摔倒/翻滚)。用投影重力比较, 对偏航不变。"""
    asset = env.scene[asset_cfg.name]
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    # 重力世界向量(-9.81, 0, 0) 用 z 轴分量比较: GRAVITY_VEC_W 在 isaaclab 是 (0,0,-9.81)
    motion_projected_gravity_b = _quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = _quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def fell(env: "ManagerBasedRLEnv", command_name: str, threshold: float = 0.25) -> torch.Tensor:
    """机器人基座落地(物理 base z 过低)。站立/深蹲约 0.35~0.85m, 0.25 以下视为摔倒。"""
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    return command.robot_anchor_pos_w[:, 2] < threshold


def out_of_bounds(env: "ManagerBasedRLEnv", command_name: str, half_size: float = 1.5) -> torch.Tensor:
    """基座 x/y 相对各自 env 中心超出 ±half_size → 终止。

    2026-08-13 V11 修复: jump_high 环境用 env_spacing=2.5 分散各 env, 机器人
    robot_anchor_pos_w 是绝对世界坐标(含 env_origins 偏移), 不能查世界坐标
    (否则 4092/4096 env 第一步就误判出界)。改为相对 env_origins 判位移。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    pos = command.robot_anchor_pos_w  # (E,3) 绝对世界坐标
    rel = pos - env.scene.env_origins  # 相对各自 env 中心
    return (rel[:, 0].abs() > half_size) | (rel[:, 1].abs() > half_size)


__all__ = ["bad_anchor_ori", "fell", "out_of_bounds"]
