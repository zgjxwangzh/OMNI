from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def jump_height_bonus(
    env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.9, scale: float = 1.0, max_excess: float = 0.5
) -> torch.Tensor:
    """Reward the anchor (base) for rising above a height threshold — i.e. getting airborne."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_z = command.robot_anchor_pos_w[:, 2]
    excess = torch.clamp(robot_z - threshold, min=0.0, max=max_excess)
    return excess * scale


def crouch_phase_height_match(
    env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.85, tol: float = 0.2, vel_thresh: float = -0.03
) -> torch.Tensor:
    """During the reference's descent into the crouch (base below `threshold` AND still
    descending, ref_vel_z < `vel_thresh`), give a LINEAR reward for matching the
    reference base height. Gated to the descending frames so it cannot pull the policy
    toward passive low tracking through the whole grounded period (the v7 failure) and
    does not fight the explosive rise into the takeoff.
    (v8: fix for v7 — reward only fires while squatting down, never at stand/rise.)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_z = command.anchor_pos_w[:, 2]
    ref_vel_z = command.anchor_lin_vel_w[:, 2]
    robot_z = command.robot_anchor_pos_w[:, 2]
    active = ((ref_z < threshold) & (ref_vel_z < vel_thresh)).float()
    err = torch.clamp(torch.abs(robot_z - ref_z) / tol, max=1.0)
    return active * (1.0 - err)


def takeoff_vertical_vel(
    env: ManagerBasedRLEnv, command_name: str, vel_thresh: float = 0.15
) -> torch.Tensor:
    """During the reference's ascent (base rising faster than `vel_thresh`), reward the
    robot's upward base velocity as a fraction of the reference's own rise. Self-
    calibrating: full reward when the robot rises at least as fast as the reference.
    (v8: fix for v7 — ratio form avoids the tiny flat-cap contribution and scales with
    the actual ascent speed of each jump.)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_vel_z = command.anchor_lin_vel_w[:, 2]
    robot_vel_z = command.robot_anchor_lin_vel_w[:, 2]
    active = (ref_vel_z > vel_thresh).float()
    ref_up = torch.clamp(ref_vel_z, min=0.0) + 1e-6
    ratio = torch.clamp(robot_vel_z / ref_up, min=0.0, max=1.0)
    return active * ratio


def upright_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """躯干姿态惩罚（v14：只罚头朝下，无竖直正向奖励）：
    - 头朝下（projected_gravity_z < 0）惩罚 |pz|
    - 竖直站立（pz ≈ 1）不惩罚（也不给正向奖励）
    - 正常跳跃前倾（pz > 0）不罚 —— v13 证明躺平惩罚(flat_threshold 0.5)与参考动作起跳前倾冲突，
      把策略吓得不敢起跳（成功率 89%→32%）；头朝下（pz<0）才是 v11 翻跟头的根因，仍罚。
    projected_gravity_z = 重力在基座坐标系 z 轴投影：竖直=1、躺平=0、头朝下=-1。"""
    asset = env.scene[asset_cfg.name]
    pz = asset.data.projected_gravity_b[:, 2]
    down_pen = torch.clamp(-pz, min=0.0)
    return down_pen


def torso_roll_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚 base 翻滚角速度（roll/pitch）平方（v12）—— 直接压制翻跟头/翻滚。
    翻跟头是旋转运动，在翻滚进行时（角速度大）即被罚，早于重力投影(z)检测到"头朝下"。
    注册时 weight 取负值才是惩罚（reward = term × weight × dt）。"""
    asset = env.scene[asset_cfg.name]
    ang_vel_w = asset.data.root_ang_vel_w  # (N,3) 世界系 base 角速度
    return torch.sum(ang_vel_w[:, :2] ** 2, dim=-1)


def leg_swing_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["hip_.*", "knee_.*", "ankle_.*"]),
    air_threshold: float = 0.95,
) -> torch.Tensor:
    """腾空时惩罚腿部关节角速度平方（v12）—— 压制腿部乱摆。
    只在 base 高度高于 air_threshold（腾空，无承载）时生效，不干扰地面起跳爆发；
    腾空时腿部多余摆动是无效运动，屈膝收腿等参考动作受影响有限。
    注册时 weight 取负值才是惩罚。"""
    command = env.command_manager.get_term(command_name)
    air = (command.robot_anchor_pos_w[:, 2] > air_threshold).float()
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        joint_ids = slice(None)
    leg_vel = asset.data.joint_vel[:, joint_ids]
    return air * torch.sum(leg_vel ** 2, dim=-1)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def torque_sum_excess(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Compute summed torque excess above a threshold (zero when under the limit)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)

    torque_sum = torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return torch.clamp(torque_sum - threshold, min=0.0)
