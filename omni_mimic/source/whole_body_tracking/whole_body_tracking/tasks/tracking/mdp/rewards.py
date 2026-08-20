from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from whole_body_tracking.envs import MyBaseRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_xyz_indexes(xyz_dim: list[str] | None) -> list[int]:
    if xyz_dim is None:
        return [0, 1, 2]
    dim_map = {'x': 0, 'y': 1, 'z': 2}
    return [dim_map[d] for d in xyz_dim if d in dim_map]


def motion_global_anchor_position_error_exp(env: MyBaseRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: MyBaseRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: MyBaseRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: MyBaseRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: MyBaseRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: MyBaseRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: MyBaseRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

########################################### feet ###########################################
def feet_spacing(
    env: MyBaseRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)

def feet_slide(
    env: MyBaseRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def swing_foot_clearance(
    env: MyBaseRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

########################################### root and joint level ###########################################
def anchor_global_position(env: MyBaseRLEnv, command_name: str, std: float, xyz_dim: list[str] | None = None, penalty: bool = False) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    dim_index = _get_xyz_indexes(xyz_dim)
    error = torch.sum(torch.square(command.anchor_pos_w[:, dim_index] - command.robot_anchor_pos_w[:, dim_index]), dim=-1)
    return 1.0 * penalty + (-1)**(penalty) * torch.exp(-error / std**2)

def anchor_yaw_exp(env: MyBaseRLEnv, command_name: str, std: float, penalty: bool = False) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    anchor_quat_yaw = yaw_quat(command.anchor_quat_w)
    robot_anchor_quat_yaw = yaw_quat(command.robot_anchor_quat_w)
    error = quat_error_magnitude(anchor_quat_yaw, robot_anchor_quat_yaw) ** 2
    return 1.0 * penalty + (-1)**(penalty) * torch.exp(-error / std**2)

def joint_pos_exp(env: MyBaseRLEnv, command_name: str, std: float, body_names: list[str] | None = None, penalty: bool = False) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if body_names is None:
        joint_ids = slice(None)  # all joints
    else:
        joint_ids = command.robot.find_joints(body_names)[0]
    # 2026-08-20 修复: sum → mean, 解决 29 关节求和导致奖励曲面平坦、梯度消失的问题
    # 参考: liuzq_jump_train_v11 跳高训练用 mean + std=0.5 成功收敛
    error = torch.mean(torch.square(command.joint_pos[:, joint_ids] - command.robot_joint_pos[:, joint_ids]), dim=-1)
    return 1.0 * penalty + (-1)**(penalty) * torch.exp(-error / std**2)

# 线性惩罚：不饱和，误差越大惩罚越大，梯度始终存在
def joint_pos_l2_penalty(env: MyBaseRLEnv, command_name: str, body_names: list[str] | None = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if body_names is None:
        joint_ids = slice(None)  # all joints
    else:
        joint_ids = command.robot.find_joints(body_names)[0]
    error = torch.mean(torch.square(command.joint_pos[:, joint_ids] - command.robot_joint_pos[:, joint_ids]), dim=-1)
    return -error  # 负值：误差越大，奖励越小（惩罚越大）

def joint_vel_exp(env: MyBaseRLEnv, command_name: str, std: float, body_names: list[str] | None = None, penalty: bool = False) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if body_names is None:
        joint_ids = slice(None)  # all joints
    else:
        joint_ids = command.robot.find_joints(body_names)[0]
    # 2026-08-20 修复: sum → mean, 与 joint_pos_exp 保持一致
    error = torch.mean(torch.square(command.joint_vel[:, joint_ids] - command.robot_joint_vel[:, joint_ids]), dim=-1)
    return 1.0 * penalty + (-1)**(penalty) * torch.exp(-error / std**2)

########################################### feet symmetry (for two feet symmetry motions) ###########################################
def feet_force_symmetry(
    env: MyBaseRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 50.0,
) -> torch.Tensor:
    """Penalize asymmetric Z-direction ground reaction forces between left and right feet.

    Only active when both feet are in contact (force > threshold), so it won't
    interfere with walking or single-leg phases. Returns the normalized force
    difference: |F_left_z - F_right_z| / max(F_left_z, F_right_z).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history: (num_envs, history_len, num_bodies, 3)
    forces_z = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2].abs()
    # body_ids should have exactly 2 entries: [left_foot, right_foot]
    left_z = forces_z[:, 0]
    right_z = forces_z[:, 1]

    both_contact = (left_z > force_threshold) & (right_z > force_threshold)
    max_force = torch.max(left_z, right_z).clamp(min=1.0)
    force_diff = torch.abs(left_z - right_z) / max_force

    return force_diff * both_contact.float()

def feet_contact_symmetry(
    env: MyBaseRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize one foot in contact while the other is in the air.

    Uses max force over the history window for robust contact detection.
    Returns 1.0 when exactly one foot is on the ground, 0.0 otherwise.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # (num_envs, history_len, num_bodies, 3)
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    max_force = forces.norm(dim=-1).max(dim=1)[0]  # (num_envs, 2)
    left_contact = max_force[:, 0] > force_threshold
    right_contact = max_force[:, 1] > force_threshold

    # XOR: exactly one foot on ground
    return (left_contact ^ right_contact).float()

