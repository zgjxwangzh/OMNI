"""跳高专项奖励函数。

在 xMimic 参考跟踪奖励之外, 针对"原地跳高"目标补充:
- 腾空高度: 鼓励跳得比参考更高(有界)。
- 原地: base_link 水平位移偏离参考锚点(x,y) -> 惩罚(防一步步往前走/侧移)。
- 重心靠后: 整体质心落后参考锚点 -> 惩罚(防后仰/后移摔倒)。
- 躯干稳定: 抑制起跳/腾空/落地的旋转不稳。
- 落地缓冲: 脚触地时向下速度越快惩罚越大(防摔)。

签名遵循 xMimic `rewards.py` 风格: `(env, ...) -> Tensor[N]`。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


def jump_height_above_standing(env: ManagerBasedRLEnv, standing_height: float, std: float = 0.2) -> torch.Tensor:
    """腾空高度奖励: 机器人 root 高于站立基准越多奖励越大(有界 [0,1))。

    阶段二备选(当前训练已禁用, 见 omni_jump_env_cfg 阶段一注释)。
    参考动作在深蹲蓄力段 root 会降到站立以下(0.37m vs 站立 0.79m), 所以
    不能用"高于参考**瞬时**高度"做奖励——那会奖励机器人不下蹲、站着不动
    (深蹲段站着=高出参考 0.42m=接近满分)。改用**固定站立基准**(参考站立帧
    的 root 高度): 深蹲/站立段奖励为 0, 只有真正起跳超过站立高度才有奖励,
    且越高越接近 1。
    """
    robot: Articulation = env.scene["robot"]
    h_robot = robot.data.root_pos_w[:, 2]
    bonus = (h_robot - standing_height).clamp(min=0.0)
    return 1.0 - torch.exp(-bonus / std)


def com_behind_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.15,
    margin: float = 0.02,
) -> torch.Tensor:
    """重心靠后惩罚: 整体质心(x)落后参考锚点(base_link)越远惩罚越大。

    原地跳高要求质心全程保持在参考位置附近(原地、不后移)。以参考锚点 x 为基准,
    质心 x 落后锚点 x 的距离(减去 margin 容差)-> 有界惩罚(返回正值, 由负权重变惩罚)。

    参考动作中整体质心 x 相对 base_link 几乎不落后(最大 ~0.004m, 见
    scripts 分析), 因此跟随参考时该项 ~0; 一旦重心后仰/后移即被惩罚。
    整体质心 = 各 body 质心按 `default_mass` 加权求和。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene["robot"]
    # default_mass 由 physx 读出可能在 CPU, 先对齐到 com_pos 所在设备
    mass = robot.data.default_mass.to(robot.device)  # (N, num_bodies)
    com_pos = robot.data.body_com_pos_w  # (N, num_bodies, 3)
    com_x = (com_pos * mass.unsqueeze(-1)).sum(dim=1) / mass.sum(dim=-1, keepdim=True)
    com_x = com_x[:, 0]
    behind = (command.anchor_pos_w[:, 0] - com_x - margin).clamp(min=0.0)
    return 1.0 - torch.exp(-behind / std)


def root_xy_anchor_penalty(env: ManagerBasedRLEnv, command_name: str = "motion", std: float = 0.3) -> torch.Tensor:
    """原地惩罚: base_link 水平位移偏离参考锚点(x,y)越远惩罚越大。

    参考跳高是原地跳(base_link 的 x/y 全程恒定, 实测 [0,0]), 水平漂移
    (一步步往前走 / 侧移)都是偏离。以参考锚点 x,y 为基准, 水平距离 ->
    有界惩罚(返回正值, 由负权重变惩罚)。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene["robot"]
    root_xy = robot.data.root_pos_w[:, :2]
    anchor_xy = command.anchor_pos_w[:, :2]
    dist = torch.norm(root_xy - anchor_xy, dim=-1)
    return 1.0 - torch.exp(-dist / std)


def action_symmetry_penalty(env: ManagerBasedRLEnv, scale: float = 0.25) -> torch.Tensor:
    """左右对称动作惩罚: 成对关节的动作应镜像对称, 不对称越大惩罚越大。

    机器人身体左右对称, 原地跳高是镜像动作, 因此成对关节(L/R)的动作应满足:
      - pitch 类(髋pitch/膝/踝pitch/肩pitch/肘pitch/腕pitch): a_L == a_R
      - roll/yaw 类(髋roll/髋yaw/踝roll/肩roll/肩yaw/肘yaw/腕roll): a_L == -a_R
    取成对不对称的均值(无量纲平方), 由负权重变惩罚。参考动作已对称化,
    因此跟参考时该项≈0; 一旦策略产出不对称动作(乱扭/偏侧)即被惩罚。
    **平方, 不是 exp(教训: exp 饱和掐死梯度)**: 原 `1-exp(-asym/0.15)` 在
    asym≈0.5 时惩罚顶到 0.96 但梯度只剩 ~0.24, 策略收到"反正罚满, 改不改没
    区别" → 对称学不会(实测全时段 -0.79 平台卡死)。平方 `(asym/scale)²`
    不封顶: 不对称越大梯度越大, 全程有学习信号。

    关节配对索引为 Isaac 实际关节序(实查 robot.joint_names):
      pitch 相等对: (0,1) (9,10) (13,14) (11,12) (21,22) (25,26)
      mirror 镜像对: (3,4) (6,7) (17,18) (15,16) (19,20) (23,24) (27,28)
    """
    actions = env.action_manager.action  # (N, num_actions=29, Isaac 序)
    pitch_pairs = [(0, 1), (9, 10), (13, 14), (11, 12), (21, 22), (25, 26)]
    mirror_pairs = [(3, 4), (6, 7), (17, 18), (15, 16), (19, 20), (23, 24), (27, 28)]
    asym = torch.zeros(actions.shape[0], device=actions.device)
    for i, j in pitch_pairs:
        asym += torch.abs(actions[:, i] - actions[:, j])
    for i, j in mirror_pairs:
        asym += torch.abs(actions[:, i] + actions[:, j])
    asym = asym / (len(pitch_pairs) + len(mirror_pairs))
    return (asym / scale) ** 2


def base_ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """躯干角速度惩罚: 抑制起跳/腾空/落地时身体旋转不稳。

    返回正值(角速度平方和), 由配置里的负权重产生惩罚(与 isaaclab 标准 L2 一致)。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_ang_vel_w), dim=-1)


def landing_hardness_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0
) -> torch.Tensor:
    """落地冲击惩罚: 脚与地面接触时, 躯干向下速度越大惩罚越大。

    落地缓冲阶段脚触地且仍在快速下坠 -> 惩罚, 促使命中后及时缓冲;
    腾空/站立阶段(脚未触地或速度很小) -> 接近 0。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene["robot"]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    in_contact = (torch.norm(forces, dim=-1) > force_threshold).any(dim=-1)
    down_speed = torch.clamp(-robot.data.root_lin_vel_w[:, 2], min=0.0)
    return torch.where(in_contact, down_speed, torch.zeros_like(down_speed))
