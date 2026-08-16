"""529 维 obs: 逐块复刻公司 SDK HighDynamic `_build_obs`(部署基准包 omni_rl_sdk.zip)。

布局(与 SDK `_compute_num_obs` + `_build_obs` 完全一致):
  [0:58]   command    = 参考当前帧 绝对 joint_pos(29) + joint_vel(29)
  [58:64]  anchor_ori = R(robot)^T @ calib @ R(ref当前帧) 的**前两列**(6, C-order)
  [64:529] 5 帧历史(旧→新), 每帧 93 = gravity(3) + ang_vel(3) + joint_pos_rel(29) + joint_vel(29) + action(29)

语义对齐(逐项核对 SDK high_dynamic_policy.py):
- joint_pos_rel = q - SDK_DEFAULT_POS(部署 yaml default_pos 映射到 policy 序)。
  不用 robot.data.default_joint_pos —— 训练 init_state 与 SDK default_pos 差 ~4e-4, 这里减硬编码常量。
- action 历史 = clamp(action_manager.action, -10, 10) = SDK 存 clip(net) 后的 last_action_policy。
  obs 在 process_action 之后算, action_manager.action = 本步施加的原始策略输出 → **无滞后**(push-then-slice)。
- 首 obs(回合重置, episode_length_buf==0): 4 个状态 buffer = 5×当前真实状态,
  action buffer = [0,0,0,0, warmup_action], warmup_action=(q-default)/scale(SDK warmup_from_state, scale=0.5)。
- yaw 校准(SDK _calibrate_init_rotation): 每回合首次 obs 时
  calib = R_z(yaw(robot_q0)) @ R_z(yaw(ref帧0))^T, 之后冻结。
  isaaclab yaw_quat 公式与 SDK _yaw_quat 逐字一致(已核实)。
- reset 检测: episode_length_buf == 0 —— _reset_idx 在 observation_manager.compute 之前跑(ManagerBasedRLEnv.step)。
- 幂等缓存: 同一步(common_step_counter)二次调用直接返回缓存, 防历史 buffer 双推进。

铁律: 训练 obs = 部署 obs, 逐字节。此函数输出 = ONNX 网络的唯一输入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- SDK high_dynamic.yaml(部署基准包 omni_rl_sdk.zip, 已核实无第二层覆盖) ---
SDK_ACTION_SCALE = 0.5   # action.scale
SDK_ACTION_CLIP = 10.0   # action.clip
HISTORY_LENGTH = 5       # observation.history_length

# SDK high_dynamic.yaml `default_pos`(motor 序) -> 映射到 `policy_joint_names` 序(= 训练 URDF/BFS 序)。
# 逐项核对 zip policy_joint_names 29 个; 勿用 run-length 简写(曾数错 28)。
SDK_DEFAULT_POS = torch.tensor(
    [
        -0.262, -0.262,   # hip_pitch_l, hip_pitch_r
        0.0, 0.0, 0.0,    # waist_yaw, hip_roll_l, hip_roll_r
        0.0, 0.0, 0.0,    # waist_roll, hip_yaw_l, hip_yaw_r
        0.0,              # waist_pitch
        0.524, 0.524,     # knee_pitch_l, knee_pitch_r
        0.3, 0.3,         # shoulder_pitch_l, shoulder_pitch_r
        -0.262, -0.262,   # ankle_pitch_l, ankle_pitch_r
        0.0, 0.0,         # shoulder_roll_l, shoulder_roll_r
        0.0, 0.0,         # ankle_roll_l, ankle_roll_r
        0.0, 0.0,         # shoulder_yaw_l, shoulder_yaw_r
        -0.7, -0.7,       # elbow_pitch_l, elbow_pitch_r
        0.0, 0.0,         # elbow_yaw_l, elbow_yaw_r
        0.0, 0.0,         # wrist_pitch_l, wrist_pitch_r
        0.0, 0.0,         # wrist_roll_l, wrist_roll_r
    ],
    dtype=torch.float32,
)
assert SDK_DEFAULT_POS.numel() == 29


def _matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    """四元数 (...,4) (w,x,y,z) -> (...,3,3) 旋转矩阵。

    纯 torch 实现, 与 isaaclab.utils.math.matrix_from_quat / SDK matrix_from_quat_numpy
    公式逐项一致(已核实)。不用 isaaclab 的版本是为了让本模块可脱离 omni 运行时
    独立导入与单测(字节一致性关卡只依赖 torch)。
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    """四元数共轭(单位四元数即逆), 与 isaaclab quat_inv 等价。"""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """四元数 (...,4) -> 绕世界 z 的偏航四元数。

    与 isaaclab.utils.math.yaw_quat / SDK _yaw_quat 公式逐字一致(已核实):
    yaw = atan2(2(wz+xy), 1-2(y²+z²)), 返回 [cos(yaw/2), 0, 0, sin(yaw/2)]。
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack(
        [torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)],
        dim=-1,
    )


def _init_state(num_envs: int, device: torch.device) -> dict:
    """历史 buffer + yaw 校准缓存。"""
    return {
        "num_envs": num_envs,
        "device": device,
        "gravity": torch.zeros(num_envs, HISTORY_LENGTH, 3, device=device),
        "ang_vel": torch.zeros(num_envs, HISTORY_LENGTH, 3, device=device),
        "joint_pos": torch.zeros(num_envs, HISTORY_LENGTH, 29, device=device),
        "joint_vel": torch.zeros(num_envs, HISTORY_LENGTH, 29, device=device),
        "action": torch.zeros(num_envs, HISTORY_LENGTH, 29, device=device),
        "calib": torch.eye(3, device=device).repeat(num_envs, 1, 1),  # per-env yaw 校准
        "step": -1,  # 已处理的 common_step_counter
        "cache": None,
    }


def _prefill(env: "ManagerBasedRLEnv", command, robot, state: dict, fresh: torch.Tensor) -> None:
    """回合重置: 5×当前真实状态 + warmup 动作 + yaw 校准(复刻 SDK _reset_history + warmup_from_state)。"""
    device = robot.device
    grav = robot.data.projected_gravity_b[fresh]  # (F,3) = R^T@[0,0,-1], 与 SDK gravity_ori 同式
    ang = robot.data.root_ang_vel_b[fresh]  # (F,3)
    q_rel = robot.data.joint_pos[fresh] - SDK_DEFAULT_POS.to(device)  # (F,29)
    dq = robot.data.joint_vel[fresh]  # (F,29)
    warmup = q_rel / SDK_ACTION_SCALE  # SDK warmup_action = (q-default)/scale

    state["gravity"][fresh] = grav[:, None, :].expand(-1, HISTORY_LENGTH, -1).clone()
    state["ang_vel"][fresh] = ang[:, None, :].expand(-1, HISTORY_LENGTH, -1).clone()
    state["joint_pos"][fresh] = q_rel[:, None, :].expand(-1, HISTORY_LENGTH, -1).clone()
    state["joint_vel"][fresh] = dq[:, None, :].expand(-1, HISTORY_LENGTH, -1).clone()
    state["action"][fresh] = 0.0
    state["action"][fresh, -1, :] = warmup  # [0,0,0,0,warmup_action]; 注意是末帧整行, 不是末关节

    # yaw 校准: calib = R_z(yaw(robot_q0)) @ R_z(yaw(ref帧0))^T (SDK _calibrate_init_rotation)
    rob_q0 = robot.data.root_quat_w[fresh]  # (F,4) reset 姿
    ref_q0 = command.motion.body_quat_w[0, command.motion_anchor_body_index]  # (4,) 参考帧0 base_link
    r_rob = _matrix_from_quat(_yaw_quat(rob_q0))  # (F,3,3)
    r_ref = _matrix_from_quat(_yaw_quat(ref_q0))  # (3,3)
    state["calib"][fresh] = r_rob @ r_ref.T


def _roll(env: "ManagerBasedRLEnv", command, robot, state: dict, mask: torch.Tensor) -> None:
    """非重置: 滚动推进历史(丢最旧、尾插当前)。"""
    device = robot.device
    grav = robot.data.projected_gravity_b[mask]
    ang = robot.data.root_ang_vel_b[mask]
    q_rel = robot.data.joint_pos[mask] - SDK_DEFAULT_POS.to(device)
    dq = robot.data.joint_vel[mask]
    act = torch.clamp(env.action_manager.action[mask], -SDK_ACTION_CLIP, SDK_ACTION_CLIP)

    state["gravity"][mask] = torch.roll(state["gravity"][mask], shifts=-1, dims=1)
    state["gravity"][mask, -1, :] = grav
    state["ang_vel"][mask] = torch.roll(state["ang_vel"][mask], shifts=-1, dims=1)
    state["ang_vel"][mask, -1, :] = ang
    state["joint_pos"][mask] = torch.roll(state["joint_pos"][mask], shifts=-1, dims=1)
    state["joint_pos"][mask, -1, :] = q_rel
    state["joint_vel"][mask] = torch.roll(state["joint_vel"][mask], shifts=-1, dims=1)
    state["joint_vel"][mask, -1, :] = dq
    state["action"][mask] = torch.roll(state["action"][mask], shifts=-1, dims=1)
    state["action"][mask, -1, :] = act


def _anchor_ori(env: "ManagerBasedRLEnv", command, robot, state: dict) -> torch.Tensor:
    """anchor_ori = (R(robot)^T @ calib @ R(ref当前帧)) 前两列, C-order(6)。

    复刻 SDK _get_anchor_ori_b: rot_b = rot_inv @ world_to_init_rot @ ref_rot; return rot_b[:, :2].reshape(-1)。
    """
    robot_q = command.robot_anchor_quat_w  # (N,4) 当前机 base_link 四元数
    ref_q = command.anchor_quat_w  # (N,4) 当前帧参考 base_link 四元数
    r_robot_t = _matrix_from_quat(_quat_inv(robot_q))  # R(q)^T, (N,3,3)
    r_ref = _matrix_from_quat(ref_q)  # (N,3,3)
    rot_b = torch.matmul(r_robot_t, torch.matmul(state["calib"], r_ref))  # (N,3,3)
    return rot_b[..., :2].reshape(-1, 6)  # 前两列, C-order(与 numpy reshape 一致)


def obs529(env: "ManagerBasedRLEnv", command_name: str = "motion") -> torch.Tensor:
    """529 维 obs, 与 SDK HighDynamic 逐字节一致。"""
    command = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    device = robot.device
    num_envs = env.num_envs

    # 懒初始化(env 数量/设备变了重建)
    state = env.extras.get("_obs529")
    if state is None or state["num_envs"] != num_envs or state["device"] != device:
        state = _init_state(num_envs, device)
        env.extras["_obs529"] = state

    # 幂等: 同一步已算过 → 返回缓存(防 recorder/双算把历史推进两遍)
    step = env.common_step_counter
    if state["step"] == step:
        return state["cache"]

    # reset 检测: _reset_idx 在 obs 之前跑, 重置 env 的 episode_length_buf == 0
    ep_len = env.episode_length_buf
    fresh = ep_len == 0

    # 回合重置: 重预填 + 重算 yaw 校准; 非重置: 滚动推进
    if fresh.any():
        _prefill(env, command, robot, state, fresh)
    mask = ~fresh
    if mask.any():
        _roll(env, command, robot, state, mask)

    # 组装 529 (SDK 顺序)
    cmd_obs = command.command  # (N,58) 参考当前帧绝对 pos+vel
    anchor_ori = _anchor_ori(env, command, robot, state)  # (N,6)
    hist = torch.cat(
        [
            state["gravity"].reshape(num_envs, -1),
            state["ang_vel"].reshape(num_envs, -1),
            state["joint_pos"].reshape(num_envs, -1),
            state["joint_vel"].reshape(num_envs, -1),
            state["action"].reshape(num_envs, -1),
        ],
        dim=1,
    )  # (N, 5*93=465)
    obs = torch.cat([cmd_obs, anchor_ori, hist], dim=1)  # (N, 529)
    if obs.shape[1] != 529:
        raise RuntimeError(f"[obs529] obs dim {obs.shape[1]} != 529")

    state["step"] = step
    state["cache"] = obs
    return obs
