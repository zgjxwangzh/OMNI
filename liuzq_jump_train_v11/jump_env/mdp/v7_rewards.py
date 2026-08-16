"""V7 奖励栈移植到 jump_high(xMimic) 框架的纯 torch 实现。

2026-08-13 V11: 把 my_omni_jump_train_v7/omni_jump_tasks_v7/mdp/rewards.py 的 26 项
奖励函数移植到 jump_high 框架。运行时零 isaaclab 依赖(纯 torch + TYPE_CHECKING
import), 便于无 sim 单测。

移植适配(相对 V7 原文):
1. `SceneEntityCfg` / `matrix_from_quat` / `quat_error_magnitude` 挪到 TYPE_CHECKING
   import; 运行时用本文件的纯 torch `_matrix_from_quat`(wxyz) / `_quat_error_magnitude`。
2. `_quat_yaw` 修正: V7 版把 wxyz 当 xyzw 拆(q[...,0]=x), 对纯 yaw 恒返 0(潜在 no-op)。
   改用 wxyz 正确公式 atan2(2(wz+xy), 1-2(y²+z²)), 与 obs529 的 `_yaw_quat` 一致。
3. 分类 D 函数(takeoff_vertical_vel / premature_jump_penalty / torso_backward_lean_penalty /
   torso_roll_penalty / flight_yaw_penalty)读 `command.motion.jump_mask` /
   `command.motion.first_jump_frame` —— 由 JumpMotionCommand 在 __init__ 内存计算提供
   (见 jump_env/mdp/commands.py)。
4. 帧号/高度阈值按 jump_high 参考(183帧/50fps, base 峰值 0.961m)重标定, 见各函数
   默认参数; 实际注册值在 omni_jump_env_cfg.py 的 RewardsCfg。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


def _matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    """四元数(...,4) (w,x,y,z) -> (...,3,3) 旋转矩阵。wxyz。"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def _quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """两个四元数(wxyz)的误差角(弧度)。"""
    dot = (q1 * q2).sum(dim=-1).clamp(min=-1.0, max=1.0)
    return 2.0 * torch.acos(dot.abs())


def _quat_yaw(q: torch.Tensor) -> torch.Tensor:
    """四元数(wxyz)绕世界 z 的偏航角: atan2(2(wz+xy), 1-2(y²+z²))。

    V11 修正: V7 版把 wxyz 当 xyzw 拆(x,y,z,w = q[...,0..3]), 对纯 yaw 恒返 0。
    这里用 wxyz 正确公式(与 obs529._yaw_quat 一致)。
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _jump_gate(env: "ManagerBasedRLEnv", command: "MotionCommand", threshold: float = 0.80) -> torch.Tensor:
    """第一次腾空窗口闸门: 首次 base 升到阈值→进入腾空窗口(持续给奖), 回落到阈值以下才锁定。

    2026-08-15 V11 修复: 原实现"首次 base>threshold 立即锁定"导致 jump_height_bonus
    只在腾空第 1 帧给奖, 后续 17 帧(base 持续 0.80+)全 0 → 策略没有"持续跳高"动机,
    只把 apex 提到刚过 0.80 就拿 1 帧奖, 不再冲高(日志 jump_height_bonus=0)。

    新逻辑(状态机, 存 env.extras):
      state: 0=未起跳 / 1=第一次腾空窗口(给奖) / 2=已锁定(本集后续 0)
      - 未起跳(0) → base>threshold 进 1(窗口开始)
      - 窗口内(1) → base<threshold(回落落地) → 锁 2
      - 已锁定(2) → 恒 0
    效果: 第一次腾空全程给高度奖励, 腾空结束后锁定, 第二次起跳不给。
    """
    key = "_jump_gate_state"  # 0/1/2
    state = env.extras.get(key)
    if state is None or state.shape[0] != env.num_envs:
        state = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.extras[key] = state
    # 新 episode 清空: ⚠️ 2026-08-15 修复重大 bug。
    # reward 计算在 step() 的 episode_length_buf+=1 之后, reset 在其后 → reward 时
    # episode_length_buf 恒 >=1, 原 `== 0` 永不成立 → gate state 从不重置 → 从第二集起
    # state 卡在 2(锁定), jump_height/takeoff_vel/tuck 恒 0(跳高奖励从未生效)。
    # 新 episode 第一帧 episode_length_buf==1(reset 置 0 → 下步 +1), 用 <=1 判定。
    fresh = env.episode_length_buf <= 1
    state = torch.where(fresh, torch.zeros_like(state), state)
    robot_z = command.robot_anchor_pos_w[:, 2]
    airborne = robot_z > threshold
    # 状态转移
    # 0 → 1: base 首次超阈值(起跳)
    enter = (state == 0) & airborne
    state = torch.where(enter, torch.ones_like(state), state)
    # 1 → 2: 窗口内 base 回落(腾空结束/落地)
    exit_win = (state == 1) & ~airborne
    state = torch.where(exit_win, torch.full_like(state, 2), state)
    env.extras[key] = state
    # gate: 窗口内(1)=1 给奖; 未起跳(0)或已锁定(2)=0
    gate = (state == 1).float()
    return gate


# ---------------------------------------------------------------------------
# 分类 A/B/C/D: 与 V7 逐字对应(仅修正四元数约定)
# ---------------------------------------------------------------------------


def track_dof_pos_exp(env: "ManagerBasedRLEnv", command_name: str, std: float = 0.5) -> torch.Tensor:
    """核心模仿奖励: 关节角跟踪(参考 joint_pos vs 机器人关节角)。

    2026-08-13 V11 修复: 原 29 关节**求和**平方误差 + std 0.3, 在真实策略下饱和归零
    (参考膝 0.29→2.18, 双膝误差²≈7, exp(-7/0.09)≈0 → 屈膝无梯度 → 策略膝盖不动)。
    改**平均**平方误差 + std 0.5: 单膝偏 0.5 rad → error_mean≈0.25, exp(-0.25/0.25)≈0.37
    → 屈膝有真实梯度, 策略才学得动下蹲。与 arm_tracking 同思路。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    error = torch.mean(torch.square(command.joint_pos - command.robot.data.joint_pos), dim=-1)
    return torch.exp(-error / std**2)


def track_root_ori_exp(env: "ManagerBasedRLEnv", command_name: str, std: float = 0.4) -> torch.Tensor:
    """基座姿态跟踪(参考 anchor_quat_w vs 机器人基座姿态)。"""
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    error = _quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def track_yaw_exp(
    env: "ManagerBasedRLEnv", command_name: str, std: float = 0.3, scale: float = 1.0
) -> torch.Tensor:
    """基座 yaw(绕世界 z)跟踪奖励: 参考 vs 机器人, exp 核。"""
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    yaw_ref = _quat_yaw(command.anchor_quat_w)
    yaw_rob = _quat_yaw(command.robot_anchor_quat_w)
    err = torch.atan2(torch.sin(yaw_ref - yaw_rob), torch.cos(yaw_ref - yaw_rob))
    return scale * torch.exp(-err**2 / std**2)


def jump_height_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    threshold: float = 0.80,
    scale: float = 1.0,
    max_excess: float = 0.8,
) -> torch.Tensor:
    """腾空奖励: 机器人基座高度超过阈值(物理值)。参考高度不跟踪, 纯激励起跳。

    2026-08-14 V11 修复: 阈值 0.85→0.80(比站立 0.782 高 0.018)。model_25400 策略
    深蹲但推蹬后 root 到不了 0.85 → 腾空奖励不触发 → 无冲高动机 → 不起跳。
    降到 0.80 让"轻微蹬起"就能触发, 形成正反馈逐步学会起跳。
    2026-08-14 加 _jump_gate: 每集只奖第一次腾空, 防"跳两次"(参考播完重采样后再跳)。
    2026-08-15 超线性化: `excess + excess**2`。线性奖励下从 0.90 冲到 1.0 只多
    0.10/步(weight 10 → +1.0), 高段边际不变, 策略冲过参考(0.961)后没额外动机。
    加二次项后 z=1.00 → 0.20+0.04=0.24 vs z=0.90 → 0.10+0.01=0.11, 越冲越高奖励
    递增越快, 直接激励突破参考 apex 0.961(目标 1.05, 需起跳速度 2.3 m/s)。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    gate = _jump_gate(env, command, threshold=0.80)
    robot_z = command.robot_anchor_pos_w[:, 2]
    excess = torch.clamp(robot_z - threshold, min=0.0, max=max_excess)
    return gate * (excess + excess**2) * scale


def takeoff_vertical_vel(
    env: "ManagerBasedRLEnv", command_name: str, vel_thresh: float = 0.3, max_vel: float = 1.8
) -> torch.Tensor:
    """起跳帧(jump_mask)内奖励机器人向上的基座速度, 激励爆发起跳。
    2026-08-14 加 _jump_gate: 每集只奖第一次腾空, 防"跳两次"。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    gate = _jump_gate(env, command, threshold=0.80)
    active = command.motion.jump_mask[command.time_steps]
    robot_vel_z = command.robot_anchor_lin_vel_w[:, 2]
    return gate * active.float() * torch.clamp(robot_vel_z, min=0.0, max=max_vel) / max_vel


def takeoff_push_power(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    max_ext_vel: float = 6.0,
    threshold: float = 0.80,
) -> torch.Tensor:
    """推蹬段奖励双腿快速伸展(膝/髋伸直), 激励爆发起跳。

    2026-08-14 V11 新增: jump_mask 窗口内奖励双腿关节的"伸展角速度"(负角速度=伸直)。
    参考推蹬段(99-115): 膝 2.18→0.16(伸展 2.02 rad), 髋 -1.97→-0.1, 膝伸展角速度
    峰值 ~11.8 rad/s。直接用关节角速度(负=伸直)激励, 比等 base 升高更早、更直接,
    教策略"爆发蹬地"。返回 [0,1], 由正权重产生奖励。
    2026-08-15 加 _jump_gate: ⚠️ 修"两次下蹲蓄力"。原无 gate, jump_mask 窗口(57-132,
    含整个下蹲段)内任意"腿伸展"都给奖 → 策略学会"下蹲-站起-再下蹲-再站起"反复刷
    (每次站起伸膝都拿伸展奖励), 不真跳。gate 只在第一次腾空窗口(0.80)给 → 下蹲段
    gate=0 刷不到分, 只能靠真正起跳拿奖励。起跳爆发主信号仍由 takeoff_vertical_vel
    (weight 10, 已 gate) 承担。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    gate = _jump_gate(env, command, threshold=threshold)
    active = command.motion.jump_mask[command.time_steps].float()
    joint_vel = env.scene[asset_cfg.name].data.joint_vel  # (E,29) rad/s, 正=屈曲, 负=伸直
    idx = [asset_cfg.joint_ids[asset_cfg.joint_names.index(n)] for n in asset_cfg.joint_names]
    ext_vel = torch.clamp(-joint_vel[:, idx], min=0.0)  # 伸展速度(正=伸直)
    power = torch.clamp(ext_vel.mean(dim=-1) / max_ext_vel, min=0.0, max=1.0)
    return gate * active * power


def premature_jump_penalty(
    env: "ManagerBasedRLEnv", command_name: str, threshold: float = 0.95
) -> torch.Tensor:
    """首个跳跃帧之前机器人基座就升得过高(提前起跳)的惩罚量(正值, 配负权重)。"""
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    robot_z = command.robot_anchor_pos_w[:, 2]
    before_jump = command.time_steps < command.motion.first_jump_frame
    violation = robot_z - threshold
    return (before_jump & (violation > 0.0)).float() * violation


def hip_spread_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg",
    threshold: float = 0.35,
    scale: float = 1.0,
) -> torch.Tensor:
    """限制大腿外展(双腿叉开): 只罚 hip_roll 超出阈值的外展量(正值, 配负权重)。"""
    joint_pos = env.scene[asset_cfg.name].data.joint_pos
    idx_l = asset_cfg.joint_ids[asset_cfg.joint_names.index("hip_roll_l_joint")]
    idx_r = asset_cfg.joint_ids[asset_cfg.joint_names.index("hip_roll_r_joint")]
    spread_l = torch.clamp(joint_pos[:, idx_l] - threshold, min=0.0)
    spread_r = torch.clamp(-joint_pos[:, idx_r] - threshold, min=0.0)
    return scale * (spread_l**2 + spread_r**2)


def torso_backward_lean_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    tolerance: float = 0.04,
    scale: float = 1.0,
    flight_threshold: float = 0.78,
    flight_cutoff_frame: int = 183,
) -> torch.Tensor:
    """起跳窗口 + 腾空 + 落地段惩罚躯干后仰(基座前向轴朝上翘起)。

    2026-08-14 V11 修复: 原门控 in_flight 用 base>flight_threshold(0.90/0.78),
    但策略落地段 base 只有 0.72-0.79 → 落地段后仰不触发。改为**纯 frame 门控**
    (腾空+落地段 = 115~183, 不依赖 base 高度), 参考落地段 fwd_z≤0.002(前倾),
    罚后仰(>0.04)不冲突。用户反馈"后仰出现在跳跃结束后"= 落地段, 现已覆盖。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    in_jump = command.motion.jump_mask[command.time_steps]
    # 腾空+落地段: frame 115~183(参考腾空 115-132 + 落地 133-183), 纯 frame 门控
    in_flight = (command.time_steps >= 115) & (command.time_steps < flight_cutoff_frame)
    active = (in_jump | in_flight).float()
    fwd_z = _matrix_from_quat(command.robot_anchor_quat_w)[..., 2, 0]
    lean = torch.clamp(fwd_z - tolerance, min=0.0)
    return active * scale * lean


def arm_back_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    airborne_threshold: float = 0.90,
    shoulder_tol: float = 0.2,
    scale: float = 1.0,
) -> torch.Tensor:
    """腾空段(z>base 阈值)惩罚肩后摆。"""
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    airborne = (command.robot_anchor_pos_w[:, 2] > airborne_threshold).float()
    joint_pos = env.scene[asset_cfg.name].data.joint_pos

    def _col(name: str) -> int:
        return asset_cfg.joint_ids[asset_cfg.joint_names.index(name)]

    sh_l = joint_pos[:, _col("shoulder_pitch_l_joint")]
    sh_r = joint_pos[:, _col("shoulder_pitch_r_joint")]
    back_l = torch.clamp(sh_l - shoulder_tol, min=0.0)
    back_r = torch.clamp(sh_r - shoulder_tol, min=0.0)
    return airborne * scale * (back_l + back_r)


def arm_tracking_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    std: float = 0.3,
) -> torch.Tensor:
    """胳膊关节跟踪奖励: 14 胳膊关节平均平方误差(非求和), exp 核。"""
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    idx = [asset_cfg.joint_ids[asset_cfg.joint_names.index(n)] for n in asset_cfg.joint_names]
    err = torch.mean(
        torch.square(command.joint_pos[:, idx] - command.robot.data.joint_pos[:, idx]), dim=-1
    )
    return torch.exp(-err / std**2)


def tuck_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    airborne_threshold: float = 0.90,
    start_frame: int = 115,
    knee_target: float = 1.5,
    hip_target: float = -0.9,
    sigma: float = 0.4,
    scale: float = 1.0,
) -> torch.Tensor:
    """腾空段收腿奖励(膝屈 + 髋屈): 指定弯曲方向的高斯目标。
    2026-08-14 加 _jump_gate: 每集只奖第一次腾空, 防"跳两次"。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    gate = _jump_gate(env, command, threshold=0.80)
    airborne = (command.robot_anchor_pos_w[:, 2] > airborne_threshold) & (
        command.time_steps > start_frame
    )
    active = airborne.float()
    joint_pos = env.scene[asset_cfg.name].data.joint_pos

    def _col(name: str) -> int:
        return asset_cfg.joint_ids[asset_cfg.joint_names.index(name)]

    knee = (joint_pos[:, _col("knee_pitch_l_joint")] + joint_pos[:, _col("knee_pitch_r_joint")]) / 2
    hip = (joint_pos[:, _col("hip_pitch_l_joint")] + joint_pos[:, _col("hip_pitch_r_joint")]) / 2
    quality = torch.exp(-(knee - knee_target) ** 2 / (2 * sigma**2)) * torch.exp(
        -(hip - hip_target) ** 2 / (2 * sigma**2)
    )
    return gate * active * scale * quality


def flight_yaw_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    airborne_threshold: float = 0.90,
    scale: float = 1.0,
) -> torch.Tensor:
    """起跳窗口+腾空段(jump_mask|z>阈值)惩罚世界系 z 轴角速度平方: 抑制向右拧转。"""
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    in_jump = command.motion.jump_mask[command.time_steps]
    in_flight = command.robot_anchor_pos_w[:, 2] > airborne_threshold
    active = (in_jump | in_flight).float()
    omega_w = env.scene[asset_cfg.name].data.root_ang_vel_w
    return active * scale * omega_w[:, 2] ** 2


def waist_yaw_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    scale: float = 1.0,
) -> torch.Tensor:
    """惩罚上半身(腰)偏航相对参考的扭转(正值, 配负权重)。"""
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    idx = asset_cfg.joint_ids[asset_cfg.joint_names.index("waist_yaw_joint")]
    waist = env.scene[asset_cfg.name].data.joint_pos[:, idx]
    ref_waist = command.joint_pos[:, idx]
    return scale * torch.abs(waist - ref_waist)


def waist_roll_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    scale: float = 1.0,
) -> torch.Tensor:
    """惩罚上半身(腰)侧倾相对参考的扭转(正值, 配负权重)。"""
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    idx = asset_cfg.joint_ids[asset_cfg.joint_names.index("waist_roll_joint")]
    waist = env.scene[asset_cfg.name].data.joint_pos[:, idx]
    ref_waist = command.joint_pos[:, idx]
    return scale * torch.abs(waist - ref_waist)


def elbow_bend_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    lo: float = 0.0,
    hi: float = 2.09,
    scale: float = 1.0,
) -> torch.Tensor:
    """全局惩罚肘部超出[lo,hi](60°-180° 允许起跳伸臂, 只防过度屈肘)。"""
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    joint_pos = env.scene[asset_cfg.name].data.joint_pos

    def _col(name: str) -> int:
        return asset_cfg.joint_ids[asset_cfg.joint_names.index(name)]

    elb_l = joint_pos[:, _col("elbow_pitch_l_joint")]
    elb_r = joint_pos[:, _col("elbow_pitch_r_joint")]
    viol_l = torch.clamp(elb_l - hi, min=0.0) + torch.clamp(lo - elb_l, min=0.0)
    viol_r = torch.clamp(elb_r - hi, min=0.0) + torch.clamp(lo - elb_r, min=0.0)
    return scale * (viol_l + viol_r)


def boundary_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    inner: float = 1.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """基座 x/y 相对各自 env 中心超出 ±inner 时惩罚(正值, 配负权重), 保持居中。

    2026-08-13 V11 修复: jump_high 环境分散在 env_origins, robot_anchor_pos_w 是
    绝对世界坐标, 须减 env_origins 判相对位移(否则所有 env 恒 >inner 全被罚)。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    pos = command.robot_anchor_pos_w
    rel = pos - env.scene.env_origins
    viol = torch.clamp(rel[:, 0].abs() - inner, min=0.0) + torch.clamp(rel[:, 1].abs() - inner, min=0.0)
    return scale * viol


def feet_force_symmetry(
    env: "ManagerBasedRLEnv",
    sensor_cfg: "SceneEntityCfg",
    force_threshold: float = 50.0,
) -> torch.Tensor:
    """罚左右脚地面反力不对称(双脚同时着地时): |F_left_z - F_right_z| / max(F_l, F_r)。"""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces_z = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2].abs()
    left_z, right_z = forces_z[:, 0], forces_z[:, 1]
    both_contact = (left_z > force_threshold) & (right_z > force_threshold)
    max_force = torch.max(left_z, right_z).clamp(min=1.0)
    force_diff = torch.abs(left_z - right_z) / max_force
    return force_diff * both_contact.float()


def feet_contact_symmetry(
    env: "ManagerBasedRLEnv",
    sensor_cfg: "SceneEntityCfg",
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """罚"单脚着地"不对称: 用历史窗口最大力判接触, 恰好一只脚着地 → 1。"""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    max_force = forces.norm(dim=-1).max(dim=1)[0]
    left_contact = max_force[:, 0] > force_threshold
    right_contact = max_force[:, 1] > force_threshold
    return (left_contact ^ right_contact).float()


def leg_symmetry_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg",
    threshold: float = 0.1,
    scale: float = 1.0,
) -> torch.Tensor:
    """罚左右腿对称关节角偏差超 threshold(正值, 配负权重)。"""
    joint_pos = env.scene[asset_cfg.name].data.joint_pos

    def _col(name: str) -> int:
        return asset_cfg.joint_ids[asset_cfg.joint_names.index(name)]

    pairs = [
        (_col("hip_pitch_l_joint"), _col("hip_pitch_r_joint"), 1),
        (_col("hip_roll_l_joint"), _col("hip_roll_r_joint"), -1),
        (_col("knee_pitch_l_joint"), _col("knee_pitch_r_joint"), 1),
        (_col("ankle_pitch_l_joint"), _col("ankle_pitch_r_joint"), 1),
        (_col("ankle_roll_l_joint"), _col("ankle_roll_r_joint"), -1),
    ]
    total = 0
    for il, ir, sign in pairs:
        diff = (joint_pos[:, il] - sign * joint_pos[:, ir]).abs()
        total = total + torch.clamp(diff - threshold, min=0.0)
    return scale * total


def takeoff_leg_symmetry_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    start_frame: int = 90,
    cutoff_frame: int = 115,
    threshold: float = 0.1,
    scale: float = 1.0,
) -> torch.Tensor:
    """推蹬段罚左右腿发力不一致(髋/膝伸展角速度差), 治腾空扭转。

    2026-08-16 V11 新增: 腾空扭转根因=推蹬段双腿发力不一致→产生绕垂直轴角动量。
    参考推蹬段(99-115) hip_pitch/knee_pitch 左右伸展速度差**0.0000**(完全对称),
    零冲突。罚左右髋/膝伸展速度差超 threshold(0.1 rad/s): 双腿同步发力=0 惩罚,
    发力不一致(一侧先蹬/更猛)→罚。只罚 hip/knee(踝参考自带轻微不对称 1.32, 不罚避冲突)。
    窗口 90~115 覆盖推蹬段, 避开腾空(115+)收腿。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    active = ((command.time_steps >= start_frame) & (command.time_steps < cutoff_frame)).float()
    joint_vel = env.scene[asset_cfg.name].data.joint_vel  # (E,29) 正=屈曲 负=伸直

    def _col(name: str) -> int:
        return asset_cfg.joint_ids[asset_cfg.joint_names.index(name)]

    # 髋/膝 pitch 左右对(伸展时两者同为负, 差 0 = 同步发力)
    pairs = [
        (_col("hip_pitch_l_joint"), _col("hip_pitch_r_joint")),
        (_col("knee_pitch_l_joint"), _col("knee_pitch_r_joint")),
    ]
    asym = 0.0
    for il, ir in pairs:
        asym = asym + (joint_vel[:, il] - joint_vel[:, ir]).abs()
    asym = asym / len(pairs)  # 平均绝对伸展速度差 (E,)
    return active * torch.clamp(asym - threshold, min=0.0) * scale


def pre_jump_foot_motion_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    start_frame: int = 0,
    cutoff_frame: int = 100,
    vel_thresh: float = 0.08,
    scale: float = 1.0,
) -> torch.Tensor:
    """起跳前(站立+下蹲段)罚双脚踝水平移动, 治"正式起跳前的小碎步"。

    2026-08-15 V11 新增: 已检测参考 npz 起跳前(0-99帧)双脚 z 恒 0.033 贴地、y 平滑
    收拢(摆幅仅 0.044), 下蹲段脚踝线速度 ≤0.05 m/s —— **碎步不是 npz 原因, 是策略
    自学**(新奖励下的探索/重心微调)。罚起跳前脚踝速度超阈值:
    参考 0.05 → vel_thresh 不触发; 碎步(快速交替踏地/踮脚)速度远高 → 罚。
    **窗口 0~99 避开推蹬段(100-114, 参考脚踝发力 0.99 m/s 属正常起跳)**。
    2026-08-16 改**全速度(含 z)**: 治"踮脚/抬脚"型碎步(原只罚水平抓不到 z 向微动)。
    参考下蹲段脚速≤0.05 含 z 不变(脚踝 z 恒定); vel_thresh 收紧 0.03 近乎零容忍。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    active = ((command.time_steps >= start_frame) & (command.time_steps < cutoff_frame)).float()
    foot_vel = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids]  # (E,2,3)
    speed = torch.norm(foot_vel, dim=-1).mean(dim=-1)  # (E,) 双脚全速度均值(含 z)
    v = torch.clamp(speed - vel_thresh, min=0.0)
    return active * v * scale


def recovery_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg" | None = None,
    start_frame: int = 115,
    cutoff_frame: int = 183,
    scale: float = 1.0,
) -> torch.Tensor:
    """腾空+落地段奖励"失衡状态正在减小"(救回倾斜 + 重心复位, 主动回正)。

    2026-08-16 V11 新增, 两次修正:
    1. 原"角速度减小"→ 策略可停在倾斜位置(角速度0但仍歪)骗分。改奖励**倾斜角减小**。
    2. 用户建议"重心综合考虑"→ **失衡状态 = 倾斜角 + 重心偏移**(落地段):
       - 腾空段(115-132, 脚离地无支撑): 失衡 = 倾斜角 |right_z|
       - 落地段(133-183, 脚着地): 失衡 = 倾斜角 + base 投影到两脚踝中心距离
       prev_instability - cur_instability > 0 = 正在回正(倾斜回竖直/重心回脚上) → 奖励。
    落地段允许脚移动去接重心(无脚静止惩罚), 脚动 → 支撑面调整 → 重心回脚上 → 奖励。
    参考全程 right_z≈0 + 重心贴脚中心 → prev≈cur≈0 → 零冲突。
    env.extras 存前一步失衡量(新 episode 重置, episode_length_buf<=1)。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    active = ((command.time_steps >= start_frame) & (command.time_steps < cutoff_frame)).float()
    R = _matrix_from_quat(command.robot_anchor_quat_w)
    tilt = R[..., 2, 1].abs()  # (E,) 基座 roll 倾斜量
    instability = tilt
    if asset_cfg is not None:
        # 落地段(脚着地)加重心偏移: base 投影到两脚踝中心距离
        com_active = ((command.time_steps >= 133) & (command.time_steps < cutoff_frame)).float()
        base_xy = command.robot_anchor_pos_w[:, :2]
        foot_xy = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids][:, :, :2]  # (E,2,2)
        center = foot_xy.mean(dim=1)  # (E,2)
        com_d = torch.norm(base_xy - center, dim=-1)  # (E,)
        instability = tilt + com_active * com_d
    key = "_recovery_prev_instab"
    prev = env.extras.get(key)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = instability.detach().clone()
    fresh = env.episode_length_buf <= 1  # 新 episode 第一帧
    prev = torch.where(fresh, instability.detach(), prev)
    env.extras[key] = instability.detach().clone()
    recovering = torch.clamp(prev - instability, min=0.0)  # 正=失衡在减小(正在回正)
    return active * recovering * scale


def torso_roll_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    tolerance: float = 0.05,
    scale: float = 1.0,
    flight_threshold: float = 0.78,
    flight_cutoff_frame: int = 183,
) -> torch.Tensor:
    """起跳窗口+腾空+落地段罚躯干左右倾斜(roll): 基座 y 轴(左)世界 z 分量绝对值超容差。

    2026-08-14 V11 修复: 同 torso_backward_lean_penalty, 门控改纯 frame(115~183),
    不依赖 base 高度, 覆盖落地段 roll 姿态。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    in_jump = command.motion.jump_mask[command.time_steps]
    in_flight = (command.time_steps >= 115) & (command.time_steps < flight_cutoff_frame)
    active = (in_jump | in_flight).float()
    R = _matrix_from_quat(command.robot_anchor_quat_w)
    right_z = R[..., 2, 1]
    roll = torch.clamp(right_z.abs() - tolerance, min=0.0)
    return active * scale * roll


def landing_angular_vel_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    start_frame: int = 133,
    cutoff_frame: int = 183,
    scale: float = 1.0,
) -> torch.Tensor:
    """落地段罚基座 roll/yaw 角速度(世界系), 治落地侧倾/拧转不稳。

    2026-08-15 V11 新增: 参考落地段(133-183)基座 roll(x)/yaw(z) 角速度全程≈0
    (只有 pitch 有前倾缓冲 +0.93/-1.07, 因此不罚 pitch)。策略落地侧后仰=偏离参考,
    罚 roll/yaw 角速度**零冲突**。torso_roll/torso_backward_lean 是位置惩罚(姿势
    已经歪了才作用), 角速度惩罚在"刚开始转"就压制, 对症落地冲击下的不稳。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    active = ((command.time_steps >= start_frame) & (command.time_steps < cutoff_frame)).float()
    ang_vel = command.robot_anchor_ang_vel_w  # (E,3) world
    # 只罚 roll(x) + yaw(z); pitch 留给参考的前倾缓冲动作
    return active * (ang_vel[..., 0] ** 2 + ang_vel[..., 2] ** 2) * scale


def landing_balance_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    sigma: float = 0.1,
    scale: float = 1.0,
    start_frame: int = 133,
    cutoff_frame: int = 183,
) -> torch.Tensor:
    """落地段奖励基座投影贴近双脚踝中心(≈重心在支撑面内), 正向引导侧倾后救回重心。

    2026-08-15 V11 新增: 参考落地段(133-183) base 投影距脚中心 mean 0.046m/max 0.072m,
    侧向 y 全程≈0(双脚踝 y ±0.121, 支撑面半宽 0.12)。策略落地侧倾 = base 投影偏出脚间
    → 摔倒。exp 核: 参考 0.046m→0.81, 偏出 0.15m→0.105(强信号)。配合
    landing_angular_vel(速度惩罚) + torso_roll(位置惩罚), 这是"歪了怎么救回来"的
    正向平衡引导, 纯 body_pos_w + anchor_pos 可算, 2.2/2.3 兼容。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    active = ((command.time_steps >= start_frame) & (command.time_steps < cutoff_frame)).float()
    base_xy = command.robot_anchor_pos_w[:, :2]  # (E,2)
    foot_xy = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids][:, :, :2]  # (E,2,2)
    center = foot_xy.mean(dim=1)  # (E,2) 两脚踝中心
    d = torch.norm(base_xy - center, dim=-1)
    return active * torch.exp(-(d**2) / sigma**2) * scale


__all__ = [
    "track_dof_pos_exp",
    "track_root_ori_exp",
    "track_yaw_exp",
    "jump_height_bonus",
    "takeoff_vertical_vel",
    "takeoff_push_power",
    "premature_jump_penalty",
    "hip_spread_penalty",
    "torso_backward_lean_penalty",
    "arm_back_penalty",
    "arm_tracking_exp",
    "tuck_bonus",
    "flight_yaw_penalty",
    "waist_yaw_penalty",
    "waist_roll_penalty",
    "elbow_bend_penalty",
    "boundary_penalty",
    "feet_force_symmetry",
    "feet_contact_symmetry",
    "leg_symmetry_penalty",
    "takeoff_leg_symmetry_penalty",
    "pre_jump_foot_motion_penalty",
    "recovery_bonus",
    "torso_roll_penalty",
    "landing_angular_vel_penalty",
    "landing_balance_bonus",
]
