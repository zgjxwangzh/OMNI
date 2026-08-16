"""
Reference Tracking Environment for OMNI 29-DOF High Dynamic Motions
===================================================================

Isaac Lab DirectRLEnv 实现，用于训练高动态动作（跳高、翻箱等）。
观测空间与官方 SDK high_dynamic_policy.py 的 _build_obs() 完全对齐，
确保训练出的 ONNX 可直接部署。

观测结构（与推理侧一致）：
    obs = [command(58), anchor_ori(6), history(5 × 93)]
    command     = ref_joint_pos(29) + ref_joint_vel(29)
    anchor_ori  = base 旋转矩阵前两列展平 (6)
    history     = gravity(3) + ang_vel(3) + joint_pos_err(29)
                  + joint_vel(29) + last_action(29)  × history_length

动作空间：
    action(29) → target = default_pos + action * action_scale

关节顺序：
    仿真内部 = motor order（URDF/actuator 顺序）
    参考数据 = policy order（high_dynamic.yaml 的 policy_joint_names）
    通过 MOTOR_TO_POLICY_IDX / POLICY_TO_MOTOR_IDX 互转

依赖：
    Isaac Lab 2.2+, rsl_rl, PyTorch, numpy
"""

import os
import numpy as np
import torch
import gymnasium as gym
from dataclasses import dataclass
from typing import Literal

from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationCfg
from isaaclab.assets import Articulation
from isaaclab.utils import configclass

# ─────────────────────────────────────────────────────────────
# 关节顺序常量
# ─────────────────────────────────────────────────────────────
NUM_JOINTS = 29

# motor order（URDF/actuator 顺序，与 bvh_retarget.py 的 OMNI_JOINTS 一致）
# [0:6] 左腿  [6:12] 右腿  [12:15] 腰  [15:22] 左臂  [22:29] 右臂

# policy order（high_dynamic.yaml 的 policy_joint_names）
# 交替排列：L/R hip_pitch, waist_yaw, L/R hip_roll, waist_roll, ...

# motor idx → policy idx 的映射
MOTOR_TO_POLICY_IDX = np.array([
    0, 6, 12,   # hip_pitch L/R, waist_yaw
    1, 7, 13,   # hip_roll  L/R, waist_roll
    2, 8, 14,   # hip_yaw   L/R, waist_pitch
    3, 9,       # knee_pitch L/R
    15, 22,     # shoulder_pitch L/R
    4, 10,      # ankle_pitch L/R
    16, 23,     # shoulder_roll L/R
    5, 11,      # ankle_roll L/R
    17, 24,     # shoulder_yaw L/R
    18, 25,     # elbow_pitch L/R
    19, 26,     # elbow_yaw L/R
    20, 27,     # wrist_pitch L/R
    21, 28,     # wrist_roll L/R
], dtype=np.int64)

# policy idx → motor idx 的逆映射
POLICY_TO_MOTOR_IDX = np.argsort(MOTOR_TO_POLICY_IDX)

# motor order 的默认关节角度（与 high_dynamic.yaml 的 default_pos 一致）
# motor order: 左腿6 + 右腿6 + 腰3 + 左臂7 + 右臂7
DEFAULT_POS_MOTOR = np.array([
    -0.262, 0.0, 0.0, 0.524, -0.262, 0.0,     # 左腿
    -0.262, 0.0, 0.0, 0.524, -0.262, 0.0,     # 右腿
    0.0, 0.0, 0.0,                              # 腰
    0.300, 0.0, 0.0, -0.700, 0.0, 0.0, 0.0,   # 左臂
    0.300, 0.0, 0.0, -0.700, 0.0, 0.0, 0.0,   # 右臂
], dtype=np.float32)

# motor order 的 PD 增益（与 high_dynamic.yaml 的 kp/kd 一致）
KP_MOTOR = np.array([
    150, 150, 150, 150, 30, 30,
    150, 150, 150, 150, 30, 30,
    150, 150, 150,
    100, 100, 50, 50, 50, 20, 20,
    100, 100, 50, 50, 50, 20, 20,
], dtype=np.float32)

KD_MOTOR = np.array([
    5, 5, 5, 5, 3, 3,
    5, 5, 5, 5, 3, 3,
    5, 5, 5,
    2, 2, 2, 2, 2, 1, 1,
    2, 2, 2, 2, 2, 1, 1,
], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 环境配置
# ─────────────────────────────────────────────────────────────
@configclass
class ReferenceTrackingEnvCfg:
    """Reference tracking 训练环境配置"""

    # ── 仿真 ──
    sim: SimulationCfg = SimulationCfg(
        dt=0.002,
        gravity=(0.0, 0.0, -9.81),
    )
    decimation: int = 8           # 每 8 个仿真步做一次策略推理（≈20ms @2ms）
    sim_device: str = "cuda:0"
    rendering_dt: float = 0.01

    # ── 动作空间 ──
    action_space: int = NUM_JOINTS
    action_scale: float = 0.5     # 与 high_dynamic.yaml 的 action.scale 一致
    action_clip: float = 10.0     # 与 high_dynamic.yaml 的 action.clip 一致

    # ── 观测空间 ──
    history_length: int = 5       # 与 high_dynamic.yaml 的 observation.history_length 一致
    # num_obs = command(2*29) + anchor_ori(6) + history(5*(3+3+3*29))
    #         = 58 + 6 + 5*93 = 529

    # ── 奖励权重 ──
    rew_joint_pos: float = 1.0       # 关节位置跟踪
    rew_joint_vel: float = 0.5       # 关节速度跟踪
    rew_body_orientation: float = 0.5  # base 朝向跟踪
    rew_action_smoothness: float = -0.001  # 动作平滑性
    rew_energy: float = -0.0001      # 能耗惩罚
    rew_alive: float = 0.1           # 存活奖励

    # 跟踪误差的指数衰减系数
    tracking_alpha_pos: float = 10.0
    tracking_alpha_vel: float = 5.0
    tracking_alpha_ori: float = 5.0

    # ── 终止条件 ──
    episode_length_s: float = 20.0   # 最大回合时长（秒）
    termination_height: float = 0.3  # base 低于此高度终止
    termination_tilt: float = 0.7    # base 倾斜超过此值终止（cos(angle)）

    # ── 运动数据 ──
    motion_dir: str = "motion_data"  # NPZ 文件目录（policy order）
    motion_files: list[str] = []     # 指定文件列表；空则加载 motion_dir 下全部

    # ── 初始化噪声 ──
    init_pos_noise: float = 0.02     # 初始位置噪声（m）
    init_joint_noise: float = 0.02   # 初始关节角噪声（rad）

    # ── 域随机化（可选，sim2real 用）──
    randomize_pd_gains: bool = False
    pd_gain_noise: float = 0.1       # PD 增益随机噪声比例


# ─────────────────────────────────────────────────────────────
# 环境类
# ─────────────────────────────────────────────────────────────
class ReferenceTrackingEnv(DirectRLEnv):
    """
    Reference tracking 环境：机器人跟踪参考动作的关节角度/速度/base朝向。

    与官方 SDK high_dynamic_policy.py 的推理管线完全对齐：
    - obs 构建逻辑 = _build_obs()
    - action 映射 = default_pos + action * scale
    - 关节顺序转换 = MOTOR_TO_POLICY_IDX / POLICY_TO_MOTOR_IDX
    """

    cfg: ReferenceTrackingEnvCfg

    def __init__(self, cfg: ReferenceTrackingEnvCfg, **kwargs):
        # 先计算 obs/act space（super().__init__ 需要）
        self._num_obs = (
            2 * NUM_JOINTS                       # command: ref_pos + ref_vel
            + 6                                   # anchor_ori
            + cfg.history_length * (3 + 3 + 3 * NUM_JOINTS)  # history
        )
        self._num_actions = cfg.action_space

        # 调用父类初始化（会创建仿真、场景等）
        super().__init__(cfg, **kwargs)

        # ── 加载运动数据 ──
        self._load_motions()

        # ── 关节顺序索引（GPU tensor）──
        self._m2p = torch.tensor(MOTOR_TO_POLICY_IDX, device=self.device, dtype=torch.long)
        self._p2m = torch.tensor(POLICY_TO_MOTOR_IDX, device=self.device, dtype=torch.long)

        # ── 默认姿态 & PD 增益（GPU tensor，motor order）──
        self._default_pos = torch.tensor(DEFAULT_POS_MOTOR, device=self.device).repeat(self.num_envs, 1)
        self._kp = torch.tensor(KP_MOTOR, device=self.device).repeat(self.num_envs, 1)
        self._kd = torch.tensor(KD_MOTOR, device=self.device).repeat(self.num_envs, 1)
        self._action_scale = cfg.action_scale

        # ── 历史缓冲区 ──
        hl = cfg.history_length
        ne = self.num_envs
        self._gravity_hist = torch.zeros(ne, hl * 3, device=self.device)
        self._ang_vel_hist = torch.zeros(ne, hl * 3, device=self.device)
        self._joint_pos_hist = torch.zeros(ne, hl * NUM_JOINTS, device=self.device)
        self._joint_vel_hist = torch.zeros(ne, hl * NUM_JOINTS, device=self.device)
        self._action_hist = torch.zeros(ne, hl * NUM_JOINTS, device=self.device)

        # ── 参考数据索引 ──
        self._motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._motion_dt = 1.0 / 30.0  # 参考数据帧间隔（假设 30fps）

        # ── 上一动作（用于平滑性奖励）──
        self._last_action = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)

        # ── 最大步数 ──
        self._max_episode_steps = int(cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))

        print(f"[RefTrackEnv] initialized: {self.num_envs} envs, "
              f"num_obs={self._num_obs}, motions={len(self._motion_joint_pos)}")

    # ═══════════════════════════════════════════════════════════
    # 运动数据加载
    # ═══════════════════════════════════════════════════════════
    def _load_motions(self):
        """从 NPZ 文件加载参考动作数据（policy order）"""
        cfg = self.cfg
        motion_files = cfg.motion_files
        if not motion_files:
            # 自动扫描 motion_dir 下所有 *_highdynamic.npz
            motion_dir = cfg.motion_dir
            if not os.path.isabs(motion_dir):
                motion_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", motion_dir)
            motion_files = sorted([
                os.path.join(motion_dir, f)
                for f in os.listdir(motion_dir)
                if f.endswith("_highdynamic.npz")
            ])

        if not motion_files:
            raise FileNotFoundError(f"未找到运动数据文件，请检查 motion_dir: {cfg.motion_dir}")

        self._motion_joint_pos = []   # list of (T, 29) policy order
        self._motion_joint_vel = []   # list of (T, 29) policy order
        self._motion_body_quat = []   # list of (T, 4) base quaternion
        self._motion_names = []

        for mf in motion_files:
            data = np.load(mf)
            jp = torch.tensor(data["joint_pos"], dtype=torch.float32, device=self.device)
            jv = torch.tensor(data["joint_vel"], dtype=torch.float32, device=self.device)
            bq = torch.tensor(data["body_quat_w"][:, 0, :], dtype=torch.float32, device=self.device)
            # bq shape: (T, 4) — 取第一个 body（base_link）

            self._motion_joint_pos.append(jp)
            self._motion_joint_vel.append(jv)
            self._motion_body_quat.append(bq)
            self._motion_names.append(os.path.basename(mf))
            print(f"  加载动作: {os.path.basename(mf)}, {jp.shape[0]} 帧")

        # 为每个环境随机分配初始动作
        self._num_motions = len(self._motion_joint_pos)

    def _get_ref_at_frame(self, motion_idx, frame_idx):
        """获取指定动作在指定帧的参考数据（policy order）"""
        motion_idx = motion_idx.clamp(0, self._num_motions - 1)
        ref_pos = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        ref_vel = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        ref_quat = torch.zeros(self.num_envs, 4, device=self.device)
        ref_quat[:, 0] = 1.0  # 默认单位四元数

        for i in range(self._num_motions):
            mask = motion_idx == i
            if not mask.any():
                continue
            T = self._motion_joint_pos[i].shape[0]
            fi = frame_idx[mask].clamp(0, T - 1)
            ref_pos[mask] = self._motion_joint_pos[i][fi]
            ref_vel[mask] = self._motion_joint_vel[i][fi]
            ref_quat[mask] = self._motion_body_quat[i][fi]

        return ref_pos, ref_vel, ref_quat

    # ═══════════════════════════════════════════════════════════
    # 核心循环
    # ═══════════════════════════════════════════════════════════
    def _reset_idx(self, env_ids):
        """重置指定环境的状态"""
        if len(env_ids) == 0:
            return

        n = len(env_ids)

        # 随机选择动作
        new_motion_idx = torch.randint(0, self._num_motions, (n,), device=self.device)
        self._motion_idx[env_ids] = new_motion_idx
        self._frame_idx[env_ids] = 0

        # 获取第一帧参考数据
        ref_pos, ref_vel, ref_quat = self._get_ref_at_frame(new_motion_idx, self._frame_idx[env_ids])

        # 将 policy order 的参考转为 motor order
        ref_motor = ref_pos[:, self._p2m]

        # 设置 base 位姿（free joint: pos(3) + quat(4)）
        # 从参考数据获取 base 朝向
        base_pos = torch.zeros(n, 3, device=self.device)
        base_pos[:, 2] = 0.82  # 默认高度

        # 添加噪声
        base_pos[:, :2] += torch.randn(n, 2, device=self.device) * self.cfg.init_pos_noise
        joint_noise = torch.randn(n, NUM_JOINTS, device=self.device) * self.cfg.init_joint_noise

        # 设置机器人状态
        # 注意：实际 API 调用取决于 Isaac Lab 版本
        # 以下为 Isaac Lab 2.2+ 的典型模式
        robot: Articulation = self.scene["robot"]
        robot.write_root_pose_to_sim(
            torch.cat([base_pos, ref_quat], dim=-1),
            env_ids=env_ids,
        )
        robot.write_joint_pos_to_sim(
            ref_motor + joint_noise,
            env_ids=env_ids,
        )
        robot.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device),
            env_ids=env_ids,
        )
        robot.write_joint_vel_to_sim(
            torch.zeros(n, NUM_JOINTS, device=self.device),
            env_ids=env_ids,
        )

        # 重置历史缓冲区
        gravity_init = torch.zeros(n, 3, device=self.device)
        gravity_init[:, 2] = -1.0  # 初始重力方向（base 朝上）
        for j in range(self.cfg.history_length):
            self._gravity_hist[env_ids, j*3:(j+1)*3] = gravity_init
            self._ang_vel_hist[env_ids, j*3:(j+1)*3] = 0.0
            self._joint_pos_hist[env_ids, j*NUM_JOINTS:(j+1)*NUM_JOINTS] = 0.0
            self._joint_vel_hist[env_ids, j*NUM_JOINTS:(j+1)*NUM_JOINTS] = 0.0
            self._action_hist[env_ids, j*NUM_JOINTS:(j+1)*NUM_JOINTS] = 0.0

        self._last_action[env_ids] = 0.0

        # 重置步数计数器
        self._episode_length_buf[env_ids] = 0

    def _pre_physics_step(self, actions):
        """在物理步进前处理动作（每个 decimation 步调用一次）"""
        self._actions = actions.clone()

        # clip + scale + default → 目标关节角（motor order）
        action_clipped = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)
        self._target_pos = self._default_pos + action_clipped * self._action_scale

    def _apply_action(self):
        """将 PD 控制目标发送给电机"""
        robot: Articulation = self.scene["robot"]
        q = robot.data.joint_pos   # (num_envs, 29) motor order
        dq = robot.data.joint_vel  # (num_envs, 29) motor order

        # PD 控制: tau = kp * (target - q) - kd * dq
        q_error = self._target_pos - q
        tau = self._kp * q_error - self._kd * dq

        # 可选：PD 增益随机化（域随机化）
        if self.cfg.randomize_pd_gains and self.common_step_counter == 0:
            noise_kp = 1.0 + (torch.rand_like(self._kp) - 0.5) * 2 * self.cfg.pd_gain_noise
            noise_kd = 1.0 + (torch.rand_like(self._kd) - 0.5) * 2 * self.cfg.pd_gain_noise
            tau = (self._kp * noise_kp) * q_error - (self._kd * noise_kd) * dq

        robot.set_joint_effort_target(tau)

    def _post_physics_step(self):
        """物理步进后的更新：推进参考帧、更新历史、计算 obs/reward/done"""
        robot: Articulation = self.scene["robot"]

        # 推进参考帧索引
        self._frame_idx += 1
        # 检查是否到达动作末尾
        for i in range(self._num_motions):
            mask = self._motion_idx == i
            T = self._motion_joint_pos[i].shape[0]
            at_end = mask & (self._frame_idx >= T)
            if at_end.any():
                # 到达末尾的环境：保持在最后一帧
                self._frame_idx[at_end] = T - 1

        # 更新历史缓冲区
        self._update_history(robot)

        # 计算 obs / reward / done
        self._compute_observations()
        self._compute_rewards()
        self._compute_dones()

    def _update_history(self, robot: Articulation):
        """更新历史缓冲区（向左滑动，新值追加到末尾）"""
        hl = self.cfg.history_length
        ne = self.num_envs

        # 获取当前状态
        quat = robot.data.root_state_w[:, 3:7]  # (ne, 4) wxyz
        ang_vel = robot.data.root_ang_vel_b  # (ne, 3) body frame
        q = robot.data.joint_pos    # (ne, 29) motor order
        dq = robot.data.joint_vel   # (ne, 29) motor order

        # 计算重力方向（body frame）
        # gravity_b = R_world_to_body @ [0, 0, -1]
        # 用四元数旋转
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(ne, -1)
        gravity_b = self._rotate_vec_by_quat(gravity_w, quat)

        # 关节误差（相对于默认姿态）
        joint_pos_err = q - self._default_pos

        # 滑动历史（左移一位，新值追加到末尾）
        for hist_buf, new_val in [
            (self._gravity_hist, gravity_b),
            (self._ang_vel_hist, ang_vel),
            (self._joint_pos_hist, joint_pos_err),
            (self._joint_vel_hist, dq),
            (self._action_hist, self._last_action),
        ]:
            # 左移：把 [1:] 移到 [:-1]
            dim1 = hist_buf.shape[1]
            hist_buf[:, :-new_val.shape[1]] = hist_buf[:, new_val.shape[1]:].clone()
            hist_buf[:, -new_val.shape[1]:] = new_val

    # ═══════════════════════════════════════════════════════════
    # 观测计算（与 high_dynamic_policy.py 的 _build_obs 对齐）
    # ═══════════════════════════════════════════════════════════
    def _compute_observations(self):
        """
        构建观测向量，与推理侧 _build_obs() 完全一致：
        obs = [command(58), anchor_ori(6), history(5×93)]
        """
        robot: Articulation = self.scene["robot"]

        # ── 1. command = ref_joint_pos(29) + ref_joint_vel(29) ──
        ref_pos_policy, ref_vel_policy, ref_quat = self._get_ref_at_frame(
            self._motion_idx, self._frame_idx
        )
        # command 在推理侧是 policy order，训练时也用 policy order
        command = torch.cat([ref_pos_policy, ref_vel_policy], dim=-1)  # (ne, 58)

        # ── 2. anchor_ori = base 旋转矩阵前两列 (6) ──
        quat = robot.data.root_state_w[:, 3:7]  # wxyz
        rot_mat = self._quat_to_matrix(quat)     # (ne, 3, 3)
        anchor_ori = rot_mat[:, :, :2].reshape(self.num_envs, -1)  # (ne, 6)

        # ── 3. history ──
        history = torch.cat([
            self._gravity_hist,     # (ne, hl*3)
            self._ang_vel_hist,     # (ne, hl*3)
            self._joint_pos_hist,   # (ne, hl*29)
            self._joint_vel_hist,   # (ne, hl*29)
            self._action_hist,      # (ne, hl*29)
        ], dim=-1)  # (ne, hl * (3+3+3*29))

        # ── 拼装 ──
        self.obs_buf["policy"] = torch.cat([command, anchor_ori, history], dim=-1)

    # ═══════════════════════════════════════════════════════════
    # 奖励计算
    # ═══════════════════════════════════════════════════════════
    def _compute_rewards(self):
        """
        Reference tracking 奖励函数：
        - 关节位置跟踪：exp(-alpha * mse(pos_error))
        - 关节速度跟踪：exp(-alpha * mse(vel_error))
        - base 朝向跟踪：exp(-alpha * ori_error)
        - 动作平滑性：-||a_t - a_{t-1}||^2
        - 能耗惩罚：-||tau||^2
        - 存活奖励：常数
        """
        robot: Articulation = self.scene["robot"]
        cfg = self.cfg

        # 获取当前状态（motor order）
        q = robot.data.joint_pos     # (ne, 29)
        dq = robot.data.joint_vel    # (ne, 29)
        quat = robot.data.root_state_w[:, 3:7]

        # 获取参考数据（policy order → motor order）
        ref_pos_policy, ref_vel_policy, ref_quat = self._get_ref_at_frame(
            self._motion_idx, self._frame_idx
        )
        ref_pos_motor = ref_pos_policy[:, self._p2m]
        ref_vel_motor = ref_vel_policy[:, self._p2m]

        # ── 关节位置跟踪 ──
        pos_error = torch.mean((q - ref_pos_motor) ** 2, dim=-1)  # (ne,)
        rew_pos = torch.exp(-cfg.tracking_alpha_pos * pos_error)

        # ── 关节速度跟踪 ──
        vel_error = torch.mean((dq - ref_vel_motor) ** 2, dim=-1)
        rew_vel = torch.exp(-cfg.tracking_alpha_vel * vel_error)

        # ── base 朝向跟踪 ──
        # 用四元数内积衡量朝向差异：|dot(q1, q2)| 越接近 1 越好
        ori_error = 1.0 - torch.abs(torch.sum(quat * ref_quat, dim=-1))  # (ne,)
        rew_ori = torch.exp(-cfg.tracking_alpha_ori * ori_error)

        # ── 动作平滑性 ──
        rew_smooth = -torch.sum((self._actions - self._last_action) ** 2, dim=-1)

        # ── 能耗惩罚 ──
        tau = self._kp * (self._target_pos - q) - self._kd * dq
        rew_energy = -torch.sum(tau ** 2, dim=-1)

        # ── 存活奖励 ──
        rew_alive = torch.ones(self.num_envs, device=self.device) * cfg.rew_alive

        # ── 加权求和 ──
        self.rew_buf = (
            cfg.rew_joint_pos * rew_pos
            + cfg.rew_joint_vel * rew_vel
            + cfg.rew_body_orientation * rew_ori
            + cfg.rew_action_smoothness * rew_smooth
            + cfg.rew_energy * rew_energy
            + rew_alive
        )

        # 保存当前动作用于下一步平滑性计算
        self._last_action = self._actions.clone()

        # ── 记录奖励分量（用于 TensorBoard）──
        self.extras["rew_pos"] = rew_pos.mean()
        self.extras["rew_vel"] = rew_vel.mean()
        self.extras["rew_ori"] = rew_ori.mean()
        self.extras["pos_error"] = pos_error.mean()

    # ═══════════════════════════════════════════════════════════
    # 终止条件
    # ═══════════════════════════════════════════════════════════
    def _compute_dones(self):
        """计算终止条件"""
        robot: Articulation = self.scene["robot"]
        cfg = self.cfg

        base_height = robot.data.root_pos_w[:, 2]  # (ne,)
        quat = robot.data.root_state_w[:, 3:7]

        # 1. 摔倒：base 高度过低
        fell = base_height < cfg.termination_height

        # 2. 倾斜过大：base 朝向的 z 轴分量太小
        rot_mat = self._quat_to_matrix(quat)
        tilt = rot_mat[:, 2, 2]  # z 轴 z 分量 = cos(tilt_angle)
        tilted = tilt < cfg.termination_tilt

        # 3. 超时
        timeout = self._episode_length_buf >= self._max_episode_steps

        # 合并
        self.reset_terminated = fell | tilted
        self.reset_time_outs = timeout

        self.reset_buf = self.reset_terminated | self.reset_time_outs

    # ═══════════════════════════════════════════════════════════
    # 工具函数
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def _quat_to_matrix(q):
        """四元数 (w,x,y,z) → 3×3 旋转矩阵"""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
            2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
            2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)
        return R

    @staticmethod
    def _rotate_vec_by_quat(v, q):
        """用四元数 q (w,x,y,z) 旋转向量 v"""
        # v' = q * (0,v) * q_conj
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # 用旋转矩阵更高效
        R = ReferenceTrackingEnv._quat_to_matrix(q)
        return torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)
