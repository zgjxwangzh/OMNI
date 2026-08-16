"""跳高专用的运动命令: 每回合从参考运动第 0 帧(站立)开始。

xMimic 默认在参考运动随机相位开始(DeepMimic 式), 适合循环的行走/奔跑。
但跳高是**离散动作**: 随机相位会导致大量回合从空中/深蹲瞬间开始、直接摔倒,
终止率极高(冒烟测试 anchor_pos 终止率 ~0.7)。强制从站立帧开始, 让每个回合
都是"完整一跳": 站立 -> 深蹲 -> 起跳 -> 腾空 -> 落地。

保留父类的 pose/velocity/joint 抖动, 让 512 个并行环境仍有多样性。
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


class JumpMotionCommand(MotionCommand):
    """运动命令: 起始相位固定为站立帧 0。

    2026-08-13 V11: 在 __init__ 里为 self.motion 补 V7 需要的 jump_mask / first_jump_frame
    (V7 的 takeoff_vertical_vel / premature_jump_penalty / torso_backward_lean_penalty /
    torso_roll_penalty / flight_yaw_penalty 依赖它们)。用 V7 同款膝角推断
    knee>1.2 | kvel<-0.8, 并按落地帧 133 截断(排除落地恢复段的误触发)。
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._compute_jump_mask()

    def _compute_jump_mask(self):
        """从参考关节角推断 jump_mask / first_jump_frame, 挂到 self.motion。"""
        # BFS 运行序: knee_pitch_l/r 在列 9/10(与 LEG_TRACK_COLS/TUCK_TRACK_COLS 一致)
        motion = self.motion
        knee = (motion.joint_pos[:, 9] + motion.joint_pos[:, 10]) / 2  # (T,)
        # 膝角速度(数值差分)
        kvel = torch.zeros_like(knee)
        kvel[1:] = (knee[1:] - knee[:-1]) * float(motion.fps)
        kvel[0] = kvel[1]
        # V7 同款: 膝屈>1.2 或 膝角速度<-0.8(快速下蹲蓄力)
        jump_mask = (knee > 1.2) | (kvel < -0.8)
        # 落地截断: jump_high 参考腾空 115~132, 133 起落地; 落地缓冲伸直段会误触发
        landing_frame = 133
        jump_mask[landing_frame:] = False
        first_jump_frame = int(jump_mask.long().argmax()) if jump_mask.any() else 0
        motion.jump_mask = jump_mask
        motion.first_jump_frame = first_jump_frame

    def _resample_command(self, env_ids):
        # 强制起始相位 = 0 (站立帧), 其余逻辑与父类一致
        phase = torch.zeros(len(env_ids), device=self.device)
        self.time_steps[env_ids] = (phase * (self.motion.time_step_total - 1)).long()

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
