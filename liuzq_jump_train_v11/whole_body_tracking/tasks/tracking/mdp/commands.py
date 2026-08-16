from __future__ import annotations

import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], joint_indexes: Sequence[int],
                 num_robot_joints: int, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        _joint_pos_npz = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        _joint_vel_npz = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)

        # Pad joint data from NPZ dimensions to full robot joint count
        # joint_indexes maps NPZ dims -> robot joint indices
        self._joint_indexes = torch.tensor(joint_indexes, dtype=torch.long, device=device)
        self._joint_pos = torch.zeros(_joint_pos_npz.shape[0], num_robot_joints, dtype=torch.float32, device=device)
        self._joint_vel = torch.zeros(_joint_vel_npz.shape[0], num_robot_joints, dtype=torch.float32, device=device)
        self._joint_pos[:, self._joint_indexes] = _joint_pos_npz
        self._joint_vel[:, self._joint_indexes] = _joint_vel_npz
        self._npz_joint_indexes = torch.arange(len(joint_indexes), dtype=torch.long, device=device)

        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self._joint_pos.shape[0]

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel

    @property
    def joint_pos_npz(self) -> torch.Tensor:
        """Return only the joints that have NPZ reference data."""
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel_npz(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # Build joint index mapping from NPZ joints to robot joints
        # NPZ has a subset of robot joints (e.g., 23 of 29)
        num_robot_joints = self.robot.num_joints
        npz_joint_names = self.cfg.joint_names
        if npz_joint_names is not None and len(npz_joint_names) > 0:
            self._npz_joint_indexes = torch.tensor(
                self.robot.find_joints(npz_joint_names, preserve_order=True)[0],
                dtype=torch.long, device=self.device
            )
        else:
            self._npz_joint_indexes = torch.arange(num_robot_joints, dtype=torch.long, device=self.device)

        self.motion = MotionLoader(
            self.cfg.motion_file, self.body_indexes, self._npz_joint_indexes, num_robot_joints, device=self.device
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Frames of the reference motion that fall inside a jump (base height above a
        # threshold, expanded by a margin). Used to bias episode starts toward the
        # jump phase via resample_jump_prob.
        self._jump_frames = self._compute_jump_frames()
        # Adaptive-curriculum state: EMA of per-segment airborne success drives the
        # jump-start probability (low success -> bias to jump starts; high -> random).
        self._jump_prob_eff = float(self.cfg.resample_jump_prob)
        self._success_ema = 0.5
        self._ep_max_z = torch.zeros(self.num_envs, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        # Include the raw (unaligned) reference root pose so policies can directly consume the
        # commanded root state in addition to the DOF targets.
        return torch.cat([self.joint_pos_npz, self.joint_vel_npz], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """Full robot joint positions (padded to match robot DOF count)."""
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Full robot joint velocities (padded to match robot DOF count)."""
        return self.motion.joint_vel[self.time_steps]

    @property
    def joint_pos_npz(self) -> torch.Tensor:
        """Only NPZ-tracked joint positions."""
        return self.motion.joint_pos_npz[self.time_steps]

    @property
    def joint_vel_npz(self) -> torch.Tensor:
        """Only NPZ-tracked joint velocities."""
        return self.motion.joint_vel_npz[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_joint_pos_npz(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self._npz_joint_indexes]

    @property
    def robot_joint_vel_npz(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self._npz_joint_indexes]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos_npz - self.robot_joint_pos_npz, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel_npz - self.robot_joint_vel_npz, dim=-1)

    def _compute_jump_frames(self) -> torch.Tensor:
        """Reference-motion frames whose anchor (base) height is above a threshold,
        expanded back by jump_approach_frames (covers the approach/crouch before the
        jump) and forward by jump_margin_frames. Used as jump-focused episode starts."""
        total = self.motion.time_step_total
        base_z = self.motion.body_pos_w[:, self.motion_anchor_body_index, 2]
        high = torch.where(base_z > self.cfg.jump_base_height_threshold)[0].cpu().numpy()
        margin = int(self.cfg.jump_margin_frames)
        approach = int(self.cfg.jump_approach_frames)
        frames = set()
        for f in high:
            lo, hi = max(0, int(f) - approach), min(total - 1, int(f) + margin)
            frames.update(range(lo, hi + 1))
        if not frames:
            return torch.arange(total, dtype=torch.long, device=self.device)
        return torch.tensor(sorted(frames), dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        # Adaptive curriculum: update success EMA from the just-finished segments and
        # move the jump-start probability toward random starts once jumps succeed often.
        if len(env_ids) > 0:
            success = self._ep_max_z[env_ids] > self.cfg.jump_success_height
            self._success_ema = 0.9 * self._success_ema + 0.1 * success.float().mean().item()
            self._jump_prob_eff = self.cfg.jump_prob_min + (self.cfg.jump_prob_max - self.cfg.jump_prob_min) * (
                1.0 - self._success_ema
            )
            self._ep_max_z[env_ids] = 0.0
        if self._jump_prob_eff > 0.0:
            # Bias a fraction of resets to start inside a jump phase.
            r = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
            jump_mask = r < self._jump_prob_eff
            n_jump = int(jump_mask.sum().item())
            jump_ids = env_ids[jump_mask]
            rest_ids = env_ids[~jump_mask]
            if n_jump > 0:
                idx = torch.randint(0, len(self._jump_frames), (n_jump,), device=self.device)
                self.time_steps[jump_ids] = self._jump_frames[idx]
            if len(rest_ids) > 0:
                phase = sample_uniform(0.0, 1.0, (len(rest_ids),), device=self.device)
                self.time_steps[rest_ids] = (phase * (self.motion.time_step_total - 1)).long()
        else:
            phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
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

    def _update_command(self):
        # Track each env's highest base height in the current motion segment.
        self._ep_max_z = torch.maximum(self._ep_max_z, self.robot_anchor_pos_w[:, 2])
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body: str = MISSING
    body_names: list[str] = MISSING
    joint_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    # Jump-phase curriculum: with resample_jump_prob probability, an episode reset
    # starts inside a jump phase (frames whose reference base height exceeds
    # jump_base_height_threshold, expanded by jump_margin_frames).
    resample_jump_prob: float = 0.0
    jump_base_height_threshold: float = 0.85
    jump_margin_frames: int = 25
    jump_approach_frames: int = 100  # frames before a jump peak to start episodes (covers approach/crouch)
    # Adaptive curriculum: a motion segment counts as success if base rose above
    # jump_success_height; the EMA drives jump-start probability between
    # [jump_prob_min, jump_prob_max].
    # v6: 0.9 -> 1.0 so fewer segments count as success -> EMA stays low ->
    # more episodes start at jump phase, focusing training on the apex.
    jump_success_height: float = 1.0
    jump_prob_min: float = 0.4
    jump_prob_max: float = 0.9

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
