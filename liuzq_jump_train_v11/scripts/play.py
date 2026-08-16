# Copyright (c) 2025-2026, The Omni Lab Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# 回放已训练的 Omni 29-DOF 跳高策略，并导出 policy.pt（JIT）/ policy.onnx。
# 用法示例：
#   python scripts/play.py --task Isaac-Jump-Flat-Omni29dof-V7-Play-v0 --headless --video --load_run <run_dir>
#   python scripts/play.py --headless --play_full_motion --load_run <run_dir>
#   （不加 --load_run 时自动加载 logs/rsl_rl/<experiment_name>/ 下最新的 checkpoint）

import argparse
import os
import sys

# -- add project root and robot model package to sys.path ----------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "omni_29dof_v260705")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -- arguments ----------------------------------------------------------------
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play an RL agent checkpoint with RSL-RL for the Omni high-jump task.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Omni-Jump-v0",
    help="Name of the task.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--motion_file", type=str, default=None, help="Override the reference motion npz file.")
parser.add_argument(
    "--play_full_motion",
    action="store_true",
    default=False,
    help="Start the reference motion at phase 0 and stop playback after one full trajectory.",
)
parser.add_argument(
    "--keep_running",
    action="store_true",
    default=False,
    help="Keep the simulation running after one full motion playback (used with --play_full_motion).",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# -- launch omniverse app ------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from packaging import version

from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
try:  # handle_deprecated_rsl_rl_checkpoint 仅存在于较新 isaaclab_rl(2.3+/main); Isaac Lab 2.2 没有
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_checkpoint as _handle_deprecated_rsl_rl_checkpoint
except ImportError:
    _handle_deprecated_rsl_rl_checkpoint = None

# import the task package: this registers the gym environments (must be after AppLauncher)
import jump_env  # noqa: F401
# RSL-RL 兼容层：统一处理 Isaac Lab main / 2.3.x 的 policy -> actor/critic 迁移
import rsl_rl_compat  # noqa: E402

installed_version = metadata.version("rsl-rl-lib")


def _restore_obs_norm_stats(ckpt_path: str, runner) -> None:
    """把旧版 rsl-rl(<5.0) checkpoint 里的观测归一化统计恢复到本地 5.x 模型上。

    handle_deprecated_rsl_rl_checkpoint 只转换策略权重(actor/critic mlp + std),
    不搬运 obs_norm_state_dict;而本地 rsl-rl 5.x 的 obs 归一化内嵌在每个
    MLPModel 里(obs_normalizer),统计是全新初始化的。这里补上,保证回放与训练时的
    观测分布一致(否则策略吃未归一化的观测,跳法会明显走样)。
    新格式 checkpoint / 无归一化统计时自动跳过。
    """
    if not os.path.isfile(ckpt_path):
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return
    alg = getattr(runner, "alg", None)
    if alg is None:
        return
    actor = getattr(alg, "actor", None)
    critic = getattr(alg, "critic", None)
    if "obs_norm_state_dict" in ckpt and actor is not None and hasattr(actor, "obs_normalizer"):
        actor.obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
        print("[INFO] 已恢复 actor 观测归一化统计")
    if "privileged_obs_norm_state_dict" in ckpt and critic is not None and hasattr(critic, "obs_normalizer"):
        critic.obs_normalizer.load_state_dict(ckpt["privileged_obs_norm_state_dict"])
        print("[INFO] 已恢复 critic 观测归一化统计")


def _prepare_full_motion_play(vec_env: RslRlVecEnvWrapper):
    """把参考动作定位到第 0 帧, 并让 episode 至少覆盖整个动作, 用于确定性回放。"""
    base_env = getattr(vec_env, "unwrapped", vec_env)
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None:
        return None, None
    try:
        motion_term = command_manager.get_term("motion")
    except KeyError:
        return None, None

    env_ids = torch.arange(motion_term.num_envs, device=motion_term.device, dtype=torch.long)
    motion_term.time_steps.zero_()  # 对齐到第 0 帧 (setter 同步 _ref_time)

    horizon_s = float(motion_term.motion.time_step_total) / motion_term.motion.fps
    if horizon_s > base_env.cfg.episode_length_s:
        base_env.cfg.episode_length_s = horizon_s + 0.5
        if hasattr(base_env, "episode_length_buf"):
            base_env.episode_length_buf.zero_()

    joint_pos = motion_term.joint_pos.clone()
    joint_vel = motion_term.joint_vel.clone()
    motion_term.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
    root_state = torch.cat(
        [motion_term.anchor_pos_w, motion_term.anchor_quat_w, torch.zeros(len(env_ids), 6, device=motion_term.device)],
        dim=-1,
    )
    motion_term.robot.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)
    return motion_term, int(motion_term.motion.time_step_total)


def main():
    # -- load configurations from the gym registry
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # 兼容迁移（policy -> actor/critic 等）在创建 runner 时由 rsl_rl_compat.build_runner_cfg 统一完成

    # -- override configurations with CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.motion_file is not None:
        print(f"[INFO] Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # -- specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # resolve checkpoint path (default: latest checkpoint of the latest run)
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(log_root_path, resume_path)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # -- create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # -- wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    motion_term = None
    motion_max_steps = None
    prev_motion_step = None
    play_env_id = 0
    if args_cli.play_full_motion:
        motion_term, motion_max_steps = _prepare_full_motion_play(env)
        if motion_term is not None:
            prev_motion_step = motion_term.time_steps[play_env_id].item()
            print(f"[INFO] play_full_motion: 从 phase 0 播放, 共 {motion_max_steps} 帧")

    # -- load checkpoint into runner
    runner_cfg = rsl_rl_compat.build_runner_cfg(agent_cfg, installed_version)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    _orig_ckpt_path = resume_path  # 转换前的原始 checkpoint(含旧版归一化统计)
    if _handle_deprecated_rsl_rl_checkpoint is not None:
        resume_path = _handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    # strict=False 仅对 rsl-rl>=4.0(其 load 接受 strict 参数, 且转换后 checkpoint 缺
    # obs_normalizer.* 键需跳过); rsl-rl 2.3.3 的 OnPolicyRunner.load 不接受 strict,
    # 且 2.3.3 无 obs_normalizer 键要跳过 → 直接严格加载。这里探测签名兼容两种。
    import inspect
    _load_sig = inspect.signature(runner.load)
    if "strict" in _load_sig.parameters:
        runner.load(resume_path, strict=False)
    else:
        runner.load(resume_path)
    # 恢复旧版 checkpoint 的观测归一化统计(rsl-rl 5.x 本地回放保真; 2.3.3 无 obs_normalizer 则 no-op)
    _restore_obs_norm_stats(_orig_ckpt_path, runner)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # -- export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(log_dir, "exported")
    print(f"[INFO] Exporting policy to: {export_model_dir}")
    if hasattr(runner, "export_policy_to_jit"):  # rsl-rl >= 5 的 runner 自带导出方法
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:  # rsl-rl 4.x 用模块函数导出
        from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

        policy_nn = runner.alg.actor_critic
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        else:
            normalizer = None
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    # 兼容: rsl-rl 2.x 的 wrapper get_observations() 返回 (obs, info) 元组; 5.x 返回 TensorDict
    _raw_obs = env.get_observations()
    obs = _raw_obs[0] if isinstance(_raw_obs, tuple) else _raw_obs
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length and not args_cli.keep_running:
                break
        if args_cli.play_full_motion and motion_term is not None:
            current_step = motion_term.time_steps[play_env_id].item()
            if current_step >= motion_max_steps - 1 and not args_cli.keep_running:
                print("[INFO] 完整动作播放结束")
                break
            prev_motion_step = current_step

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
