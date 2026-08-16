#!/usr/bin/env python
"""V11 跳高高度探针: headless 跑若干 episode, 记录 base 峰值高度。

判断 model 是否真的腾空、能跳多高。修复自 2026-08-14 之前卡死的版本:
- 简化循环, 用 env.step 直接驱动(不重复调 get_observations)
- 手动加载权重(兼容 2.3.3 model_state_dict → 5.x MLPModel)
用法:
  python scripts/eval_jump_height.py --checkpoint /home/liuziqi/model_31700.pt --episodes 3
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "omni_29dof_v260705")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="V11 jump height probe.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="Omni-Jump-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import importlib.metadata as metadata  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import jump_env  # noqa: E402,F401
import rsl_rl_compat  # noqa: E402


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cfg = rsl_rl_compat.build_runner_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)

    # 手动加载权重(2.3.3 model_state_dict → 5.x)
    ckpt = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    msd = ckpt["model_state_dict"]
    actor_sd = {f"mlp.{k[len('actor.'):]}": v for k, v in msd.items() if k.startswith("actor.")}
    critic_sd = {f"mlp.{k[len('critic.'):]}": v for k, v in msd.items() if k.startswith("critic.")}
    if "obs_norm_state_dict" in ckpt and hasattr(runner.alg.actor, "obs_normalizer"):
        runner.alg.actor.obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
    if "privileged_obs_norm_state_dict" in ckpt and hasattr(runner.alg.critic, "obs_normalizer"):
        runner.alg.critic.obs_normalizer.load_state_dict(ckpt["privileged_obs_norm_state_dict"])
    runner.alg.actor.load_state_dict(actor_sd, strict=False)
    runner.alg.critic.load_state_dict(critic_sd, strict=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    eps_len = int(env.unwrapped.max_episode_length)
    max_steps = args_cli.episodes * eps_len
    root_h_all = []
    obs, _ = env.reset()
    for i in range(max_steps):
        _raw = env.get_observations()
        obs_t = _raw[0] if isinstance(_raw, tuple) else _raw
        if isinstance(obs_t, dict):
            obs_t = obs_t["policy"]
        with torch.no_grad():
            actions = policy(obs_t)
        env.step(actions)
        root_h_all.extend(robot.data.root_pos_w[:, 2].cpu().tolist())

    root_h_all = np.array(root_h_all)
    apex = root_h_all.max()
    n_ep = max(1, len(root_h_all) // eps_len)
    ep_apex = [root_h_all[i * eps_len:(i + 1) * eps_len].max() for i in range(min(n_ep, args_cli.episodes))]
    print(f"\n=== V11 高度探针结果 ===")
    print(f"  episodes: {min(n_ep, args_cli.episodes)}, envs: {args_cli.num_envs}")
    print(f"  全程 base 峰值: {apex:.3f} m")
    print(f"  >0.80m 帧占比: {np.mean(root_h_all > 0.80):.1%}")
    print(f"  >0.82m 帧占比: {np.mean(root_h_all > 0.82):.1%}")
    print(f"  >0.90m 帧占比: {np.mean(root_h_all > 0.90):.1%}")
    print(f"  逐 episode 峰值: {[round(x,3) for x in ep_apex]}")
    simulation_app.close()


if __name__ == "__main__":
    main()
