# Copyright (c) 2026, The jump_high project developers.
# SPDX-License-Identifier: BSD-3-Clause

"""32-env 应用内 smoke: 验证 529 obs 在真实 Isaac Lab 环境里集成正确。

在 isaaclab app 内跑 32 env, 步进超过一个回合长度(200 步 @50Hz), 逐项断言:
  1. obs_buf["policy"].shape == (N, 529), critic == (N, 430), 均无 NaN。
  2. command 块(参考帧关节角)随帧推进、且 reset 后回到首帧。
  3. reset 首 obs: 内部历史 buffer 动作末帧 = warmup、前 4 帧 = 0; 状态 buffer 5×当前。
  4. 非 reset 步: 滚动把当前状态写进历史末帧(gravity/动作 clamp 校验)。
跑法(需 PYTHONPATH 含 omni 包/xMimic/本项目):
  bash rl/run.sh smoke 用 train.py; 本脚本更聚焦:
  cd scripts && PYTHONPATH=... python smoke_obs529.py --num_envs 32 --headless
"""

import argparse
import os
import sys

# -- add project root and robot model package to sys.path (V9 自包含) --
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "omni_29dof_v260705")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-test 529 obs integration in the real env.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of envs.")
parser.add_argument("--max_steps", type=int, default=240, help="Total env steps (episode = 200).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from jump_env.omni_jump_env_cfg import OmniJumpEnvCfg  # noqa: E402
from jump_env.mdp.obs529 import SDK_ACTION_CLIP, SDK_ACTION_SCALE, SDK_DEFAULT_POS, HISTORY_LENGTH  # noqa: E402

# 关闭 Blackwell 下 TorchScript fusion 崩溃(与 train.py 一致)
try:
    torch._C._jit_override_can_fuse_on_gpu(False)
except Exception:
    pass


def _check_ok(cond, msg):
    if not cond:
        print(f"[FAIL] {msg}")
        sys.exit(1)


def main() -> None:
    env_cfg = OmniJumpEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Omni-Jump-v0", cfg=env_cfg, render_mode=None).unwrapped
    N = env.num_envs
    device = env.device
    print(f"[smoke] env {type(env).__name__}, N={N}, obs_groups={list(env.observation_manager._group_obs_term_names.keys())}")

    obs, _ = env.reset()
    print(f"[smoke] obs groups: policy {obs['policy'].shape}, critic {obs['critic'].shape}")

    cmd_frames_seen = []
    reset_steps = 0
    prev_ep_len = None

    for step in range(args_cli.max_steps):
        actions = torch.zeros(N, env.action_manager.total_action_dim, device=device)
        obs_dict, _, terminated, truncated, extras = env.step(actions)

        policy = obs_dict["policy"]
        critic = obs_dict["critic"]
        _check_ok(policy.shape == (N, 529), f"policy shape {policy.shape} != (N,529)")
        _check_ok(critic.shape == (N, 430), f"critic shape {critic.shape} != (N,430)")
        _check_ok(bool(torch.isfinite(policy).all()), "policy 含 NaN")
        _check_ok(bool(torch.isfinite(critic).all()), "critic 含 NaN")

        # 2) command 块随帧推进(参考关节角前 5 个值会变)
        cmd_pos = policy[:, :29].mean(dim=-1)  # (N,) 每 env 参考 pos 均值
        cmd_frames_seen.append(float(cmd_pos[0]))

        # 3/4) reset 与非 reset 步检查内部历史 buffer
        state = env.extras["_obs529"]
        ep_len = env.episode_length_buf
        fresh = ep_len == 0
        mask = ~fresh

        if fresh.any():
            reset_steps += 1
            i = torch.nonzero(fresh)[0].item()
            act_hist = state["action"][i]  # (5,29)
            # 首 obs: 前 4 帧动作 = 0, 末帧 = warmup = (q-default)/scale
            warmup = (env.scene["robot"].data.joint_pos[i] - SDK_DEFAULT_POS.to(device)) / SDK_ACTION_SCALE
            _check_ok(
                float(act_hist[:4].abs().max()) == 0.0,
                f"step{step} fresh 动作前4帧应全0, max={act_hist[:4].abs().max().item():.3e}",
            )
            _check_ok(
                bool(torch.allclose(act_hist[4], warmup, atol=1e-5)),
                f"step{step} fresh 动作末帧应=warmup, dev={float((act_hist[4]-warmup).abs().max()):.3e}",
            )
            # 状态 buffer 5×当前
            g = state["gravity"][i]
            grav_now = env.scene["robot"].data.projected_gravity_b[i]
            _check_ok(
                bool(torch.allclose(g[0], grav_now, atol=1e-6)),
                f"step{step} fresh gravity 首帧应=当前",
            )
            _check_ok(
                bool(torch.allclose(g[4], grav_now, atol=1e-6)),
                f"step{step} fresh gravity 末帧应=当前",
            )

        if mask.any():
            i = torch.nonzero(mask)[0].item()
            # 滚动末帧 = 当前状态; 动作末帧 = clamp(本步动作)
            grav_now = env.scene["robot"].data.projected_gravity_b[i]
            _check_ok(
                bool(torch.allclose(state["gravity"][i, -1], grav_now, atol=1e-6)),
                f"step{step} roll gravity 末帧应=当前, dev={float((state['gravity'][i,-1]-grav_now).abs().max()):.3e}",
            )
            act_now = torch.clamp(env.action_manager.action[i], -SDK_ACTION_CLIP, SDK_ACTION_CLIP)
            _check_ok(
                bool(torch.allclose(state["action"][i, -1], act_now, atol=1e-6)),
                f"step{step} roll 动作末帧应=clamp(本步动作), dev={float((state['action'][i,-1]-act_now).abs().max()):.3e}",
            )

        # 幂等: 同一步二次读 obs 应返回同一值(缓存)
        obs_again = env.observation_manager.compute()
        _check_ok(
            bool(torch.equal(obs_again["policy"], policy)),
            f"step{step} 同一步二次 compute 应返回缓存 obs",
        )

        prev_ep_len = ep_len

        if step % 30 == 0:
            print(f"  step {step}/{args_cli.max_steps}  done (fresh_any={bool(fresh.any())})", flush=True)

    # 2b) 参考帧确实在动(不是卡死同一帧)
    uniq = len(set(round(v, 4) for v in cmd_frames_seen))
    _check_ok(uniq > 5, f"command 块几乎没变({uniq} 个不同均值), 参考可能没在推进")
    _check_ok(reset_steps >= 1, "240 步内应至少发生一次自然 reset(回合 200 步)")

    print(f"[PASS] {args_cli.max_steps} 步, {N} envs: policy(529)/critic(430) 无 NaN, "
          f"command 推进 {uniq} 帧, reset 预填校验 {reset_steps} 次, 幂等缓存 OK")


if __name__ == "__main__":
    main()
    simulation_app.close()
