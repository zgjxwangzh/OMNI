# Copyright (c) 2025-2026, The Omni Lab Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# 训练 Omni 29-DOF 跳高策略（RSL-RL PPO）。
# 用法示例：
#   python scripts/train.py --task Isaac-Jump-Flat-Omni29dof-V7-v0 --headless --num_envs 4096
#   python scripts/train.py --headless --num_envs 4096 --max_iterations 10000
#
# 说明：本脚本把工程根目录与机器人模型包（omni_29dof_v260705）加入 sys.path，
#       因此从任意目录执行均可。日志写入 logs/rsl_rl/<experiment_name>/。

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

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL for the Omni 29-DOF high-jump task.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Omni-Jump-v0",
    help="Name of the task.",
)
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment and agent (-1 for random).")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--motion_file", type=str, default=None, help="Override the reference motion npz file.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
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

"""Check for supported RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")
# rsl-rl-lib 版本必须与 Isaac Lab 匹配(见 README §2 / setup_environment.sh):
#   Isaac Lab 2.2.x → 2.3.3 ; Isaac Lab 2.3+/main → 4.x/5.x
# 这里只提示, 不强制退出(2.2 与 main 的 rsl-rl 版本跨度很大)。
print(f"[INFO] rsl-rl-lib version: {installed_version}")
if version.parse(installed_version) >= version.parse("4.0.0"):
    print("[INFO] rsl-rl-lib >= 4.0(Isaac Lab 2.3+/main 接口), 训练走新接口路径。")
else:
    print("[INFO] rsl-rl-lib < 4.0(Isaac Lab 2.2 接口), 训练走旧接口路径。")

"""Rest everything follows."""

import logging
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# import the task package: this registers the gym environments (must be after AppLauncher)
import jump_env  # noqa: F401
# RSL-RL 兼容层：统一处理 Isaac Lab main / 2.3.x 的 policy -> actor/critic 迁移
import rsl_rl_compat  # noqa: E402

# logger
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _checkpoint_has_empty_optimizer(ckpt_path: str) -> bool:
    """判断 checkpoint 是否是 convert_rsl5_to_233.py 转换版(optimizer 为空)。"""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        opt = ckpt.get("optimizer_state_dict", None)
        if opt is None:
            return False
        # 转换版 optimizer = {"state": {}, "param_groups": []}
        return isinstance(opt, dict) and opt.get("param_groups") == [] and opt.get("state") == {}
    except Exception:
        return False


def main():
    # -- load configurations from the gym registry
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # -- override motion file if given
    if args_cli.motion_file is not None:
        print(f"[INFO] Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # -- override configurations with CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # 兼容迁移（policy -> actor/critic 等）在创建 runner 时由 rsl_rl_compat.build_runner_cfg 统一完成

    # set the environment seed and device
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )
    # multi-gpu training configuration
    # note: torchrun 会为每个进程设置 LOCAL_RANK / RANK / WORLD_SIZE 环境变量
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # -- logging directory
    # note: 多卡时各进程使用相同 log_dir，但 RSL-RL 只有 rank 0 写日志/保存 checkpoint
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # -- resolve resume checkpoint before creating a new log dir
    # 2026-08-18: 绕过 get_checkpoint_path 的路径匹配 bug, 直接构造路径
    resume_path = None
    if agent_cfg.resume:
        _lr = agent_cfg.load_run
        _ck = agent_cfg.load_checkpoint or "model.pt"
        if os.path.isabs(_lr):
            # 绝对路径: 直接用
            _run_dir = _lr
        elif os.path.sep in _lr:
            # 含分隔符(如 "logs/rsl_rl/xxx"): 相对 CWD 解析
            _run_dir = os.path.abspath(_lr)
        else:
            # 纯目录名(如 "2026-08-17_01-47-39"): 在 log_root_path 下查找
            _run_dir = os.path.join(log_root_path, _lr)
        resume_path = os.path.join(_run_dir, _ck)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}\n"
                f"  load_run={_lr}, checkpoint={_ck}\n"
                f"  Hint: run `ls {_run_dir}` to verify available checkpoints."
            )
        print(f"[INFO]: Resolved resume path: {resume_path}")

    # -- create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # -- wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # -- create runner from rsl-rl
    runner_cfg = rsl_rl_compat.build_runner_cfg(agent_cfg, installed_version)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    if getattr(app_launcher, "local_rank", 0) == 0:
        runner.add_git_repo_to_log(__file__)

    # load the checkpoint
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # 同格式续训(2.3.3 checkpoint, 如服务器自训的 model_6500): 直接 runner.load()。
        # 仅当 checkpoint 是跨版本转换版(convert_rsl5_to_233.py 生成, optimizer 为空)
        # 时, 才需要跳过 optimizer 加载。
        _need_skip_opt = _checkpoint_has_empty_optimizer(resume_path)
        if _need_skip_opt:
            # 转换版: 只加载 actor/critic, 跳过 optimizer
            try:
                import inspect
                _sig = inspect.signature(runner.load)
                if "load_cfg" in _sig.parameters:
                    runner.load(
                        resume_path,
                        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": True},
                    )
                else:
                    # 旧版 runner.load 无 load_cfg: 直接完整加载(转换版 optimizer 空,
                    # 但 2.3.3 同格式续训主要场景不涉及, 跨版本场景建议用 load_cfg 支持版本)
                    runner.load(resume_path)
            except (TypeError, KeyError):
                runner.load(resume_path)
        else:
            # 同格式 2.3.3 checkpoint: 完整加载(含 optimizer)
            runner.load(resume_path)

    # dump the configuration into log-directory (仅 rank 0，多卡共享同一目录)
    if getattr(app_launcher, "local_rank", 0) == 0:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # -- train
    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds", flush=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
