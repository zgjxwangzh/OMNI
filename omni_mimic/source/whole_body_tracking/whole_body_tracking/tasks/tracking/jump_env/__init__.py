"""Gym registration for Omni-Jump-OmniMimic-v0 (移植自室友 liuzq 的跳高训练方案)。

Import 本包即自动注册 gym 环境:
    from whole_body_tracking.tasks.tracking import jump_env  # noqa: F401
    env = gym.make("Omni-Jump-OmniMimic-v0")
"""

import os
import sys

# ---------------------------------------------------------------------------
# sys.path 设置: 确保子模块可导入
# 1) omni_mimic source/ → whole_body_tracking.tasks.tracking.* 可导入
# 2) 项目根 → omni_29dof_v260705.robots.* 可导入(机器人模型)
# ---------------------------------------------------------------------------
_pkg_dir = os.path.dirname(os.path.abspath(__file__))           # .../tracking/
_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(_pkg_dir)))  # .../omni_mimic/source/
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(_pkg_dir)))))               # 项目根目录

if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gymnasium as gym

from . import agents  # noqa: F401
from .omni_jump_env_cfg import OmniJumpEnvCfg

# ---------------------------------------------------------------------------
# Register Gym environment
# ---------------------------------------------------------------------------
gym.register(
    id="Omni-Jump-OmniMimic-v0",
    entry_point="whole_body_tracking.tasks.tracking.omni_jump_env:OmniJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.omni_jump_env_cfg:OmniJumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:OmniJumpPPORunnerCfg",
    },
)
