# Copyright (c) 2026, The jump_high project developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Omni 29-DOF high-jump training task."""

import gymnasium as gym

from . import agents  # noqa: F401
from .omni_jump_env_cfg import OmniJumpEnvCfg

##
# Register Gym environment.
##

gym.register(
    id="Omni-Jump-v0",
    entry_point="jump_env.omni_jump_env:OmniJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.omni_jump_env_cfg:OmniJumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniJumpPPORunnerCfg",
    },
)
