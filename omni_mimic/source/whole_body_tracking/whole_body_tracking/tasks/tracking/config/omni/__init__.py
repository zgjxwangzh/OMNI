import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-Omni-v0",
    entry_point="whole_body_tracking.envs:MyBaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Omni-Delayed-v0",
    entry_point="whole_body_tracking.envs:MyBaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniDelayedFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Omni-Hist-v0",
    entry_point="whole_body_tracking.envs:MyBaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniHistFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Omni-Hist-Delayed-v0",
    entry_point="whole_body_tracking.envs:MyBaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniHistDelayedFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0",
    entry_point="whole_body_tracking.envs:MyBaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniDelayedDCMotorHistFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniFlatPPORunnerCfg",
    },
)

# Box (box scene)
gym.register(
    id="Tracking-Box-Omni-Hist-v0",
    entry_point="whole_body_tracking.envs:MyBaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniBoxHistFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Box-Omni-Hist-Delayed-DCMotor-v0",
    entry_point="whole_body_tracking.envs:MyBaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniBoxDelayedDCMotorHistFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniFlatPPORunnerCfg",
    },
)

#  =============================== Play ===============================
gym.register(
    id="Tracking-Flat-Omni-Play",
    entry_point="whole_body_tracking.envs:MyBaseRLEnv",
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniFlatPlayEnvCfg,
        "rsl_rl_cfg_entry_point":f"{agents.__name__}.rsl_rl_ppo_cfg:OmniFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Omni-Hist-Play",
    entry_point="whole_body_tracking.envs:MyBaseRLEnv",
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniHistFlatPlayEnvCfg,
        "rsl_rl_cfg_entry_point":f"{agents.__name__}.rsl_rl_ppo_cfg:OmniFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Omni-Hist-Delayed-DCMotor-Play",
    entry_point="whole_body_tracking.envs:MyBaseRLEnv",
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniDelayedDCMotorHistFlatPlayEnvCfg,
        "rsl_rl_cfg_entry_point":f"{agents.__name__}.rsl_rl_ppo_cfg:OmniFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Box-Omni-Hist-Delayed-DCMotor-Play",
    entry_point="whole_body_tracking.envs:MyBaseRLEnv",
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniBoxDelayedDCMotorHistFlatPlayEnvCfg,
        "rsl_rl_cfg_entry_point":f"{agents.__name__}.rsl_rl_ppo_cfg:OmniFlatPPORunnerCfg",
    },
)

