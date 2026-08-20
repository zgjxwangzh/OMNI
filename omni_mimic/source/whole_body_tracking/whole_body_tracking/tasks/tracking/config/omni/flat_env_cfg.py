"""
flat_env_cfg.py - omni_29dof_nohead_noshoe

配置说明:
  - 可控关节: 29 个 (全部活动关节, 同 G1)
  - 锁定关节 0 个 (shoe_pitch, head 在 URDF 中已被合并, 不参与控制)
"""

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import TerminationTermCfg as DoneTerm

# TODO: update to 55p
from whole_body_tracking.robots.omni_29dof_nohead_noshoe import OMNI_ACTION_SCALE, OMNI_CYLINDER_CFG, OMNI_DELAYED_CFG

# from whole_body_tracking.robots.omni_29dof_nohead_noshoe_dcmotor_58p import OMNI_DCMOTOR_ACTION_SCALE, OMNI_DCMOTOR_CFG
from whole_body_tracking.robots.omni_29dof_nohead_noshoe_dcmotor import OMNI_DCMOTOR_ACTION_SCALE, OMNI_DCMOTOR_CFG

from whole_body_tracking.tasks.tracking.config.omni.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_omni_cfg import TrackingEnvCfg
from whole_body_tracking.tasks.tracking.tracking_env_omni_cfg import TrackingHistEnvCfg
from whole_body_tracking.tasks.tracking.tracking_env_omni_cfg import MySceneCfgWithBox

import whole_body_tracking.tasks.tracking.mdp as mdp

# omni 29 actuated joints, for underactuated system with passive joints
# usage: self.commands.motion.controlled_joint_names = CONTROLLED_JOINT_NAMES , None by default
CONTROLLED_JOINT_NAMES = [
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",

    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",

    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",

    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",

    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]


@configclass
class OmniFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = OMNI_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = OMNI_ACTION_SCALE
        
class OmniHistFlatEnvCfg(TrackingHistEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = OMNI_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = OMNI_ACTION_SCALE

@configclass
class OmniDelayedFlatEnvCfg(OmniFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = OMNI_DELAYED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class OmniHistDelayedFlatEnvCfg(OmniHistFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = OMNI_DELAYED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class OmniDelayedDCMotorHistFlatEnvCfg(TrackingHistEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = OMNI_DCMOTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = OMNI_DCMOTOR_ACTION_SCALE


# ----------------------------------- Box (box scene) configurations --------------------------------------
BOX_SIZE = (0.4, 0.8, 0.8)          # (长, 宽, 高) 米
ROBOT_REL_BOX_XY = (0.0, 0.0)       # 机器人 root 相对于 box 中心的 xy 偏移 (米)

def _apply_box_scene(cfg, box_size=BOX_SIZE, robot_rel_box_xy=ROBOT_REL_BOX_XY):
    """Apply box scene overrides to an env cfg (call inside __post_init__)."""
    cfg.scene.box.spawn.size = box_size
    box_x = -robot_rel_box_xy[0]
    box_y = -robot_rel_box_xy[1]
    box_z = box_size[2] / 2.0
    cfg.scene.box.init_state.pos = (box_x, box_y, box_z)
    cfg.commands.motion.pose_range = {
        "x": (-0.03, 0.03),
        "y": (-0.03, 0.03),
        "z": (-0.005, 0.005),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.1, 0.1),
    }

@configclass
class OmniBoxHistFlatEnvCfg(OmniHistFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene = MySceneCfgWithBox(num_envs=4096, env_spacing=3.5)
        self.scene.robot = OMNI_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _apply_box_scene(self)

@configclass
class OmniBoxDelayedDCMotorHistFlatEnvCfg(OmniDelayedDCMotorHistFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene = MySceneCfgWithBox(num_envs=4096, env_spacing=3.5)
        self.scene.robot = OMNI_DCMOTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _apply_box_scene(self)

# -----------------------------------------------------------------------------------------
#                                     Play Env Configs 
# -----------------------------------------------------------------------------------------
_ZERO_6DOF = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
              "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}

def _apply_play_overrides(cfg):
    """Disable domain-rand and set zero-range commands for play/eval (call inside __post_init__)."""
    cfg.episode_length_s = 100000.0
    cfg.terminations.anchor_pos = None
    cfg.terminations.anchor_ori = None
    cfg.terminations.ee_body_pos = None
    cfg.terminations.illegal_contact = None
    cfg.terminations.base_ang_vel_exceed = None
    cfg.events.physics_material = None
    cfg.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)
    cfg.events.base_com = None
    cfg.events.push_robot = None
    cfg.events.add_base_mass = None
    cfg.events.actuator_gains = None
    cfg.events.scale_link_mass = None
    cfg.events.random_joint_params = None
    cfg.events.arm_com = None
    cfg.commands.motion.pose_range = dict(_ZERO_6DOF)
    cfg.commands.motion.velocity_range = dict(_ZERO_6DOF)
    cfg.commands.motion.joint_position_range = (0.0, 0.0)
    cfg.commands.motion.play_mode = True
    cfg.observations.policy.enable_corruption = False

@configclass
class OmniFlatPlayEnvCfg(OmniFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)

@configclass
class OmniHistFlatPlayEnvCfg(OmniHistFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)

@configclass
class OmniBoxHistFlatPlayEnvCfg(OmniBoxHistFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)

@configclass
class OmniDelayedDCMotorHistFlatPlayEnvCfg(OmniDelayedDCMotorHistFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)
@configclass
class OmniBoxDelayedDCMotorHistFlatPlayEnvCfg(OmniBoxDelayedDCMotorHistFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)

