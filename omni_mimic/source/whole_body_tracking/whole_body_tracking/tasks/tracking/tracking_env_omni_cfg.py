from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
# from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
# from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.envs import MyBaseRLEnvCfg

import whole_body_tracking.tasks.tracking.mdp as mdp

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

PUSH_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=TerrainGeneratorCfg(
    #         size=(8.0, 8.0),
    #         border_width=20.0,
    #         num_rows=4,
    #         num_cols=10,
    #         horizontal_scale=0.1,
    #         vertical_scale=0.005,
    #         slope_threshold=0.75,
    #         use_cache=False,
    #         sub_terrains={
    #             "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
    #                 proportion=1.0,
    #                 noise_range=(0.01, 0.04),
    #                 noise_step=0.01,
    #                 border_width=0.25,
    #             ),
    #         },
    #         curriculum=True,
    #     ),
    
    #     max_init_terrain_level=5,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     debug_vis=False,
    # )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


@configclass
class MySceneCfgWithBox(MySceneCfg):
    """Scene with an additional static box (platform) for box training."""

    box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.3, 0.1),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        command=["joint_pos", "joint_vel"],
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False, # True
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        anchor_body_name = "base_link",
        body_names = [
            "base_link",
            "hip_roll_l_link",
            "knee_pitch_l_link",
            "ankle_roll_l_link",
            "hip_roll_r_link",
            "knee_pitch_r_link",
            "ankle_roll_r_link",
            "waist_pitch_link",
            "shoulder_roll_l_link",
            "elbow_pitch_l_link",
            "wrist_roll_l_link",
            "shoulder_roll_r_link",
            "elbow_pitch_r_link",
            "wrist_roll_r_link",
        ],
        start_from_begin_prob=0.05,
        start_from_default_prob=0.01,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))    # 1.5 for worse joints
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class ObservationsHistCfg:
    """Observation specifications with history for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=5, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,history_length=5,noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel,history_length=5,noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,history_length=5,noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action,history_length=5)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=5)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,history_length=5)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel,history_length=5)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,history_length=5)
        actions = ObsTerm(func=mdp.last_action,history_length=5)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.4),
            "dynamic_friction_range": (0.2, 1.2),
            "restitution_range": (0.0, 0.3),    # (0.0, 0.5)
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link"),
            "com_range": {"x": (-0.08, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link"),
            "mass_distribution_params": (-2.0, 4.0),
            "operation": "add",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    random_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.4, 1.6),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # maybe useless ?
    arm_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[
                "shoulder_roll_l_link", "shoulder_roll_r_link",
            ]),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": PUSH_VELOCITY_RANGE},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- Batch 6: 指数+线性结合，科学计算权重 --
    # 根因分析:
    #   penalty=10/20/30: 策略在 1.68 满足，曾到 1.35 但回升
    #   penalty=50/100: 策略"躺平"，放弃所有跟踪
    #
    # 科学计算:
    #   躺平(error=2.0): penalty = -W×0.138, 其他奖励≈0
    #   努力(error=1.0): penalty = -W×0.034, 其他奖励≈+2.8
    #   努力>躺平: W < 2.8/0.104 = 27
    #
    # 方案: exp(w=8, std=0.5) 提供初始梯度 + linear(w=15) 防止满足
    #   error=1.78→1.35: exp梯度1.2 + linear梯度0.78 = 2.0/step ✓
    #   不会躺平: W=15 < 27 ✓
    track_joint_pos = RewTerm(
        func=mdp.joint_pos_exp,
        weight=8.0,
        params={"command_name": "motion", "std": 0.5},
    )
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_l2_penalty,
        weight=15.0,
        params={"command_name": "motion"},
    )
    track_joint_vel = RewTerm(
        func=mdp.joint_vel_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 2.0},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1000.0)  # -200→-1000 防止策略通过摔倒作弊
    # -- 保留: 基座级跟踪（与关节级不冲突，不同层面）--
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    # -- Batch 1 移除: body 级跟踪被 joint 级替代 --
    # motion_body_pos = RewTerm(
    #     func=mdp.motion_relative_body_position_error_exp,
    #     weight=1.0,
    #     params={"command_name": "motion", "std": 0.3},
    # )
    # motion_body_ori = RewTerm(
    #     func=mdp.motion_relative_body_orientation_error_exp,
    #     weight=1.0,
    #     params={"command_name": "motion", "std": 0.4},
    # )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.5, # 1.0
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.5, # 1.0
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.2)  # -0.1→-0.2 温和抑制
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,  # -3.0→-5.0 适度加强
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # joint_torques_l2 = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     weight=-3e-5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    # )
    # contact_forces = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-0.003,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=[
    #                 "ankle_roll_l_link",
    #                 "ankle_roll_r_link", 
    #                 "shoe_pitch_l_link",
    #                 "shoe_pitch_r_link",
    #             ],
    #         ),
    #         "threshold": 500.0,
    #     },
    # )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.2,  # -0.1→-0.2 适度加强
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                    body_names=[
                        r"^(?!ankle_roll_l_link$)(?!ankle_roll_r_link$)(?!shoe_pitch_l_link$)(?!shoe_pitch_r_link$)(?!wrist_roll_l_link$)(?!wrist_roll_r_link$).+$"
                    ],
            ),
            "threshold": 1.0,
        },
    )
    # joint_pos_tracking_legs = RewTerm(
    #     func=mdp.joint_pos_exp,
    #     weight=0.3,
    #     params={
    #         "command_name": "motion",
    #         "std": 0.5,
    #         "body_names": [
    #             "hip_pitch_l_joint", "hip_pitch_r_joint",
    #             "knee_pitch_l_joint", "knee_pitch_r_joint",
    #             "ankle_pitch_l_joint", "ankle_pitch_r_joint",
    #         ],
    #     },
    # )
    # feet_force_sym = RewTerm(
    #     func=mdp.feet_force_symmetry,
    #     weight=-0.005,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["ankle_roll_l_link", "ankle_roll_r_link"],
    #         ),
    #         "force_threshold": 50.0,
    #     },
    # )
    # feet_contact_sym = RewTerm(
    #     func=mdp.feet_contact_symmetry,
    #     weight=-0.002,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["ankle_roll_l_link", "ankle_roll_r_link"],
    #         ),
    #         "force_threshold": 10.0,
    #     },
    # )
    
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.3},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.9},
    )
    # -- Batch 1 移除: 手腕位移大导致 75% 回合被杀，阻碍手臂摆动学习 --
    # ee_body_pos = DoneTerm(
    #     func=mdp.bad_motion_body_pos_z_only,
    #     params={
    #         "command_name": "motion",
    #         "threshold": 0.3,
    #         "body_names": [
    #             "ankle_roll_l_link",
    #             "ankle_roll_r_link",
    #             "wrist_roll_l_link",
    #             "wrist_roll_r_link",
    #         ],
    #     },
    # )
    # illegal_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "threshold": 1.0,
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=[".*waist_pitch.*"],
    #         ),
    #     },
    # )
    # base_ang_vel_exceed = DoneTerm(
    #     func=mdp.base_ang_vel_exceed,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 800 * 3.14159265 / 180.0},
    # )
    
    # tracking_complete = DoneTerm(
    #     func=mdp.tracking_complete,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "end_time": 1.0},
    #     time_out=True,
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_motion)
    pass

##
# Environment configuration
##


@configclass
class TrackingEnvCfg(MyBaseRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    clip_actions: float = 10.0

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

@configclass
class TrackingHistEnvCfg(TrackingEnvCfg):
    observations: ObservationsHistCfg = ObservationsHistCfg()
