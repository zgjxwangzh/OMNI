"""Omni 29-DOF 原地跳高跟踪环境配置。

基于 xMimic whole_body_tracking 的 `TrackingEnvCfg`, 注入 Omni 机器人
(`OMNI_DCMOTOR_IDENTIFIED_CFG`), 参考运动 = `motion/jump_high_firstjump_50fps.npz`
(50fps/183 帧, 与部署 SDK HighDynamic 50Hz 同频)。

2026-08-11 重构(529 obs + 50Hz):
- 控制 50Hz(decimation=4, dt=0.005→200Hz 物理), 参考 50fps。
- 策略 obs = 529 维(SDK HighDynamic 布局, 5帧历史), 见 jump_env/mdp/obs529.py。
- 模型对齐 SDK: 脚踝质量 0.365kg、default 位姿 -0.262/0.524、action scale 0.5。
- enable_corruption=False(不加 obs 噪声)。critic(特权)430 维不动。


策略(阶段一): **纯模仿参考动作**。主要奖励 = xMimic 参考跟踪(motion_* 系列),
`JumpMotionCommand` 保证每回合从参考站立帧 0 开始、时刻 t 逐帧对齐参考帧 t,
即"时刻对应参考动作模拟"。不再叠加任何"结果型"跳高奖励 —— 之前叠加的
`jump_height`(高于参考瞬时高度)在深蹲段会奖励"站着不动", 和"跟着参考跳"
自相矛盾, 是本轮训练学不动的根因。阶段二(2026-08-09)补了三个跳高专项奖赏
`takeoff_completion_bonus` / `standing_too_long_penalty` / `airborne_leg_tuck`
(见 `__post_init__` 注释), 目标不是替代模仿, 而是把"站着不蹲也能刷低分"的
局部盆地挖掉、把腾空收腿(成绩关键)教出来。摔倒(躯干 < 0.25m)判死兜底:
不纵容摔倒, 但也不因动作小偏差过早砍掉回合(终止阈值放宽到 1.0m)。

物理: Isaac Sim 默认重力 (0,0,-9.81), 这里显式声明(用户要求训练引入物理规则、有重力)。
"""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

from jump_env.mdp.commands import JumpMotionCommand
from jump_env.mdp.obs529 import obs529
# V11: V7 终止函数(bad_anchor_ori/fell/out_of_bounds, jump_high vendored mdp 没有)
from jump_env.mdp.v7_terminations import bad_anchor_ori, fell, out_of_bounds
# V11: V7 奖励栈(26 项, 见 v7_rewards.py)。jump_high 原自定义奖励(jump_phase_rewards/
# jump_rewards)已全部移除, 换成 V7 算法。
from jump_env.mdp.v7_rewards import (
    arm_back_penalty,
    arm_tracking_exp,
    boundary_penalty,
    elbow_bend_penalty,
    feet_contact_symmetry,
    feet_force_symmetry,
    flight_yaw_penalty,
    hip_spread_penalty,
    jump_height_bonus,
    landing_angular_vel_penalty,
    landing_balance_bonus,
    leg_symmetry_penalty,
    premature_jump_penalty,
    pre_jump_foot_motion_penalty,
    recovery_bonus,
    takeoff_leg_symmetry_penalty,
    takeoff_vertical_vel,
    takeoff_push_power,
    torso_backward_lean_penalty,
    torso_roll_penalty,
    track_dof_pos_exp,
    track_root_ori_exp,
    track_yaw_exp,
    tuck_bonus,
    waist_roll_penalty,
    waist_yaw_penalty,
)
from robots.omni_29dof_nohead_noshoe_dcmotor_identified import (
    OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE,
    OMNI_DCMOTOR_IDENTIFIED_CFG,
)

# 参考运动: 第一跳参考, 50fps 版(部署 SDK HighDynamic 50Hz 同频, 隔帧抽取自 100fps 母本)
# 2026-08-12 V9: 支持环境变量 JUMP_HIGH_MOTION_FILE 覆盖(换 npz 不须改代码)。
_DEFAULT_MOTION_FILE = str(Path(__file__).resolve().parent.parent / "motion" / "jump_high_firstjump_50fps.npz")
MOTION_FILE = os.environ.get("JUMP_HIGH_MOTION_FILE", _DEFAULT_MOTION_FILE)

# 机器人 body 关节树序(实查自 Isaac Lab, 即 npz 索引顺序)
OMNI_BODY_NAMES = [
    "base_link",
    "hip_pitch_l_link",
    "hip_pitch_r_link",
    "waist_yaw_link",
    "hip_roll_l_link",
    "hip_roll_r_link",
    "waist_roll_link",
    "hip_yaw_l_link",
    "hip_yaw_r_link",
    "waist_pitch_link",
    "knee_pitch_l_link",
    "knee_pitch_r_link",
    "shoulder_pitch_l_link",
    "shoulder_pitch_r_link",
    "ankle_pitch_l_link",
    "ankle_pitch_r_link",
    "shoulder_roll_l_link",
    "shoulder_roll_r_link",
    "ankle_roll_l_link",
    "ankle_roll_r_link",
    "shoulder_yaw_l_link",
    "shoulder_yaw_r_link",
    "elbow_pitch_l_link",
    "elbow_pitch_r_link",
    "elbow_yaw_l_link",
    "elbow_yaw_r_link",
    "wrist_pitch_l_link",
    "wrist_pitch_r_link",
    "wrist_roll_l_link",
    "wrist_roll_r_link",
]

# 兜底终止: 躯干跌到该高度以下视为摔倒
MIN_ROOT_HEIGHT = 0.25


@configclass
class OmniJumpEnvCfg(TrackingEnvCfg):
    """原地跳高跟踪环境配置。"""

    def __post_init__(self):
        super().__post_init__()

        # --- 机器人: DelayedDCMotor 真实作动器, 受重力 ---
        self.scene.robot = OMNI_DCMOTOR_IDENTIFIED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE

        # --- 物理规则: 显式声明重力(用户要求) ---
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.scene.robot.spawn.rigid_props.disable_gravity = False

        # --- 50Hz 控制 + 200Hz 物理(对齐部署 SDK HighDynamic: run_interval 0.0025, decimation 8) ---
        # dt=0.005 不变, decimation 2→4 → 策略步 0.02s(50Hz), 参考 50fps 逐帧对齐。
        self.decimation = 4
        self.sim.render_interval = 4

        # --- 平地跳高: 撤掉 xMimic 默认的斜坡噪声生成地形 ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # --- 参考运动: jump_high_firstjump.npz(第一跳, 左右严格镜像) ---
        self.commands.motion.motion_file = MOTION_FILE
        self.commands.motion.anchor_body = "base_link"
        self.commands.motion.body_names = OMNI_BODY_NAMES
        # 2026-08-12 V9: joint_names 显式置空 —— 剪枝后 vendored 的 xMimic
        # MotionCommandCfg 默认 joint_names=dataclasses.MISSING, 其 __init__ 里
        # `if joint_names is not None and len(joint_names)>0` 会对 MISSING 调 len()
        # 崩溃; 显式 [] → 走 `arange(num_robot_joints)` 全关节路径, npz 29 列全用。
        self.commands.motion.joint_names = []
        # 服务器 headless 安全: 关掉命令 debug 可视化
        self.commands.motion.debug_vis = False
        # 跳高是离散动作: 每回合从站立帧 0 开始做完整一跳(而非随机相位)
        self.commands.motion.class_type = JumpMotionCommand

        # --- 回合长度: 50Hz 下一跳 183 步 + 落地余量 ---
        # 2026-08-14: 4.0→3.70(185 步)。参考动作 183 帧@50fps=3.66s 播完(第183步)后
        # _update_command 触发重采样回帧0, 原 4.0s(200步)重采样后剩17步→机器人跳第二次。
        # 3.70s(185步)重采样后只剩2步→来不及完整起跳, 一集一跳。
        self.episode_length_s = 3.70

        # --- 修复 DexEVT 特有硬编码: omni 没有 "pelvis" body ---
        self.events.base_com.params["asset_cfg"].body_names = "base_link"

        # --- 2026-08-12 V9 服务器安全 DR(镜像 V7 经验, 绕服务器 Isaac Lab 2.2.x
        #     reset 子集 write_*_to_sim shape mismatch bug) ---
        # xMimic TrackingEnvCfg 默认 7 项 EventCfg: randomize_rigid_body_mass_others
        # 是 mode="reset"(逐 episode), 服务器 2.2.x 在部分 env_ids 下写会崩 →
        # 改 startup(启动随机一次); push_robot(mode="interval") 对离散跳高还可能在
        # 起跳瞬间推一把, 直接禁用。其余 startup 项(randomize_joint_params /
        # physics_material / add_joint_default_pos / base_com)保持默认;
        # randomize_actuator_gains(mode="reset") 保持 —— V7 验证其走 set_* 函数参数
        # 形式, 不受子集写 bug 影响。
        if "randomize_rigid_body_mass_others" in self.events.__dict__:
            self.events.randomize_rigid_body_mass_others.mode = "startup"
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None

        # --- V11: V7 奖励栈(26 项, 唯一栈) ---
        # 移植自 my_omni_jump_train_v7/omni_jump_tasks_v7/jump_env_cfg.py 的 RewardsCfg。
        # 帧号/高度阈值已按 jump_high 参考(183帧/50fps, base 峰值 0.961m, 腾空 115~132)
        # 重标定。jump_high 原自定义奖励(motion_*/takeoff_completion/standing_*/airborne_*)
        # 全部移除。

        # -- 模仿跟踪
        self.rewards.track_dof_pos = RewTerm(
            func=track_dof_pos_exp, weight=3.0, params={"command_name": "motion", "std": 0.5}
        )
        self.rewards.track_root_ori = RewTerm(
            func=track_root_ori_exp, weight=2.0, params={"command_name": "motion", "std": 0.4}
        )
        # 专用 yaw 跟踪(V11 修正 wxyz 公式)
        self.rewards.track_yaw = RewTerm(
            func=track_yaw_exp,
            weight=2.5,
            params={"command_name": "motion", "std": 0.25, "scale": 1.0},
        )
        # 腰 yaw/roll 约束(治上半身拧转/侧倾)
        self.rewards.waist_yaw_penalty = RewTerm(
            func=waist_yaw_penalty,
            weight=-0.5,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"]),
                "scale": 1.0,
            },
        )
        self.rewards.waist_roll_penalty = RewTerm(
            func=waist_roll_penalty,
            weight=-1.5,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint"]),
                "scale": 1.0,
            },
        )
        # 胳膊跟踪(非饱和, 平均误差)
        self.rewards.arm_tracking = RewTerm(
            func=arm_tracking_exp,
            weight=2.0,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
                        "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
                        "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
                        "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
                    ],
                ),
                "std": 0.3,
            },
        )

        # -- 跳高塑造(物理量)
        # 腾空高度: base 峰值 0.961m, 阈值 0.85(覆盖 116~131 共 16 帧)
        self.rewards.jump_height_bonus = RewTerm(
            func=jump_height_bonus,
            # 2026-08-15: 增强腾空高度 weight 5→10 + 超线性(excess+excess², 函数内实现)。
            # max_excess 0.8 上限覆盖到 base 1.6m。目标 apex 1.05m(参考 0.961)。
            # 2026-08-16: 突破局部最优, 9->15 强制策略探索起跳
            # 2026-08-17: 80->60, 进一步降低跳高激励
            # 80/40 跑了 200 步后 leg_symmetry 从 -0.45 恶化到 -0.50, 说明仍不够低
            weight=60.0,
            params={"command_name": "motion", "threshold": 0.79, "scale": 1.0, "max_excess": 0.8},
        )
        # 起跳爆发速度(jump_mask 窗口)
        self.rewards.takeoff_vertical_vel = RewTerm(
            func=takeoff_vertical_vel,
            # 2026-08-15: max_vel 1.8→2.3(1.8=参考峰值, 达到就满奖, 无梯度冲更高;
            # 目标 apex 1.05m 需起跳速度 2.3 m/s, 打开超参考的冲高梯度)
            # 2026-08-16: 突破局部最优, 9->12 强化起跳爆发梯度
            # 2026-08-17: 40->30, 同步降低, 配合 jump_height_bonus 调整
            weight=30.0,
            params={"command_name": "motion", "vel_thresh": 0.3, "max_vel": 2.3},
        )
        # 2026-08-14 新增: 推蹬段奖励双腿快速伸展(膝/髋伸直), 教爆发起跳
        self.rewards.takeoff_push_power = RewTerm(
            func=takeoff_push_power,
            # 2026-08-15: max_ext_vel 6→4(参考推蹬6关节均值1.98 离满奖太远, 伸展更快没
            # 梯度; 降到4让更强伸展落进可及范围, 配合 max_vel 2.3 冲更高)
            # 2026-08-15: 加 _jump_gate(threshold 0.80), 堵死"下蹲段反复伸展刷分"→两次下蹲
            weight=3.0,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "hip_pitch_l_joint", "hip_pitch_r_joint",
                        "knee_pitch_l_joint", "knee_pitch_r_joint",
                        "ankle_pitch_l_joint", "ankle_pitch_r_joint",
                    ],
                ),
                "max_ext_vel": 4.0,
                "threshold": 0.80,
            },
        )
        # 提前起跳惩罚(first_jump_frame=57 之前)
        self.rewards.premature_jump_penalty = RewTerm(
            func=premature_jump_penalty,
            weight=-1.0,
            params={"command_name": "motion", "threshold": 0.95},
        )
        # 腾空收腿(膝屈 1.5 + 髋屈 -0.9, start_frame=115 腾空起点)
        self.rewards.tuck_bonus = RewTerm(
            func=tuck_bonus,
            # 2026-08-16: 突破局部最优, 5→8 强化腾空收腿激励
            weight=8.0,
            # 2026-08-14: airborne_threshold 0.90→0.82(用户选定; 站立 0.782<0.82 不误触,
            # 需策略腾空超过 0.82 才触发收腿奖励——配合 takeoff_vertical_vel 7.0 冲高)
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "knee_pitch_l_joint", "knee_pitch_r_joint",
                        "hip_pitch_l_joint", "hip_pitch_r_joint",
                    ],
                ),
                "airborne_threshold": 0.82,
                "start_frame": 115,
                "knee_target": 1.5,
                "hip_target": -0.9,
                "sigma": 0.4,
                "scale": 1.0,
            },
        )
        # 腾空段抑制向右拧转(世界 z 轴角速度)
        self.rewards.flight_yaw_penalty = RewTerm(
            func=flight_yaw_penalty,
            # 2026-08-16: 加强 -0.15→-0.3(89700 反馈腾空扭转; 参考腾空 yaw 角速度=0 零冲突)
            weight=-0.3,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot"),
                "airborne_threshold": 0.90,
                "scale": 1.0,
            },
        )
        # 失衡回稳奖励: 腾空(倾斜角)+落地(倾斜角+重心偏移)在减小就给奖, 主动救回失衡
        self.rewards.recovery_bonus = RewTerm(
            func=recovery_bonus,
            weight=0.3,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["ankle_roll_l_link", "ankle_roll_r_link"]
                ),
                "start_frame": 115,
                "cutoff_frame": 185,
                "scale": 1.0,
            },
        )

        # -- 姿态约束
        # 反肘: 全局允许起跳伸臂(60°-180°)
        self.rewards.elbow_bend_penalty = RewTerm(
            func=elbow_bend_penalty,
            weight=-1.5,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["elbow_pitch_l_joint", "elbow_pitch_r_joint"],
                ),
                "lo": 0.0,
                "hi": 2.09,
                "scale": 1.0,
            },
        )
        # 限制大腿外展(双腿叉开)
        self.rewards.leg_spread_penalty = RewTerm(
            func=hip_spread_penalty,
            weight=-8.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["hip_roll_l_joint", "hip_roll_r_joint"]),
                "threshold": 0.35,
                "scale": 1.0,
            },
        )
        # 躯干后仰(起跳窗口+腾空, 只罚后仰不冲突)
        self.rewards.torso_backward_lean_penalty = RewTerm(
            func=torso_backward_lean_penalty,
            # 2026-08-16: 加强 -8→-10 + tol 0.04→0.02(89700 反馈落地后有后仰;
            # 参考落地段 fwd_z≤0.002 前倾, 罚后仰(>0.02)仍不冲突)
            weight=-10.0,
            params={
                "command_name": "motion",
                "tolerance": 0.02,
                "scale": 1.0,
                "flight_threshold": 0.78,
                "flight_cutoff_frame": 185,
            },
        )
        # 起跳前(站立+下蹲段)罚双脚踝水平移动, 治"正式起跳前的小碎步"
        # (参考 npz 起跳前脚踝 z 恒 0.033 贴地/下蹲段脚速≤0.05, 碎步是策略自学的)
        self.rewards.pre_jump_foot_motion_penalty = RewTerm(
            func=pre_jump_foot_motion_penalty,
            # 2026-08-16: 加强 -1.0→-2.0 + vel_thresh 0.06→0.03(89700 反馈小碎步未完全消失,
            # 用户要求"起跳前脚一点都不能动"; 函数已改全速度含z治踮脚)
            weight=-2.0,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["ankle_roll_l_link", "ankle_roll_r_link"]
                ),
                "start_frame": 0,
                "cutoff_frame": 100,
                "vel_thresh": 0.03,
                "scale": 1.0,
            },
        )
        # 躯干左右倾斜(roll)
        self.rewards.torso_roll_penalty = RewTerm(
            func=torso_roll_penalty,
            # 2026-08-16: -4→-5(落地侧倾位置惩罚再加强)
            weight=-5.0,
            params={
                "command_name": "motion",
                "tolerance": 0.05,
                "scale": 1.0,
                # 2026-08-14: flight_threshold 0.90→0.78, cutoff 132→183(覆盖落地段)
                "flight_threshold": 0.78,
                "flight_cutoff_frame": 185,
            },
        )
        # 落地段罚基座 roll/yaw 角速度(速度型, 治落地冲击下的侧倾/拧转不稳)
        self.rewards.landing_angular_vel_penalty = RewTerm(
            func=landing_angular_vel_penalty,
            # 2026-08-16: 加强 -0.5→-0.8(70300 反馈落地稳定度还需增强)
            weight=-0.8,
            params={"command_name": "motion", "start_frame": 133, "cutoff_frame": 185, "scale": 1.0},
        )
        # 落地段奖励基座投影贴近双脚中心(重心在支撑面内), 正向引导侧倾后救回
        self.rewards.landing_balance_bonus = RewTerm(
            func=landing_balance_bonus,
            # 2026-08-16: 加强 +2→+3(70300 反馈落地稳定度还需增强)
            weight=3.0,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["ankle_roll_l_link", "ankle_roll_r_link"]
                ),
                "sigma": 0.1,
                "scale": 1.0,
                "start_frame": 133,
                "cutoff_frame": 185,
            },
        )
        # 腾空段惩罚肩后摆
        self.rewards.arm_back_penalty = RewTerm(
            func=arm_back_penalty,
            weight=-3.0,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["shoulder_pitch_l_joint", "shoulder_pitch_r_joint"],
                ),
                "airborne_threshold": 0.90,
                "shoulder_tol": 0.2,
                "scale": 1.0,
            },
        )

        # -- 左右对称(起跳/落地)
        self.rewards.feet_force_symmetry = RewTerm(
            func=feet_force_symmetry,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["ankle_roll_l_link", "ankle_roll_r_link"],
                ),
                "force_threshold": 50.0,
            },
        )
        self.rewards.feet_contact_symmetry = RewTerm(
            func=feet_contact_symmetry,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["ankle_roll_l_link", "ankle_roll_r_link"],
                ),
                "force_threshold": 10.0,
            },
        )
        self.rewards.leg_symmetry_penalty = RewTerm(
            func=leg_symmetry_penalty,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "hip_pitch_l_joint", "hip_pitch_r_joint",
                        "hip_roll_l_joint", "hip_roll_r_joint",
                        "knee_pitch_l_joint", "knee_pitch_r_joint",
                        "ankle_pitch_l_joint", "ankle_pitch_r_joint",
                        "ankle_roll_l_joint", "ankle_roll_r_joint",
                    ],
                ),
                "threshold": 0.1,
                "scale": 1.0,
            },
        )
        # 推蹬段罚左右腿发力不一致(髋/膝伸展速度差), 治腾空扭转(参考差 0 零冲突)
        self.rewards.takeoff_leg_symmetry_penalty = RewTerm(
            func=takeoff_leg_symmetry_penalty,
            weight=-2.0,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "hip_pitch_l_joint", "hip_pitch_r_joint",
                        "knee_pitch_l_joint", "knee_pitch_r_joint",
                    ],
                ),
                "start_frame": 90,
                "cutoff_frame": 115,
                "threshold": 0.1,
                "scale": 1.0,
            },
        )

        # -- 常规
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
        self.rewards.boundary_penalty = RewTerm(
            func=boundary_penalty,
            weight=-1.0,
            params={"command_name": "motion", "inner": 1.0, "scale": 1.0},
        )
        # 通用项: action_rate/torque/limits/contacts 由基类 RewardsCfg 提供(保留)

        # --- 2026-08-13 V11 修复: 移除基类非 V7 奖励项(唯一栈) ---
        # 基类 TrackingEnvCfg 自带 motion_*(body 跟踪模仿)/crouch_phase_height_match/
        # upright_penalty, 不是 V7 的 26 项, 且 motion_body_* 依赖 body 跟踪(V7 不跟踪
        # body, 会与 track_dof_pos 等矛盾)。全部置 None, 只保留 V7 26 项 + 通用项。
        # 通用项(action_rate_l2/joint_torque_l2/joint_limit/joint_vel_limit/
        # undesired_contacts)保留 —— 与 V7 一致。
        # 移除基类非 V7 项: motion_*(body 跟踪)/crouch_phase_height_match/upright_penalty。
        # ⚠️ 2026-08-14 修复: 之前误把 jump_height_bonus/takeoff_vertical_vel 也置 None
        # (它们是 V7 核心腾空激励, V7 注册在后已覆盖基类同名, 不应删) → 导致训练日志
        # 缺这两项、策略收不到起跳信号、不起跳。现在只删基类的 motion_*/crouch/upright。
        for _term in [
            "crouch_phase_height_match",  # 非 V7
            "upright_penalty",          # 非 V7
            "motion_global_anchor_pos",
            "motion_global_anchor_ori",
            "motion_body_pos",
            "motion_body_ori",
            "motion_body_lin_vel",
            "motion_body_ang_vel",
        ]:
            if hasattr(self.rewards, _term):
                setattr(self.rewards, _term, None)

        # --- V11: V7 终止(替换 jump_high 的 body 跟踪终止, 防 V9 回合被误杀) ---
        # V7 奖励不跟踪 body 位置, 依赖 body 跟踪的 ee_body_pos/anchor_pos 会误杀回合
        # → 移除, 换 V7 的 bad_anchor_ori / fell / out_of_bounds。
        # 基类已有 time_out。这里显式替换。
        self.terminations.bad_anchor_ori = DoneTerm(
            func=bad_anchor_ori,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 1.2},
        )
        self.terminations.fell = DoneTerm(
            func=fell,
            params={"command_name": "motion", "threshold": 0.25},
        )
        self.terminations.out_of_bounds = DoneTerm(
            func=out_of_bounds,
            params={"command_name": "motion", "half_size": 1.5},
        )
        # 移除基类自带/依赖 body 跟踪的终止(基类 TrackingEnvCfg 自带
        # anchor_pos/ee_body_pos/anchor_ori), 避免与 V7 的 bad_anchor_ori 重复判死
        # (V9 教训: anchor_ori 曾占 55% 终止, 重复会过度杀回合)。
        for _tname in ["anchor_pos", "ee_body_pos", "anchor_ori"]:
            if hasattr(self.terminations, _tname):
                setattr(self.terminations, _tname, None)
        # 保留 root_height 兜底(摔倒)
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": MIN_ROOT_HEIGHT}
        )

        # --- obs: 529 维(SDK HighDynamic 布局, 5帧历史), 替换掉 160 单帧训练 obs ---
        # 铁律: 训练 obs = 部署 obs(逐字节)。529 = command(58) + anchor_ori(6) +
        # 5帧×(gravity 3 + ang_vel 3 + joint_pos_rel 29 + joint_vel 29 + action 29),
        # 复刻 SDK high_dynamic_policy.py `_build_obs`(见 jump_env/mdp/obs529.py)。
        for _term_name in [
            "command",
            "motion_anchor_pos_b",
            "motion_anchor_ori_b",
            "base_lin_vel",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ]:
            setattr(self.observations.policy, _term_name, None)
        self.observations.policy.obs529 = ObsTerm(func=obs529)
        self.observations.policy.enable_corruption = False

        # --- 50Hz 下动作平滑惩罚减半(xMimic LOW_FREQ_SCALE=0.5 同款) ---
        self.rewards.action_rate_l2.weight *= 0.5
