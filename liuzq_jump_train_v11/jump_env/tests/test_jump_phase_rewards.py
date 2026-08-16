"""jump_phase_rewards.py 的纯 torch 单测(不依赖 omni)。

用假 env 对象注入所需数据:
  env.scene["robot"].data.root_pos_w          -> 机器人 base_link 世界高度 (N,3)
  env.command_manager.get_term("motion")       -> .time_steps (N,)  + .anchor_pos_w (N,3)
  env.scene.sensors["contact_forces"].data.net_forces_w_history -> (N, H, B, 3)

关键物理事实(来自参考 firstjump, 50fps 帧号折半):
  - 腾空窗口 115~132 帧; 循环长度 183
  - 下蹲段(15~100)参考 root 从 0.782 降到最低 0.435
  - 腾空段(115~132)参考 root 升到 0.785~0.962
  - 站立时踝/脚接触力 >190N, 离地后 ~0
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

MODULE_PATH = "/home/zyy/jump_high/jump_env/jump_phase_rewards.py"
spec = importlib.util.spec_from_file_location("jump_phase_rewards", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

REF_LEN = 183
T_TO, T_LAND = 115, 132
T_CROUCH, MARGIN = 15, 0.06

# ---- fake env 装配 ----
class FakeSensor:
    def __init__(self, forces_history):
        # forces_history: (N, H, B, 3)
        self.data = SimpleNamespace(net_forces_w_history=torch.tensor(forces_history, dtype=torch.float32))


class FakeCommand:
    def __init__(self, time_steps, anchor_pos_w, anchor_quat_w=None, robot_anchor_quat_w=None, body_pos_w=None):
        self.time_steps = torch.tensor(time_steps, dtype=torch.long)
        self.anchor_pos_w = torch.tensor(anchor_pos_w, dtype=torch.float32)
        # 四元数 (w,x,y,z); 缺省 = 单位四元数(朝上立正)
        self.anchor_quat_w = torch.tensor(anchor_quat_w, dtype=torch.float32) if anchor_quat_w is not None else \
            torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(time_steps), dtype=torch.float32)
        self.robot_anchor_quat_w = torch.tensor(robot_anchor_quat_w, dtype=torch.float32) if robot_anchor_quat_w is not None else \
            self.anchor_quat_w.clone()
        # 参考 body 世界位置 (N,30,3, OMNI_BODY_NAMES 序); 缺省 0 占位
        self.body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32) if body_pos_w is not None else \
            torch.zeros(len(time_steps), 30, 3)


class FakeRobot:
    def __init__(self, root_pos_w, body_pos_w=None, joint_vel=None, root_lin_vel_w=None, body_lin_vel_w=None):
        # body_pos_w: (N, B, 3) 可选; 缺省时假设每个 body 高度 = root 高度
        self.data = SimpleNamespace(root_pos_w=torch.tensor(root_pos_w, dtype=torch.float32))
        if body_pos_w is not None:
            self.data.body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32)
        # standing_fidget_penalty 需要脚 body 线速度(世界系)
        if body_lin_vel_w is not None:
            self.data.body_lin_vel_w = torch.tensor(body_lin_vel_w, dtype=torch.float32)
        # 旧版 fidget(关节速度口径)遗留参数, 保留兼容
        if joint_vel is not None:
            self.data.joint_vel = torch.tensor(joint_vel, dtype=torch.float32)
        if root_lin_vel_w is not None:
            self.data.root_lin_vel_w = torch.tensor(root_lin_vel_w, dtype=torch.float32)


class FakeScene:
    def __init__(self, robot, sensors):
        self._robot = robot
        self._sensors = sensors

    def __getitem__(self, key):
        assert key == "robot"
        return self._robot

    @property
    def sensors(self):
        return self._sensors


class FakeEnv:
    def __init__(self, command, robot, sensor):
        self.command_manager = SimpleNamespace(get_term=lambda name: command)
        self.scene = FakeScene(robot, {"contact_forces": sensor})


class FakeSensorCfg:
    name = "contact_forces"
    body_ids = [0, 1]  # ankle_roll_l/r 已解析后的索引


def make_env(root_h, time_steps, ref_root_h, foot_forces, body_min_h=None):
    """foot_forces: 每脚每历史帧的力幅值, 形状 (N, H, 2)。
    body_min_h: (N,) 每个 env 的最低 link 高度; 缺省时用 root 高度。"""
    N = len(root_h)
    H = len(foot_forces[0])
    # 合成三维力: 只留 z 分量, 幅值=给定值
    forces = torch.zeros(N, H, 2, 3)
    for n in range(N):
        for h in range(H):
            for b in range(2):
                forces[n, h, b, 2] = foot_forces[n][h][b]
    # 构造 body_pos_w: 3 个 body, 最低 link 高度 = body_min_h (或 root 高度)
    if body_min_h is None:
        body_pos_w = None
    else:
        body_pos_w = []
        for n in range(N):
            bm = body_min_h[n]
            body_pos_w.append([[0, 0, bm], [0, 0, bm + 0.3], [0, 0, bm + 0.5]])
    env = FakeEnv(
        command=FakeCommand(time_steps, [[0, 0, r] for r in ref_root_h]),
        robot=FakeRobot([[0, 0, r] for r in root_h], body_pos_w=body_pos_w),
        sensor=FakeSensor(forces),
    )
    return env


def close(a, b, tol=1e-5):
    assert torch.allclose(a, b, atol=tol), f"expect {b} got {a}"


# ---------- takeoff_completion_bonus ----------
def test_takeoff_airborne_in_window():
    # 窗口内(帧 120)双脚全程离地 -> 1
    env = make_env([0.9], [120], [0.95], [[[0, 0], [0, 0], [0, 0]]])
    close(mod.takeoff_completion_bonus(env, FakeSensorCfg), torch.tensor([1.0]))


def test_takeoff_contact_in_window():
    # 窗口内但一只脚接触(如落地) -> 0
    env = make_env([0.9], [120], [0.95], [[[0, 0], [190, 0], [0, 0]]])
    close(mod.takeoff_completion_bonus(env, FakeSensorCfg), torch.tensor([0.0]))


def test_takeoff_airborne_outside_window():
    # 帧 10(站立段)离地 -> 0; 窗口外跳不算
    env = make_env([0.9], [10], [0.95], [[[0, 0], [0, 0], [0, 0]]])
    close(mod.takeoff_completion_bonus(env, FakeSensorCfg), torch.tensor([0.0]))


def test_takeoff_loop_wraps():
    # 帧 298 = 183+115, 取模后仍在窗口内 -> 1
    env = make_env([0.9], [298], [0.95], [[[0, 0], [0, 0], [0, 0]]])
    close(mod.takeoff_completion_bonus(env, FakeSensorCfg), torch.tensor([1.0]))


def test_takeoff_loops_no_double_count_off_window():
    # 帧 367 = 2×183+1, 取模 1(第二圈站立段) -> 0
    env = make_env([0.9], [184], [0.95], [[[0, 0], [0, 0], [0, 0]]])
    close(mod.takeoff_completion_bonus(env, FakeSensorCfg), torch.tensor([0.0]))


# ---------- standing_too_long_penalty ----------
def test_standing_good_tracking():
    # 机器人完美贴着参考 -> 不罚
    env = make_env([0.60], [75], [0.60], [[[190, 190], [190, 190], [190, 190]]])
    close(mod.standing_too_long_penalty(env), torch.tensor([0.0]))


def test_standing_lag_within_margin():
    # 高出参考 0.05m < margin 0.06 -> 不罚
    env = make_env([0.65], [75], [0.60], [[[190, 190], [190, 190], [190, 190]]])
    close(mod.standing_too_long_penalty(env), torch.tensor([0.0]))


def test_standing_too_long_linear():
    # 深蹲段站着不动(root 0.782, 参考 0.435) -> 线性罚 0.782-0.435-0.06=0.287
    env = make_env([0.782], [75], [0.435], [[[190, 190], [190, 190], [190, 190]]])
    close(mod.standing_too_long_penalty(env), torch.tensor([0.287]))


def test_standing_no_penalty_in_stand_phase():
    # 站立段(帧 10)站得比参考高也不罚(还没到该蹲的时候)
    env = make_env([0.79], [10], [0.782], [[[190, 190], [190, 190], [190, 190]]])
    close(mod.standing_too_long_penalty(env), torch.tensor([0.0]))


def test_standing_no_penalty_in_air():
    # 腾空段参考比机器人高 -> 0(这段时间交给 takeoff bonus)
    env = make_env([0.80], [120], [0.95], [[[0, 0], [0, 0], [0, 0]]])
    close(mod.standing_too_long_penalty(env), torch.tensor([0.0]))


def test_standing_wraps_loop():
    # 帧 75+183=258 取模回 75, 仍在深蹲 -> 罚
    env = make_env([0.782], [258], [0.435], [[[190, 190], [190, 190], [190, 190]]])
    close(mod.standing_too_long_penalty(env), torch.tensor([0.287]))


# ---------- airborne_leg_tuck ----------
def test_tuck_in_window_airborne():
    # 窗口内(帧 240)离地 + 最低 link 0.40m -> 奖励 0.40
    env = make_env([0.9], [120], [0.95], [[[0, 0], [0, 0], [0, 0]]], body_min_h=[0.40])
    close(mod.airborne_leg_tuck(env, FakeSensorCfg), torch.tensor([0.40]))


def test_tuck_no_contact():
    # 窗口内但一只脚还接触 -> 0(没收腿状态无奖)
    env = make_env([0.9], [120], [0.95], [[[0, 0], [190, 0], [0, 0]]], body_min_h=[0.40])
    close(mod.airborne_leg_tuck(env, FakeSensorCfg), torch.tensor([0.0]))


def test_tuck_outside_window():
    # 帧 20(站立段)离地但不在窗口 -> 0
    env = make_env([0.9], [10], [0.95], [[[0, 0], [0, 0], [0, 0]]], body_min_h=[0.40])
    close(mod.airborne_leg_tuck(env, FakeSensorCfg), torch.tensor([0.0]))


def test_tuck_capped_at_reference():
    # 最低 link 0.50 > cap 0.462 -> 封顶 0.462(不鼓励偏离参考的极致收腿)
    env = make_env([0.9], [120], [0.95], [[[0, 0], [0, 0], [0, 0]]], body_min_h=[0.50])
    close(mod.airborne_leg_tuck(env, FakeSensorCfg), torch.tensor([0.462]))


def test_tuck_loop_wraps():
    # 帧 298 = 183+115, 取模后仍在窗口内 -> 正常发奖
    env = make_env([0.9], [298], [0.95], [[[0, 0], [0, 0], [0, 0]]], body_min_h=[0.30])
    close(mod.airborne_leg_tuck(env, FakeSensorCfg), torch.tensor([0.30]))


def test_tuck_no_tuck_low_link():
    # 离地但最低 link 还很低(0.06, 没收腿) -> 奖励小, 说明差在收腿
    env = make_env([0.9], [120], [0.95], [[[0, 0], [0, 0], [0, 0]]], body_min_h=[0.06])
    close(mod.airborne_leg_tuck(env, FakeSensorCfg), torch.tensor([0.06]))


# ---------- arm_deviation_penalty ----------
ARM_COLS = [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]  # 运行时序


def make_jp_env(time_steps, ref_arm, act_arm):
    """ref_arm / act_arm: (N, 14) 手臂关节值; 其余关节都置 0。"""
    ref_arm = torch.tensor(ref_arm, dtype=torch.float32)
    act_arm = torch.tensor(act_arm, dtype=torch.float32)
    N = len(time_steps)
    jp_ref = torch.zeros(N, 29)
    jp_act = torch.zeros(N, 29)
    jp_ref[:, ARM_COLS] = ref_arm
    jp_act[:, ARM_COLS] = act_arm
    cmd = SimpleNamespace(
        time_steps=torch.tensor(time_steps, dtype=torch.long),
        joint_pos=jp_ref,
        anchor_pos_w=torch.zeros(N, 3),
    )
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=jp_act, root_pos_w=torch.zeros(N, 3)))
    return FakeEnv(command=cmd, robot=robot, sensor=FakeSensor(torch.zeros(N, 3, 2, 3)))


def test_arm_perfect_tracking_zero():
    # 手臂完全贴合参考 -> 0
    ref = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2, 1.3, -1.4]
    env = make_jp_env([75], [ref], [ref])
    close(mod.arm_deviation_penalty(env), torch.tensor([0.0]))


def test_arm_constant_deviation_linear():
    # 14 个手臂关节全部 +0.3 -> 平均绝对偏差 = 0.3(线性, 不饱和)
    ref = [0.0] * 14
    act = [0.3] * 14
    env = make_jp_env([75], [ref], [act])
    close(mod.arm_deviation_penalty(env), torch.tensor([0.3]))


def test_arm_large_deviation_still_linear():
    # 偏差 2.0 rad(模仿项早就饱和了) -> 仍线性罚 2.0
    ref = [0.0] * 14
    act = [2.0] * 14
    env = make_jp_env([75], [ref], [act])
    close(mod.arm_deviation_penalty(env), torch.tensor([2.0]))


def test_arm_only_arm_columns_count():
    # 只有腿/躯干偏差(非手臂列), 手臂贴合参考 -> 0(证明只罚手臂列)
    ref_arm = [0.0] * 14
    act_arm = [0.0] * 14
    env = make_jp_env([75], [ref_arm], [act_arm])
    env.scene["robot"].data.joint_pos[0, 0] = 5.0   # hip_pitch_l 大偏差
    env.scene["robot"].data.joint_pos[0, 9] = 5.0   # knee_l 大偏差
    close(mod.arm_deviation_penalty(env), torch.tensor([0.0]))


def test_arm_mixed_mean():
    # 一半关节偏 0.2、一半偏 0.4 -> 平均 0.3
    ref = [0.0] * 14
    act = [0.2] * 7 + [0.4] * 7
    env = make_jp_env([75], [ref], [act])
    close(mod.arm_deviation_penalty(env), torch.tensor([0.3]))


# ---------- locked_joint_penalty ----------
LOCKED_COLS = [2, 5, 17, 18]  # waist_yaw / waist_roll / ankle_roll_L/R(运行时序)


def make_locked_env(time_steps, ref_locked, act_locked):
    """ref_locked / act_locked: (N, 4) 锁死关节值; 其余关节都置 0。"""
    ref_locked = torch.tensor(ref_locked, dtype=torch.float32)
    act_locked = torch.tensor(act_locked, dtype=torch.float32)
    N = len(time_steps)
    jp_ref = torch.zeros(N, 29)
    jp_act = torch.zeros(N, 29)
    jp_ref[:, LOCKED_COLS] = ref_locked
    jp_act[:, LOCKED_COLS] = act_locked
    cmd = SimpleNamespace(
        time_steps=torch.tensor(time_steps, dtype=torch.long),
        joint_pos=jp_ref,
        anchor_pos_w=torch.zeros(N, 3),
    )
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=jp_act, root_pos_w=torch.zeros(N, 3)))
    return FakeEnv(command=cmd, robot=robot, sensor=FakeSensor(torch.zeros(N, 3, 2, 3)))


def test_locked_perfect_zero():
    # 参考锁 0, 实际也锁 0 -> 0
    env = make_locked_env([75], [0.0] * 4, [0.0] * 4)
    close(mod.locked_joint_penalty(env), torch.tensor([0.0]))


def test_locked_linear_penalty():
    # 4 个锁死关节全部偏 0.5 -> 平均绝对偏差 0.5(线性, 不饱和)
    env = make_locked_env([75], [0.0] * 4, [0.5] * 4)
    close(mod.locked_joint_penalty(env), torch.tensor([0.5]))


def test_locked_large_deviation_still_linear():
    # 偏 1.5 rad(模仿项早饱和) -> 仍线性罚 1.5
    env = make_locked_env([75], [0.0] * 4, [1.5] * 4)
    close(mod.locked_joint_penalty(env), torch.tensor([1.5]))


def test_locked_only_locked_columns():
    # 只有非锁死列(如 knee/hip)偏差, 4 个锁死列贴合 -> 0
    env = make_locked_env([75], [0.0] * 4, [0.0] * 4)
    env.scene["robot"].data.joint_pos[0, 0] = 5.0   # hip_pitch_l
    env.scene["robot"].data.joint_pos[0, 9] = 5.0   # knee_l
    close(mod.locked_joint_penalty(env), torch.tensor([0.0]))


def test_locked_mixed_mean():
    # 两列偏 0.2、两列偏 0.6 -> 平均 0.4
    env = make_locked_env([75], [0.0] * 4, [0.2, 0.2, 0.6, 0.6])
    close(mod.locked_joint_penalty(env), torch.tensor([0.4]))


# ---------- leg_tracking_penalty ----------
LEG_COLS = [0, 1, 9, 10, 13, 14, 17, 18, 8]  # hip_pitch L/R, knee L/R, ankle_pitch L/R, ankle_roll L/R, waist_pitch


def make_leg_env(time_steps, ref_legs, act_legs):
    """ref_legs / act_legs: (N, 9) 腿部关节值; 其余关节都置 0。"""
    ref_legs = torch.tensor(ref_legs, dtype=torch.float32)
    act_legs = torch.tensor(act_legs, dtype=torch.float32)
    N = len(time_steps)
    jp_ref = torch.zeros(N, 29)
    jp_act = torch.zeros(N, 29)
    jp_ref[:, LEG_COLS] = ref_legs
    jp_act[:, LEG_COLS] = act_legs
    cmd = SimpleNamespace(
        time_steps=torch.tensor(time_steps, dtype=torch.long),
        joint_pos=jp_ref,
        anchor_pos_w=torch.zeros(N, 3),
    )
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=jp_act, root_pos_w=torch.zeros(N, 3)))
    return FakeEnv(command=cmd, robot=robot, sensor=FakeSensor(torch.zeros(N, 3, 2, 3)))


def test_leg_perfect_tracking_zero():
    # 推蹬窗内腿部完全贴合参考 -> 0
    ref = [0.1, -0.2, 1.0, -1.0, 0.3, -0.3, 0.0, 0.0, 0.5]
    env = make_leg_env([110], [ref], [ref])
    close(mod.leg_tracking_penalty(env), torch.tensor([0.0]))


def test_leg_dive_forward_linear():
    # 前扑: 腿部关节全部停在深蹲值、参考已伸直 -> 线性罚平均偏差 0.5
    ref = [0.0] * 9
    act = [0.5] * 9
    env = make_leg_env([110], [ref], [act])
    close(mod.leg_tracking_penalty(env), torch.tensor([0.5]))


def test_leg_large_deviation_still_linear():
    # 偏差 1.5 rad(模仿项早饱和) -> 仍线性罚 1.5
    ref = [0.0] * 9
    act = [1.5] * 9
    env = make_leg_env([110], [ref], [act])
    close(mod.leg_tracking_penalty(env), torch.tensor([1.5]))


def test_leg_outside_window_zero():
    # 站立段(帧 10)腿怎么偏都不罚; 窗口外不干扰站立/深蹲
    ref = [0.0] * 9
    act = [1.0] * 9
    env = make_leg_env([10], [ref], [act])
    close(mod.leg_tracking_penalty(env), torch.tensor([0.0]))


def test_leg_only_window_start():
    # 推蹬窗起点 95 帧(深蹲底): 偏差也罚(从伸直开始拉)
    ref = [0.0] * 9
    act = [0.4] * 9
    env = make_leg_env([95], [ref], [act])
    close(mod.leg_tracking_penalty(env), torch.tensor([0.4]))


def test_leg_wraps_loop():
    # 95+183=278 取模回 95, 仍在窗口 -> 罚
    ref = [0.0] * 9
    act = [0.4] * 9
    env = make_leg_env([278], [ref], [act])
    close(mod.leg_tracking_penalty(env), torch.tensor([0.4]))


# ---------- airborne_tuck_tracking_penalty ----------
TUCK_COLS = [9, 10, 13, 14]  # knee L/R, ankle_pitch L/R(运行时序)


def make_tuck_env(time_steps, ref_tuck, act_tuck):
    """ref_tuck / act_tuck: (N, 4) 收腿关节值; 其余关节都置 0。"""
    ref_tuck = torch.tensor(ref_tuck, dtype=torch.float32)
    act_tuck = torch.tensor(act_tuck, dtype=torch.float32)
    N = len(time_steps)
    jp_ref = torch.zeros(N, 29)
    jp_act = torch.zeros(N, 29)
    jp_ref[:, TUCK_COLS] = ref_tuck
    jp_act[:, TUCK_COLS] = act_tuck
    cmd = SimpleNamespace(
        time_steps=torch.tensor(time_steps, dtype=torch.long),
        joint_pos=jp_ref,
        anchor_pos_w=torch.zeros(N, 3),
    )
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=jp_act, root_pos_w=torch.zeros(N, 3)))
    return FakeEnv(command=cmd, robot=robot, sensor=FakeSensor(torch.zeros(N, 3, 2, 3)))


def test_tuck_tracking_perfect_zero():
    # 腾空窗内膝/踝完全贴参考钟形收腿轨迹 -> 0
    ref = [1.5, 1.5, 0.4, 0.4]
    env = make_tuck_env([124], [ref], [ref])
    close(mod.airborne_tuck_tracking_penalty(env), torch.tensor([0.0]))


def test_tuck_tracking_extended_legs_calibrated():
    # model_29299 的真实失败模式: 膝停在伸直 0.29、踝 0(没收腿), 参考已在最高点
    # 收腿(膝 1.89/踝 0.49, 帧124) -> 平均偏差 (1.60×2 + 0.49×2)/4 = 1.045
    env = make_tuck_env([124], [1.89, 1.89, 0.49, 0.49], [0.29, 0.29, 0.0, 0.0])
    close(mod.airborne_tuck_tracking_penalty(env), torch.tensor([1.045]))


def test_tuck_tracking_large_deviation_still_linear():
    # 偏差 2.0 rad(模仿项早饱和) -> 仍线性罚 2.0(线性不饱和, 远距保留梯度)
    ref = [0.0] * 4
    act = [2.0] * 4
    env = make_tuck_env([124], [ref], [act])
    close(mod.airborne_tuck_tracking_penalty(env), torch.tensor([2.0]))


def test_tuck_tracking_outside_window_zero():
    # 站立段(帧 10)腿怎么偏都不罚; 窗口外不干扰站立/深蹲/推蹬
    ref = [0.0] * 4
    act = [1.0] * 4
    env = make_tuck_env([10], [ref], [act])
    close(mod.airborne_tuck_tracking_penalty(env), torch.tensor([0.0]))


def test_tuck_tracking_push_phase_zero():
    # 推蹬窗(帧 105, 窗口外)膝伸直不收腿 -> 0(收腿约束不碰推蹬伸腿)
    ref = [0.0] * 4
    act = [0.0] * 4
    env = make_tuck_env([105], [ref], [act])
    close(mod.airborne_tuck_tracking_penalty(env), torch.tensor([0.0]))


def test_tuck_tracking_only_tuck_columns():
    # 只有非收腿列(hip/waist)偏差, 膝/踝贴参考 -> 0(只罚膝/踝)
    ref = [0.0] * 4
    act = [0.0] * 4
    env = make_tuck_env([124], [ref], [act])
    env.scene["robot"].data.joint_pos[0, 0] = 5.0   # hip_pitch_l 大偏差
    env.scene["robot"].data.joint_pos[0, 8] = 5.0   # waist_pitch 大偏差
    close(mod.airborne_tuck_tracking_penalty(env), torch.tensor([0.0]))


def test_tuck_tracking_mixed_mean():
    # 两列偏 0.2、两列偏 0.6 -> 平均 0.4
    env = make_tuck_env([124], [0.0] * 4, [0.2, 0.2, 0.6, 0.6])
    close(mod.airborne_tuck_tracking_penalty(env), torch.tensor([0.4]))


def test_tuck_tracking_wraps_loop():
    # 124+183=307 取模回 124, 仍在窗口 -> 罚
    ref = [0.0] * 4
    act = [0.5] * 4
    env = make_tuck_env([307], [ref], [act])
    close(mod.airborne_tuck_tracking_penalty(env), torch.tensor([0.5]))


# ---------- body_lean_penalty ----------
def q_from_pitch(pitch_rad):
    """绕 Y 轴转 pitch 的四元数 (w,x,y,z)"""
    c, s = torch.cos(torch.tensor(pitch_rad) / 2), torch.sin(torch.tensor(pitch_rad) / 2)
    return [float(c), 0.0, float(s), 0.0]


def make_lean_env(time_steps, ref_pitch, act_pitch):
    """ref_pitch / act_pitch: (N,) base_link 俯仰角; 缺省单位四元数 = 0°。"""
    N = len(time_steps)
    ref_q = torch.tensor([q_from_pitch(p) for p in ref_pitch], dtype=torch.float32)
    act_q = torch.tensor([q_from_pitch(p) for p in act_pitch], dtype=torch.float32)
    cmd = SimpleNamespace(
        time_steps=torch.tensor(time_steps, dtype=torch.long),
        anchor_pos_w=torch.zeros(N, 3),
        anchor_quat_w=ref_q,
        robot_anchor_quat_w=act_q,
    )
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(N, 29), root_pos_w=torch.zeros(N, 3)))
    return FakeEnv(command=cmd, robot=robot, sensor=FakeSensor(torch.zeros(N, 3, 2, 3)))


def test_lean_airborne_zero():
    # 腾空窗口内前后倾都=0(参考要求) -> 0
    env = make_lean_env([125], [0.0], [0.0])
    close(mod.body_lean_penalty(env), torch.tensor([0.0]))


def test_lean_airborne_lean_back():
    # 腾空窗口内机器人后仰 -0.3 rad, 参考 0 -> 罚 0.3(线性)
    env = make_lean_env([125], [0.0], [-0.3])
    close(mod.body_lean_penalty(env), torch.tensor([0.3]))


def test_lean_airborne_lean_forward():
    # 腾空窗口内还保持下蹲的前倾 +0.4(没转回 0) -> 罚 0.4
    env = make_lean_env([125], [0.0], [0.4])
    close(mod.body_lean_penalty(env), torch.tensor([0.4]))


def test_lean_outside_window_zero():
    # 下蹲段(参考 +0.5 前倾, 机器人也 +0.5)窗口外 -> 0
    env = make_lean_env([50], [0.5], [0.5])
    close(mod.body_lean_penalty(env), torch.tensor([0.0]))


def test_lean_crouch_deviation_not_penalized():
    # 下蹲段机器人 pitch 偏了也不罚(窗口外, 下蹲前倾是合理的)
    env = make_lean_env([50], [0.5], [0.0])
    close(mod.body_lean_penalty(env), torch.tensor([0.0]))


def test_lean_push_phase_penalized():
    # 推蹬段(帧 105)机器人还保持深蹲前倾 +0.4、参考已回 0 -> 罚(窗口已延伸到推蹬)
    env = make_lean_env([105], [0.0], [0.4])
    close(mod.body_lean_penalty(env), torch.tensor([0.4]))


def test_lean_large_deviation_still_linear():
    # 后仰 1.2 rad(模仿项早饱和) -> 仍线性罚 1.2
    env = make_lean_env([125], [0.0], [-1.2])
    close(mod.body_lean_penalty(env), torch.tensor([1.2]))


def test_lean_loop_wraps():
    # 腾空窗口帧循环取模后仍然生效(115+183=298 也在窗口内)
    env = make_lean_env([298], [0.0], [-0.2])
    close(mod.body_lean_penalty(env), torch.tensor([0.2]))


def make_fidget_env(time_steps, foot_lin_vel, act_foot_z, ref_foot_z=None):
    """standing_fidget_penalty 专用(脚度量)。脚 = OMNI_BODY_NAMES 序 18/19
    (ankle_roll_l/r_link)。foot_lin_vel: (N,2,3) 脚世界速度 m/s;
    act_foot_z: (N,2) 实际脚世界高度 m; ref_foot_z: (N,2) 参考脚世界高度 m
    (缺省 = act, 即"参考脚未抬起"基线)。其余 body 用 0 占位。"""
    N = len(time_steps)
    flv = torch.tensor(foot_lin_vel, dtype=torch.float32)
    a_fz = torch.tensor(act_foot_z, dtype=torch.float32)
    r_fz = torch.tensor(ref_foot_z, dtype=torch.float32) if ref_foot_z is not None else a_fz.clone()
    c0, c1 = mod.FOOT_COLS  # 18, 19
    # 实际脚: body_pos_w 高度 + body_lin_vel_w 速度
    body_pos_w = torch.zeros(N, 30, 3)
    body_pos_w[:, c0, 2] = a_fz[:, 0]
    body_pos_w[:, c1, 2] = a_fz[:, 1]
    body_lin_vel_w = torch.zeros(N, 30, 3)
    body_lin_vel_w[:, c0, :] = flv[:, 0, :]
    body_lin_vel_w[:, c1, :] = flv[:, 1, :]
    # 参考脚: command.body_pos_w 高度
    cmd_body_pos_w = torch.zeros(N, 30, 3)
    cmd_body_pos_w[:, c0, 2] = r_fz[:, 0]
    cmd_body_pos_w[:, c1, 2] = r_fz[:, 1]
    return FakeEnv(
        command=FakeCommand(time_steps, [[0, 0, 0.78]] * N, body_pos_w=cmd_body_pos_w),
        robot=FakeRobot([[0, 0, 0.78]] * N, body_pos_w=body_pos_w, body_lin_vel_w=body_lin_vel_w),
        sensor=FakeSensor(torch.zeros(N, 3, 2, 3)),
    )


# ---------- standing_fidget_penalty(脚度量: 滑动 + 离地, 规则 1.3.7) ----------
def test_fidget_still_zero():
    # 站立窗(帧 5)脚完全静止 + 脚未抬起 -> 0
    env = make_fidget_env([5], [[[0, 0, 0], [0, 0, 0]]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([0.0]))


def test_fidget_slip_below_margin_zero():
    # 脚水平滑速 0.01 < slip_margin 0.02 -> 0(PD/接触噪声进死区)
    env = make_fidget_env([5], [[[0.01, 0.0, 0.0], [0.01, 0.0, 0.0]]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([0.0]))


def test_fidget_slip_penalized():
    # 双脚水平滑 0.5 m/s -> 每脚 excess (0.5-0.02)=0.48, 平方 (0.48/0.1)²=23.04, 两脚均值
    env = make_fidget_env([5], [[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([23.04]))


def test_fidget_slip_quadratic_superlinear():
    # 滑动 1.0 -> excess 0.98, 平方 (0.98/0.1)²=96.04(超线性, 大滑动代价暴涨)
    env = make_fidget_env([5], [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([96.04]))


def test_fidget_slip_only_horizontal():
    # 脚垂直速度(上升)不算"滑动"—— 脚抬起由高度(lift)抓, 别重复计
    env = make_fidget_env([5], [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([0.0]))


def test_fidget_one_foot_slip_only_mean():
    # 单脚滑动(另一只钉地)-> 两脚均值 23.04/2=11.52(脚粒度, 一只动就罚)
    env = make_fidget_env([5], [[[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([11.52]))


def test_fidget_lift_penalized():
    # 脚离地: (0.15-0.033-0.02)=0.097 平方 (0.097/0.1)²=0.9409, 两脚均值
    env = make_fidget_env([5], [[[0, 0, 0], [0, 0, 0]]], [[0.15, 0.15]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([0.9409]))


def test_fidget_lift_below_margin_zero():
    # 脚高 0.043 仅比参考高 0.01 < lift_margin 0.02 -> 0
    env = make_fidget_env([5], [[[0, 0, 0], [0, 0, 0]]], [[0.043, 0.043]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([0.0]))


def test_fidget_one_foot_lift_only_mean():
    # 单脚离地(另一只钉地)-> 两脚均值 0.9409/2 = 0.47045
    env = make_fidget_env([5], [[[0, 0, 0], [0, 0, 0]]], [[0.15, 0.033]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([0.9409 / 2]))


def test_fidget_slip_and_lift_add():
    # 平方下 slip 与 lift 两项相加: (0.48/0.1)² + (0.097/0.1)² = 23.04 + 0.9409 = 23.9809
    env = make_fidget_env([5], [[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], [[0.15, 0.15]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([23.9809]))


def test_fidget_outside_window_zero():
    # 下蹲(帧 20)脚乱动 -> 0(不干扰起跳)
    env = make_fidget_env([20], [[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], [[0.15, 0.15]])
    close(mod.standing_fidget_penalty(env), torch.tensor([0.0]))


def test_fidget_boundary_15_zero():
    # 帧 15(下蹲起点)脚乱动 -> 0
    env = make_fidget_env([15], [[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], [[0.15, 0.15]])
    close(mod.standing_fidget_penalty(env), torch.tensor([0.0]))


def test_fidget_boundary_14_penalized():
    # 帧 14(站立最后一帧)滑动 0.5 -> 罚 23.04
    env = make_fidget_env([14], [[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([23.04]))


def test_fidget_wraps_loop():
    # 帧 183 -> 取模 0(回绕回站立)脚离地 -> 罚 0.9409
    env = make_fidget_env([183], [[[0, 0, 0], [0, 0, 0]]], [[0.15, 0.15]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([0.9409]))


def test_fidget_quadratic_small_noise_cheap():
    # 0.05 m/s(噪声级, 略超 margin 0.02)-> (0.03/0.1)²=0.09, 几乎免费(不误罚微动)
    env = make_fidget_env([5], [[[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env), torch.tensor([0.09]))


def test_fidget_quadratic_doubling_more_than_doubles():
    # 滑动 0.1 -> (0.08/0.1)²=0.64, 0.2 -> (0.18/0.1)²=3.24: 翻倍 → 惩罚翻 5 倍(超线性)
    env = make_fidget_env(
        [5, 5],
        [[[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]],
        [[0.033, 0.033], [0.033, 0.033]],
    )
    close(mod.standing_fidget_penalty(env), torch.tensor([0.64, 3.24]))


def test_fidget_scale_param():
    # scale=0.2: 滑动 0.5 -> (0.48/0.2)²=5.76(scale 控制非线性陡峭度)
    env = make_fidget_env([5], [[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], [[0.033, 0.033]])
    close(mod.standing_fidget_penalty(env, scale=0.2), torch.tensor([5.76]))


# ---------- standing_symmetry_penalty(站立窗动作左右对称, 治"单脚先动"根因) ----------
def make_sym_env(time_steps, actions):
    """standing_symmetry_penalty 专用: actions (N,29) 策略动作(运行时序 BFS)。"""
    N = len(time_steps)
    cmd = SimpleNamespace(
        time_steps=torch.tensor(time_steps, dtype=torch.long),
        anchor_pos_w=torch.zeros(N, 3),
        body_pos_w=torch.zeros(N, 30, 3),
        joint_pos=torch.zeros(N, 29),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros(N, 29),
            root_pos_w=torch.zeros(N, 3),
            body_pos_w=torch.zeros(N, 30, 3),
            body_lin_vel_w=torch.zeros(N, 30, 3),
        )
    )
    env = FakeEnv(command=cmd, robot=robot, sensor=FakeSensor(torch.zeros(N, 3, 2, 3)))
    env.action_manager = SimpleNamespace(action=torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in actions]))
    return env


def make_sym_actions():
    """13 对完全对称的动作向量(相等对相等 / 镜像对反号)。"""
    a = torch.zeros(29)
    for i, j in mod.STAND_SYM_PITCH_PAIRS:
        a[i] = a[j] = 0.3
    for i, j in mod.STAND_SYM_MIRROR_PAIRS:
        a[i] = 0.5
        a[j] = -0.5
    return a


def _sym_expected(single_pair_dev):
    """单对偏差 single_pair_dev 对应的平方惩罚值: ((dev/13)/0.3)²。"""
    return float(((single_pair_dev / 13) / 0.3) ** 2)


def test_sym_symmetric_zero():
    # 站立帧 5, 13 对完全对称 -> 0
    env = make_sym_env([5], [make_sym_actions()])
    close(mod.standing_symmetry_penalty(env), torch.tensor([0.0]))


def test_sym_pitch_pair_asym_penalized():
    # ankle_pitch L(13)=0.5 R(14)=0 -> 单对偏差 0.5, 归一化 /13, 有界 1-exp
    a = make_sym_actions()
    a[13], a[14] = 0.5, 0.0
    env = make_sym_env([5], [a])
    close(mod.standing_symmetry_penalty(env), torch.tensor([_sym_expected(0.5)]))


def test_sym_mirror_pair_misaligned():
    # ankle_roll L(17)=0.3 R(18)=0.3(应反号却同号)-> 和 0.6
    a = make_sym_actions()
    a[17], a[18] = 0.3, 0.3
    env = make_sym_env([5], [a])
    close(mod.standing_symmetry_penalty(env), torch.tensor([_sym_expected(0.6)]))


def test_sym_outside_window_zero():
    # 下蹲帧 20 动作不对称 -> 0(不约束起跳段)
    a = make_sym_actions()
    a[13], a[14] = 0.5, 0.0
    env = make_sym_env([20], [a])
    close(mod.standing_symmetry_penalty(env), torch.tensor([0.0]))


def test_sym_boundary():
    # 帧 15(下蹲起点)不罚; 帧 14(站立末帧)罚
    a = make_sym_actions()
    a[13], a[14] = 0.5, 0.0
    env15 = make_sym_env([15], [a])
    close(mod.standing_symmetry_penalty(env15), torch.tensor([0.0]))
    env14 = make_sym_env([14], [a])
    close(mod.standing_symmetry_penalty(env14), torch.tensor([_sym_expected(0.5)]))


def test_sym_scale_param():
    # scale 越大惩罚越小(更宽容, 平方罚的分母)
    a = make_sym_actions()
    a[13], a[14] = 0.5, 0.0
    env = make_sym_env([5], [a])
    assert mod.standing_symmetry_penalty(env, scale=0.15)[0] > mod.standing_symmetry_penalty(env, scale=0.3)[0]


def test_sym_wraps_loop():
    # 帧 183 -> 取模 0(回绕回站立)动作不对称 -> 罚
    a = make_sym_actions()
    a[13], a[14] = 0.5, 0.0
    env = make_sym_env([183], [a])
    close(mod.standing_symmetry_penalty(env), torch.tensor([_sym_expected(0.5)]))


# ---------- standing_invalid_termination(策略 A: 站立窗脚动 -> 判无效硬终止) ----------
def _term(mod_ref, env):
    return bool(mod_ref.standing_invalid_termination(env)[0].item())


def test_term_still_no_terminate():
    # 站立帧脚静止 + 未抬起 -> 不判无效
    env = make_fidget_env([5], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[0.033, 0.033]])
    assert not _term(mod, env)


def test_term_slip_below_thresh_no_terminate():
    # 滑速 0.05 < slip_thresh 0.10(接触/PD 噪声级) -> 不判无效
    env = make_fidget_env([5], [[[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]], [[0.033, 0.033]])
    assert not _term(mod, env)


def test_term_slip_above_thresh_terminate():
    # 滑速 0.15 > slip_thresh 0.10(真滑动) -> 判无效
    env = make_fidget_env([5], [[[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]]], [[0.033, 0.033]])
    assert _term(mod, env)


def test_term_lift_below_thresh_no_terminate():
    # 脚高 0.043 仅比参考高 0.010 < lift_thresh 0.05 -> 不判无效
    env = make_fidget_env([5], [[[0, 0, 0], [0, 0, 0]]], [[0.043, 0.043]], [[0.033, 0.033]])
    assert not _term(mod, env)


def test_term_lift_above_thresh_terminate():
    # 脚抬到 0.10(比参考高 0.067 > 0.05) -> 判无效
    env = make_fidget_env([5], [[[0, 0, 0], [0, 0, 0]]], [[0.10, 0.10]], [[0.033, 0.033]])
    assert _term(mod, env)


def test_term_single_foot_terminate():
    # 只有一只脚滑动(另一只钉地) -> 也判无效("单脚先动")
    env = make_fidget_env([5], [[[0.15, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[0.033, 0.033]])
    assert _term(mod, env)


def test_term_outside_window_no_terminate():
    # 下蹲帧 20 大滑动 -> 窗口外, 不判死(不干扰起跳)
    env = make_fidget_env([20], [[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], [[0.033, 0.033]])
    assert not _term(mod, env)


def test_term_grace_period_no_terminate():
    # 复位后前 2 帧(0/1)即使大滑动也不判死(防接触瞬态 100% 误杀)
    env0 = make_fidget_env([0], [[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], [[0.033, 0.033]])
    assert not _term(mod, env0)
    env1 = make_fidget_env([1], [[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]], [[0.033, 0.033]])
    assert not _term(mod, env1)
    # 第 3 帧(2)起正常判死
    env2 = make_fidget_env([2], [[[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]]], [[0.033, 0.033]])
    assert _term(mod, env2)


def test_term_boundary_t15():
    # 帧 14(站立末帧)滑 0.15 -> 判死; 帧 15(下蹲起点) -> 不判死
    env14 = make_fidget_env([14], [[[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]]], [[0.033, 0.033]])
    assert _term(mod, env14)
    env15 = make_fidget_env([15], [[[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]]], [[0.033, 0.033]])
    assert not _term(mod, env15)


def test_term_wraps_loop():
    # 帧 183 取模 0 = 回绕回站立, 但在 grace 前 2 帧内 -> 不判死
    env0 = make_fidget_env([183], [[[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]]], [[0.033, 0.033]])
    assert not _term(mod, env0)
    # 帧 186 取模 3(已过 grace, 仍在站立窗) -> 判死
    env3 = make_fidget_env([186], [[[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]]], [[0.033, 0.033]])
    assert _term(mod, env3)


def test_term_custom_thresh():
    # 滑速 0.12: slip_thresh=0.15 不判死, slip_thresh=0.10 判死(阈值参数生效)
    env = make_fidget_env([5], [[[0.12, 0.0, 0.0], [0.12, 0.0, 0.0]]], [[0.033, 0.033]])
    assert not _term_sl_thresh(mod, env, 0.15)
    assert _term_sl_thresh(mod, env, 0.10)


def _term_sl_thresh(mod_ref, env, thresh):
    return bool(mod_ref.standing_invalid_termination(env, slip_thresh=thresh)[0].item())


def test_term_custom_grace():
    # 帧 3 滑 0.15: grace=4 不判死, grace=2 判死(grace 参数生效)
    env = make_fidget_env([3], [[[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]]], [[0.033, 0.033]])
    assert not bool(mod.standing_invalid_termination(env, grace=4)[0].item())
    assert bool(mod.standing_invalid_termination(env, grace=2)[0].item())


def test_term_foot_below_ref_not_lift():
    # 脚低于参考(踩实) -> lift 无负值不计, 不判无效
    env = make_fidget_env([5], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[0.01, 0.01]], [[0.033, 0.033]])
    assert not _term(mod, env)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS {t.__name__}")
    print(f"\n{passed}/{len(tests)} 通过")


if __name__ == "__main__":
    main()
