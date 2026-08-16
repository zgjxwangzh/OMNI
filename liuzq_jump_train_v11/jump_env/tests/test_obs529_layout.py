"""529 obs 字节一致性关卡(纯 numpy + torch, 无需 omni)。

铁律: 训练 obs = 部署 obs, 逐字节。本测试把 SDK HighDynamic `_build_obs` 用 numpy
逐行复刻(`SdkReplica`), 再把 `jump_env/mdp/obs529.py` 跑在一个假 env 上, 断言两者
逐元素一致 —— obs529 的组装/校准/历史推进必须与 SDK 公式逐位吻合。

数据来源: 用真实 50fps 参考运动 `motion/jump_high_firstjump_50fps.npz` 的帧作
"当前机器人状态 + 当前参考帧", 保证不是空转的合成值。

布局(SDK _compute_num_obs, num_actions=29, history_length=5):
  [0:58]   command    = 参考当前帧 绝对 joint_pos(29)+joint_vel(29)
  [58:64]  anchor_ori = R(robot)^T @ calib @ R(ref) 前两列, C-order(6)
  [64:79]  gravity 历史(5×3)   [79:94]   ang_vel 历史(5×3)
  [94:239] joint_pos_rel 历史(5×29)  [239:384] joint_vel 历史(5×29)
  [384:529] action 历史(5×29)
  合计 58+6+5×93 = 529。
"""

from __future__ import annotations

import importlib.util
import os
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 常量: 全部来自部署基准包 omni_rl_sdk.zip policy/high_dynamic/config/high_dynamic.yaml
# ---------------------------------------------------------------------------
NUM_ACTIONS = 29
HISTORY_LENGTH = 5
ACTION_SCALE = 0.5   # action.scale
ACTION_CLIP = 10.0   # action.clip
NUM_OBS = 529
COMMAND_DIM = 2 * NUM_ACTIONS   # 58
ANCHOR_DIM = 6
HISTORY_FRAME_DIM = 3 + 3 + 3 * NUM_ACTIONS  # 93
ACTION_SLICE = slice(NUM_OBS - HISTORY_LENGTH * NUM_ACTIONS, NUM_OBS)  # 384:529

MOTION_PATH = "/home/zyy/jump_high/motion/jump_high_firstjump_50fps.npz"

# SDK dof.default_pos 是 motor 序, 经 policy_joint_names 逆映射到 policy 序(= 训练
# URDF/BFS 序)。逐项核对 zip policy_joint_names 29 个; 勿用 run-length 简写(曾数错)。
DEFAULT_POS = np.array(
    [
        -0.262, -0.262,   # hip_pitch_l, hip_pitch_r
        0.0, 0.0, 0.0,    # waist_yaw, hip_roll_l, hip_roll_r
        0.0, 0.0, 0.0,    # waist_roll, hip_yaw_l, hip_yaw_r
        0.0,              # waist_pitch
        0.524, 0.524,     # knee_pitch_l, knee_pitch_r
        0.3, 0.3,         # shoulder_pitch_l, shoulder_pitch_r
        -0.262, -0.262,   # ankle_pitch_l, ankle_pitch_r
        0.0, 0.0,         # shoulder_roll_l, shoulder_roll_r
        0.0, 0.0,         # ankle_roll_l, ankle_roll_r
        0.0, 0.0,         # shoulder_yaw_l, shoulder_yaw_r
        -0.7, -0.7,       # elbow_pitch_l, elbow_pitch_r
        0.0, 0.0,         # elbow_yaw_l, elbow_yaw_r
        0.0, 0.0,         # wrist_pitch_l, wrist_pitch_r
        0.0, 0.0,         # wrist_roll_l, wrist_roll_r
    ],
    dtype=np.float32,
)
assert DEFAULT_POS.shape == (29,)


def matrix_from_quat_np(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵(与 SDK matrix_from_quat_numpy 同式)。"""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_quat_np(q: np.ndarray) -> np.ndarray:
    """SDK _yaw_quat 逐字复刻: yaw=atan2(2(wz+xy), 1-2(y²+z²)), 返回绕 z 四元数。"""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul_yaw(q: np.ndarray, yaw_rad: float) -> np.ndarray:
    """给四元数绕世界 z 转 yaw(用于构造带偏航的机器人姿态)。"""
    c, s = np.cos(yaw_rad / 2), np.sin(yaw_rad / 2)
    qy = np.array([c, 0.0, 0.0, s])
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = qy
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


class SdkReplica:
    """SDK HighDynamic 策略的 numpy 复刻(_reset_history / warmup_from_state /
    _update_history_from_state / _calibrate_init_rotation / _get_anchor_ori_b / _build_obs)。"""

    def __init__(self, default_pos: np.ndarray = DEFAULT_POS):
        self.default_pos = np.asarray(default_pos, dtype=np.float32)
        self.scale = ACTION_SCALE
        self.clip = ACTION_CLIP
        self.history_length = HISTORY_LENGTH
        self.gravity_hist = deque(maxlen=self.history_length)
        self.ang_vel_hist = deque(maxlen=self.history_length)
        self.joint_pos_hist = deque(maxlen=self.history_length)
        self.joint_vel_hist = deque(maxlen=self.history_length)
        self.action_hist = deque(maxlen=self.history_length)
        self.world_to_init_rot = np.eye(3, dtype=np.float64)
        self._init_calibrated = False
        self.last_action_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)

    def reset_history(self, state: dict) -> None:
        """SDK _reset_history(state_cmd): 5×当前真实状态 + action 全 0。"""
        for h in (self.gravity_hist, self.ang_vel_hist, self.joint_pos_hist,
                  self.joint_vel_hist, self.action_hist):
            h.clear()
        gravity = np.asarray(state["gravity"], dtype=np.float32).copy()
        ang_vel = np.asarray(state["ang_vel"], dtype=np.float32).copy()
        joint_pos = np.asarray(state["q"], dtype=np.float32) - self.default_pos
        joint_vel = np.asarray(state["dq"], dtype=np.float32).copy()
        for _ in range(self.history_length):
            self.gravity_hist.append(gravity.copy())
            self.ang_vel_hist.append(ang_vel.copy())
            self.joint_pos_hist.append(joint_pos.copy())
            self.joint_vel_hist.append(joint_vel.copy())
            self.action_hist.append(np.zeros(NUM_ACTIONS, dtype=np.float32))

    def warmup_from_state(self, state: dict) -> None:
        """SDK warmup_from_state: 预填历史 + last_action_policy=(q-default)/scale。"""
        self.reset_history(state)
        q_policy = np.asarray(state["q"], dtype=np.float32)
        scale = np.float32(self.scale) if self.scale != 0.0 else np.float32(1.0)
        self.last_action_policy = ((q_policy - self.default_pos) / scale).astype(np.float32)

    def update_history_from_state(self, state: dict) -> None:
        """SDK _update_history_from_state: 尾插当前状态 + last_action_policy(= 上一步 clip 后的动作)。"""
        self.gravity_hist.append(np.asarray(state["gravity"], dtype=np.float32).copy())
        self.ang_vel_hist.append(np.asarray(state["ang_vel"], dtype=np.float32).copy())
        self.joint_pos_hist.append(
            (np.asarray(state["q"], dtype=np.float32) - self.default_pos).astype(np.float32)
        )
        self.joint_vel_hist.append(np.asarray(state["dq"], dtype=np.float32).astype(np.float32))
        self.action_hist.append(self.last_action_policy.copy().astype(np.float32))

    def set_last_action(self, action: np.ndarray) -> None:
        """get_action 后: last_action_policy = clip(net, -clip, clip)。"""
        self.last_action_policy = np.clip(
            np.asarray(action, dtype=np.float32), -self.clip, self.clip
        ).astype(np.float32)

    def calibrate(self, robot_quat: np.ndarray, ref_q0: np.ndarray) -> None:
        """SDK _calibrate_init_rotation: 首次冻结, 之后不变。"""
        if self._init_calibrated:
            return
        init_to_anchor_rot = matrix_from_quat_np(yaw_quat_np(np.asarray(ref_q0, dtype=np.float64)))
        world_to_anchor_rot = matrix_from_quat_np(yaw_quat_np(np.asarray(robot_quat, dtype=np.float64)))
        self.world_to_init_rot = world_to_anchor_rot @ init_to_anchor_rot.T
        self._init_calibrated = True

    def anchor_ori_b(self, robot_quat: np.ndarray, ref_quat: np.ndarray) -> np.ndarray:
        """SDK _get_anchor_ori_b: rot_b = R(robot)^T @ calib @ R(ref), 前两列 C-order。"""
        rot_inv = matrix_from_quat_np(np.asarray(robot_quat, dtype=np.float64)).T
        ref_rot = matrix_from_quat_np(np.asarray(ref_quat, dtype=np.float64))
        rot_b = rot_inv @ self.world_to_init_rot @ ref_rot
        return rot_b[:, :2].reshape(-1).astype(np.float32)

    def build_obs(self, ref_pos, ref_vel, robot_quat, ref_quat) -> np.ndarray:
        """SDK _build_obs: 分块拼接 529。ref_pos/vel = 参考当前帧绝对关节角/速。"""
        command = np.concatenate(
            [np.asarray(ref_pos).reshape(-1), np.asarray(ref_vel).reshape(-1)]
        ).astype(np.float32)
        anchor_ori = self.anchor_ori_b(robot_quat, ref_quat)
        obs = np.concatenate(
            [
                command,
                anchor_ori,
                np.concatenate(list(self.gravity_hist)),
                np.concatenate(list(self.ang_vel_hist)),
                np.concatenate(list(self.joint_pos_hist)),
                np.concatenate(list(self.joint_vel_hist)),
                np.concatenate(list(self.action_hist)),
            ],
            axis=0,
        ).astype(np.float32)
        assert obs.shape[0] == NUM_OBS
        return obs


# ---------------------------------------------------------------------------
# 假 env: 一个共享 env 对象, 通过 refresh() 换状态(obs529 的懒状态存 env.extras,
# 必须在同一对象上跨步推进才保真)。
# ---------------------------------------------------------------------------
class FakeMotion:
    def __init__(self, body_quat_w):
        self.body_quat_w = torch.tensor(body_quat_w, dtype=torch.float32)


class FakeCommand:
    def __init__(self, body_quat_w, num_envs=1):
        self.motion = FakeMotion(body_quat_w)
        self.motion_anchor_body_index = 0
        self.num_envs = num_envs
        self.command = None
        self.robot_anchor_quat_w = None
        self.anchor_quat_w = None


class FakeRobot:
    def __init__(self, device, num_envs):
        self.device = device
        self.data = SimpleNamespace(
            projected_gravity_b=torch.zeros(num_envs, 3),
            root_ang_vel_b=torch.zeros(num_envs, 3),
            joint_pos=torch.zeros(num_envs, 29),
            joint_vel=torch.zeros(num_envs, 29),
            root_quat_w=torch.zeros(num_envs, 4),
        )


class FakeScene:
    def __init__(self, robot):
        self._robot = robot

    def __getitem__(self, key):
        assert key == "robot"
        return self._robot


class FakeEnv:
    def __init__(self, robot, command, num_envs=1):
        self.scene = FakeScene(robot)
        self.command_manager = SimpleNamespace(get_term=lambda name: command)
        self.action_manager = SimpleNamespace(action=torch.zeros(num_envs, 29))
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.common_step_counter = -1
        self.num_envs = num_envs
        self.extras = {}


def _b(a, n: int) -> np.ndarray:
    """(D,) -> (1,D); (n,D) 保持; 首维为 1 且需 n>1 时广播到 (n,D)。"""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a[None]
    if a.shape[0] == 1 and n > 1:
        a = np.broadcast_to(a, (n,) + a.shape[1:]).copy()
    return a


def refresh(env: FakeEnv, st: dict, ref_pos, ref_vel, ref_q, action, ep_len, step) -> None:
    """在共享 env 上换状态: 机器人数据 + 命令 + 动作 + 回合步计数。"""
    n = env.num_envs
    r = env.scene._robot
    r.data.projected_gravity_b = torch.tensor(_b(st["gravity"], n))
    r.data.root_ang_vel_b = torch.tensor(_b(st["ang_vel"], n))
    r.data.joint_pos = torch.tensor(_b(st["q"], n))
    r.data.joint_vel = torch.tensor(_b(st["dq"], n))
    r.data.root_quat_w = torch.tensor(_b(st["robot_quat"], n))
    c = env.command_manager.get_term("motion")
    c.command = torch.tensor(np.concatenate([_b(ref_pos, n), _b(ref_vel, n)], axis=-1))
    c.robot_anchor_quat_w = torch.tensor(_b(st["robot_quat"], n))
    c.anchor_quat_w = torch.tensor(_b(ref_q, n))
    env.action_manager.action = torch.tensor(_b(action, n))
    env.episode_length_buf = torch.tensor(ep_len, dtype=torch.long)
    env.common_step_counter = step


def run_obs529(env) -> np.ndarray:
    return obs529_mod.obs529(env).cpu().numpy()


def make_state(k: int, yaw: float) -> dict:
    """k: 参考帧号(机器人状态 = 参考 k 帧); yaw: 机器人偏航偏移。"""
    robot_q = quat_mul_yaw(_REF_QUAT[k], yaw)
    gravity_b = (matrix_from_quat_np(robot_q).T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)
    return {
        "q": _REF_POS[k],
        "dq": _REF_VEL[k],
        "gravity": gravity_b,
        "ang_vel": np.array([0.1, -0.2, 0.3], dtype=np.float32),
        "robot_quat": robot_q.astype(np.float32),
    }


# 观测负载 obs529(importlib 直载, 避开 jump_env/__init__ 的 omni 依赖链)
_MOD_PATH = os.path.join(os.path.dirname(__file__), "..", "mdp", "obs529.py")
_spec = importlib.util.spec_from_file_location("obs529_module", _MOD_PATH)
obs529_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(obs529_mod)

_NPZ = np.load(MOTION_PATH, allow_pickle=True)
_REF_POS = _NPZ["joint_pos"]      # (183,29)
_REF_VEL = _NPZ["joint_vel"]      # (183,29)
_REF_QUAT = _NPZ["body_quat_w"][:, 0, :]  # (183,4) base_link
_BODY_QUAT = _NPZ["body_quat_w"]   # (183,30,4)


def _fresh_env(num_envs=1) -> FakeEnv:
    robot = FakeRobot("cpu", num_envs)
    cmd = FakeCommand(_BODY_QUAT, num_envs=num_envs)
    return FakeEnv(robot, cmd, num_envs=num_envs)


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
def test_num_obs_529():
    assert COMMAND_DIM + ANCHOR_DIM + HISTORY_LENGTH * HISTORY_FRAME_DIM == NUM_OBS


def test_sdk_default_pos_matches_yaml():
    """obs529.SDK_DEFAULT_POS(policy 序) 与 yaml default_pos(motor 序) 映射一致。"""
    got = obs529_mod.SDK_DEFAULT_POS.numpy()
    assert np.allclose(got, DEFAULT_POS, atol=1e-6), "obs529.SDK_DEFAULT_POS 与测试副本不一致"
    # motor 序锚点(yaml default_pos -> policy 序), 防整表移位:
    #   motor[0]  hip_pitch_l      -> policy[0]  = -0.262
    #   motor[3]  knee_pitch_l     -> policy[9]  =  0.524
    #   motor[15] shoulder_pitch_l -> policy[11] =  0.300
    #   motor[18] elbow_pitch_l    -> policy[21] = -0.700
    # float32 表示, 用容差比较(0.3 在 float32 下是 0.3000000119…)
    assert abs(got[0] - (-0.262)) < 1e-5 and abs(got[9] - 0.524) < 1e-5
    assert abs(got[11] - 0.3) < 1e-5 and abs(got[21] - (-0.7)) < 1e-5


def test_sdk_constants():
    assert obs529_mod.SDK_ACTION_SCALE == 0.5
    assert obs529_mod.SDK_ACTION_CLIP == 10.0
    assert obs529_mod.HISTORY_LENGTH == 5
    assert obs529_mod.SDK_DEFAULT_POS.numel() == 29


def test_fresh_obs_byte_exact():
    """回合首 obs(预填 + warmup + 校准)与 SDK 复刻逐元素一致。"""
    k = 0
    st = make_state(k, yaw=0.4)
    env = _fresh_env()
    refresh(env, st, _REF_POS[k], _REF_VEL[k], _REF_QUAT[k], np.zeros(NUM_ACTIONS), [0], 0)
    obs = run_obs529(env)  # fresh(ep_len==0)

    # 期望: SDK warmup -> 首次 update_history(尾插 warmup 动作) -> build
    rep = SdkReplica()
    rep.warmup_from_state(st)
    rep.calibrate(st["robot_quat"], _BODY_QUAT[0, 0])
    rep.update_history_from_state(st)
    expected = rep.build_obs(_REF_POS[k], _REF_VEL[k], st["robot_quat"], _REF_QUAT[k])

    assert obs.shape == (1, NUM_OBS)
    assert np.allclose(obs[0], expected, atol=1e-5), f"fresh obs 偏差 max={np.abs(obs[0]-expected).max():.2e}"


def test_roll_obs_byte_exact():
    """非首 obs(历史滚动 + 动作尾插 clip(a) + 校准冻结)与 SDK 复刻逐元素一致。"""
    env = _fresh_env()
    rep = SdkReplica()

    # 第 1 步 fresh: 建立校准 + 历史
    k0 = 0
    st0 = make_state(k0, yaw=0.4)
    refresh(env, st0, _REF_POS[k0], _REF_VEL[k0], _REF_QUAT[k0], np.zeros(NUM_ACTIONS), [0], 0)
    obs0 = run_obs529(env)
    rep.warmup_from_state(st0)
    rep.calibrate(st0["robot_quat"], _BODY_QUAT[0, 0])
    rep.update_history_from_state(st0)
    assert np.allclose(obs0[0], rep.build_obs(_REF_POS[k0], _REF_VEL[k0], st0["robot_quat"], _REF_QUAT[k0]), atol=1e-5)

    # 第 2 步非 fresh: 状态/偏航都变, 上一步动作 a_prev
    k1 = 5
    st1 = make_state(k1, yaw=0.9)
    a_prev = np.array([3.0, -4.0, 5.0, 1.5, 2.5] + [0.0] * 24, dtype=np.float32)
    refresh(env, st1, _REF_POS[k1], _REF_VEL[k1], _REF_QUAT[k1], a_prev, [1], 1)
    obs = run_obs529(env)  # 非 fresh -> 滚动

    rep.set_last_action(a_prev)
    rep.update_history_from_state(st1)
    expected = rep.build_obs(_REF_POS[k1], _REF_VEL[k1], st1["robot_quat"], _REF_QUAT[k1])

    assert obs.shape == (1, NUM_OBS)
    assert np.allclose(obs[0], expected, atol=1e-5), f"roll obs 偏差 max={np.abs(obs[0]-expected).max():.2e}"


def test_warmup_action_formula():
    """首 obs 动作历史末槽 = (q-default)/scale, 前 4 槽 = 0。"""
    k = 0
    st = make_state(k, yaw=0.0)
    env = _fresh_env()
    refresh(env, st, _REF_POS[k], _REF_VEL[k], _REF_QUAT[k], np.zeros(NUM_ACTIONS), [0], 0)
    obs = run_obs529(env)

    act_hist = obs[0, ACTION_SLICE].reshape(HISTORY_LENGTH, NUM_ACTIONS)
    expected_warmup = (_REF_POS[k] - DEFAULT_POS) / ACTION_SCALE
    assert np.allclose(act_hist[0:4], 0.0, atol=1e-6)
    assert np.allclose(act_hist[4], expected_warmup, atol=1e-6)


def test_action_history_clipped():
    """滚动时动作历史末槽 = clamp(上一步动作, -clip, clip)。"""
    env = _fresh_env()
    k0 = 0
    st0 = make_state(k0, yaw=0.0)
    refresh(env, st0, _REF_POS[k0], _REF_VEL[k0], _REF_QUAT[k0], np.zeros(NUM_ACTIONS), [0], 0)
    run_obs529(env)

    # 上一步动作含越界 15.0(>clip 10)
    a_prev = np.array([15.0, -3.0] + [0.0] * 27, dtype=np.float32)
    k1 = 3
    st1 = make_state(k1, yaw=0.0)
    refresh(env, st1, _REF_POS[k1], _REF_VEL[k1], _REF_QUAT[k1], a_prev, [1], 1)
    obs = run_obs529(env)
    act_hist = obs[0, ACTION_SLICE].reshape(HISTORY_LENGTH, NUM_ACTIONS)
    assert act_hist[4, 0] == 10.0  # clamp 15 -> 10
    assert act_hist[4, 1] == -3.0


def test_anchor_ori_first_two_columns_corder():
    """anchor_ori 6 元 = R(ref)^T@calib@R(ref') 前两列 C-order(偏航 0 -> calib=I)。"""
    k = 0
    st = make_state(k, yaw=0.0)
    env = _fresh_env()
    refresh(env, st, _REF_POS[k], _REF_VEL[k], _REF_QUAT[k], np.zeros(NUM_ACTIONS), [0], 0)
    obs = run_obs529(env)

    ref_rot = matrix_from_quat_np(_REF_QUAT[k])  # calib=I(robot/ref 偏航均为 0), R(robot)^T=I
    expect = ref_rot[:, :2].reshape(-1).astype(np.float32)
    got = obs[0, COMMAND_DIM : COMMAND_DIM + ANCHOR_DIM]
    assert np.allclose(got, expect, atol=1e-5), f"anchor_ori 偏差 max={np.abs(got-expect).max():.2e}"


def test_batch_shape():
    """N=4 fresh->roll 后仍 (4,529), 且与 SDK 逐 env 一致。"""
    env = _fresh_env(num_envs=4)
    rep = SdkReplica()

    # fresh: 4 env 同帧 0, 同偏航 0
    k0 = 0
    st0 = make_state(k0, 0.0)
    refresh(env, st0, _REF_POS[k0], _REF_VEL[k0], _REF_QUAT[k0],
            np.zeros((4, NUM_ACTIONS)), [0] * 4, 0)
    obs0 = run_obs529(env)
    assert obs0.shape == (4, NUM_OBS)
    rep.warmup_from_state(st0)
    rep.calibrate(st0["robot_quat"], _BODY_QUAT[0, 0])
    rep.update_history_from_state(st0)
    assert np.allclose(obs0[0], rep.build_obs(_REF_POS[k0], _REF_VEL[k0], st0["robot_quat"], _REF_QUAT[k0]), atol=1e-5)

    # roll: 4 env 不同帧/偏航/动作
    ks = [2, 4, 6, 8]
    yaws = [0.0, 0.3, -0.3, 0.6]
    qs = [make_state(ks[i], yaws[i]) for i in range(4)]
    a_prev = np.stack([np.array([float(i + 1)] + [0.0] * 28) for i in range(4)]).astype(np.float32)
    refresh(
        env,
        {k_: np.stack([qs[i][k_] for i in range(4)]) for k_ in ("q", "dq", "gravity", "ang_vel", "robot_quat")},
        np.stack([_REF_POS[k] for k in ks]),
        np.stack([_REF_VEL[k] for k in ks]),
        np.stack([_REF_QUAT[k] for k in ks]),
        a_prev,
        [1] * 4, 1,
    )
    obs = run_obs529(env)
    assert obs.shape == (4, NUM_OBS)

    rep.set_last_action(a_prev[0])
    rep.update_history_from_state(qs[0])
    expected0 = rep.build_obs(_REF_POS[ks[0]], _REF_VEL[ks[0]], qs[0]["robot_quat"], _REF_QUAT[ks[0]])
    assert np.allclose(obs[0], expected0, atol=1e-5)


def test_idempotent_cache():
    """同一步二次调用返回同一 obs(防 recorder/双算推两遍历史)。"""
    env = _fresh_env()
    k0 = 0
    st0 = make_state(k0, yaw=0.0)
    refresh(env, st0, _REF_POS[k0], _REF_VEL[k0], _REF_QUAT[k0], np.zeros(NUM_ACTIONS), [0], 0)
    obs1 = run_obs529(env)
    obs2 = run_obs529(env)  # 同 step -> 缓存, 不再滚
    assert np.array_equal(obs1, obs2)

    # 推进到 step 1 后动作历史真的滚了(末槽从 warmup 变 clamp(0))
    k1 = 1
    st1 = make_state(k1, yaw=0.0)
    refresh(env, st1, _REF_POS[k1], _REF_VEL[k1], _REF_QUAT[k1], np.zeros(NUM_ACTIONS), [1], 1)
    obs3 = run_obs529(env)
    assert not np.array_equal(obs3[0, ACTION_SLICE], obs2[0, ACTION_SLICE]), "动作历史应随 step 滚动"


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
