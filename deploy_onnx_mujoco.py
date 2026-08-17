#!/usr/bin/env python3
"""
纯 Python ONNX → MuJoCo 部署验证脚本

用途：
  将 omni_mimic 训练导出的 ONNX 模型在 MuJoCo 仿真中运行，
  验证 ONNX 模型能否正确驱动机器人执行参考动作。

原理：
  1. 加载 ONNX 模型（含嵌入的 normalizer + policy 网络）
  2. 从 ONNX metadata 读取关节顺序、默认位置、action_scale 等参数
  3. 在 MuJoCo 中加载 MJCF 机器人模型
  4. 每步构建观测（与 high_dynamic_policy.py 一致）→ ONNX 推理 → PD 力矩控制
  5. MuJoCo 可视化窗口实时显示

用法：
    PY=/opt/homebrew/Caskroom/miniforge/base/envs/omni_deploy/bin/python3.11

    # 基本用法（ONNX + NPZ 必须指定）
    $PY deploy_onnx_mujoco.py \
        --onnx path/to/model.onnx \
        --motion path/to/reference.npz

    # 指定动作名
    $PY deploy_onnx_mujoco.py \
        --onnx logs/rsl_rl/jump06/exported/jump06_5000.onnx \
        --motion training_data/跳高06_chr00_training.npz

    # 不弹窗，只跑仿真并打印统计
    $PY deploy_onnx_mujoco.py --onnx model.onnx --motion ref.npz --no_gui

    # 录制视频
    $PY deploy_onnx_mujoco.py --onnx model.onnx --motion ref.npz --record --output video.mp4
"""
import argparse
import os
import sys
from collections import deque

import numpy as np
import onnx
import onnxruntime as ort
import yaml

# ═══════════════════════════════════════════════════════════════
# 关节顺序映射（与 high_dynamic.yaml 中定义一致）
# ═══════════════════════════════════════════════════════════════

# MuJoCo MJCF / env-omni31.yaml 中的 motor（硬件）顺序
MOTOR_JOINT_NAMES = [
    "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint", "elbow_yaw_l_joint",
    "wrist_pitch_l_joint", "wrist_roll_l_joint",
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint", "elbow_yaw_r_joint",
    "wrist_pitch_r_joint", "wrist_roll_r_joint",
]

# ONNX metadata 中的 policy 顺序
POLICY_JOINT_NAMES = [
    "hip_pitch_l_joint", "hip_pitch_r_joint", "waist_yaw_joint",
    "hip_roll_l_joint", "hip_roll_r_joint", "waist_roll_joint",
    "hip_yaw_l_joint", "hip_yaw_r_joint", "waist_pitch_joint",
    "knee_pitch_l_joint", "knee_pitch_r_joint",
    "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",
    "ankle_pitch_l_joint", "ankle_pitch_r_joint",
    "shoulder_roll_l_joint", "shoulder_roll_r_joint",
    "ankle_roll_l_joint", "ankle_roll_r_joint",
    "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_l_joint", "elbow_pitch_r_joint",
    "elbow_yaw_l_joint", "elbow_yaw_r_joint",
    "wrist_pitch_l_joint", "wrist_pitch_r_joint",
    "wrist_roll_l_joint", "wrist_roll_r_joint",
]

NUM_JOINTS = 29

# motor_order → policy_order 的索引映射
POLICY_TO_MOTOR_IDX = np.array(
    [MOTOR_JOINT_NAMES.index(name) for name in POLICY_JOINT_NAMES], dtype=np.int32
)
MOTOR_TO_POLICY_IDX = np.argsort(POLICY_TO_MOTOR_IDX)

# ═══════════════════════════════════════════════════════════════
# 默认参数（与训练配置一致）
# ═══════════════════════════════════════════════════════════════

# 训练用默认关节位置（motor order）
DEFAULT_JOINT_POS_MOTOR = np.array([
    -0.262, 0.0, 0.0, 0.524, -0.262, 0.0,     # 左腿
    -0.262, 0.0, 0.0, 0.524, -0.262, 0.0,      # 右腿
    0.0, 0.0, 0.0,                               # 腰部
    0.300, 0.0, 0.0, -0.700, 0.0, 0.0, 0.0,    # 左臂
    0.300, 0.0, 0.0, -0.700, 0.0, 0.0, 0.0,    # 右臂
], dtype=np.float32)

# 2026-08-18 v3: 改用 <position> 执行器后，PD 增益由 MJCF 内置
# kp/kd 写入 omni_29dof.xml 的 <position kp= kv=> 属性
# 来自 SDK high_dynamic.yaml (策略运行时实际输出值, 非 env-omni31.yaml sim 列)
# deploy 脚本只需设 ctrl = target_pos，MuJoCo 内置位置伺服自动计算力矩
# 以下保留为参考，不再用于力矩计算
_KP_MOTOR_REF = np.array([
    150.0, 150.0, 150.0, 150.0, 30.0, 30.0,
    150.0, 150.0, 150.0, 150.0, 30.0, 30.0,
    150.0, 150.0, 150.0,
    100.0, 100.0, 50.0, 50.0, 50.0, 20.0, 20.0,
    100.0, 100.0, 50.0, 50.0, 50.0, 20.0, 20.0,
], dtype=np.float32)

_KD_MOTOR_REF = np.array([
    5.0, 5.0, 5.0, 5.0, 3.0, 3.0,
    5.0, 5.0, 5.0, 5.0, 3.0, 3.0,
    5.0, 5.0, 5.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
], dtype=np.float32)

# 控制参数
CONTROL_DT = 0.02       # 策略推理间隔 20ms (50Hz)
# DECIMATION 和 SIM_DT 将在加载模型后动态计算
HISTORY_LENGTH = 5      # 历史帧数


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def yaw_quat(q):
    """提取四元数的 yaw 分量，返回 yaw-only 四元数 [w, x, y, z]"""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)


def quat_to_mat(q):
    """四元数 [w,x,y,z] → 3x3 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def load_onnx_metadata(onnx_path):
    """从 ONNX metadata 读取训练参数"""
    model = onnx.load(onnx_path)
    meta = {}
    for entry in model.metadata_props:
        key = entry.key
        value = entry.value
        if "," in value:
            try:
                meta[key] = np.array([float(x) for x in value.split(",")], dtype=np.float32)
            except ValueError:
                meta[key] = value.split(",")
        else:
            try:
                meta[key] = float(value)
            except ValueError:
                meta[key] = value
    return meta


def verify_mujoco_joint_order(model):
    """验证 MuJoCo 模型关节顺序与预期一致"""
    import mujoco
    mjcf_joint_names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        mjcf_joint_names.append(name if name else f"act_{i}")

    # 只比较前 29 个 actuator
    mjcf_29 = mjcf_joint_names[:NUM_JOINTS]
    if mjcf_29 != MOTOR_JOINT_NAMES:
        print("  ⚠ MJCF 关节顺序与预期不一致！")
        print(f"    预期: {MOTOR_JOINT_NAMES[:5]}...")
        print(f"    实际: {mjcf_29[:5]}...")
        return False
    print(f"  ✓ MJCF 关节顺序验证通过 ({NUM_JOINTS} 个关节)")
    return True


# ═══════════════════════════════════════════════════════════════
# 核心：ONNX Policy + 观测构建
# ═══════════════════════════════════════════════════════════════

class OnnxPolicy:
    """封装 ONNX 推理 + 观测构建，与 high_dynamic_policy.py 逻辑一致"""

    def __init__(self, onnx_path, motion_path=None, default_pos_motor=None, action_scale_val=None):
        self.onnx_path = onnx_path
        self.motion_path = motion_path
        self.motion_data = None
        self.motion_total_steps = 0

        # 加载 ONNX metadata
        print(f"\n═══ 加载 ONNX 模型 ═══")
        self.meta = load_onnx_metadata(onnx_path)
        self._print_metadata()

        # 默认位置（policy order）
        # 注意：ONNX metadata 中的 default_joint_pos 是 policy order（与 joint_names 一致）
        if "default_joint_pos" in self.meta and isinstance(self.meta["default_joint_pos"], np.ndarray):
            default_meta = self.meta["default_joint_pos"]
            if len(default_meta) == NUM_JOINTS:
                # metadata 中的 default_joint_pos 已经是 policy order
                self.default_pos_policy = default_meta.copy()
                print(f"  ✓ 使用 ONNX metadata 中的 default_joint_pos (policy order)")
            else:
                self.default_pos_policy = DEFAULT_JOINT_POS_MOTOR[POLICY_TO_MOTOR_IDX]
                print(f"  ⚠ metadata default_joint_pos 长度不匹配，使用硬编码值")
        else:
            self.default_pos_policy = DEFAULT_JOINT_POS_MOTOR[POLICY_TO_MOTOR_IDX]
            print(f"  ⚠ ONNX 无 default_joint_pos metadata，使用硬编码值")

        # action scale（policy order）
        # 注意：ONNX metadata 中的 action_scale 也是 policy order
        if "action_scale" in self.meta and isinstance(self.meta["action_scale"], np.ndarray):
            scale_meta = self.meta["action_scale"]
            if len(scale_meta) == NUM_JOINTS:
                self.action_scale_policy = scale_meta.copy()
                print(f"  ✓ 使用 ONNX metadata 中的 action_scale (policy order): {scale_meta[0]:.3f}")
            else:
                self.action_scale_policy = np.full(NUM_JOINTS, action_scale_val or 0.25, dtype=np.float32)
                print(f"  ⚠ metadata action_scale 长度不匹配，使用默认值")
        else:
            val = action_scale_val if action_scale_val is not None else 0.25
            self.action_scale_policy = np.full(NUM_JOINTS, val, dtype=np.float32)
            print(f"  ⚠ ONNX 无 action_scale metadata，使用默认值 {val}")

        # 避免除零
        self.action_scale_policy[self.action_scale_policy == 0.0] = 1.0

        # 2026-08-18 v3: 改用 <position> 执行器后不再需要外部 PD 计算
        # MJCF 中的 <position kp= kv=> 已内置位置伺服
        print(f"  执行器模式: <position> (MuJoCo 内置位置伺服)")
        print(f"  kp/kd 由 MJCF 属性控制，无需外部 PD 计算")

        # 初始化 ONNX session
        print(f"\n  初始化 ONNXRuntime session...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shapes = {inp.name: inp.shape for inp in self.session.get_inputs()}
        print(f"  ✓ ONNX session 创建成功")
        print(f"    输入: {self.input_names} shapes={self.input_shapes}")
        print(f"    输出: {self.output_names}")

        # 计算观测维度
        self.num_obs = 2 * NUM_JOINTS + 6 + HISTORY_LENGTH * (3 + 3 + 3 * NUM_JOINTS)
        print(f"    预期观测维度: {self.num_obs}")

        # 验证输入维度
        obs_shape = self.input_shapes.get(self.input_names[0], [])
        if len(obs_shape) > 1 and isinstance(obs_shape[1], int):
            if obs_shape[1] != self.num_obs:
                print(f"  ⚠ ONNX 输入维度 {obs_shape[1]} != 预期 {self.num_obs}")
            else:
                print(f"  ✓ ONNX 输入维度匹配")

        # 加载参考动作 NPZ（omni_mimic 框架必须）
        if motion_path and os.path.isfile(motion_path):
            self._load_motion(motion_path)
        else:
            print(f"  ⚠ 未提供 NPZ 参考动作文件，参考轨迹将保持默认站立姿态")
            print(f"     omni_mimic 框架的 ONNX 需要 NPZ 文件才能正确构建观测！")

        # 历史缓冲区
        self.reset()

    def _load_motion(self, motion_path):
        """加载 NPZ 参考动作文件"""
        print(f"\n  加载参考动作: {motion_path}")
        data = np.load(motion_path)
        required = ('joint_pos', 'joint_vel', 'body_quat_w')
        for key in required:
            if key not in data:
                raise ValueError(f"NPZ 缺少必要字段: {key}")

        jp = np.asarray(data['joint_pos'], dtype=np.float32)
        jv = np.asarray(data['joint_vel'], dtype=np.float32)
        bq = np.asarray(data['body_quat_w'], dtype=np.float32)

        if jp.shape[1] != NUM_JOINTS:
            raise ValueError(f"joint_pos 列数 {jp.shape[1]} != {NUM_JOINTS}")

        # joint_pos/joint_vel 是 policy order（与训练一致）
        self.motion_joint_pos = jp
        self.motion_joint_vel = jv
        self.motion_body_quat_w = bq  # (T, num_bodies, 4)
        self.motion_total_steps = jp.shape[0]
        self.num_bodies = bq.shape[1]
        self.anchor_body_index = 0  # base_link

        print(f"  ✓ 参考动作: {self.motion_total_steps} 帧, {self.num_bodies} bodies")
        print(f"    joint_pos range: [{jp.min():.3f}, {jp.max():.3f}]")

    def _get_ref_at_step(self, step):
        """从 NPZ 获取当前帧的参考轨迹（与 SDK high_dynamic_policy.py 一致）"""
        if self.motion_total_steps <= 0:
            return
        idx = min(max(int(step), 0), self.motion_total_steps - 1)
        self.ref_joint_pos = self.motion_joint_pos[idx:idx+1].copy()
        self.ref_joint_vel = self.motion_joint_vel[idx:idx+1].copy()
        self.ref_body_quat_w = self.motion_body_quat_w[idx:idx+1].copy()  # (1, num_bodies, 4)

    def _print_metadata(self):
        """打印 ONNX metadata 摘要"""
        for key in ["joint_names", "default_joint_pos", "action_scale",
                     "observation_names", "body_names", "anchor_body_name"]:
            if key in self.meta:
                val = self.meta[key]
                if isinstance(val, np.ndarray):
                    print(f"  {key}: shape={val.shape}, first5={val[:5]}")
                elif isinstance(val, list):
                    print(f"  {key}: {val[:5]}... (len={len(val)})")
                else:
                    print(f"  {key}: {val}")

    def reset(self):
        """重置所有内部状态"""
        self.step = 0
        self.last_action_policy = np.zeros(NUM_JOINTS, dtype=np.float32)

        # 参考轨迹（从 NPZ 获取，如果没有 NPZ 则用默认站立姿态）
        self.ref_joint_pos = DEFAULT_JOINT_POS_MOTOR[POLICY_TO_MOTOR_IDX].reshape(1, -1).copy()
        self.ref_joint_vel = np.zeros((1, NUM_JOINTS), dtype=np.float32)
        self.ref_body_quat_w = None
        self.world_to_init_rot = np.eye(3, dtype=np.float64)
        self._init_calibrated = False

        # 如果有 NPZ，初始化第一帧参考
        if self.motion_total_steps > 0:
            self._get_ref_at_step(0)

        # 历史缓冲区
        self.gravity_hist = deque(maxlen=HISTORY_LENGTH)
        self.ang_vel_hist = deque(maxlen=HISTORY_LENGTH)
        self.joint_pos_hist = deque(maxlen=HISTORY_LENGTH)
        self.joint_vel_hist = deque(maxlen=HISTORY_LENGTH)
        self.action_hist = deque(maxlen=HISTORY_LENGTH)

        # 用默认值填充历史
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        ang_vel = np.zeros(3, dtype=np.float32)
        joint_pos = np.zeros(NUM_JOINTS, dtype=np.float32)
        joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        for _ in range(HISTORY_LENGTH):
            self.gravity_hist.append(gravity.copy())
            self.ang_vel_hist.append(ang_vel.copy())
            self.joint_pos_hist.append(joint_pos.copy())
            self.joint_vel_hist.append(joint_vel.copy())
            self.action_hist.append(np.zeros(NUM_JOINTS, dtype=np.float32))

    def warmup_from_state(self, q_motor, dq_motor, gravity_ori, ang_vel):
        """用当前真实状态预热历史缓冲区"""
        q_policy = q_motor[POLICY_TO_MOTOR_IDX]
        dq_policy = dq_motor[POLICY_TO_MOTOR_IDX]
        gravity = gravity_ori.astype(np.float32).copy()
        av = ang_vel.astype(np.float32).copy()
        jp = (q_policy - self.default_pos_policy).astype(np.float32)
        jv = dq_policy.astype(np.float32)

        self.gravity_hist.clear()
        self.ang_vel_hist.clear()
        self.joint_pos_hist.clear()
        self.joint_vel_hist.clear()
        self.action_hist.clear()

        for _ in range(HISTORY_LENGTH):
            self.gravity_hist.append(gravity.copy())
            self.ang_vel_hist.append(av.copy())
            self.joint_pos_hist.append(jp.copy())
            self.joint_vel_hist.append(jv.copy())
            self.action_hist.append(np.zeros(NUM_JOINTS, dtype=np.float32))

        # 计算初始 action 使第一帧输出接近当前状态
        scale = self.action_scale_policy.copy()
        scale[scale == 0.0] = 1.0
        self.last_action_policy = ((q_policy - self.default_pos_policy) / scale).astype(np.float32)

    def _calibrate_init_rotation(self, robot_quat):
        """校准初始旋转（yaw-only 对齐）"""
        if self._init_calibrated or self.ref_body_quat_w is None:
            return
        ref_quat = self.ref_body_quat_w[0, self.anchor_body_index]  # shape (4,)
        init_to_anchor_rot = quat_to_mat(yaw_quat(ref_quat))
        world_to_anchor_rot = quat_to_mat(yaw_quat(robot_quat))
        self.world_to_init_rot = world_to_anchor_rot @ init_to_anchor_rot.T
        self._init_calibrated = True

    def _get_anchor_ori_b(self, robot_quat):
        """计算机器人 base 相对于参考的 yaw-only 方向误差（body frame）"""
        if self.ref_body_quat_w is None:
            # 初始状态，无参考轨迹，返回零
            return np.zeros(6, dtype=np.float32)
        ref_quat = self.ref_body_quat_w[0, self.anchor_body_index]  # shape (4,)
        rot_inv = quat_to_mat(robot_quat).T
        ref_rot = quat_to_mat(ref_quat)
        rot_b = rot_inv @ self.world_to_init_rot @ ref_rot
        return rot_b[:, :2].reshape(-1).astype(np.float32)

    def build_observation(self, q_motor, dq_motor, gravity_ori, ang_vel):
        """
        构建观测向量（与 high_dynamic_policy.py 完全一致）

        参数:
            q_motor: 关节位置 (29,) motor order
            dq_motor: 关节速度 (29,) motor order
            gravity_ori: 重力方向在 body frame 的表示 (3,)
            ang_vel: 角速度 (3,)

        返回:
            inputs: dict 用于 ONNX 推理
        """
        # 转为 policy order
        q_policy = q_motor[POLICY_TO_MOTOR_IDX].astype(np.float32)
        dq_policy = dq_motor[POLICY_TO_MOTOR_IDX].astype(np.float32)

        # 更新历史缓冲区
        self.gravity_hist.append(gravity_ori.astype(np.float32).copy())
        self.ang_vel_hist.append(ang_vel.astype(np.float32).copy())
        self.joint_pos_hist.append((q_policy - self.default_pos_policy).astype(np.float32))
        self.joint_vel_hist.append(dq_policy.astype(np.float32))
        self.action_hist.append(self.last_action_policy.astype(np.float32).copy())

        # 构建观测
        command = np.concatenate([
            self.ref_joint_pos.reshape(-1),
            self.ref_joint_vel.reshape(-1),
        ]).astype(np.float32)

        # anchor orientation（需要 robot quat）
        # 这里 gravity_ori 是 body-frame 的重力方向，可以从中推断 base 方向
        # 但更准确的方式是直接用 base 四元数
        # 暂时用简化方式：从重力方向估计
        anchor_ori_obs = np.zeros(6, dtype=np.float32)  # 会在外部用更准确的方式计算

        obs_parts = [
            command,
            anchor_ori_obs,
            np.concatenate(list(self.gravity_hist)),
            np.concatenate(list(self.ang_vel_hist)),
            np.concatenate(list(self.joint_pos_hist)),
            np.concatenate(list(self.joint_vel_hist)),
            np.concatenate(list(self.action_hist)),
        ]
        obs = np.concatenate(obs_parts).astype(np.float32)

        if obs.shape[0] != self.num_obs:
            raise RuntimeError(f"观测维度 {obs.shape[0]} != 预期 {self.num_obs}")

        time_step = np.array([[min(self.step + 1, 99999)]], dtype=np.float32)

        inputs = {self.input_names[0]: obs.reshape(1, -1)}
        if len(self.input_names) > 1:
            inputs[self.input_names[1]] = time_step

        return obs, inputs

    def build_observation_with_base_quat(self, q_motor, dq_motor, base_quat, ang_vel):
        """
        使用 base 四元数构建精确的观测

        参数:
            q_motor: 关节位置 (29,) motor order
            dq_motor: 关节速度 (29,) motor order
            base_quat: base 四元数 [w,x,y,z] 世界坐标系
            ang_vel: 角速度 (3,) body frame
        """
        # 转为 policy order
        q_policy = q_motor[POLICY_TO_MOTOR_IDX].astype(np.float32)
        dq_policy = dq_motor[POLICY_TO_MOTOR_IDX].astype(np.float32)

        # 更新历史缓冲区
        # gravity_ori: 在 body frame 中的重力方向
        base_mat = quat_to_mat(base_quat)
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_body = (base_mat.T @ gravity_world).astype(np.float32)

        self.gravity_hist.append(gravity_body)
        self.ang_vel_hist.append(ang_vel.astype(np.float32).copy())
        self.joint_pos_hist.append((q_policy - self.default_pos_policy).astype(np.float32))
        self.joint_vel_hist.append(dq_policy.astype(np.float32))
        self.action_hist.append(self.last_action_policy.astype(np.float32).copy())

        # anchor orientation
        self._calibrate_init_rotation(base_quat.astype(np.float32))
        anchor_ori_obs = self._get_anchor_ori_b(base_quat.astype(np.float32))

        # 命令
        command = np.concatenate([
            self.ref_joint_pos.reshape(-1),
            self.ref_joint_vel.reshape(-1),
        ]).astype(np.float32)

        obs_parts = [
            command,
            anchor_ori_obs,
            np.concatenate(list(self.gravity_hist)),
            np.concatenate(list(self.ang_vel_hist)),
            np.concatenate(list(self.joint_pos_hist)),
            np.concatenate(list(self.joint_vel_hist)),
            np.concatenate(list(self.action_hist)),
        ]
        obs = np.concatenate(obs_parts).astype(np.float32)

        if obs.shape[0] != self.num_obs:
            raise RuntimeError(f"观测维度 {obs.shape[0]} != 预期 {self.num_obs}")

        time_step = np.array([[min(self.step + 1, 99999)]], dtype=np.float32)

        inputs = {self.input_names[0]: obs.reshape(1, -1)}
        if len(self.input_names) > 1:
            inputs[self.input_names[1]] = time_step

        return obs, inputs

    def get_action(self, inputs):
        """
        ONNX 推理 + action 后处理

        返回:
            actions_motor: 关节目标位置 (29,) motor order
        """
        # 推理
        outputs = self.session.run(self.output_names, inputs)
        output_dict = {name: arr for name, arr in zip(self.output_names, outputs)}

        # 提取 action
        action_policy = output_dict.get("actions", outputs[0])
        action_policy = action_policy.reshape(-1)[:NUM_JOINTS].astype(np.float32)

        # clip
        action_policy = np.clip(action_policy, -10.0, 10.0)
        self.last_action_policy = action_policy.copy()

        # default_pos + action * scale → target（policy order）
        target_policy = self.default_pos_policy + action_policy * self.action_scale_policy

        # policy order → motor order
        actions_motor = np.zeros(NUM_JOINTS, dtype=np.float32)
        actions_motor[POLICY_TO_MOTOR_IDX] = target_policy

        # 推进参考轨迹步数（从 NPZ 读取下一帧）
        self.step += 1
        if self.motion_total_steps > 0:
            self._get_ref_at_step(self.step)

        return actions_motor

    # compute_torques 已移除 — <position> 执行器由 MuJoCo 内置伺服控制


# ═══════════════════════════════════════════════════════════════
# MuJoCo 仿真主循环
# ═══════════════════════════════════════════════════════════════

def run_simulation(policy, model_path, total_steps=None, gui=True):
    """在 MuJoCo 中运行 ONNX 策略"""
    import time
    import mujoco
    import mujoco.viewer  # 显式导入 viewer 模块

    print(f"\n═══ MuJoCo 仿真 ═══")

    # 加载模型
    if not os.path.isfile(model_path):
        print(f"  ✗ 模型文件不存在: {model_path}")
        return None

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    print(f"  模型加载成功: {model_path}")
    print(f"  DOF: {mj_model.nq} qpos, {mj_model.nv} qvel, {mj_model.nu} actuators")

    # 验证关节顺序
    verify_mujoco_joint_order(mj_model)

    # 动态计算 decimation（基于 MJCF 的实际 timestep）
    global DECIMATION, SIM_DT
    physics_dt = mj_model.opt.timestep
    DECIMATION = max(1, round(CONTROL_DT / physics_dt))
    SIM_DT = physics_dt  # 使用 MJCF 的实际 timestep
    print(f"  物理 timestep: {physics_dt}s, 计算 decimation: {DECIMATION}, 控制周期: {CONTROL_DT}s")

    # 初始化机器人姿态（使用默认关节位置）
    default_pos_motor = DEFAULT_JOINT_POS_MOTOR
    mj_data.qpos[7:7 + NUM_JOINTS] = default_pos_motor
    mj_data.qpos[2] = 0.82  # base 初始高度
    mujoco.mj_forward(mj_model, mj_data)

    # 读取初始状态
    q_motor = mj_data.qpos[7:7 + NUM_JOINTS].astype(np.float32)
    dq_motor = mj_data.qvel[6:6 + NUM_JOINTS].astype(np.float32)
    base_quat = mj_data.qpos[3:7].astype(np.float32)
    gyro_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
    if gyro_sid >= 0:
        ang_vel = mj_data.sensordata[mj_model.sensor_adr[gyro_sid]:mj_model.sensor_adr[gyro_sid]+mj_model.sensor_dim[gyro_sid]].astype(np.float32)
    else:
        ang_vel = np.zeros(3, dtype=np.float32)

    # 预热策略
    policy.reset()
    policy.warmup_from_state(q_motor, dq_motor,
                             np.array([0.0, 0.0, -1.0], dtype=np.float32),
                             ang_vel)
    # 先给一个初始参考（通过一次空推理获取 ONNX 输出的第一帧参考）
    _, init_inputs = policy.build_observation_with_base_quat(q_motor, dq_motor, base_quat, ang_vel)
    policy.get_action(init_inputs)
    policy.step = 0  # 重置步数

    print(f"\n  开始仿真 (control_dt={CONTROL_DT}s, decimation={DECIMATION})...")

    # 统计
    heights = []
    actions_log = []
    step_count = 0
    max_steps = total_steps or int(10.0 / CONTROL_DT)  # 默认跑 10 秒

    # MuJoCo 可视化
    realtime_factor = 1.0  # 1.0 = 实时, 0.5 = 半速, 2.0 = 两倍速
    if gui:
        try:
            viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
            # 相机设置：平视、适中距离
            viewer.cam.lookat[:] = [0.0, 0.0, 0.7]   # 看向机器人腰部高度
            viewer.cam.distance = 3.5                  # 距离 3.5m（适中）
            viewer.cam.elevation = -5                  # 略微俯视（-5°，接近平视）
            viewer.cam.azimuth = 90                    # 侧面视角
            print(f"  ✓ MuJoCo 可视化窗口已打开（关闭窗口结束仿真，实时速率={realtime_factor}x）")
        except Exception as e:
            print(f"  ⚠ 无法打开可视化窗口: {e}")
            print("  继续无 GUI 仿真...")
            gui = False

    # 实时同步变量
    sim_start_time = time.time() if gui else None

    while step_count < max_steps:
        # 构建观测
        q_motor = mj_data.qpos[7:7 + NUM_JOINTS].astype(np.float32)
        dq_motor = mj_data.qvel[6:6 + NUM_JOINTS].astype(np.float32)
        base_quat = mj_data.qpos[3:7].astype(np.float32)

        gyro_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
        if gyro_sid >= 0:
            ang_vel = mj_data.sensordata[mj_model.sensor_adr[gyro_sid]:mj_model.sensor_adr[gyro_sid]+mj_model.sensor_dim[gyro_sid]].astype(np.float32).copy()
        else:
            ang_vel = np.zeros(3, dtype=np.float32)

        # 每 DECIMATION 步推理一次
        if step_count % DECIMATION == 0:
            obs, inputs = policy.build_observation_with_base_quat(
                q_motor, dq_motor, base_quat, ang_vel
            )
            target_pos_motor = policy.get_action(inputs)
            actions_log.append(target_pos_motor.copy())

        # 2026-08-18 v3: <position> 执行器 — ctrl 直接设为目标位置
        # MuJoCo 内置位置伺服: force = kp*(ctrl-q) - kv*dq
        mj_data.ctrl[:NUM_JOINTS] = target_pos_motor

        # 仿真步进
        mujoco.mj_step(mj_model, mj_data)
        step_count += 1

        # 记录
        heights.append(mj_data.qpos[2])

        # 可视化更新
        if gui:
            try:
                # 实时同步：让仿真速度匹配真实时间
                sim_time = mj_data.time  # 使用 MuJoCo 实际仿真时间
                wall_elapsed = time.time() - sim_start_time
                target_wall_time = sim_time / realtime_factor
                sleep_time = target_wall_time - wall_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                viewer.sync()
                if not viewer.is_running():
                    break
            except Exception:
                break

    # 统计结果
    heights = np.array(heights)
    actions_log = np.array(actions_log)

    print(f"\n═══ 仿真结果 ═══")
    print(f"  总步数: {step_count}")
    print(f"  仿真时间: {mj_data.time:.2f}s")  # 使用 MuJoCo 实际时间
    print(f"  Base 高度: {heights.min():.3f} ~ {heights.max():.3f} m (均值 {heights.mean():.3f})")
    print(f"  推理次数: {len(actions_log)}")

    if len(actions_log) > 0:
        print(f"\n  动作统计 (motor order):")
        print(f"    均值: {actions_log.mean(axis=0)[:5]}...")
        print(f"    标准差: {actions_log.std(axis=0)[:5]}...")
        print(f"    范围: [{actions_log.min():.3f}, {actions_log.max():.3f}]")

    # 检查异常
    if np.any(np.isnan(mj_data.qpos)):
        print(f"\n  ✗ 检测到 NaN！仿真可能不稳定")
    else:
        print(f"\n  ✓ 仿真完成，无 NaN")

    if heights.max() > 3.0:
        print(f"  ⚠ Base 高度超过 3m，可能飞了")
    if heights.min() < 0.1:
        print(f"  ⚠ Base 高度低于 0.1m，可能摔了")

    # 关闭 viewer
    if gui:
        try:
            viewer.close()
        except Exception:
            pass

    return {
        "heights": heights,
        "actions": actions_log,
        "total_steps": step_count,
    }


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ONNX → MuJoCo 部署验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--onnx", required=True, help="ONNX 模型文件路径")
    parser.add_argument("--motion", default=None,
                        help="参考动作 NPZ 文件路径（omni_mimic 框架必须指定）")
    parser.add_argument("--model", default="omni_29dof_mjc/mjcf/omni_29dof.xml",
                        help="MuJoCo MJCF 模型路径")
    parser.add_argument("--no_gui", action="store_true", help="不打开可视化窗口")
    parser.add_argument("--steps", type=int, default=None,
                        help="总仿真步数（默认 10 秒 = 5000 步）")
    parser.add_argument("--action_scale", type=float, default=None,
                        help="覆盖 action_scale（默认从 ONNX metadata 读取）")
    args = parser.parse_args()

    # 检查文件
    if not os.path.isfile(args.onnx):
        print(f"✗ ONNX 文件不存在: {args.onnx}")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model) if not os.path.isabs(args.model) else args.model
    if not os.path.isfile(model_path):
        print(f"✗ MJCF 模型文件不存在: {model_path}")
        sys.exit(1)

    print(f"  ONNX: {args.onnx}")
    print(f"  MJCF: {model_path}")
    if args.motion:
        print(f"  NPZ:  {args.motion}")

    # 创建策略
    policy = OnnxPolicy(args.onnx, motion_path=args.motion, action_scale_val=args.action_scale)

    # 运行仿真
    result = run_simulation(policy, model_path, total_steps=args.steps, gui=not args.no_gui)

    if result is None:
        print("\n✗ 仿真失败")
        sys.exit(1)
    else:
        print("\n═══ 完成 ═══")


if __name__ == "__main__":
    main()
