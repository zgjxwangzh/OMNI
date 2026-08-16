#!/usr/bin/env python3
"""
高动态动作 MuJoCo 可视化播放（通用脚本）

支持两种模式：
  1. PD 跟踪模式：用 PD 控制器跟踪参考 NPZ（验证动作可行性）
  2. ONNX 策略模式：加载训练好的 ONNX 策略 + training NPZ（验证训练效果）

ONNX 策略模式从 ONNX 元信息自动读取所有参数（joint_names、stiffness、damping、
default_pos、action_scale、body_names），不需要外部 config YAML。

使用方法：
    # PD 跟踪模式（不需要 ONNX）
    mjpython play_high_dynamic.py --motion retargeted/跳高06_chr00.npz

    # ONNX 策略模式（需要训练好的 ONNX + training NPZ）
    mjpython play_high_dynamic.py --onnx jump06.onnx --motion training_data/跳高06_chr00_training.npz

    # 无头模式（不显示窗口，只记录数据）
    mjpython play_high_dynamic.py --motion retargeted/跳高06_chr00.npz --headless

注意：macOS 上必须用 mjpython 而非 python 才能打开可视化窗口。
"""
import argparse
import os
import sys
import time
from collections import deque
import numpy as np
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print(" MuJoCo 未安装，请 pip install mujoco")
    sys.exit(1)


# ── 工具函数 ───────────────────────────────────────────────────────────────

def quat_to_mat(q):
    """[w,x,y,z] → 3×3 旋转矩阵"""
    w, x, y, z = q
    s = 2.0 / (w*w + x*x + y*y + z*z)
    return np.array([
        [1 - s*(y*y+z*z),   s*(x*y - z*w),     s*(x*z + y*w)],
        [s*(x*y + z*w),     1 - s*(x*x+z*z),   s*(y*z - x*w)],
        [s*(x*z - y*w),     s*(y*z + x*w),     1 - s*(x*x+y*y)],
    ])


def yaw_only_quat(q):
    """提取 yaw 分量，返回 [w,0,0,z] 形式的四元数"""
    w, x, y, z = q
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])


def quat_to_yaw(q):
    """从四元数提取 yaw 角（弧度）"""
    w, x, y, z = q
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def gravity_from_quat(q):
    """从四元数计算重力方向（projected gravity）"""
    w, x, y, z = q
    return np.array([
        2*(-z*x + w*y),
        -2*(z*y + w*x),
        1 - 2*(w*w + z*z),
    ], dtype=np.float32)


# ─── ONNX 策略模式 ───────────────────────────────────────────────────────────

def run_onnx_policy(mj_model, mj_data, onnx_path, motion_path=None,
                    sim_dt=0.001, viewer=None, headless=False, realtime=False, slowmo=1.0,
                    decimation=8, total_steps_override=None):
    """ONNX 策略模式：加载 ONNX + training NPZ，在 MuJoCo 中仿真"""
    import onnxruntime as ort

    # 1. 加载 ONNX
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map

    # 解析元信息
    joint_names = meta["joint_names"].split(",")
    body_names = meta["body_names"].split(",")
    stiffness = np.array([float(x) for x in meta["joint_stiffness"].split(",")], dtype=np.float64)
    damping = np.array([float(x) for x in meta["joint_damping"].split(",")], dtype=np.float64)
    default_pos = np.array([float(x) for x in meta["default_joint_pos"].split(",")], dtype=np.float64)
    action_scale = np.array([float(x) for x in meta["action_scale"].split(",")], dtype=np.float64)
    anchor_body_name = meta["anchor_body_name"]

    n_act = len(joint_names)  # 29
    n_bodies = len(body_names)  # 14

    # 验证 ONNX 输入输出维度
    inp = session.get_inputs()
    out = session.get_outputs()
    assert inp[0].shape[1] == 529, f"ONNX obs dim = {inp[0].shape[1]}, expected 529"
    assert out[0].shape[1] == n_act, f"ONNX action dim = {out[0].shape[1]}, expected {n_act}"

    # 构建 body_name → index 映射（用于定位 anchor body）
    body_name_to_idx = {name: i for i, name in enumerate(body_names)}
    anchor_idx = body_name_to_idx[anchor_body_name]

    print(f"✓ ONNX 模型加载: {onnx_path}")
    print(f"  obs={inp[0].shape[1]}, action={out[0].shape[1]}")
    print(f"  joints={n_act}, bodies={n_bodies}")
    print(f"  anchor_body={anchor_body_name} (idx={anchor_idx})")
    print(f"  action_scale={action_scale[0]}")

    # 2. 加载 training NPZ（可选，仅用于确定总帧数和 body 映射）
    if motion_path is not None:
        data = np.load(motion_path)
        for key in ("joint_pos", "joint_vel", "body_quat_w"):
            assert key in data, f"NPZ 缺少 key: {key}"

        ref_joint_pos = data["joint_pos"].astype(np.float64)    # (T, 29) policy order
        ref_joint_vel = data["joint_vel"].astype(np.float64)    # (T, 29) policy order
        ref_body_quat = data["body_quat_w"].astype(np.float64)  # (T, 14, 4)
        T = ref_joint_pos.shape[0]

        assert ref_joint_pos.shape[1] == n_act, f"NPZ joint dim = {ref_joint_pos.shape[1]}, expected {n_act}"
        # 构建 body_name → URDF link index 映射
        urdf_link_names = [
            "ankle_pitch_l_link", "ankle_pitch_r_link", "ankle_roll_l_link", "ankle_roll_r_link",
            "base_link",
            "elbow_pitch_l_link", "elbow_pitch_r_link", "elbow_yaw_l_link", "elbow_yaw_r_link",
            "hip_pitch_l_link", "hip_pitch_r_link", "hip_roll_l_link", "hip_roll_r_link",
            "hip_yaw_l_link", "hip_yaw_r_link", "knee_pitch_l_link", "knee_pitch_r_link",
            "shoulder_pitch_l_link", "shoulder_pitch_r_link", "shoulder_roll_l_link", "shoulder_roll_r_link",
            "shoulder_yaw_l_link", "shoulder_yaw_r_link", "waist_pitch_link", "waist_roll_link", "waist_yaw_link",
            "wrist_pitch_l_link", "wrist_pitch_r_link", "wrist_roll_l_link", "wrist_roll_r_link",
        ]
        urdf_name_to_idx = {name: i for i, name in enumerate(urdf_link_names)}

        npz_body_indices = []
        for name in body_names:
            if name in urdf_name_to_idx:
                npz_body_indices.append(urdf_name_to_idx[name])
            else:
                print(f"  ⚠ body '{name}' 不在 URDF link 列表中，跳过")
        npz_body_indices = np.array(npz_body_indices, dtype=np.int32)

        assert ref_body_quat.shape[1] == len(urdf_link_names), \
            f"NPZ body dim = {ref_body_quat.shape[1]}, expected {len(urdf_link_names)} (URDF links)"
        assert len(npz_body_indices) == n_bodies, \
            f"Mapped {len(npz_body_indices)} bodies, expected {n_bodies}"

        anchor_npz_idx = npz_body_indices[anchor_idx]

        print(f"✓ 参考动作加载: {motion_path}")
        print(f"  帧数={T}, fps≈30, 时长={T/30:.1f}s")
        print(f"  NPZ bodies={ref_body_quat.shape[1]}, ONNX needs={n_bodies}, mapped={len(npz_body_indices)}")
        print(f"  anchor NPZ index={anchor_npz_idx}\n")
    else:
        # 无 NPZ：默认跑 ~13 秒（与跳高训练 NPZ 时长相当）
        T = 400
        print(f"✓ 未提供 NPZ，参考轨迹完全来自 ONNX 输出")
        print(f"  默认仿真帧数={T}（~{T/30:.1f}s）\n")

    # 3. 初始化 MuJoCo（使用 MJCF 默认 timestep=0.001s）
    # sim_dt 保持 MJCF 默认值 0.001s
    motion_dt = 1.0 / 30
    n_substeps = max(1, int(round(motion_dt / sim_dt)))  # 33 substeps per motion frame

    # 设置初始姿态（与组员的 deploy_onnx_mujoco.py 完全一致）
    policy_to_motor = np.argsort(np.array([
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9,
        15, 22, 4, 10, 16, 23, 5, 11,
        17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ]))
    motor_to_policy = np.array([
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9,
        15, 22, 4, 10, 16, 23, 5, 11,
        17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ])

    # 注意：不设置 qpos[3:7]（base 四元数），使用 MJCF 默认 identity
    # 组员脚本也只设置 qpos[2]=0.82 和 qpos[7:36]（关节位置）
    mj_data.qpos[2] = 0.82
    mj_data.qpos[7:7+n_act] = default_pos[policy_to_motor]
    mujoco.mj_forward(mj_model, mj_data)

    # 4. Warmup：用真实状态初始化 history（匹配官方 SDK 的 warmup_from_state）
    hist_len = 5
    gravity_hist = deque(maxlen=hist_len)
    ang_vel_hist = deque(maxlen=hist_len)
    joint_pos_hist = deque(maxlen=hist_len)
    joint_vel_hist = deque(maxlen=hist_len)
    action_hist = deque(maxlen=hist_len)

    # 获取初始状态
    init_base_quat = mj_data.qpos[3:7].astype(np.float64)
    init_ang_vel = mj_data.qvel[3:6].astype(np.float32)
    init_q_motor = mj_data.qpos[7:7+n_act].astype(np.float32)
    init_dq_motor = mj_data.qvel[6:6+n_act].astype(np.float32)
    init_q_policy = init_q_motor[policy_to_motor]
    init_dq_policy = init_dq_motor[policy_to_motor]

    # 计算初始 action（从当前关节位置反推）
    scale_safe = action_scale.copy()
    scale_safe[scale_safe == 0] = 1.0
    init_action_policy = ((init_q_policy - default_pos.astype(np.float32)) / scale_safe).astype(np.float32)

    # 填充 history（与组员的 warmup_from_state 完全一致：真实状态 + action=0）
    init_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    # 从 gyro sensor 读取角速度（与组员 deploy_onnx_mujoco.py 一致）
    _gyro_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
    if _gyro_sid >= 0:
        _adr = mj_model.sensor_adr[_gyro_sid]
        _dim = mj_model.sensor_dim[_gyro_sid]
        init_ang_vel = mj_data.sensordata[_adr:_adr+_dim].astype(np.float32).copy()
    else:
        init_ang_vel = np.zeros(3, dtype=np.float32)
    init_q_policy = init_q_motor[policy_to_motor]
    init_dq_policy = init_dq_motor[policy_to_motor]
    init_joint_pos = (init_q_policy - default_pos.astype(np.float32)).astype(np.float32)
    init_joint_vel = init_dq_policy.astype(np.float32)
    init_action_zero = np.zeros(n_act, dtype=np.float32)

    for _ in range(hist_len):
        gravity_hist.append(init_gravity.copy())
        ang_vel_hist.append(init_ang_vel.copy())
        joint_pos_hist.append(init_joint_pos.copy())
        joint_vel_hist.append(init_joint_vel.copy())
        action_hist.append(init_action_zero.copy())

    # 不做旋转校准（与组员的 deploy_onnx_mujoco.py 一致，不使用 NPZ 的 yaw）
    world_to_init_yaw = 0.0

    if motion_path is not None:
        print(f"✓ 参考动作加载：{motion_path}")
        print(f"  帧数={T}, fps≈30, 时长={T/30:.1f}s")
        print(f"  NPZ bodies={ref_body_quat.shape[1]}, ONNX needs={n_bodies}, mapped={len(npz_body_indices)}")
        print(f"  anchor NPZ index={anchor_npz_idx}")
    else:
        print(f"✓ 参考轨迹完全来自 ONNX 输出")
        print(f"  仿真帧数={T}（~{T/30:.1f}s）")
    print(f"  warmup: history 已用真实状态初始化")
    print(f"  旋转校准：已禁用（与组员一致）\n")

    # 5. 初始推理：从 ONNX 输出获取首帧参考轨迹（匹配组员的 deploy_onnx_mujoco.py）
    print(f"  执行初始推理获取 ONNX 参考轨迹...")
    init_gravity_obs = gravity_from_quat(init_base_quat).astype(np.float32)
    # 用默认参考值做第一次推理
    ref_jp_onnx = default_pos.astype(np.float32).copy()  # policy order
    ref_jv_onnx = np.zeros(n_act, dtype=np.float32)
    ref_quat_onnx = init_base_quat.astype(np.float32).copy()

    # 构建初始 obs（anchor_ori 使用完整 ref_quat，与组员 _get_anchor_ori_b 一致）
    ref_rot = quat_to_mat(ref_quat_onnx)
    robot_rot_inv = quat_to_mat(init_base_quat).T
    init_rot_mat = np.eye(3)  # world_to_init_yaw=0 → identity
    anchor_ori_b = (robot_rot_inv @ init_rot_mat @ ref_rot)[:, :2].reshape(-1).astype(np.float32)

    command_init = np.concatenate([ref_jp_onnx, ref_jv_onnx]).astype(np.float32)
    obs_init = np.concatenate([
        command_init,                       # 58
        anchor_ori_b,                       # 6
        np.concatenate(list(gravity_hist)), # 15
        np.concatenate(list(ang_vel_hist)), # 15
        np.concatenate(list(joint_pos_hist)),  # 145
        np.concatenate(list(joint_vel_hist)),  # 145
        np.concatenate(list(action_hist)),     # 145
    ]).astype(np.float32)

    outputs_init = session.run(None, {"obs": obs_init.reshape(1, -1), "time_step": np.array([[1]], dtype=np.float32)})
    out_names = [o.name for o in session.get_outputs()]
    output_dict_init = {n: a for n, a in zip(out_names, outputs_init)}

    # 从 ONNX 输出提取参考轨迹（policy order）
    if "joint_pos" in output_dict_init:
        ref_jp_onnx = output_dict_init["joint_pos"].reshape(-1)[:n_act].astype(np.float32)
    if "joint_vel" in output_dict_init:
        ref_jv_onnx = output_dict_init["joint_vel"].reshape(-1)[:n_act].astype(np.float32)
    if "body_quat_w" in output_dict_init:
        bq = output_dict_init["body_quat_w"]
        if bq.ndim == 2:
            bq = bq.reshape(1, -1, 4)
        ref_quat_onnx = bq[0, 0].astype(np.float32)

    print(f"  ✓ ONNX 参考轨迹已获取: ref_jp={ref_jp_onnx[:3]}, ref_quat={ref_quat_onnx}")

    # 6. 仿真循环（三层解耦：物理 1000Hz / 策略 50Hz / 参考轨迹来自 ONNX 输出）
    heights = []
    print(f"\n═══ ONNX 策略模式 ═══")
    print(f"  sim_dt={sim_dt}s, decimation={decimation}, 策略频率={1/(sim_dt*decimation):.0f}Hz")
    print(f"  参考轨迹: 来自 ONNX 输出（与组员 deploy_onnx_mujoco.py 一致）\n")

    last_action_policy = init_action_policy.copy()
    target_motor = (default_pos + ref_jp_onnx[policy_to_motor] * 0).astype(np.float64)  # 初始目标
    ref_dt = 1.0 / 30  # NPZ 帧率，仅用于计算总步数
    if total_steps_override is not None:
        total_steps = int(total_steps_override * ref_dt / sim_dt)
        print(f"  总仿真帧数: {total_steps_override}（--steps 覆盖）")
    else:
        total_steps = int(T * ref_dt / sim_dt)  # 总物理步数

    for step in range(total_steps):
        # 当前状态
        q_motor = mj_data.qpos[7:7+n_act].astype(np.float64)
        dq_motor = mj_data.qvel[6:6+n_act].astype(np.float64)
        base_quat = mj_data.qpos[3:7].astype(np.float64)

        # 转换到 policy order
        q_policy = q_motor[policy_to_motor]
        dq_policy = dq_motor[policy_to_motor]

        # 读取角速度（每步，与组员一致：从 gyro sensor 读取）
        _gyro_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
        if _gyro_sid >= 0:
            _adr = mj_model.sensor_adr[_gyro_sid]
            _dim = mj_model.sensor_dim[_gyro_sid]
            ang_vel = mj_data.sensordata[_adr:_adr+_dim].astype(np.float32).copy()
        else:
            ang_vel = np.zeros(3, dtype=np.float32)

        # 更新历史缓冲区（每物理步，与组员 build_observation_with_base_quat 一致）
        _base_mat = quat_to_mat(base_quat)
        _gravity_body = (_base_mat.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)
        gravity_hist.append(_gravity_body)
        ang_vel_hist.append(ang_vel)
        joint_pos_hist.append((q_policy - default_pos.astype(np.float32)))
        joint_vel_hist.append(dq_policy.astype(np.float32))
        action_hist.append(last_action_policy.astype(np.float32))

        # 策略调用（每 decimation 步）
        if step % decimation == 0:
            # 构建 obs（参考轨迹来自 ONNX 输出，不用 NPZ）
            command = np.concatenate([ref_jp_onnx, ref_jv_onnx]).astype(np.float32)

            # anchor_ori_b（使用完整 ref_quat，与组员 _get_anchor_ori_b 一致）
            ref_rot = quat_to_mat(ref_quat_onnx)
            robot_rot_inv = quat_to_mat(base_quat).T
            init_rot_mat = np.eye(3)  # world_to_init_yaw=0 → identity
            anchor_ori_b = (robot_rot_inv @ init_rot_mat @ ref_rot)[:, :2].reshape(-1).astype(np.float32)

            # 拼装 obs
            obs = np.concatenate([
                command,                        # 58
                anchor_ori_b,                   # 6
                np.concatenate(list(gravity_hist)),    # 15
                np.concatenate(list(ang_vel_hist)),    # 15
                np.concatenate(list(joint_pos_hist)),  # 145
                np.concatenate(list(joint_vel_hist)),  # 145
                np.concatenate(list(action_hist)),     # 145
            ]).astype(np.float32)

            assert obs.shape[0] == 529, f"obs dim = {obs.shape[0]}, expected 529"

            # ONNX 推理
            time_step = np.array([[step + 1]], dtype=np.float32)
            outputs = session.run(None, {"obs": obs.reshape(1, -1), "time_step": time_step})
            out_names = [o.name for o in session.get_outputs()]
            output_dict = {n: a for n, a in zip(out_names, outputs)}

            action_policy = output_dict.get("actions", outputs[0])
            action_policy = action_policy.reshape(-1)[:n_act].astype(np.float32)
            action_policy = np.clip(action_policy, -10.0, 10.0)
            last_action_policy = action_policy.copy()

            # 目标位置 = default + action * scale（policy order → motor order）
            target_pos_policy = default_pos.astype(np.float32) + action_policy * action_scale.astype(np.float32)
            target_motor = target_pos_policy[policy_to_motor].astype(np.float64)

            # 从 ONNX 输出更新参考轨迹（policy order）
            if "joint_pos" in output_dict:
                ref_jp_onnx = output_dict["joint_pos"].reshape(-1)[:n_act].astype(np.float32)
            if "joint_vel" in output_dict:
                ref_jv_onnx = output_dict["joint_vel"].reshape(-1)[:n_act].astype(np.float32)
            if "body_quat_w" in output_dict:
                bq = output_dict["body_quat_w"]
                if bq.ndim == 2:
                    bq = bq.reshape(1, -1, 4)
                ref_quat_onnx = bq[0, 0].astype(np.float32)

        # PD 控制（每物理步更新）
        q_error = target_motor - q_motor
        tau = stiffness * q_error - damping * dq_motor
        mj_data.ctrl[:] = tau

        # 物理步进
        mujoco.mj_step(mj_model, mj_data)

        # 记录和可视化（每 motion frame 一次）
        if step % n_substeps == 0:
            frame_idx = step // n_substeps
            heights.append(mj_data.qpos[2])
            if viewer is not None:
                viewer.sync()
                if realtime:
                    time.sleep(slowmo / 30)

            if (frame_idx + 1) % 50 == 0:
                h = mj_data.qpos[2]
                act_norm = np.linalg.norm(last_action_policy)
                print(f"  帧 {frame_idx+1}/{T}  高度={h:.3f}m  action_norm={act_norm:.3f}")

            if mj_data.qpos[2] < 0.2:
                print(f"\n  ⚠ 机器人摔倒 @ 帧 {frame_idx+1}，高度={mj_data.qpos[2]:.3f}m")
                break

    # 统计
    heights = np.array(heights)
    print(f"\n═══ 结果统计 ═══")
    print(f"  仿真帧数: {len(heights)}/{T}")
    print(f"  高度范围: {heights.min():.3f}m → {heights.max():.3f}m")
    print(f"  初始高度: {heights[0]:.3f}m")
    print(f"  最终高度: {heights[-1]:.3f}m")

    if heights.min() < 0.3:
        print(f"  ⚠ 机器人摔倒了")
    elif heights.max() > 2.0:
        print(f"  ⚠ 机器人飞太高了")
    else:
        print(f"  ✓ 机器人保持直立")


# ─── PD 跟踪模式 ────────────────────────────────────────────────────────────

# policy order → motor order
MOTOR_TO_POLICY_IDX = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9,
    15, 22, 4, 10, 16, 23, 5, 11,
    17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])
POLICY_TO_MOTOR_IDX = np.argsort(MOTOR_TO_POLICY_IDX)


def run_pd_tracking(mj_model, mj_data, joint_pos, body_quat_w,
                    kp_scale=1.0, kd_scale=1.0, sim_dt=0.002,
                    viewer=None, headless=False, realtime=False, slowmo=1.0):
    """PD 跟踪模式：用 PD 控制器跟踪参考 NPZ"""
    T = joint_pos.shape[0]
    n_act = mj_model.nu

    kp_base = np.array([
        150.0, 150.0, 150.0, 150.0, 30.0, 30.0,
        150.0, 150.0, 150.0, 150.0, 30.0, 30.0,
        150.0, 150.0, 150.0,
        100.0, 100.0, 50.0, 50.0, 50.0, 20.0, 20.0,
        100.0, 100.0, 50.0, 50.0, 50.0, 20.0, 20.0,
    ], dtype=np.float64)
    kd_base = np.array([
        5.0, 5.0, 5.0, 5.0, 3.0, 3.0,
        5.0, 5.0, 5.0, 5.0, 3.0, 3.0,
        5.0, 5.0, 5.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
    ], dtype=np.float64)

    kp = kp_base * kp_scale
    kd = kd_base * kd_scale

    # 初始化
    if body_quat_w is not None:
        mj_data.qpos[3:7] = body_quat_w[0, 0]
    else:
        mj_data.qpos[3:7] = [1, 0, 0, 0]
    mj_data.qpos[7:7+n_act] = joint_pos[0][POLICY_TO_MOTOR_IDX]
    mujoco.mj_forward(mj_model, mj_data)

    motion_dt = 1.0 / 30
    n_substeps = max(1, int(round(motion_dt / sim_dt)))
    mj_model.opt.timestep = sim_dt

    heights = []
    tracking_errors = []

    print(f"═══ PD 跟踪模式 ═══")
    print(f"  帧数: {T}, sim_dt={sim_dt}s, substeps={n_substeps}")
    print(f"  kp_scale={kp_scale}, kd_scale={kd_scale}\n")

    for frame_idx in range(T):
        ref_motor = joint_pos[frame_idx][POLICY_TO_MOTOR_IDX]

        q_current = mj_data.qpos[7:7+n_act]
        dq_current = mj_data.qvel[6:6+n_act]
        q_error = ref_motor - q_current
        tau = kp * q_error - kd * dq_current

        max_torque = np.array([
            140, 140, 90, 140, 50, 50,
            140, 140, 90, 140, 50, 50,
            90, 50, 50,
            25, 25, 25, 25, 25, 10, 10,
            25, 25, 25, 25, 25, 10, 10,
        ], dtype=np.float64)
        tau = np.clip(tau, -max_torque, max_torque)
        mj_data.ctrl[:] = tau

        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        q_after = mj_data.qpos[7:7+n_act]
        frame_error = np.sqrt(np.mean((ref_motor - q_after) ** 2))
        tracking_errors.append(frame_error)
        heights.append(mj_data.qpos[2])

        if viewer is not None:
            viewer.sync()
            if realtime:
                time.sleep(slowmo / 30)
            else:
                time.sleep(sim_dt * n_substeps)

        if (frame_idx + 1) % 50 == 0:
            h = mj_data.qpos[2]
            print(f"  帧 {frame_idx+1}/{T}  高度={h:.3f}m  误差={frame_error:.3f}rad")

        if mj_data.qpos[2] < 0.2:
            print(f"\n  ⚠ 机器人摔倒 @ 帧 {frame_idx+1}")
            break

    heights = np.array(heights)
    tracking_errors = np.array(tracking_errors)

    print(f"\n═══ 结果统计 ═══")
    print(f"  平均跟踪误差: {tracking_errors.mean():.3f} rad")
    print(f"  最大跟踪误差: {tracking_errors.max():.3f} rad")
    print(f"  高度范围: {heights.min():.3f}m → {heights.max():.3f}m")

    if tracking_errors.mean() > 0.5:
        print(f"  ⚠ 跟踪误差大（PD 基线预期结果，需要 RL 训练）")
    else:
        print(f"  ✓ 跟踪良好")


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="高动态动作 MuJoCo 可视化播放")
    parser.add_argument("--model", default="omni_29dof_mjc/mjcf/omni_29dof.xml")
    parser.add_argument("--motion", default=None, help="参考动作 NPZ 路径（ONNX 模式下可选）")
    parser.add_argument("--onnx", default=None, help="ONNX 策略路径（不提供则用 PD 模式）")
    parser.add_argument("--steps", type=int, default=None, help="ONNX 模式总仿真帧数（无 NPZ 时默认 400）")
    parser.add_argument("--kp_scale", type=float, default=1.0, help="PD 刚度缩放")
    parser.add_argument("--kd_scale", type=float, default=1.0, help="PD 阻尼缩放")
    parser.add_argument("--sim_dt", type=float, default=0.001, help="仿真步长（默认 0.001s 匹配 MJCF）")
    parser.add_argument("--decimation", type=int, default=20, help="策略 decimation（默认 20，0.001s*20=0.02s=50Hz）")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--realtime", action="store_true", help="实时速度（每帧等待 ~33ms）")
    parser.add_argument("--slowmo", type=float, default=1.0, help="慢动作倍率（3.0 = 3 倍慢放，仅 realtime 模式生效）")
    parser.add_argument("--pause", action="store_true", help="仿真结束后保持窗口打开（按 Q 关闭）")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载 MuJoCo 模型
    model_path = os.path.join(script_dir, args.model)
    if not os.path.isfile(model_path):
        print(f"✗ 模型文件不存在: {model_path}")
        sys.exit(1)

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    print(f"✓ MuJoCo 模型加载: {mj_model.nq} qpos, {mj_model.nu} actuators")
    print(f"  重力: {mj_model.opt.gravity}\n")

    # 加载参考动作（ONNX 模式下可选）
    motion_path = None
    if args.motion:
        motion_path = os.path.join(script_dir, args.motion)
        if not os.path.isfile(motion_path):
            print(f"✗ 动作文件不存在: {motion_path}")
            sys.exit(1)

    if args.onnx:
        # ONNX 模式：NPZ 可选
        pass
    elif motion_path is None:
        print(f"✗ PD 跟踪模式需要 --motion 参数")
        sys.exit(1)
    else:
        # PD 模式：必须加载 NPZ
        test_data = np.load(motion_path)
        if "joint_pos" in test_data and "joint_vel" in test_data and "body_quat_w" in test_data:
            use_training_format = True
            print(f"✓ 检测到 training NPZ 格式")
        else:
            use_training_format = False
            print(f"✓ 检测到 retargeted NPZ 格式")

        if use_training_format:
            joint_pos = test_data["joint_angles"] if "joint_angles" in test_data else test_data["joint_pos"]
            joint_vel = test_data.get("joint_vel", np.zeros_like(joint_pos))
            body_quat_w = test_data.get("body_quat_w", None)
        else:
            joint_pos = test_data["joint_angles"]
            joint_vel = np.zeros_like(joint_pos)
            body_quat_w = None

        print(f"  帧数: {joint_pos.shape[0]}, 关节数: {joint_pos.shape[1]}\n")

    # 启动可视化
    viewer = None
    if not args.headless:
        try:
            viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
            print("✓ MuJoCo 可视化窗口已打开")

            # 设置 45 度视角（兼顾高度和姿态观察）
            viewer.cam.azimuth = 135     # 45 度侧后方
            viewer.cam.elevation = -20   # 俯视
            viewer.cam.distance = 3.5    # 距离
            viewer.cam.lookat[:] = [0, 0, 0.8]  # 看向机器人中心
            viewer.sync()
            print("  视角: 45 度侧后方 (azimuth=135, elevation=-20)\n")
        except Exception as e:
            print(f" 无法打开可视化窗口: {e}")
            print("  切换到无头模式（或用 mjpython 运行）\n")
            args.headless = True

    # 运行仿真
    if args.onnx:
        onnx_path = os.path.join(script_dir, args.onnx)
        if not os.path.isfile(onnx_path):
            print(f"✗ ONNX 文件不存在: {onnx_path}")
            sys.exit(1)
        run_onnx_policy(mj_model, mj_data, onnx_path, motion_path,
                       sim_dt=args.sim_dt, viewer=viewer, headless=args.headless, realtime=args.realtime, slowmo=args.slowmo, decimation=args.decimation,
                       total_steps_override=args.steps)
    else:
        run_pd_tracking(mj_model, mj_data, joint_pos, body_quat_w,
                       kp_scale=args.kp_scale, kd_scale=args.kd_scale,
                       sim_dt=args.sim_dt, viewer=viewer, headless=args.headless, realtime=args.realtime, slowmo=args.slowmo)

    if viewer is not None:
        if args.pause:
            print("\n仿真结束。窗口保持打开，按 Q 或关闭窗口退出...")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
        viewer.close()
        print("\n✓ 可视化窗口已关闭")


if __name__ == "__main__":
    main()
