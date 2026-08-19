#!/usr/bin/env python3
"""L1 站立诊断: 逐秒记录 base 高度/倾角/动作幅度, 定位摔倒时刻与行为模式"""
import numpy as np
import mujoco
from collections import deque

from deploy_onnx_mujoco import (
    OnnxPolicy, NUM_JOINTS, DEFAULT_JOINT_POS_MOTOR, CONTROL_DT,
    quat_to_mat, POLICY_TO_MOTOR_IDX,
)

ONNX = "/Users/condenast/Downloads/policy-17700step.onnx"  # v11 权重 + 外部 lzq normalizer
LZQ_ONNX = "/Users/condenast/Downloads/policy-lzq.onnx"      # 仅用于提取 normalizer
NPZ = "training_data/jump_high_firstjump_50fps.npz"
MJCF = "omni_29dof_mjc/mjcf/omni_29dof.xml"
DELAY = 3
TOTAL_S = 8.0

policy = OnnxPolicy(ONNX, motion_path=NPZ, action_scale_val=0.5)

# --- 外挂 normalizer: 从 policy-lzq 提取 mean/div, 包住 get_action ---
import onnx as _onnx
from onnx import numpy_helper as _nh
_lzq_inits = {i.name: _nh.to_array(i) for i in _onnx.load(LZQ_ONNX).graph.initializer}
_norm_mean = _lzq_inits['obs_normalizer._mean'].reshape(-1).astype(np.float32)
_norm_div = _lzq_inits['onnx::Div_24'].reshape(-1).astype(np.float32)
_orig_get_action = policy.get_action
def _normalized_get_action(inputs):
    key = policy.input_names[0]
    inputs = dict(inputs)
    inputs[key] = ((inputs[key] - _norm_mean) / _norm_div).astype(np.float32)
    return _orig_get_action(inputs)
policy.get_action = _normalized_get_action
print("[diag] 已外挂 lzq normalizer 到 17700 策略")
mj_model = mujoco.MjModel.from_xml_path(MJCF)
mj_data = mujoco.MjData(mj_model)
DECIMATION = max(1, round(CONTROL_DT / mj_model.opt.timestep))

# 训练时每回合重置到参考帧0姿态; 部署必须同样从 NPZ 帧0 初始化(policy序 → motor序)
npz_init = np.load(NPZ)
q0_policy = npz_init['joint_pos'][0].astype(np.float32)
q0_motor = np.zeros(NUM_JOINTS, dtype=np.float32)
q0_motor[POLICY_TO_MOTOR_IDX] = q0_policy

mj_data.qpos[7:7 + NUM_JOINTS] = q0_motor
mj_data.qpos[2] = float(npz_init['body_pos_w'][0, 0, 2])  # 参考帧0 base 高度
mj_data.qpos[3:7] = npz_init['body_quat_w'][0, 0].astype(np.float64)  # 参考帧0 base 姿态
mujoco.mj_forward(mj_model, mj_data)

gyro_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
def read_ang_vel():
    if gyro_sid >= 0:
        adr = mj_model.sensor_adr[gyro_sid]
        dim = mj_model.sensor_dim[gyro_sid]
        return mj_data.sensordata[adr:adr + dim].astype(np.float32).copy()
    return np.zeros(3, dtype=np.float32)

q = mj_data.qpos[7:7 + NUM_JOINTS].astype(np.float32)
dq = mj_data.qvel[6:6 + NUM_JOINTS].astype(np.float32)
bq = mj_data.qpos[3:7].astype(np.float32)
policy.reset()
policy.warmup_from_state(q, dq, np.array([0, 0, -1], dtype=np.float32), read_ang_vel())
_, init_inputs = policy.build_observation_with_base_quat(q, dq, bq, read_ang_vel())
policy.get_action(init_inputs)
policy.step = 0

ctrl_buf = deque()
applied = q0_motor.copy()  # 缓冲未填满前保持参考帧0姿态
max_steps = int(TOTAL_S / mj_model.opt.timestep)
step = 0
sec_h, sec_tilt, sec_act_std = [], [], []
fall_step = None

while step < max_steps:
    q = mj_data.qpos[7:7 + NUM_JOINTS].astype(np.float32)
    dq = mj_data.qvel[6:6 + NUM_JOINTS].astype(np.float32)
    bq = mj_data.qpos[3:7].astype(np.float32)
    ang_vel = read_ang_vel()

    if step % DECIMATION == 0:
        _, inputs = policy.build_observation_with_base_quat(q, dq, bq, ang_vel)
        tgt = policy.get_action(inputs)
        if DELAY > 0:
            ctrl_buf.append(tgt.copy())
            if len(ctrl_buf) > DELAY:
                applied = ctrl_buf.popleft()
        else:
            applied = tgt
        sec_act_std.append(np.std(applied - DEFAULT_JOINT_POS_MOTOR))

    mj_data.ctrl[:NUM_JOINTS] = applied
    mujoco.mj_step(mj_model, mj_data)
    step += 1

    h = mj_data.qpos[2]
    sec_h.append(h)
    # base 倾角: z 轴与世界 z 的夹角
    R = quat_to_mat(mj_data.qpos[3:7])
    sec_tilt.append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    if fall_step is None and h < 0.4:
        fall_step = step
        print(f"  >>> 首次低于 0.4m: t={step * mj_model.opt.timestep:.2f}s (step {step})")

sec_len = int(1.0 / mj_model.opt.timestep)
print(f"\n  {'秒':>4} | {'高度均值':>8} | {'高度最小':>8} | {'倾角最大':>8} | {'动作std':>8}")
for i in range(int(TOTAL_S)):
    sl = slice(i * sec_len, (i + 1) * sec_len)
    hh, tt = np.array(sec_h)[sl], np.array(sec_tilt)[sl]
    aa = np.array(sec_act_std)[i * sec_len // DECIMATION:(i + 1) * sec_len // DECIMATION]
    print(f"  {i:>4} | {hh.mean():>8.3f} | {hh.min():>8.3f} | {tt.max():>7.1f}° | {aa.mean() if len(aa) else 0:>8.3f}")

print(f"\n  摔倒时刻: {'t=%.2fs' % (fall_step * mj_model.opt.timestep) if fall_step else '未摔倒'}")
