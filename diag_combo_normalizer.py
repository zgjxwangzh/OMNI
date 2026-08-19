#!/usr/bin/env python3
"""组合测试: 17700 actor 权重 + policy-lzq 内嵌 normalizer, 完美跟踪 obs。
若误差接近 0 → 两者同源(同一训练run的不同checkpoint), 该组合可用。
"""
import numpy as np
import onnx
from onnx import numpy_helper
from collections import deque
from deploy_onnx_mujoco import OnnxPolicy, NUM_JOINTS, quat_to_mat, HISTORY_LENGTH

LZQ = "/Users/condenast/Downloads/policy-lzq.onnx"
S17 = "/Users/condenast/Downloads/policy-17700step.onnx"
NPZ = "training_data/jump_high_firstjump_50fps.npz"

lzq = {i.name: numpy_helper.to_array(i) for i in onnx.load(LZQ).graph.initializer}
mean = lzq['obs_normalizer._mean'].reshape(-1)   # (529,)
div = lzq['onnx::Div_24'].reshape(-1)            # (529,) = sqrt(var+eps) 或 std

import onnxruntime as ort
sess17 = ort.InferenceSession(S17, providers=["CPUExecutionProvider"])
sess_lzq = ort.InferenceSession(LZQ, providers=["CPUExecutionProvider"])

npz = np.load(NPZ)
jp = np.asarray(npz['joint_pos'], dtype=np.float32)
jv = np.asarray(npz['joint_vel'], dtype=np.float32)
bq = np.asarray(npz['body_quat_w'], dtype=np.float32)
T = jp.shape[0]

# default/scale 用部署硬编码(policy 序)
policy = OnnxPolicy(S17, motion_path=NPZ, action_scale_val=0.5)
default_pos = policy.default_pos_policy
scale = policy.action_scale_policy
policy._calibrate_init_rotation(bq[0, 0])
calib = policy.world_to_init_rot

grav = np.array([0, 0, -1], dtype=np.float32)
angv = np.zeros(3, dtype=np.float32)
h_grav = deque([grav.copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
h_ang = deque([angv.copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
h_jp = deque([(jp[0] - default_pos).copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
h_jv = deque([jv[0].copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
warmup = ((jp[0] - default_pos) / scale).astype(np.float32)
h_act = deque([np.zeros(NUM_JOINTS, dtype=np.float32)] * 4 + [warmup.copy()], maxlen=HISTORY_LENGTH)

def eval_policy(run_fn, label):
    errs = []
    # 重新初始化历史
    h_grav.clear(); h_ang.clear(); h_jp.clear(); h_jv.clear(); h_act.clear()
    for _ in range(HISTORY_LENGTH):
        h_grav.append(grav.copy()); h_ang.append(angv.copy())
        h_jp.append((jp[0] - default_pos).copy()); h_jv.append(jv[0].copy())
        h_act.append(np.zeros(NUM_JOINTS, dtype=np.float32))
    h_act.append(warmup.copy())
    for t in range(T - 1):
        command = np.concatenate([jp[t], jv[t]]).astype(np.float32)
        rot_b = quat_to_mat(bq[t, 0]).T @ calib @ quat_to_mat(bq[t, 0])
        anchor = rot_b[:, :2].reshape(-1).astype(np.float32)
        obs = np.concatenate([command, anchor,
                              np.concatenate(list(h_grav)), np.concatenate(list(h_ang)),
                              np.concatenate(list(h_jp)), np.concatenate(list(h_jv)),
                              np.concatenate(list(h_act))]).astype(np.float32)
        raw = run_fn(obs).astype(np.float32)
        got = default_pos + scale * np.clip(raw, -10, 10)
        errs.append(np.abs(got - jp[t + 1]))
        h_grav.append(grav.copy()); h_ang.append(angv.copy())
        h_jp.append((jp[t + 1] - default_pos).copy()); h_jv.append(jv[t + 1].copy())
        h_act.append(np.clip(raw, -10, 10).astype(np.float32))
    errs = np.asarray(errs)
    print(f"[{label}] mean={errs.mean():.3f} median={np.median(errs):.3f} max={errs.max():.3f} "
          f"前5步={errs[:5].mean(axis=1).round(3)}")
    return errs

def lzq_run(obs):
    out = sess_lzq.run(['actions'], {'obs': obs.reshape(1, -1)})
    return np.asarray(out[0][0])

def combo_run(obs):
    norm = ((obs - mean) / div).astype(np.float32)
    out = sess17.run(['actions'], {'obs': norm.reshape(1, -1)})
    return np.asarray(out[0][0])

def s17_raw_run(obs):
    out = sess17.run(['actions'], {'obs': obs.reshape(1, -1)})
    return np.asarray(out[0][0])

print("完美跟踪测试 (误差=输出目标 vs 下一帧参考, rad):")
eval_policy(lzq_run, "policy-lzq 原始")
eval_policy(combo_run, "17700 + lzq normalizer")
eval_policy(s17_raw_run, "17700 无归一化(对照)")
