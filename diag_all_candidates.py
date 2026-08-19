#!/usr/bin/env python3
"""对全部 v11 架构候选(529维输入)做完美跟踪+相位扫描:
- policy1.onnx (自带 normalizer, 与 lzq 相同统计)
- policy-lzq.onnx (自带)
- policy.onnx + lzq normalizer (cos 0.993 ~ 17700)
- 17700 + lzq normalizer
判据: 任一相位 k 下平均误差 < 0.15 rad → 该策略为有效收敛策略
"""
import numpy as np
import onnx
from onnx import numpy_helper
from collections import deque
import onnxruntime as ort
from deploy_onnx_mujoco import NUM_JOINTS, quat_to_mat, HISTORY_LENGTH, DEFAULT_JOINT_POS_MOTOR, POLICY_TO_MOTOR_IDX

NPZ = "training_data/jump_high_firstjump_50fps.npz"
LZQ = "/Users/condenast/Downloads/policy-lzq.onnx"
_lzq = {i.name: numpy_helper.to_array(i) for i in onnx.load(LZQ).graph.initializer}
MEAN = _lzq['obs_normalizer._mean'].reshape(-1).astype(np.float32)
DIV = _lzq['onnx::Div_24'].reshape(-1).astype(np.float32)

npz = np.load(NPZ)
jp = np.asarray(npz['joint_pos'], dtype=np.float32)
jv = np.asarray(npz['joint_vel'], dtype=np.float32)
bq = np.asarray(npz['body_quat_w'], dtype=np.float32)
T = jp.shape[0]

# policy 序 default (与 SDK_DEFAULT_POS 一致)
DEFAULT_POLICY = np.zeros(NUM_JOINTS, dtype=np.float32)
DEFAULT_POLICY[:] = DEFAULT_JOINT_POS_MOTOR[POLICY_TO_MOTOR_IDX]
SCALE = np.full(NUM_JOINTS, 0.5, dtype=np.float32)

# calib: robot=ref帧0 → I
angv = np.zeros(3, dtype=np.float32)
# 逐帧 body-frame 重力 = R(ref)^T @ [0,0,-1] (参考姿态有俯仰时不再是常量)
_Rs = np.array([quat_to_mat(bq[t, 0]) for t in range(T)])           # (T,3,3)
GRAV_B = (_Rs.transpose(0, 2, 1) @ np.array([0, 0, -1], dtype=np.float32)).astype(np.float32)


def rollout_targets(run_fn):
    h_grav = deque([GRAV_B[0].copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
    h_ang = deque([angv.copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
    h_jp = deque([(jp[0] - DEFAULT_POLICY).copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
    h_jv = deque([jv[0].copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
    warmup = ((jp[0] - DEFAULT_POLICY) / SCALE).astype(np.float32)
    h_act = deque([np.zeros(NUM_JOINTS, dtype=np.float32)] * 4 + [warmup.copy()], maxlen=HISTORY_LENGTH)
    targets = []
    for t in range(T - 1):
        command = np.concatenate([jp[t], jv[t]]).astype(np.float32)
        Rt = quat_to_mat(bq[t, 0])
        anchor = (Rt.T @ Rt)[:, :2].reshape(-1).astype(np.float32)  # calib=I → R^T R = I
        obs = np.concatenate([command, anchor,
                              np.concatenate(list(h_grav)), np.concatenate(list(h_ang)),
                              np.concatenate(list(h_jp)), np.concatenate(list(h_jv)),
                              np.concatenate(list(h_act))]).astype(np.float32)
        raw = run_fn(obs).astype(np.float32)
        targets.append((DEFAULT_POLICY + SCALE * np.clip(raw, -10, 10)).copy())
        h_grav.append(GRAV_B[t + 1].copy()); h_ang.append(angv.copy())
        h_jp.append((jp[t + 1] - DEFAULT_POLICY).copy()); h_jv.append(jv[t + 1].copy())
        h_act.append(np.clip(raw, -10, 10).astype(np.float32))
    return np.asarray(targets)


def make_run(path, ext_norm=False):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    def run(obs):
        if ext_norm:
            obs = ((obs - MEAN) / DIV).astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: obs.reshape(1, -1)})
        return np.asarray(out[0][0])
    return run


CANDIDATES = [
    ("policy1.onnx (自带norm)", "/Users/condenast/Downloads/policy1.onnx", False),
    ("policy-lzq (自带norm)", LZQ, False),
    ("policy.onnx + lzq norm", "/Users/condenast/Downloads/policy.onnx", True),
    ("17700 + lzq norm", "/Users/condenast/Downloads/policy-17700step.onnx", True),
]

for label, path, ext in CANDIDATES:
    tgt = rollout_targets(make_run(path, ext))
    errs = [np.abs(tgt[:T - 1 - k] - jp[k:T - 1]).mean() for k in range(6)]
    best_k = int(np.argmin(errs))
    verdict = "<<< 有效!" if errs[best_k] < 0.15 else ""
    print(f"{label:<28} 相位误差={['%.2f' % e for e in errs]}  best k={best_k} ({errs[best_k]:.3f}) {verdict}")
