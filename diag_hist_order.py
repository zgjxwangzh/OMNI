#!/usr/bin/env python3
"""完美跟踪变体穷举: 在列序已定案(NPZ=policy/BFS序)基础上,
系统测试 obs 构造的剩余自由度:
  V1: 历史旧→新 (当前实现)
  V2: 历史新→旧 (帧序反转)
  V3: 无归一化直灌 (对照)
  V4: anchor_ori 用真实 calib(R_z yaw) 而非 I
判据: 平均误差 < 0.15 rad
"""
import numpy as np
import onnx
from onnx import numpy_helper
from collections import deque
import onnxruntime as ort
from deploy_onnx_mujoco import NUM_JOINTS, quat_to_mat, HISTORY_LENGTH

NPZ = "training_data/jump_high_firstjump_50fps.npz"
LZQ = "/Users/condenast/Downloads/policy-lzq.onnx"
S17 = "/Users/condenast/Downloads/policy-17700step.onnx"
_lzq = {i.name: numpy_helper.to_array(i) for i in onnx.load(LZQ).graph.initializer}
MEAN = _lzq['obs_normalizer._mean'].reshape(-1).astype(np.float32)
DIV = _lzq['onnx::Div_24'].reshape(-1).astype(np.float32)

npz = np.load(NPZ)
jp = np.asarray(npz['joint_pos'], dtype=np.float32)
jv = np.asarray(npz['joint_vel'], dtype=np.float32)
bq = np.asarray(npz['body_quat_w'], dtype=np.float32)
T = jp.shape[0]

DEFAULT = np.array([-0.262,-0.262, 0,0,0, 0,0,0, 0, 0.524,0.524, 0.3,0.3,
                    -0.262,-0.262, 0,0, 0,0, 0,0, -0.7,-0.7, 0,0, 0,0, 0,0], dtype=np.float32)
SCALE = np.full(NUM_JOINTS, 0.5, dtype=np.float32)

# 逐帧 body-frame 重力
_Rs = np.array([quat_to_mat(bq[t, 0]) for t in range(T)])
GRAV_B = (_Rs.transpose(0, 2, 1) @ np.array([0, 0, -1], dtype=np.float32)).astype(np.float32)

# calib: SDK _calibrate_init_rotation = R_z(yaw(robot_q0)) @ R_z(yaw(ref_q0))^T
# 完美跟踪 robot=ref → calib = I (除非 robot_q0 yaw != ref_q0 yaw)。这里算真实的。
def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
def Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=np.float32)
CALIB = Rz(yaw_of(bq[0, 0])) @ Rz(yaw_of(bq[0, 0])).T  # = I, 但保留接口


def rollout(run_fn, hist_order='old2new', use_norm=True, use_calib=False):
    h_grav = deque([GRAV_B[0].copy()]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
    h_ang = deque([np.zeros(3, np.float32)]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
    h_jp = deque([(jp[0]-DEFAULT).copy()]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
    h_jv = deque([jv[0].copy()]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
    warmup = ((jp[0]-DEFAULT)/SCALE).astype(np.float32)
    h_act = deque([np.zeros(NUM_JOINTS, np.float32)]*4 + [warmup.copy()], maxlen=HISTORY_LENGTH)
    targets = []
    for t in range(T-1):
        command = np.concatenate([jp[t], jv[t]]).astype(np.float32)
        Rt = quat_to_mat(bq[t, 0])
        if use_calib:
            rot_b = Rt.T @ CALIB @ Rt
        else:
            rot_b = np.eye(3, dtype=np.float32)
        anchor = rot_b[:, :2].reshape(-1).astype(np.float32)
        lists = [h_grav, h_ang, h_jp, h_jv, h_act]
        if hist_order == 'new2old':
            lists = [deque(reversed(list(d)), maxlen=HISTORY_LENGTH) for d in lists]
        obs = np.concatenate([command, anchor] +
                             [np.concatenate(list(d)) for d in lists]).astype(np.float32)
        if use_norm:
            obs = ((obs - MEAN) / DIV).astype(np.float32)
        raw = run_fn(obs).astype(np.float32)
        targets.append((DEFAULT + SCALE*np.clip(raw, -10, 10)).copy())
        h_grav.append(GRAV_B[t+1].copy()); h_ang.append(np.zeros(3, np.float32))
        h_jp.append((jp[t+1]-DEFAULT).copy()); h_jv.append(jv[t+1].copy())
        h_act.append(np.clip(raw, -10, 10).astype(np.float32))
    return np.asarray(targets)


def score(tgt):
    errs = [np.abs(tgt[:T-1-k] - jp[k:T-1]).mean() for k in range(6)]
    return errs


sess_lzq = ort.InferenceSession(LZQ, providers=["CPUExecutionProvider"])
sess17 = ort.InferenceSession(S17, providers=["CPUExecutionProvider"])
def run_lzq(o): return np.asarray(sess_lzq.run(None, {sess_lzq.get_inputs()[0].name: o.reshape(1,-1)})[0][0])
def run_17(o): return np.asarray(sess17.run(None, {sess17.get_inputs()[0].name: o.reshape(1,-1)})[0][0])

TESTS = [
    ("lzq 旧→新 +norm",      run_lzq, dict(hist_order='old2new', use_norm=False)),  # lzq 内嵌 norm
    ("lzq 新→旧 +norm",      run_lzq, dict(hist_order='new2old', use_norm=False)),
    ("17700+lzqnorm 旧→新",  run_17, dict(hist_order='old2new', use_norm=True)),
    ("17700+lzqnorm 新→旧",  run_17, dict(hist_order='new2old', use_norm=True)),
    ("17700 无norm 旧→新",   run_17, dict(hist_order='old2new', use_norm=False)),
    ("lzq 旧→新 calib",      run_lzq, dict(hist_order='old2new', use_norm=False, use_calib=True)),
]
for label, fn, kw in TESTS:
    tgt = rollout(fn, **kw)
    errs = score(tgt)
    k = int(np.argmin(errs))
    print(f"{label:<24} errs={['%.2f'%e for e in errs]} best k={k} ({errs[k]:.3f}) {'<<< 有效!' if errs[k]<0.15 else ''}")

# 首帧 raw action 检查
print("\n== 首帧 raw action 分布 (lzq) ==")
t0 = rollout(run_lzq, hist_order='old2new')
print("输出目标-参考帧1: mean|%.3f| max|%.3f|" % (np.abs(t0[0]-jp[1]).mean(), np.abs(t0[0]-jp[1]).max()))
print("逐关节误差:", np.abs(t0[0]-jp[1]).round(2))
