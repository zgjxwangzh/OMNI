#!/usr/bin/env python3
"""决定性测试: 用 NPZ 参考轨迹构造"完美跟踪"obs 喂给 policy-lzq。
若策略是该动作的收敛跟踪器, 则 t 步输出 action 应满足:
    default + scale*action ≈ ref_pos[t+1]  (下一帧目标姿态)
若输出与期望差很远 → 策略/normalizer 不匹配, 必须回服务器重新导出。
"""
import numpy as np
from deploy_onnx_mujoco import (
    OnnxPolicy, NUM_JOINTS, quat_to_mat, yaw_quat, HISTORY_LENGTH,
)

ONNX = "/Users/condenast/Downloads/policy-lzq.onnx"
NPZ = "training_data/jump_high_firstjump_50fps.npz"

policy = OnnxPolicy(ONNX, motion_path=NPZ, action_scale_val=0.5)
npz = np.load(NPZ)
jp = np.asarray(npz['joint_pos'], dtype=np.float32)   # (T,29) policy 序
jv = np.asarray(npz['joint_vel'], dtype=np.float32)
bq = np.asarray(npz['body_quat_w'], dtype=np.float32)  # (T,B,4)
T = jp.shape[0]

default_pos = policy.default_pos_policy
scale = policy.action_scale_policy

# yaw 校准: 机器人初始化到 ref 帧0 → calib = I (与训练 _prefill 公式一致)
policy.reset()
policy._calibrate_init_rotation(bq[0, 0])

grav = np.array([0, 0, -1], dtype=np.float32)
angv = np.zeros(3, dtype=np.float32)

from collections import deque
h_grav = deque([grav.copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
h_ang = deque([angv.copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
h_jp = deque([(jp[0] - default_pos).copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
h_jv = deque([jv[0].copy() for _ in range(HISTORY_LENGTH)], maxlen=HISTORY_LENGTH)
warmup = ((jp[0] - default_pos) / scale).astype(np.float32)
h_act = deque([np.zeros(NUM_JOINTS, dtype=np.float32)] * (HISTORY_LENGTH - 1) + [warmup.copy()], maxlen=HISTORY_LENGTH)

errs, acts, targets = [], [], []
for t in range(T - 1):
    policy.step = t
    policy._get_ref_at_step(t)
    command = np.concatenate([jp[t], jv[t]]).astype(np.float32)
    # anchor_ori: robot=ref[t] → rot_b = R(ref)^T @ calib @ R(ref)
    rot_b = quat_to_mat(bq[t, 0]).T @ policy.world_to_init_rot @ quat_to_mat(bq[t, 0])
    anchor = rot_b[:, :2].reshape(-1).astype(np.float32)
    obs = np.concatenate([command, anchor,
                          np.concatenate(list(h_grav)), np.concatenate(list(h_ang)),
                          np.concatenate(list(h_jp)), np.concatenate(list(h_jv)),
                          np.concatenate(list(h_act))]).astype(np.float32)
    assert obs.shape[0] == policy.num_obs, obs.shape

    import onnxruntime as ort
    out = policy.session.run(policy.output_names, {policy.input_names[0]: obs.reshape(1, -1)})
    raw = np.asarray(out[0][0], dtype=np.float32)
    acts.append(raw.copy())
    targets.append((default_pos + scale * np.clip(raw, -10, 10)).copy())

    # 期望: 下一帧目标
    expect_target = jp[t + 1]
    got_target = default_pos + scale * np.clip(raw, -10, 10)
    errs.append(np.abs(got_target - expect_target))

    # 滚动历史(完美跟踪: 状态=参考)
    h_grav.append(grav.copy()); h_ang.append(angv.copy())
    h_jp.append((jp[t + 1] - default_pos).copy())
    h_jv.append(jv[t + 1].copy())
    h_act.append(np.clip(raw, -10, 10).astype(np.float32))

errs = np.asarray(errs)  # (T-1, 29)
targets = np.asarray(targets)  # (T-1, 29)
# 相位扫描: 策略可能因延迟训练而输出超前 k 步的目标
print("相位扫描 (目标 vs ref[t+k] 平均误差 rad):")
for k in range(0, 7):
    n = min(len(targets), T - 1 - k)
    e = np.abs(targets[:n] - jp[k:k + n]).mean()
    print(f"  k={k}: {e:.3f}")
print()
print(f"按 k=1 统计: 策略输出目标 vs 下一帧参考姿态 误差 (rad):")
print(f"  全局 mean={errs.mean():.3f}  median={np.median(errs):.3f}  max={errs.max():.3f}")
print(f"  前10步 mean err: {errs[:10].mean(axis=1).round(3)}")
print(f"  起跳段(帧60-80) mean err: {errs[60:80].mean():.3f}")
print(f"  动作幅度 std: {np.asarray(acts).std(axis=0).mean():.3f} (健康收敛策略站立段应<0.3)")

# 逐关节平均误差 top5
per = errs.mean(axis=0)
order = np.argsort(per)[::-1]
names = ["hip_pitch_l","hip_roll_l","hip_yaw_l","knee_pitch_l","ankle_pitch_l","ankle_roll_l",
         "hip_pitch_r","hip_roll_r","hip_yaw_r","knee_pitch_r","ankle_pitch_r","ankle_roll_r",
         "waist_yaw","waist_roll","waist_pitch",
         "shoulder_pitch_l","shoulder_roll_l","shoulder_yaw_l","elbow_pitch_l","elbow_yaw_l",
         "wrist_pitch_l","wrist_roll_l",
         "shoulder_pitch_r","shoulder_roll_r","shoulder_yaw_r","elbow_pitch_r","elbow_yaw_r",
         "wrist_pitch_r","wrist_roll_r"]
print("  误差最大 8 个关节(policy序):")
for i in order[:8]:
    print(f"    {names[i]:<18} {per[i]:.3f}")
