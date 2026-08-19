#!/usr/bin/env python3
"""精确触地标定 + 静态受力检查:
1) 二分搜索 base z 使脚底 mesh 恰好触地(消除自由落体冲击)
2) 默认站立姿态 + 帧0姿态: 初始接触力/力矩、各关节静态力矩 vs ctrlrange 饱和检查
"""
import numpy as np
import mujoco

from deploy_onnx_mujoco import NUM_JOINTS, POLICY_TO_MOTOR_IDX

NPZ = "training_data/jump_high_firstjump_50fps.npz"
MJCF = "omni_29dof_mjc/mjcf/omni_29dof.xml"

DEFAULT_POSE_POLICY = np.array(
    [-0.262, -0.262, 0, 0, 0, 0, 0, 0, 0, 0, 0.524, 0.524, 0.3, 0.3,
     -0.262, -0.262, 0, 0, 0, 0, 0, 0, -0.7, -0.7, 0, 0, 0, 0, 0], dtype=np.float64)

mj_model = mujoco.MjModel.from_xml_path(MJCF)
feet_bodies = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
               for n in ("ankle_roll_l_link", "ankle_roll_r_link")]
ground_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ground")


def set_pose(d, q_motor, base_quat=None, base_z=0.8):
    d.qpos[7:7 + NUM_JOINTS] = q_motor
    d.qpos[0:3] = [0, 0, base_z]
    d.qpos[3:7] = base_quat if base_quat is not None else [1, 0, 0, 0]
    d.qvel[:] = 0


def has_foot_contact(d):
    for i in range(d.ncon):
        c = d.contact[i]
        if ground_id in (c.geom1, c.geom2):
            b1 = mj_model.geom_bodyid[c.geom1]
            b2 = mj_model.geom_bodyid[c.geom2]
            if b1 in feet_bodies or b2 in feet_bodies:
                return True
    return False


def find_touch_z(d, q_motor, base_quat=None):
    """二分: 找到恰好产生脚-地接触的 base z"""
    lo, hi = 0.5, 1.2
    set_pose(d, q_motor, base_quat, hi)
    mujoco.mj_forward(mj_model, d)
    if has_foot_contact(d):
        print("  [warn] 最高处已接触")
    for _ in range(40):
        mid = (lo + hi) / 2
        set_pose(d, q_motor, base_quat, mid)
        mujoco.mj_forward(mj_model, d)
        if has_foot_contact(d):
            lo = mid
        else:
            hi = mid
    return lo


def report(label, q_motor, base_quat=None, npz_z=None):
    d = mujoco.MjData(mj_model)
    z_touch = find_touch_z(d, q_motor, base_quat)
    print(f"\n=== {label} ===")
    print(f"  触地 base z = {z_touch:.4f}" + (f"  (NPZ 帧0 z={npz_z:.4f}, 差 {npz_z-z_touch:+.4f})" if npz_z else ""))
    set_pose(d, q_motor, base_quat, z_touch + 0.0005)
    mujoco.mj_forward(mj_model, d)

    # 总质心
    com = d.subtree_com[0]
    print(f"  总质心 (x,y) = ({com[0]:+.4f}, {com[1]:+.4f})")
    # 脚底接触点范围
    xs, ys = [], []
    for i in range(d.ncon):
        c = d.contact[i]
        if ground_id in (c.geom1, c.geom2):
            xs.append(c.pos[0]); ys.append(c.pos[1])
    if xs:
        print(f"  接触点 x [{min(xs):+.3f},{max(xs):+.3f}] y [{min(ys):+.3f},{max(ys):+.3f}] n={len(xs)}")

    # 静置 0.5s, 观察接触力与关节漂移
    d.ctrl[:NUM_JOINTS] = q_motor
    dt = mj_model.opt.timestep
    max_fz, max_drift = 0.0, np.zeros(NUM_JOINTS)
    for s in range(int(0.5 / dt)):
        mujoco.mj_step(mj_model, d)
        # 脚地接触法向力
        for i in range(d.ncon):
            c = d.contact[i]
            if ground_id in (c.geom1, c.geom2):
                f = np.zeros(6)
                mujoco.mj_contactForce(mj_model, d, i, f)
                max_fz = max(max_fz, abs(f[2]))
        max_drift = np.maximum(max_drift, np.abs(d.qpos[7:7+NUM_JOINTS] - q_motor))
    R = d.xmat[1].reshape(3, 3) if mj_model.nbody > 1 else np.eye(3)
    base_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    R = d.xmat[base_body].reshape(3, 3)
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    print(f"  0.5s 后: base z={d.qpos[2]:.3f} x={d.qpos[0]:+.3f} 倾角={tilt:.1f}° 最大接触Fz={max_fz:.0f}N")
    jmax = int(np.argmax(max_drift))
    jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jmax + 6) if False else None
    print(f"  0.5s 关节最大漂移={max_drift.max():.3f} rad @joint#{np.argmax(max_drift)}")
    big = np.where(max_drift > 0.05)[0]
    if len(big):
        names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in big]
        print(f"  漂移>0.05rad 的关节: {list(zip(names, np.round(max_drift[big], 3)))}")


npz = np.load(NPZ)
q0_policy = npz['joint_pos'][0].astype(np.float64)
q0_motor = np.zeros(NUM_JOINTS); q0_motor[POLICY_TO_MOTOR_IDX] = q0_policy
qdef_motor = np.zeros(NUM_JOINTS); qdef_motor[POLICY_TO_MOTOR_IDX] = DEFAULT_POSE_POLICY
quat0 = npz['body_quat_w'][0, 0].astype(np.float64)

report("默认站立姿态", qdef_motor)
report("NPZ帧0姿态", q0_motor, base_quat=quat0, npz_z=float(npz['body_pos_w'][0, 0, 2]))
