#!/usr/bin/env python3
"""开环增益扫描诊断:
1) 帧0姿态的静态平衡性: 质心水平投影是否在双脚支撑多边形内
2) 恒定 ctrl = NPZ帧0 姿态(无策略), 扫描 ankle/hip kp 倍数,
   区分"无反馈倒立摆物理必然摔"与"MuJoCo 下增益不足"
"""
import numpy as np
import mujoco

from deploy_onnx_mujoco import NUM_JOINTS, POLICY_TO_MOTOR_IDX, MOTOR_TO_POLICY_IDX, quat_to_mat

NPZ = "training_data/jump_high_firstjump_50fps.npz"
MJCF = "omni_29dof_mjc/mjcf/omni_29dof.xml"
TOTAL_S = 3.0


def load_npz_frame0():
    npz = np.load(NPZ)
    q0_policy = npz['joint_pos'][0].astype(np.float64)
    q0_motor = np.zeros(NUM_JOINTS)
    q0_motor[POLICY_TO_MOTOR_IDX] = q0_policy
    return npz, q0_motor


# SDK 默认站立姿态 (policy 序, 与 obs529.py SDK_DEFAULT_POS 一致)
DEFAULT_POSE_POLICY = np.array(
    [-0.262, -0.262, 0, 0, 0, 0, 0, 0, 0, 0, 0.524, 0.524, 0.3, 0.3,
     -0.262, -0.262, 0, 0, 0, 0, 0, 0, -0.7, -0.7, 0, 0, 0, 0, 0], dtype=np.float64)
DEFAULT_POSE_MOTOR = np.zeros(NUM_JOINTS)
DEFAULT_POSE_MOTOR[POLICY_TO_MOTOR_IDX] = DEFAULT_POSE_POLICY


def build_model(q0_motor, gain_mult_ankle, gain_mult_hip):
    m = mujoco.MjModel.from_xml_path(MJCF)
    # position 执行器: gainprm[0]=kp; 对 ankle/hip 类执行器按倍数缩放
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    for i, nm in enumerate(names):
        base_kp = m.actuator_gainprm[i, 0]
        if "ankle" in nm:
            m.actuator_gainprm[i, 0] = base_kp * gain_mult_ankle
            m.actuator_biasprm[i, 1] = -base_kp * gain_mult_ankle  # bias = -kp*ctrl - kv*vel
        elif "hip" in nm or "knee" in nm or "waist" in nm:
            m.actuator_gainprm[i, 0] = base_kp * gain_mult_hip
            m.actuator_biasprm[i, 1] = -base_kp * gain_mult_hip
    return m


def init_state(m, npz, q0_motor):
    d = mujoco.MjData(m)
    d.qpos[7:7 + NUM_JOINTS] = q0_motor
    d.qpos[2] = float(npz['body_pos_w'][0, 0, 2])
    d.qpos[3:7] = npz['body_quat_w'][0, 0]
    mujoco.mj_forward(m, d)
    # 修正初始高度: 让最低脚底 geom 恰好触地(消除 NPZ/MJCF 几何高度差)
    min_z = None
    for g in range(m.ngeom):
        if m.geom_bodyid[g] and "ankle" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""):
            # 用 geom 位置近似; 精确触地由首次 mj_forward 的接触决定, 这里粗调
            pass
    # 用接触检测: 逐步抬升直到无穿透 —— 简化: 用 subtree com + 脚底最低点
    feet = [b for b in range(m.nbody) if "ankle_roll" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
    if feet:
        zs = [d.xpos[b][2] for b in feet]
        # ankle_roll body 中心不是脚底, 用 geom 下缘: 找这些 body 的 geom
        foot_zmin = 1e9
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in feet:
                # box/capsule 近似: geom pos z - size 最大维
                low = d.geom_xpos[g][2] - max(m.geom_size[g])
                foot_zmin = min(foot_zmin, low)
        d.qpos[2] += (0.001 - foot_zmin)
        mujoco.mj_forward(m, d)
    return d


def run_hold(m, d, q0_motor):
    dt = m.opt.timestep
    max_steps = int(TOTAL_S / dt)
    d.ctrl[:NUM_JOINTS] = q0_motor
    fall_t = None
    max_disp = 0.0
    max_tilt = 0.0
    x0, y0 = d.qpos[0], d.qpos[1]
    for s in range(max_steps):
        mujoco.mj_step(m, d)
        h = d.qpos[2]
        if fall_t is None and h < 0.4:
            fall_t = s * dt
        R = quat_to_mat(d.qpos[3:7])
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        max_tilt = max(max_tilt, tilt)
        max_disp = max(max_disp, np.hypot(d.qpos[0] - x0, d.qpos[1] - y0))
    return fall_t, max_disp, max_tilt, d.qpos[2]


def static_balance_check(m, d):
    """质心水平投影 vs 双脚支撑多边形(矩形近似: 两脚底中心连线扩展)。"""
    com = d.subtree_com[1] if m.nbody > 1 else None
    # subtree_com[0] 是全体; 用世界总质心
    com = d.subtree_com[0]
    feet = [b for b in range(m.nbody) if "ankle_roll" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
    pts = []
    for b in feet:
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b:
                c = d.geom_xpos[g]
                # 脚底四角近似: geom 尺寸
                sx, sy = m.geom_size[g][0], m.geom_size[g][1]
                for dx in (-sx, sx):
                    for dy in (-sy, sy):
                        pts.append((c[0] + dx, c[1] + dy))
    pts = np.array(pts)
    if len(pts) < 3:
        print("  [warn] 脚底几何不足, 无法做支撑多边形检查")
        return
    # 凸包: 用简单方法判断 COM 是否在凸包内(2D)
    def in_hull(p, hull):
        from scipy.spatial import Delaunay
        return Delaunay(hull).find_simplex(p) >= 0
    try:
        inside = in_hull(com[:2], pts)
    except Exception:
        # scipy 不可用: 退化为包围盒检查
        inside = (pts[:, 0].min() <= com[0] <= pts[:, 0].max()) and (pts[:, 1].min() <= com[1] <= pts[:, 1].max())
        print("  [warn] scipy 不可用, 用包围盒近似")
    print(f"  总质心 (x,y,z) = ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f})")
    print(f"  脚底支撑包围盒: x [{pts[:,0].min():+.3f}, {pts[:,0].max():+.3f}], y [{pts[:,1].min():+.3f}, {pts[:,1].max():+.3f}]")
    print(f"  质心是否在支撑区内: {'是' if inside else '否 <<<'}")


def trace_run(m, npz, q0_motor, label, dur=1.2):
    """细粒度轨迹: base x/z/pitch/yaw, 脚底滑移, 各时刻状态。"""
    d = init_state(m, npz, q0_motor)
    d.ctrl[:NUM_JOINTS] = q0_motor
    dt = m.opt.timestep
    x0, y0 = d.qpos[0], d.qpos[1]
    feet = [b for b in range(m.nbody) if "ankle_roll" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
    fx0 = {b: d.xpos[b][0] for b in feet}
    print(f"\n  --- 轨迹跟踪 [{label}] ---")
    print(f"   t    base_x  base_z  pitch°  yaw°  | 脚x滑移L/R")
    n = int(dur / dt)
    for s in range(n):
        mujoco.mj_step(m, d)
        if s % int(0.05 / dt) == 0:
            R = quat_to_mat(d.qpos[3:7])
            pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            fslip = " ".join(f"{d.xpos[b][0]-fx0[b]:+.3f}" for b in feet)
            print(f"  {s*dt:>4.2f}  {d.qpos[0]-x0:>+.3f}  {d.qpos[2]:>.3f}  {pitch:>+6.1f} {yaw:>+5.1f} | {fslip}")
    print(f"  末态 qvel[:6] = {np.array2string(d.qvel[:6], precision=3)}")


def main():
    npz, q0_motor = load_npz_frame0()
    m0 = build_model(q0_motor, 1.0, 1.0)
    d0 = init_state(m0, npz, q0_motor)
    print("=== 1) 帧0姿态静态平衡检查 ===")
    static_balance_check(m0, d0)
    trace_run(m0, npz, q0_motor, "帧0姿态, 标准增益")

    # 默认站立姿态(SDK default_pos)对照
    q_def = DEFAULT_POSE_MOTOR.copy()
    md = build_model(q_def, 1.0, 1.0)
    trace_run(md, npz, q_def, "SDK默认站立姿态, 标准增益", dur=3.0)

    print("\n=== 2) 开环保持(恒定 ctrl)增益扫描 ===")
    print(f"  {'ankle倍':>7} {'hip/knee倍':>9} | {'摔倒时刻':>9} {'最大位移':>8} {'最大倾角':>8} {'末高度':>7}")
    for ga in (1.0, 2.0, 4.0):
        for gh in (1.0, 2.0):
            m = build_model(q0_motor, ga, gh)
            d = init_state(m, npz, q0_motor)
            ft, md, mt, h_end = run_hold(m, d, q0_motor)
            print(f"  {ga:>7.1f} {gh:>9.1f} | {('t=%.2fs' % ft) if ft else '未摔倒':>9} {md:>7.3f}m {mt:>7.1f}° {h_end:>6.3f}")


if __name__ == "__main__":
    main()
