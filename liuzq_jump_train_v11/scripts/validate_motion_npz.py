#!/usr/bin/env python
# Copyright (c) 2026, The Omni Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""校验训练参考动作 npz(xMimic 格式), 供换新 npz 用。

用法:
  python validate_motion_npz.py <npz路径> [--print-frames]

校验项(对应 xMimic MotionLoader / jump_env 硬依赖):
  1. keys 齐: fps / joint_pos / joint_vel / body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w
  2. shapes: (T,29) / (T,29) / (T,30,3) / (T,30,4) / (T,30,3) / (T,30,3)
  3. **fps 必须 == 50**(xMimic MotionCommand._update_command 每策略步 time_steps += 1,
     按 50fps 播; 非 50fps 需改 motion command 支持小数帧, 超出跳高迁移范围)
  4. 帧 0 站立: 双脚不悬浮(body_pos_w 里脚踝高度≈0 附近, 由 -p 打印判断)
  5. --print-frames: 打印 root(body 0)高度剖面 + 脚踝(body 18/19)高度, 供重算
     站立/下蹲/推蹬/腾空相位窗口(见 jump_phase_rewards.py 硬编码 15/95/115/132)

纯 numpy 实现, 不依赖 isaaclab/omni, 服务器可直接跑。
"""

import argparse
import sys

import numpy as np

# 与 jump_env/omni_jump_env_cfg.py 的 OMNI_BODY_NAMES 一致(30 body 关节树序)
OMNI_BODY_NAMES = [
    "base_link",            # 0
    "hip_pitch_l_link",     # 1
    "hip_pitch_r_link",     # 2
    "waist_yaw_link",       # 3
    "hip_roll_l_link",      # 4
    "hip_roll_r_link",      # 5
    "waist_roll_link",      # 6
    "hip_yaw_l_link",       # 7
    "hip_yaw_r_link",       # 8
    "waist_pitch_link",     # 9
    "knee_pitch_l_link",    # 10
    "knee_pitch_r_link",    # 11
    "shoulder_pitch_l_link",  # 12
    "shoulder_pitch_r_link",  # 13
    "ankle_pitch_l_link",   # 14
    "ankle_pitch_r_link",   # 15
    "shoulder_roll_l_link",  # 16
    "shoulder_roll_r_link",  # 17
    "ankle_roll_l_link",    # 18 (脚)
    "ankle_roll_r_link",    # 19 (脚)
    "shoulder_yaw_l_link",  # 20
    "shoulder_yaw_r_link",  # 21
    "elbow_pitch_l_link",   # 22
    "elbow_pitch_r_link",   # 23
    "elbow_yaw_l_link",     # 24
    "elbow_yaw_r_link",     # 25
    "wrist_pitch_l_link",   # 26
    "wrist_pitch_r_link",   # 27
    "wrist_roll_l_link",    # 28
    "wrist_roll_r_link",    # 29
]

REQUIRED_KEYS = [
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]

# 每个 key 期望的 shape 后缀(除 T 外)
EXPECTED_SHAPE = {
    "joint_pos": (29,),
    "joint_vel": (29,),
    "body_pos_w": (30, 3),
    "body_quat_w": (30, 4),
    "body_lin_vel_w": (30, 3),
    "body_ang_vel_w": (30, 3),
}


def main():
    parser = argparse.ArgumentParser(description="Validate xMimic-format motion npz for high-jump training.")
    parser.add_argument("npz", help="Path to the .npz motion file")
    parser.add_argument("--print-frames", "-p", action="store_true", help="Print root/foot height profile for phase-window recalibration")
    args = parser.parse_args()

    data = np.load(args.npz)
    errors = []

    # -- 1. keys --
    missing = [k for k in REQUIRED_KEYS if k not in data.files]
    if missing:
        print(f"[FAIL] 缺 keys: {missing}")
        sys.exit(1)

    # -- 2. shapes --
    T = data["joint_pos"].shape[0]
    for k, suffix in EXPECTED_SHAPE.items():
        actual = data[k].shape
        expected = (T,) + suffix
        if actual != expected:
            errors.append(f"{k}: shape {actual} != 期望 {expected}")

    # 所有 body 数组 T 一致
    for k in ["body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
        if data[k].shape[0] != T:
            errors.append(f"{k}: 帧数 {data[k].shape[0]} != joint_pos 帧数 {T}")

    # -- 3. fps == 50 --
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    if fps != 50:
        errors.append(f"fps = {fps}, 必须 == 50(与 50Hz 策略步逐帧对齐)")

    # -- 4. 无 NaN/Inf --
    for k in REQUIRED_KEYS:
        if k == "fps":
            continue
        arr = data[k]
        if not np.isfinite(arr).all():
            errors.append(f"{k}: 含 NaN/Inf")

    # -- 5. 关节列数/顺序提示 --
    if data["joint_pos"].shape[1] != 29:
        errors.append(f"joint_pos 列数 {data['joint_pos'].shape[1]} != 29(URDF 关节数)")

    if errors:
        print(f"[FAIL] {args.npz}")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"[OK] {args.npz}")
    print(f"  T={T} 帧, fps={fps}(50Hz 对齐 ✓), joint {data['joint_pos'].shape[1]} 列, body {data['body_pos_w'].shape[1]} 个")

    # -- 6. 帧 0 站立检查 --
    foot_z0 = float(np.mean(data["body_pos_w"][0, 18:20, 2]))
    root_z0 = float(data["body_pos_w"][0, 0, 2])
    print(f"  帧0: root_z={root_z0:.3f}m, 脚踝_z={foot_z0:.3f}m (站立: 脚踝≈贴地, 参考文件脚踝~0.02-0.05)")
    if foot_z0 > 0.15:
        print(f"  [WARN] 帧0 脚踝高 {foot_z0:.3f}m, 可能非站立姿态(会破坏 JumpMotionCommand 锁帧0)")

    # -- 7. 相位窗口参考(root 高度剖面, 供重算 15/95/115/132) --
    if args.print_frames:
        root_z = data["body_pos_w"][:, 0, 2]
        foot_z = data["body_pos_w"][:, 18:20, 2].mean(axis=1)
        print("\n  帧  root_z   脚踝_z   说明")
        print("  " + "-" * 40)
        for i in range(T):
            note = ""
            if i in (0, 15, 95, 115, 124, 132, 150):
                note = " <- 当前硬编码窗口边界"
            print(f"  {i:4d}  {root_z[i]:.3f}  {foot_z[i]:.3f}{note}")
        print("\n  [提示] 若新 npz 的帧数/起跳时刻与 183 帧不同, 必须重算 jump_phase_rewards.py 的")
        print("         t_stand/t_to/t_land/ref_len 和 airborne_leg_tuck 的窗口/收腿峰值帧。")


if __name__ == "__main__":
    main()
