#!/usr/bin/env python3
"""一次性补丁: 将 omni_29dof.xml 的执行器/关节参数对齐训练配置
(robots/omni_29dof_nohead_noshoe_dcmotor_identified.py 的 DelayedDCMotorCfg)。
幂等: 用正则匹配任意旧值, 可重复运行。"""
import re

XML = "omni_29dof_mjc/mjcf/omni_29dof.xml"

# joint: (armature, damping(=viscous_friction))  来自训练 actuator cfg
JOINT = {
    "hip_pitch_l_joint": (0.025875279679894447, 1.5535),
    "hip_pitch_r_joint": (0.025875279679894447, 1.5535),
    "hip_roll_l_joint":  (0.05765285715460777, 1.3766),
    "hip_roll_r_joint":  (0.05765285715460777, 1.3766),
    "knee_pitch_l_joint": (0.06578928977251053, 0.9860),
    "knee_pitch_r_joint": (0.06578928977251053, 0.9860),
    "hip_yaw_l_joint":   (0.056901562958955765, 1.6786),
    "hip_yaw_r_joint":   (0.056901562958955765, 1.6786),
    "ankle_pitch_l_joint": (0.010765427723526955, 0.8179),
    "ankle_pitch_r_joint": (0.010765427723526955, 0.8179),
    "ankle_roll_l_joint": (0.10608603060245514, 0.8722),
    "ankle_roll_r_joint": (0.10608603060245514, 0.8722),
    "waist_yaw_joint":   (0.02, 0.5),
    "waist_roll_joint":  (0.04, 0.5),
    "waist_pitch_joint": (0.04, 0.5),
    "shoulder_pitch_l_joint": (0.01, 0.25), "shoulder_pitch_r_joint": (0.01, 0.25),
    "shoulder_roll_l_joint":  (0.01, 0.25), "shoulder_roll_r_joint":  (0.01, 0.25),
    "shoulder_yaw_l_joint":   (0.01, 0.25), "shoulder_yaw_r_joint":   (0.01, 0.25),
    "elbow_pitch_l_joint":    (0.01, 0.25), "elbow_pitch_r_joint":    (0.01, 0.25),
    "elbow_yaw_l_joint":      (0.01, 0.25), "elbow_yaw_r_joint":      (0.01, 0.25),
    "wrist_pitch_l_joint":    (0.004, 0.25), "wrist_pitch_r_joint":   (0.004, 0.25),
    "wrist_roll_l_joint":     (0.004, 0.25), "wrist_roll_r_joint":    (0.004, 0.25),
}

# actuator: (kp=stiffness, kv=damping, forcerange=effort_limit(peak_torque))
ACT = {
    "hip_pitch_l_joint": (120, 5, 140), "hip_pitch_r_joint": (120, 5, 140),
    "hip_roll_l_joint":  (120, 5, 140), "hip_roll_r_joint":  (120, 5, 140),
    "knee_pitch_l_joint": (120, 5, 140), "knee_pitch_r_joint": (120, 5, 140),
    "hip_yaw_l_joint":   (100, 5, 90),  "hip_yaw_r_joint":   (100, 5, 90),
    "ankle_pitch_l_joint": (30, 3, 50), "ankle_pitch_r_joint": (30, 3, 50),
    "ankle_roll_l_joint":  (30, 3, 50), "ankle_roll_r_joint":  (30, 3, 50),
    "waist_yaw_joint":   (100, 5, 90),
    "waist_roll_joint":  (120, 5, 110), "waist_pitch_joint": (120, 5, 110),
    "shoulder_pitch_l_joint": (50, 2, 25), "shoulder_pitch_r_joint": (50, 2, 25),
    "shoulder_roll_l_joint":  (50, 2, 25), "shoulder_roll_r_joint":  (50, 2, 25),
    "shoulder_yaw_l_joint":   (50, 2, 25), "shoulder_yaw_r_joint":   (50, 2, 25),
    "elbow_pitch_l_joint":    (50, 2, 25), "elbow_pitch_r_joint":    (50, 2, 25),
    "elbow_yaw_l_joint":      (50, 2, 25), "elbow_yaw_r_joint":      (50, 2, 25),
    "wrist_pitch_l_joint":    (5, 1, 10),  "wrist_pitch_r_joint":    (5, 1, 10),
    "wrist_roll_l_joint":     (5, 1, 10),  "wrist_roll_r_joint":     (5, 1, 10),
}

src = open(XML).read()
orig = src

# 1) joint: armature 对齐 + 加 damping (训练 viscous_friction)
for jn, (arm, dmp) in JOINT.items():
    pat = re.compile(
        rf'(<joint name="{jn}"[^/]*?)armature="[\d.eE+-]+"( *damping="[\d.eE+-]+")?(/>)')
    m = pat.search(src)
    if not m:
        print(f"  [!] joint {jn} 未匹配")
        continue
    src = pat.sub(rf'\1armature="{arm:.6f}" damping="{dmp:.4f}"\3', src, count=1)

# 2) actuator: kp/kv 对齐训练 + forcelimited/forcerange (训练 effort_limit)
for an, (kp, kv, flim) in ACT.items():
    pat = re.compile(
        rf'(<position name="{an}" joint="{an}" )kp="[\d.]+" kv="[\d.]+"( ctrllimited="true" ctrlrange="[-\d. ]+")?( forcelimited="true" forcerange="[-\d. ]+")?(/>)')
    m = pat.search(src)
    if not m:
        print(f"  [!] actuator {an} 未匹配")
        continue
    src = pat.sub(
        rf'\1kp="{kp:.1f}" kv="{kv:.1f}"\2 forcelimited="true" forcerange="{-flim} {flim}"\4'.replace('\\4', ''),
        src, count=1)

# 3) 更新说明注释
old_note = "    <!-- 2026-08-18: 改用 position 执行器，与 SDK omni_31.xml 一致 -->\n    <!-- kp/kd 来自 SDK high_dynamic.yaml (策略运行时实际输出值) -->\n    <!-- 注意: 这些值与 env-omni31.yaml sim 列不同! -->\n    <!--   knee: 150/5 (非200/5) | ankle: 30/3 (非20/2) -->\n    <!--   waist: 150/5 (非100/5) | shoulder: 100/2 (非50/2) -->"
new_note = """    <!-- 2026-08-18: kp/kv/armature/damping/forcerange 全部对齐训练配置 -->
    <!-- 来源: robots/omni_29dof_nohead_noshoe_dcmotor_identified.py (DelayedDCMotorCfg) -->
    <!-- hip/knee=120/5 hip_yaw/waist_yaw=100/5 ankle=30/3 waist_rp=120/5 shoulder/elbow=50/2 wrist=5/1 -->
    <!-- 关键修正: armature 从统一 0.01 改为逐关节辨识值(ankle_roll 0.106), 加 viscous damping -->"""
if old_note in src:
    src = src.replace(old_note, new_note)
else:
    print("  [!] 旧注释块未匹配, 跳过注释更新")

assert src != orig, "没有任何修改生效"
open(XML, "w").write(src)
print("补丁写入完成")
