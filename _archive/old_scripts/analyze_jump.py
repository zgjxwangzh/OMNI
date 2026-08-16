"""分析跳高 BVH 的动作阶段：助跑 / 起跳 / 腾空 / 落地"""
import sys
import numpy as np
sys.path.insert(0, '/Users/condenast/Downloads/omni_29dof_v260705')
from bvh_retarget import BVHParser, C_BVH_TO_URDF

p = BVHParser('/Users/condenast/Downloads/omni_29dof_v260705/第一组 跳高 翻箱/跳高06_chr00.bvh')
n = p.n_frames
dt = p.frame_time

roots = []
foot_l_h = []
foot_r_h = []
for f in range(n):
    jd = p.get_joint_data(f)
    g = p.get_global_transforms(jd)
    roots.append(g['Hips'][1])
    foot_l_h.append(g['LeftFoot'][1][1])   # BVH Y = 高度
    foot_r_h.append(g['RightFoot'][1][1])

roots = np.array(roots) / 100.0
foot_l = np.array(foot_l_h) / 100.0
foot_r = np.array(foot_r_h) / 100.0

height = roots[:, 1]
horiz = np.sqrt(roots[:, 0] ** 2 + roots[:, 2] ** 2)
speed = np.concatenate([[0], np.diff(roots[:, 0]) ** 2 + np.diff(roots[:, 2]) ** 2]) ** 0.5 / dt

print(f"\n总时长 {n * dt:.2f}s ({n} 帧)")
print(f"根高度: {height.min():.2f} ~ {height.max():.2f} m")

# 每秒统计
print(f"\n{'t(s)':>5} {'根高':>6} {'水平速':>6} {'左脚高':>6} {'右脚高':>6}  状态")
for s in range(int(n * dt) + 1):
    i0, i1 = int(s / dt), min(int((s + 1) / dt), n)
    if i0 >= n:
        break
    sl = slice(i0, i1)
    both_off = np.mean((foot_l[sl] > foot_l.min() + 0.08) & (foot_r[sl] > foot_r.min() + 0.08))
    state = '腾空' if both_off > 0.3 else ('移动' if speed[sl].mean() > 0.5 else '静止/蹲')
    print(f"{s:>5} {height[sl].mean():>6.2f} {speed[sl].mean():>6.2f} "
          f"{foot_l[sl].mean():>6.2f} {foot_r[sl].mean():>6.2f}  {state}")

# 找腾空段（双脚离地）
fl_min, fr_min = foot_l.min(), foot_r.min()
airborne = (foot_l > fl_min + 0.08) & (foot_r > fr_min + 0.08)
if airborne.any():
    idx = np.where(airborne)[0]
    # 找最长连续段
    splits = np.where(np.diff(idx) > 5)[0]
    segs = np.split(idx, splits + 1)
    seg = max(segs, key=len)
    print(f"\n最长腾空段: 帧 {seg[0]}~{seg[-1]} (t={seg[0]*dt:.2f}~{seg[-1]*dt:.2f}s)")
    print(f"建议裁剪窗口: 帧 {max(0, seg[0]-int(1.0/dt))} ~ {min(n, seg[-1]+int(1.0/dt))}")
else:
    print("\n未检测到明显腾空段")
