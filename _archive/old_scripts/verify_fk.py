"""本地 FK 验证：用 retarget 的关节角 + URDF 运动学画火柴人快照"""
import re
import sys
import numpy as np

sys.path.insert(0, '/Users/condenast/Downloads/omni_29dof_v260705')

URDF = 'assets/omni_29dof_nohead_noshoe/urdf/omni_29dof_nohead_noshoe_merged_modify_feet.urdf'


def rpy_to_mat(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_axis(a, t):
    c, s = np.cos(t), np.sin(t)
    x, y, z = a
    return np.array([
        [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def parse_urdf(path):
    txt = open(path).read()
    joints = {}
    for m in re.finditer(r'<joint name="([^"]+)" type="([^"]+)">(.*?)</joint>', txt, re.S):
        name, jtype, body = m.groups()
        origin = re.search(r'<origin xyz="([^"]*)" rpy="([^"]*)"', body)
        axis_m = re.search(r'<axis xyz="([^"]*)"', body)
        parent = re.search(r'<parent link="([^"]*)"', body)
        child = re.search(r'<child link="([^"]*)"', body)
        xyz = [float(v) for v in origin.group(1).split()] if origin else [0, 0, 0]
        rpy = [float(v) for v in origin.group(2).split()] if origin else [0, 0, 0]
        axis = [float(v) for v in axis_m.group(1).split()] if axis_m else [0, 0, 1]
        joints[name] = dict(type=jtype, xyz=xyz, rpy=rpy, axis=axis,
                            parent=parent.group(1), child=child.group(1))
    return joints


def fk(joints, angles, root_pos, root_quat):
    """返回 {link_name: (R, pos)}"""
    T = {'base_link': (quat_to_mat(root_quat), np.asarray(root_pos))}
    # 拓扑序：重复扫描直到所有关节处理完
    done = set()
    for _ in range(len(joints) + 1):
        for jn, j in joints.items():
            if jn in done or j['parent'] not in T:
                continue
            R_p, p_p = T[j['parent']]
            R_o = rpy_to_mat(*j['rpy'])
            p_o = np.array(j['xyz'])
            if j['type'] != 'fixed' and jn in angles:
                R_j = rot_axis(np.array(j['axis']), angles[jn])
            else:
                R_j = np.eye(3)
            R_c = R_p @ R_o @ R_j
            p_c = p_p + R_p @ p_o
            T[j['child']] = (R_c, p_c)
            done.add(jn)
    return T


def main():
    joints = parse_urdf(URDF)
    data = np.load('retargeted/跳高06_chr00_v5.npz')
    ja = data['joint_angles']
    names = list(data['joint_names'])
    rp = data['root_positions']
    rq = data['root_rotations']
    n = len(ja)

    # 数值检查：站立段踝关节世界高度
    print("帧    根高   左踝z  右踝z")
    for f in [0, 30, 60, 120, 180, 210, 220, 240, 280, 340, 390]:
        T = fk(joints, {nm: ja[f, i] for i, nm in enumerate(names)}, rp[f], rq[f])
        lz = T['ankle_roll_l_link'][1][2]
        rz = T['ankle_roll_r_link'][1][2]
        print(f"{f:>3}  {rp[f][2]:.2f}  {lz:>6.3f} {rz:>6.3f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 不可用，跳过绘图")
        return

    frames = [0, 100, 160, 200, 220, 250, 300, 380]
    fig = plt.figure(figsize=(16, 8))
    segs_def = [
        ('base_link', 'hip_pitch_l_link'), ('hip_pitch_l_link', 'knee_pitch_l_link'),
        ('knee_pitch_l_link', 'ankle_pitch_l_link'), ('ankle_pitch_l_link', 'ankle_roll_l_link'),
        ('base_link', 'hip_pitch_r_link'), ('hip_pitch_r_link', 'knee_pitch_r_link'),
        ('knee_pitch_r_link', 'ankle_pitch_r_link'), ('ankle_pitch_r_link', 'ankle_roll_r_link'),
        ('base_link', 'waist_yaw_link'), ('waist_yaw_link', 'waist_roll_link'),
        ('waist_roll_link', 'waist_pitch_link'),
        ('waist_pitch_link', 'shoulder_pitch_l_link'), ('shoulder_pitch_l_link', 'elbow_pitch_l_link'),
        ('elbow_pitch_l_link', 'elbow_yaw_l_link'), ('elbow_yaw_l_link', 'wrist_roll_l_link'),
        ('waist_pitch_link', 'shoulder_pitch_r_link'), ('shoulder_pitch_r_link', 'elbow_pitch_r_link'),
        ('elbow_pitch_r_link', 'elbow_yaw_r_link'), ('elbow_yaw_r_link', 'wrist_roll_r_link'),
    ]
    for k, f in enumerate(frames):
        # 符号连续
        q = rq[f].copy()
        T = fk(joints, {nm: ja[f, i] for i, nm in enumerate(names)}, rp[f], q)
        ax = fig.add_subplot(2, 4, k + 1, projection='3d')
        for a, b in segs_def:
            if a in T and b in T:
                pa, pb = T[a][1], T[b][1]
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], 'b-')
        # 地面
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(0, 1.6)
        ax.set_title(f'frame {f}')
    plt.savefig('retarget_check.png', dpi=80)
    print("已保存 retarget_check.png")


if __name__ == '__main__':
    main()
