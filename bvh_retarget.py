#!/usr/bin/env python3
"""
BVH 动捕数据 → OMNI 29-DOF 关节角度重定向 (v4)

修复 v3 的根本性 bug：
- v3 用 axis-angle 投影法独立提取各轴角度，数学上不正确
- v4 使用正确的欧拉角分解，匹配 URDF 关节链顺序

数学推导：
  BVH 所有关节使用 Y-X-Z 内蕴欧拉角：R_bvh = Ry(y) @ Rx(x) @ Rz(z)
  坐标转换 C @ R_bvh @ C^T 后：
    C @ Ry(θ) @ C^T = Ry(-θ)
    C @ Rx(θ) @ C^T = Rx(-θ)
    C @ Rz(θ) @ C^T = Rz(-θ)
  所以 R_urdf = Rz(-z) @ Rx(-x) @ Ry(-y)  → Z-X-Y 内蕴顺序

  URDF 髋关节链 pitch→roll→yaw：
    R_chain = Ry(pitch) @ Rx(roll) @ Rz(yaw)  → Y-X-Z 内蕴顺序

  因此：
    髋/肩（3DOF 链）：对 R_urdf 做 Y-X-Z 分解 → (pitch, roll, yaw)
    腰（3DOF 链 yaw→roll→pitch）：对 R_urdf 做 Y-X-Z 分解 → (yaw_from_Y, roll_from_X, pitch_from_Z)
    膝/踝（单/双轴）：对 R_urdf 做 Y-X-Z 分解取对应分量

用法：
    python bvh_retarget.py --input "数据1/上楼梯01_chr00.bvh" --output retargeted/上楼梯01.npz
    python bvh_retarget.py --input "数据1/" --output retargeted/  # 批量处理

依赖：pip install numpy scipy
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path


# ═══════════════════════════════════════════════════════
# OMNI 29-DOF 关节配置（从 URDF 提取）
# ═══════════════════════════════════════════════════════

OMNI_JOINTS = [
    # 左腿 (6)
    "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    # 右腿 (6)
    "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    # 腰 (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # 左臂 (7)
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint", "elbow_yaw_l_joint",
    "wrist_pitch_l_joint", "wrist_roll_l_joint",
    # 右臂 (7)
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint", "elbow_yaw_r_joint",
    "wrist_pitch_r_joint", "wrist_roll_r_joint",
]

OMNI_JOINT_LIMITS = {
    "hip_pitch_l_joint": (-2.6864, 2.6864),
    "hip_roll_l_joint": (-0.52, 2.96),
    "hip_yaw_l_joint": (-2.75, 2.75),
    "knee_pitch_l_joint": (0, 2.87),
    "ankle_pitch_l_joint": (-0.87, 0.52),
    "ankle_roll_l_joint": (-0.26, 0.26),
    "hip_pitch_r_joint": (-2.6864, 2.6864),
    "hip_roll_r_joint": (-2.96, 0.52),
    "hip_yaw_r_joint": (-2.75, 2.75),
    "knee_pitch_r_joint": (0, 2.87),
    "ankle_pitch_r_joint": (-0.87, 0.52),
    "ankle_roll_r_joint": (-0.26, 0.26),
    "waist_yaw_joint": (-2.7, 2.7),
    "waist_roll_joint": (-0.52, 0.52),
    "waist_pitch_joint": (-0.52, 0.52),
    "shoulder_pitch_l_joint": (-3.14, 2.7),
    "shoulder_roll_l_joint": (-0.52, 2.355),
    "shoulder_yaw_l_joint": (-2.61, 2.61),
    "elbow_pitch_l_joint": (-2.61, 0.52),
    "elbow_yaw_l_joint": (-2.09, 2.09),
    "wrist_pitch_l_joint": (-1.57, 1.57),
    "wrist_roll_l_joint": (-1.57, 1.57),
    "shoulder_pitch_r_joint": (-3.14, 2.7),
    "shoulder_roll_r_joint": (-2.355, 0.52),
    "shoulder_yaw_r_joint": (-2.61, 2.61),
    "elbow_pitch_r_joint": (-2.61, 0.52),
    "elbow_yaw_r_joint": (-2.09, 2.09),
    "wrist_pitch_r_joint": (-1.57, 1.57),
    "wrist_roll_r_joint": (-1.57, 1.57),
}

# ═══════════════════════════════════════════════════════
# 坐标系转换工具
# ═══════════════════════════════════════════════════════

# BVH → URDF 坐标转换矩阵
# BVH: +X=left, +Y=up, +Z=forward (人物面朝 +Z)
# URDF: +X=forward, +Y=left, +Z=up
# 映射：BVH_Z → URDF_X, BVH_X → URDF_Y, BVH_Y → URDF_Z
# （保证左右不镜像、det=+1 的真旋转）
C_BVH_TO_URDF = np.array([
    [0, 0, 1],   # URDF_X = BVH_Z (forward)
    [1, 0, 0],   # URDF_Y = BVH_X (left)
    [0, 1, 0],   # URDF_Z = BVH_Y (up)
])
C_URDF_TO_BVH = C_BVH_TO_URDF.T  # 正交矩阵的转置 = 逆

# 机器人 base→踝 腿长（URDF origin 累加：0.252+0.176+0.3）
ROBOT_LEG_LEN = 0.728

# URDF 关节轴
AXIS_X = np.array([1, 0, 0])
AXIS_Y = np.array([0, 1, 0])
AXIS_Z = np.array([0, 0, 1])


def bvh_to_urdf_rotation(R_bvh):
    """将 BVH 空间的旋转矩阵转换到 URDF 空间"""
    return C_BVH_TO_URDF @ R_bvh @ C_URDF_TO_BVH


def euler_yxz_to_mat(y_deg, x_deg, z_deg):
    """BVH 的 Y-X-Z 内蕴欧拉角 → 旋转矩阵

    BVH channels 顺序: Yrotation Xrotation Zrotation
    内蕴旋转: R = Ry(y) @ Rx(x) @ Rz(z)
    """
    y, x, z = np.radians(y_deg), np.radians(x_deg), np.radians(z_deg)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)
    cz, sz = np.cos(z), np.sin(z)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


# ═══════════════════════════════════════════════════════
# 正确的欧拉角分解（v4 核心修复）
# ═══════════════════════════════════════════════════════

def decompose_yxz_intrinsic(R):
    """Y-X-Z 内蕴欧拉角分解

    给定 R = Ry(y) @ Rx(x) @ Rz(z)，提取 (y, x, z)。

    推导：
    R = [[cy*cz-sy*sx*sz,  -cy*sz-sy*sx*cz,  -sy*cx],
         [cx*sz,             cx*cz,             sx    ],
         [sy*cz+cy*sx*sz,  -sy*sz+cy*sx*cz,   cy*cx ]]

    提取公式：
      x = arcsin(R[1,2])
      y = atan2(-R[0,2], R[2,2])
      z = atan2(-R[1,0], R[1,1])
    """
    # 提取 x (中间角)
    sx = np.clip(R[1, 2], -1.0, 1.0)
    x = np.arcsin(sx)

    cx = np.cos(x)
    if abs(cx) > 1e-6:
        # 正常情况
        y = np.arctan2(-R[0, 2], R[2, 2])
        z = np.arctan2(-R[1, 0], R[1, 1])
    else:
        # 万向锁: x ≈ ±π/2
        y = 0.0
        if sx > 0:  # x ≈ π/2
            z = np.arctan2(R[0, 1], R[0, 0])
        else:  # x ≈ -π/2
            z = np.arctan2(-R[0, 1], -R[0, 0])

    return y, x, z


def decompose_zxy_intrinsic(R):
    """Z-X-Y 内蕴欧拉角分解

    给定 R = Rz(z) @ Rx(x) @ Ry(y)，提取 (z, x, y)。

    推导：
    R = [[cz*cy+sz*sx*sy,   cz*sy*sx-sz*cy,   cx*sy],
         [cx*sz,              cx*cz,             -sx  ],
         [-sz*cy+cz*sx*sy,   sz*sy+cz*cy*sx,    cx*cy]]

    提取公式：
      x = arcsin(-R[1,2])
      z = atan2(R[1,0], R[1,1])
      y = atan2(-R[0,2], R[2,2])
    """
    sx = np.clip(-R[1, 2], -1.0, 1.0)
    x = np.arcsin(sx)

    cx = np.cos(x)
    if abs(cx) > 1e-6:
        z = np.arctan2(R[1, 0], R[1, 1])
        y = np.arctan2(-R[0, 2], R[2, 2])
    else:
        z = 0.0
        if sx > 0:  # x ≈ π/2
            y = np.arctan2(R[2, 0], R[0, 0])
        else:
            y = np.arctan2(-R[2, 0], -R[0, 0])

    return z, x, y


# ═══════════════════════════════════════════════════════
# 几何法腿部 IK（v5：基于关节位置，消除 hip_roll 外撇）
# ═══════════════════════════════════════════════════════

def _rpy_to_mat(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


_LEG_CHAIN_CACHE = {}


def load_leg_chain(side):
    """从 URDF 解析腿链 6 个关节的 (xyz, rpy, axis)，缓存"""
    if side in _LEG_CHAIN_CACHE:
        return _LEG_CHAIN_CACHE[side]
    import re
    urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'assets', 'omni_29dof_nohead_noshoe', 'urdf',
                        'omni_29dof_nohead_noshoe_merged_modify_feet.urdf')
    txt = open(urdf).read()
    chain = []
    for jn in ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll']:
        m = re.search(r'<joint name="' + jn + '_' + side + r'_joint"[^>]*>(.*?)</joint>', txt, re.S)
        body = m.group(1)
        o = re.search(r'<origin xyz="([^"]*)" rpy="([^"]*)"', body)
        a = re.search(r'<axis xyz="([^"]*)"', body)
        xyz = [float(v) for v in o.group(1).split()]
        rpy = [float(v) for v in o.group(2).split()]
        axis = [float(v) for v in a.group(1).split()]
        chain.append((np.array(xyz), np.array(rpy), np.array(axis)))
    _LEG_CHAIN_CACHE[side] = chain
    return chain


def robot_ankle_z_rel(ja_row, joint_idx_map, side):
    """用 URDF 腿链 FK 计算踝关节相对 base 的 Z 高度

    用于地面约束：关节限位裁剪蹲深后，防止脚掌穿地。
    """
    chain = load_leg_chain(side)
    angles = [
        ja_row[joint_idx_map[f'hip_pitch_{side}_joint']],
        ja_row[joint_idx_map[f'hip_roll_{side}_joint']],
        ja_row[joint_idx_map[f'hip_yaw_{side}_joint']],
        ja_row[joint_idx_map[f'knee_pitch_{side}_joint']],
        ja_row[joint_idx_map[f'ankle_pitch_{side}_joint']],
        ja_row[joint_idx_map[f'ankle_roll_{side}_joint']],
    ]
    pos = np.zeros(3)
    R = np.eye(3)
    for (xyz, rpy, axis), a in zip(chain, angles):
        pos = pos + R @ xyz
        R = R @ _rpy_to_mat(*rpy) @ _rot_axis(axis, a)
    return pos


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _rot_axis(a, t):
    c, s = np.cos(t), np.sin(t)
    x, y, z = a
    return np.array([
        [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]])


def rot_mat_to_quat(R):
    """旋转矩阵 → 四元数 (w,x,y,z)"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def geometric_leg_angles(hip_pos, knee_pos, ankle_pos):
    """从 URDF 空间的髋/膝/踝位置几何计算腿部关节角
    
    URDF: X-forward, Y-left, Z-up。大腿零位朝 -Z（向下）。
    返回：(hip_pitch, hip_roll, knee_pitch)
    """
    thigh = np.asarray(knee_pos) - np.asarray(hip_pos)
    shin = np.asarray(ankle_pos) - np.asarray(knee_pos)
    tn = np.linalg.norm(thigh)
    sn = np.linalg.norm(shin)
    if tn < 1e-6 or sn < 1e-6:
        return 0.0, 0.0, 0.0
    thigh = thigh / tn
    shin = shin / sn

    # hip_pitch：矢状面（X-Z）内大腿相对垂直的夹角
    # URDF hip_pitch 正 → 大腿朝 -X（后摆），所以前摆为负
    hip_pitch = np.arctan2(-thigh[0], -thigh[2])

    # hip_roll：额状面（Y）侧倾。垂直跳应接近 0
    hip_roll = np.arcsin(np.clip(thigh[1], -1.0, 1.0))

    # knee_pitch：大腿与小腿夹角，后弯为正
    dot = np.clip(np.dot(thigh, shin), -1.0, 1.0)
    ang = np.arccos(dot)
    cross = np.cross(thigh, shin)
    knee_pitch = ang if cross[1] > 0 else -ang
    knee_pitch = max(0.0, knee_pitch)

    return hip_pitch, hip_roll, knee_pitch


# ═══════════════════════════════════════════════════════
# BVH 解析器
# ═══════════════════════════════════════════════════════

class BVHParser:
    def __init__(self, filepath):
        self.joints = []
        self.joint_channels = {}
        self.joint_offsets = {}
        self.joint_parent = {}
        self.frame_time = 0.0
        self.frames = None
        self.n_frames = 0
        self._parse(filepath)

    def _parse(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        idx = 0
        current_joint = None
        joint_stack = []
        self._in_end_site = False

        while idx < len(lines):
            line = lines[idx].strip()

            if line.startswith('ROOT ') or line.startswith('JOINT '):
                name = line.split(None, 1)[1].strip()
                self.joints.append(name)
                current_joint = name
                if joint_stack:
                    self.joint_parent[name] = joint_stack[-1]
                else:
                    self.joint_parent[name] = None
                joint_stack.append(name)

            elif line.startswith('OFFSET') and current_joint:
                parts = line.split()
                # End Site 块内的 OFFSET 不属于关节，跳过
                if not self._in_end_site:
                    self.joint_offsets[current_joint] = (
                        float(parts[1]), float(parts[2]), float(parts[3])
                    )

            elif line.startswith('CHANNELS') and current_joint:
                parts = line.split()
                channels = parts[2:]
                self.joint_channels[current_joint] = channels

            elif line == '}':
                # End Site 块的右括号不弹关节栈，否则会破坏父子层次
                if self._in_end_site:
                    self._in_end_site = False
                elif joint_stack:
                    joint_stack.pop()

            elif line.startswith('End Site'):
                self._in_end_site = True

            elif line.startswith('MOTION'):
                break

            idx += 1

        # 解析帧数据
        idx += 1
        frame_data = []
        while idx < len(lines):
            line = lines[idx].strip()
            if line.startswith('Frames:'):
                self.n_frames = int(line.split(':')[1].strip())
            elif line.startswith('Frame Time:'):
                self.frame_time = float(line.split(':')[1].strip())
            elif line:
                values = [float(x) for x in line.split()]
                frame_data.append(values)
            idx += 1

        self.frames = np.array(frame_data)
        print(f"  [BVH] {self.n_frames} 帧，{len(self.joints)} 关节，"
              f"{self.frames.shape[1]} 通道，{1.0/self.frame_time:.0f}fps")

    def get_joint_data(self, frame_idx):
        """获取某帧所有关节的旋转 (度) 和根位置"""
        data = {}
        offset = 0
        for jname in self.joints:
            channels = self.joint_channels.get(jname, [])
            values = self.frames[frame_idx, offset:offset + len(channels)]
            data[jname] = {
                'channels': channels,
                'values': values,
            }
            offset += len(channels)
        return data

    def get_rotation_matrix(self, joint_data, joint_name):
        """获取某关节的 BVH 局部旋转矩阵"""
        if joint_name not in joint_data:
            return np.eye(3)
        jdata = joint_data[joint_name]
        channels = jdata['channels']
        values = jdata['values']

        y_idx = x_idx = z_idx = 0
        for i, ch in enumerate(channels):
            if 'Yrotation' in ch: y_idx = i
            elif 'Xrotation' in ch: x_idx = i
            elif 'Zrotation' in ch: z_idx = i

        return euler_yxz_to_mat(values[y_idx], values[x_idx], values[z_idx])

    def get_global_transforms(self, joint_data):
        """正向运动学：计算每个关节的全局旋转和全局位置（BVH 空间，cm）
        
        返回：{joint_name: (R_global, pos_global)}
        """
        result = {}
        for jname in self.joints:
            jdata = joint_data[jname]
            channels = jdata['channels']
            values = jdata['values']

            # 局部旋转
            R_local = self.get_rotation_matrix(joint_data, jname)

            # 局部平移：本 BVH 格式每个关节都有 position 通道，且已包含 offset
            # （frame0 时 position == offset）。若 position 通道存在则只用它，
            # 否则回退到 header 的 OFFSET，避免骨骼长度被双倍计算。
            ox, oy, oz = self.joint_offsets.get(jname, (0, 0, 0))
            px = py = pz = None
            for i, ch in enumerate(channels):
                if 'Xposition' in ch: px = values[i]
                elif 'Yposition' in ch: py = values[i]
                elif 'Zposition' in ch: pz = values[i]
            if px is not None:
                pos_local = np.array([px, py, pz])
            else:
                pos_local = np.array([ox, oy, oz])

            parent = self.joint_parent.get(jname)
            if parent is None or parent not in result:
                R_g = R_local
                pos_g = pos_local
            else:
                R_p, pos_p = result[parent]
                R_g = R_p @ R_local
                pos_g = pos_p + R_p @ pos_local

            result[jname] = (R_g, pos_g)
        return result


# ═══════════════════════════════════════════════════════
# 重定向核心逻辑 (v4: 正确欧拉角分解)
# ═══════════════════════════════════════════════════════

def compute_bvh_bone_direction(parser, joint_data, bvh_joint_name):
    """计算 BVH 关节的骨骼方向（从父关节指向子关节）在 BVH 局部坐标系中
    
    返回单位向量。
    """
    offset = parser.joint_offsets.get(bvh_joint_name, (0, -1, 0))
    # BVH offset 是从父到子的向量（在父关节的局部坐标系中）
    bone_dir_bvh = np.array(offset)
    norm = np.linalg.norm(bone_dir_bvh)
    if norm > 1e-6:
        bone_dir_bvh /= norm
    else:
        bone_dir_bvh = np.array([0, -1, 0])  # 默认朝下
    return bone_dir_bvh


def compute_urdf_hip_angles_for_bone_direction(bone_dir_urdf):
    """给定 URDF 空间中大腿骨的方向，反算 hip_pitch/roll/yaw
    
    URDF 大腿在零位时朝 -Z（向下）。
    hip_pitch(Y) → hip_roll(X) → hip_yaw(Z) 链。
    R_total = Ry(pitch) @ Rx(roll) @ Rz(yaw)
    bone_dir_zero = [0, 0, -1] (URDF 零位时大腿朝下)
    bone_dir = R_total @ bone_dir_zero
    
    我们需要找到 (pitch, roll, yaw) 使得 R_total @ [0,0,-1] = bone_dir_urdf
    """
    # 目标方向
    target = bone_dir_urdf / (np.linalg.norm(bone_dir_urdf) + 1e-10)
    
    # 零位方向
    zero_dir = np.array([0, 0, -1])
    
    # 使用轴-角方法：找到从零位到目标的旋转
    # cross = zero_dir × target
    cross = np.cross(zero_dir, target)
    dot = np.dot(zero_dir, target)
    
    # 旋转角度
    angle = np.arctan2(np.linalg.norm(cross), dot)
    
    if angle < 1e-6:
        return 0.0, 0.0, 0.0  # 已经在零位
    
    # 旋转轴
    axis = cross / (np.linalg.norm(cross) + 1e-10)
    
    # 现在需要将这个绕任意轴的旋转分解为 Y-X-Z 欧拉角
    # 构造旋转矩阵
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    ux, uy, uz = axis
    
    # Rodrigues 公式
    R = np.array([
        [cos_a + ux*ux*(1-cos_a),     ux*uy*(1-cos_a) - uz*sin_a, ux*uz*(1-cos_a) + uy*sin_a],
        [uy*ux*(1-cos_a) + uz*sin_a,  cos_a + uy*uy*(1-cos_a),    uy*uz*(1-cos_a) - ux*sin_a],
        [uz*ux*(1-cos_a) - uy*sin_a,  uz*uy*(1-cos_a) + ux*sin_a, cos_a + uz*uz*(1-cos_a)]
    ])
    
    # Y-X-Z 分解
    pitch, roll, yaw = decompose_yxz_intrinsic(R)
    return pitch, roll, yaw


def compute_rest_pose_offsets(parser):
    """计算 BVH 零位 → URDF 零位的偏移量
    
    使用相对旋转法：
    1. BVH rest pose 的所有关节旋转为 I（单位矩阵）
    2. BVH 运动帧的局部旋转 R_bvh 就是相对于 rest pose 的旋转
    3. 转换到 URDF：R_urdf_rel = C @ R_bvh @ C^T
    4. 对 URDF 链做欧拉分解，得到相对于 URDF rest pose 的角度
    
    因为 BVH rest pose 所有关节角度为 0，所以 offset = 0。
    但 BVH 和 URDF 的骨骼朝向不同，这会导致问题。
    
    实际上，offset 应该反映 URDF rest pose 与 BVH rest pose 的差异。
    但由于两者骨骼结构不同，无法用简单角度偏移解决。
    
    这里返回全零数组，实际偏移在 retarget 函数中通过相对旋转处理。
    """
    return np.zeros(len(OMNI_JOINTS))


def retarget(bvh_path, output_path):
    print(f"\n{'='*60}")
    print(f"处理：{bvh_path}")
    print(f"{'='*60}")

    parser = BVHParser(bvh_path)
    n_frames = parser.n_frames
    n_joints = len(OMNI_JOINTS)
    joint_angles = np.zeros((n_frames, n_joints))
    root_positions = np.zeros((n_frames, 3))
    root_rotations = np.zeros((n_frames, 4))
    foot_heights = np.zeros((n_frames, 2))
    yaw_rel_arr = np.zeros(n_frames)

    joint_idx_map = {name: i for i, name in enumerate(OMNI_JOINTS)}

    # 参考朝向：取中间帧（动作阶段）的骨盆 yaw，
    # 消除动捕标定姿势与表演朝向之间的 90° 偏置
    jd_mid = parser.get_joint_data(n_frames // 2)
    R_mid_u = bvh_to_urdf_rotation(
        parser.get_rotation_matrix(jd_mid, parser.joints[0]))
    yaw_ref = np.arctan2(R_mid_u[1, 0], R_mid_u[0, 0])
    cy_ref, sy_ref = np.cos(-yaw_ref), np.sin(-yaw_ref)
    R_yaw_ref_inv = np.array([[cy_ref, -sy_ref, 0], [sy_ref, cy_ref, 0], [0, 0, 1]])
    print(f"  [朝向] 动作阶段参考 yaw={np.degrees(yaw_ref):.0f}°（将归一化）")

    # 骨骼长度（从 BVH offset 获取）
    thigh_length = abs(parser.joint_offsets.get('RightLeg', (0, -45, 0))[1])  # cm
    shin_length = abs(parser.joint_offsets.get('RightFoot', (0, -42, 0))[1])   # cm
    print(f"  [骨骼] 大腿={thigh_length:.1f}cm, 小腿={shin_length:.1f}cm")
    
    # 计算零位偏移（现在返回全零，实际通过相对旋转处理）
    rest_offsets = compute_rest_pose_offsets(parser)
    print(f"  [零位偏移] 使用相对旋转法（BVH rest pose 所有关节角度为 0）")

    for frame in range(n_frames):
        jdata = parser.get_joint_data(frame)

        # ── 正向运动学：全局关节位置（BVH cm → URDF m）──
        gtransforms = parser.get_global_transforms(jdata)
        gpos_urdf = {}
        for jname, (R_g, pos_g) in gtransforms.items():
            gpos_urdf[jname] = R_yaw_ref_inv @ (C_BVH_TO_URDF @ pos_g / 100.0)

        # ── 根节点位姿：只保留相对 yaw，去除 pitch/roll 前倾 ──
        # 演员蹲跳时骨盆前倾 20-26°，直接复现到机器人上观感过度；
        # 改为 upright base，前倾交给腰关节表达
        R_hips_g, pos_hips = gtransforms['Hips']
        R_root_full = bvh_to_urdf_rotation(R_hips_g)
        yaw_rel = np.arctan2(R_root_full[1, 0], R_root_full[0, 0]) - yaw_ref
        yaw_rel_arr[frame] = yaw_rel
        cy, sy = np.cos(yaw_rel), np.sin(yaw_rel)
        R_root = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        root_positions[frame] = C_BVH_TO_URDF @ pos_hips / 100.0
        # 水平分量转到归一化朝向系（与新 base yaw 对齐）
        root_positions[frame] = R_yaw_ref_inv @ root_positions[frame]
        root_rotations[frame] = rot_mat_to_quat(R_root)

        # 关节位置转到骨盆局部系（机器人 base 系）：
        # 消除人体世界朝向（如面向 BVH +X）对腿部角度的污染
        root_pos = root_positions[frame]
        gpos_local = {
            jname: R_root.T @ (p - root_pos)
            for jname, p in gpos_urdf.items()
        }

        # ════════════════════════════════════════════
        # 腿部（v5 几何法）：从髋/膝/踝全局位置计算 hip_pitch/roll/knee_pitch
        #   消除角度映射的轴耦合，hip_roll 不再外撇
        # ════════════════════════════════════════════
        for side, upleg, leg, foot in [
            ('r', 'RightUpLeg', 'RightLeg', 'RightFoot'),
            ('l', 'LeftUpLeg', 'LeftLeg', 'LeftFoot')
        ]:
            hip_p = gpos_local[upleg]
            knee_p = gpos_local[leg]
            ankle_p = gpos_local[foot]
            hip_pitch, hip_roll, knee_pitch = geometric_leg_angles(hip_p, knee_p, ankle_p)

            joint_angles[frame, joint_idx_map[f'hip_pitch_{side}_joint']] = hip_pitch
            joint_angles[frame, joint_idx_map[f'hip_roll_{side}_joint']] = hip_roll
            joint_angles[frame, joint_idx_map[f'knee_pitch_{side}_joint']] = knee_pitch

            # hip_yaw：仍用角度法（绕大腿长轴，几何法不易提取）
            R_bvh = parser.get_rotation_matrix(jdata, upleg)
            R_urdf = bvh_to_urdf_rotation(R_bvh)
            _, _, z_angle = decompose_yxz_intrinsic(R_urdf)
            joint_angles[frame, joint_idx_map[f'hip_yaw_{side}_joint']] = z_angle

        # ════════════════════════════════════════════
        # 踝关节：BVH Foot → URDF ankle_pitch(Y) + ankle_roll(X)
        # ════════════════════════════════════════════
        for side, bvh_ankle in [('r', 'RightFoot'), ('l', 'LeftFoot')]:
            R_bvh = parser.get_rotation_matrix(jdata, bvh_ankle)
            R_urdf = bvh_to_urdf_rotation(R_bvh)

            y_angle, x_angle, z_angle = decompose_yxz_intrinsic(R_urdf)
            # 减去零位偏移
            joint_angles[frame, joint_idx_map[f'ankle_pitch_{side}_joint']] = y_angle - rest_offsets[joint_idx_map[f'ankle_pitch_{side}_joint']]
            joint_angles[frame, joint_idx_map[f'ankle_roll_{side}_joint']] = x_angle - rest_offsets[joint_idx_map[f'ankle_roll_{side}_joint']]

        # ════════════════════════════════════════════
        # 腰部：BVH Spine → URDF waist_yaw(Z)→waist_roll(X)→waist_pitch(Y)
        #
        # URDF 链: Ry(yaw) @ Rx(roll) @ Rz(pitch) → 不对！
        # 实际 URDF 链: waist_yaw(Z) → waist_roll(X) → waist_pitch(Y)
        # R_chain = Rz(yaw) @ Rx(roll) @ Ry(pitch) → Z-X-Y 内蕴
        # ════════════════════════════════════════════
        R_bvh_spine = parser.get_rotation_matrix(jdata, 'Spine')
        R_urdf_spine = bvh_to_urdf_rotation(R_bvh_spine)

        # Z-X-Y 分解，减去零位偏移
        z_angle, x_angle, y_angle = decompose_zxy_intrinsic(R_urdf_spine)
        joint_angles[frame, joint_idx_map['waist_yaw_joint']] = z_angle - rest_offsets[joint_idx_map['waist_yaw_joint']]
        joint_angles[frame, joint_idx_map['waist_roll_joint']] = x_angle - rest_offsets[joint_idx_map['waist_roll_joint']]
        joint_angles[frame, joint_idx_map['waist_pitch_joint']] = y_angle - rest_offsets[joint_idx_map['waist_pitch_joint']]

        # ════════════════════════════════════════════
        # 左臂：BVH Shoulder+Arm → URDF shoulder_pitch(Y)→roll(X)→yaw(Z)
        #
        # BVH 有 Shoulder (锁骨) + Arm (上臂) 两个关节
        # URDF 只有 shoulder_pitch→roll→yaw 三个关节
        # 合并：R_combined = R_shoulder @ R_arm
        # URDF 链: Ry(pitch) @ Rx(roll) @ Rz(yaw) → Y-X-Z
        # ════════════════════════════════════════════
        R_bvh_shoulder_l = parser.get_rotation_matrix(jdata, 'LeftShoulder')
        R_bvh_arm_l = parser.get_rotation_matrix(jdata, 'LeftArm')
        R_bvh_combined_l = R_bvh_shoulder_l @ R_bvh_arm_l
        R_urdf_shoulder_l = bvh_to_urdf_rotation(R_bvh_combined_l)

        y_angle, x_angle, z_angle = decompose_yxz_intrinsic(R_urdf_shoulder_l)
        # 减去零位偏移
        joint_angles[frame, joint_idx_map['shoulder_pitch_l_joint']] = y_angle - rest_offsets[joint_idx_map['shoulder_pitch_l_joint']]
        joint_angles[frame, joint_idx_map['shoulder_roll_l_joint']] = x_angle - rest_offsets[joint_idx_map['shoulder_roll_l_joint']]
        joint_angles[frame, joint_idx_map['shoulder_yaw_l_joint']] = z_angle - rest_offsets[joint_idx_map['shoulder_yaw_l_joint']]

        # 左肘：BVH ForeArm → URDF elbow_pitch(Y) + elbow_yaw(Z)
        R_bvh_elbow_l = parser.get_rotation_matrix(jdata, 'LeftForeArm')
        R_urdf_elbow_l = bvh_to_urdf_rotation(R_bvh_elbow_l)
        y_angle, x_angle, z_angle = decompose_yxz_intrinsic(R_urdf_elbow_l)
        # 减去零位偏移
        joint_angles[frame, joint_idx_map['elbow_pitch_l_joint']] = y_angle - rest_offsets[joint_idx_map['elbow_pitch_l_joint']]
        joint_angles[frame, joint_idx_map['elbow_yaw_l_joint']] = z_angle - rest_offsets[joint_idx_map['elbow_yaw_l_joint']]

        # 左腕：BVH Hand → URDF wrist_pitch(Y) + wrist_roll(X)
        R_bvh_wrist_l = parser.get_rotation_matrix(jdata, 'LeftHand')
        R_urdf_wrist_l = bvh_to_urdf_rotation(R_bvh_wrist_l)
        y_angle, x_angle, z_angle = decompose_yxz_intrinsic(R_urdf_wrist_l)
        # 减去零位偏移
        joint_angles[frame, joint_idx_map['wrist_pitch_l_joint']] = y_angle - rest_offsets[joint_idx_map['wrist_pitch_l_joint']]
        joint_angles[frame, joint_idx_map['wrist_roll_l_joint']] = x_angle - rest_offsets[joint_idx_map['wrist_roll_l_joint']]

        # ════════════════════════════════════════════
        # 右臂
        # ════════════════════════════════════════════
        R_bvh_shoulder_r = parser.get_rotation_matrix(jdata, 'RightShoulder')
        R_bvh_arm_r = parser.get_rotation_matrix(jdata, 'RightArm')
        R_bvh_combined_r = R_bvh_shoulder_r @ R_bvh_arm_r
        R_urdf_shoulder_r = bvh_to_urdf_rotation(R_bvh_combined_r)

        y_angle, x_angle, z_angle = decompose_yxz_intrinsic(R_urdf_shoulder_r)
        # 减去零位偏移
        joint_angles[frame, joint_idx_map['shoulder_pitch_r_joint']] = y_angle - rest_offsets[joint_idx_map['shoulder_pitch_r_joint']]
        joint_angles[frame, joint_idx_map['shoulder_roll_r_joint']] = x_angle - rest_offsets[joint_idx_map['shoulder_roll_r_joint']]
        joint_angles[frame, joint_idx_map['shoulder_yaw_r_joint']] = z_angle - rest_offsets[joint_idx_map['shoulder_yaw_r_joint']]

        R_bvh_elbow_r = parser.get_rotation_matrix(jdata, 'RightForeArm')
        R_urdf_elbow_r = bvh_to_urdf_rotation(R_bvh_elbow_r)
        y_angle, x_angle, z_angle = decompose_yxz_intrinsic(R_urdf_elbow_r)
        # 减去零位偏移
        joint_angles[frame, joint_idx_map['elbow_pitch_r_joint']] = y_angle - rest_offsets[joint_idx_map['elbow_pitch_r_joint']]
        joint_angles[frame, joint_idx_map['elbow_yaw_r_joint']] = z_angle - rest_offsets[joint_idx_map['elbow_yaw_r_joint']]

        R_bvh_wrist_r = parser.get_rotation_matrix(jdata, 'RightHand')
        R_urdf_wrist_r = bvh_to_urdf_rotation(R_bvh_wrist_r)
        y_angle, x_angle, z_angle = decompose_yxz_intrinsic(R_urdf_wrist_r)
        # 减去零位偏移
        joint_angles[frame, joint_idx_map['wrist_pitch_r_joint']] = y_angle - rest_offsets[joint_idx_map['wrist_pitch_r_joint']]
        joint_angles[frame, joint_idx_map['wrist_roll_r_joint']] = x_angle - rest_offsets[joint_idx_map['wrist_roll_r_joint']]

        # ── 脚部高度：直接用 FK 全局位置（URDF Z = 高度）──
        foot_heights[frame, 0] = gpos_urdf['LeftFoot'][2]
        foot_heights[frame, 1] = gpos_urdf['RightFoot'][2]

    # ════════════════════════════════════════════
    # 后处理
    # ════════════════════════════════════════════

    # 裁剪开头标定段：从面向参考朝向的站立帧开始
    aligned = np.where(np.abs(yaw_rel_arr) < np.radians(15))[0]
    start = int(aligned[0]) if len(aligned) else 0
    if start > 0:
        joint_angles = joint_angles[start:]
        root_positions = root_positions[start:]
        root_rotations = root_rotations[start:]
        foot_heights = foot_heights[start:]
        n_frames = len(joint_angles)
        print(f"  [裁剪] 去掉开头标定段 {start} 帧，剩 {n_frames} 帧")

    # 根轨迹缩放：人机腿长比（否则机器人腿短会悬空）
    human_leg = thigh_length + shin_length
    scale = ROBOT_LEG_LEN / (human_leg / 100.0)
    root_positions *= scale
    foot_heights *= scale
    print(f"  [缩放] 人腿={human_leg/100:.2f}m → 机器人腿={ROBOT_LEG_LEN}m, scale={scale:.3f}")

    # 关节限位裁剪
    clipped_count = 0
    for i, jname in enumerate(OMNI_JOINTS):
        lower, upper = OMNI_JOINT_LIMITS[jname]
        before = joint_angles[:, i].copy()
        joint_angles[:, i] = np.clip(joint_angles[:, i], lower, upper)
        clipped_count += np.sum(before != joint_angles[:, i])

    if clipped_count > 0:
        print(f"  [裁剪] {clipped_count} 个值超出关节限位")

    # 膝盖角度软限制：保留爆发力（110° = 1.92 rad）
    # 实际机器人能深蹲且有爆发力，腾空稳定性通过 ang_vel 权重控制
    KNEE_SOFT_LIMIT = 1.92  # 110°
    knee_clipped = 0
    for side in ['l', 'r']:
        idx = joint_idx_map[f'knee_pitch_{side}_joint']
        before = joint_angles[:, idx].copy()
        joint_angles[:, idx] = np.clip(joint_angles[:, idx], 0, KNEE_SOFT_LIMIT)
        knee_clipped += np.sum(before != joint_angles[:, idx])
    if knee_clipped > 0:
        print(f"  [膝盖限制] {knee_clipped} 帧超过 {np.degrees(KNEE_SOFT_LIMIT):.0f}°，已裁剪")

    # 低通滤波平滑
    try:
        from scipy.signal import butter, filtfilt
        fs = 1.0 / parser.frame_time
        cutoff = min(6.0, fs * 0.4)  # 确保不超过奈奎斯特频率
        nyq = 0.5 * fs
        b, a = butter(4, cutoff / nyq, btype='low')
        for i in range(n_joints):
            if n_frames > 30:
                joint_angles[:, i] = filtfilt(b, a, joint_angles[:, i])
        # 再次裁剪
        for i, jname in enumerate(OMNI_JOINTS):
            lower, upper = OMNI_JOINT_LIMITS[jname]
            joint_angles[:, i] = np.clip(joint_angles[:, i], lower, upper)
        print(f"  [平滑] Butterworth 低通，{cutoff:.1f}Hz")
    except ImportError:
        print("  [!] scipy 未安装，跳过平滑")

    # 地面约束：裁剪/平滑后蹲深变浅，脚会穿地，按穿地量抬高根高度
    ref_z = None
    for frame in range(n_frames):
        Rr = quat_to_mat(root_rotations[frame])
        rel_l = robot_ankle_z_rel(joint_angles[frame], joint_idx_map, 'l')
        rel_r = robot_ankle_z_rel(joint_angles[frame], joint_idx_map, 'r')
        zl = root_positions[frame, 2] + (Rr @ rel_l)[2]
        zr = root_positions[frame, 2] + (Rr @ rel_r)[2]
        zmin = min(zl, zr)
        if ref_z is None:
            ref_z = zmin  # 第一帧站立 = 地面参考
        deficit = ref_z - zmin
        if deficit > 0:
            root_positions[frame, 2] += deficit

    # 脚部接触标签
    foot_contact = np.zeros((n_frames, 2), dtype=np.int32)
    min_foot_h = foot_heights.min(axis=0)
    for frame in range(n_frames):
        for foot in range(2):
            h = foot_heights[frame, foot] - min_foot_h[foot]
            if h < 0.05:
                foot_contact[frame, foot] = 1

    contact_ratio = foot_contact.sum(axis=0) / n_frames
    print(f"  [接触] 左脚着地={contact_ratio[0]:.1%}, 右脚={contact_ratio[1]:.1%}")

    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(
        output_path,
        joint_angles=joint_angles,
        root_positions=root_positions,
        root_rotations=root_rotations,
        foot_contact=foot_contact,
        foot_heights=foot_heights,
        joint_names=np.array(OMNI_JOINTS),
        frame_time=parser.frame_time,
        n_frames=n_frames,
        source_file=os.path.basename(bvh_path),
    )

    file_size = os.path.getsize(output_path) / 1024
    print(f"  [✓] {output_path} ({file_size:.0f} KB, {n_frames}帧×{n_joints}关节)")

    # 统计
    print(f"\n  {'关节名':<30s} {'min':>7s} {'max':>7s} {'range':>7s}")
    print(f"  {'-'*54}")
    for i, jname in enumerate(OMNI_JOINTS):
        jmin, jmax = joint_angles[:, i].min(), joint_angles[:, i].max()
        print(f"  {jname:<30s} {jmin:>7.3f} {jmax:>7.3f} {jmax-jmin:>7.3f}")

    return joint_angles


def main():
    parser = argparse.ArgumentParser(description="BVH → OMNI 29-DOF 重定向 v4")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        bvh_files = sorted(input_path.glob("*.bvh"))
        print(f"找到 {len(bvh_files)} 个 BVH 文件")
        for bvh_file in bvh_files:
            out_file = output_path / f"{bvh_file.stem}.npz"
            try:
                retarget(str(bvh_file), str(out_file))
            except Exception as e:
                print(f"  [错误] {bvh_file.name}: {e}")
                import traceback; traceback.print_exc()
        print(f"\n完成：{len(bvh_files)} 个文件 → {output_path}")
    else:
        retarget(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
