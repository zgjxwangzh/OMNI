# OMNI 29-DOF 人形机器人 — 动捕重定向与策略训练项目

> 最后更新：2026-08-10

## 项目概述

本项目将 OMNI 29-DOF 人形机器人的动捕数据（BVH）重定向为机器人关节角度，并转换为官方 RL SDK 的 `high_dynamic` 框架所需的 NPZ 格式，用于后续强化学习训练和真机部署。

### 数据管线

```
BVH 动捕数据 → bvh_retarget.py → retargeted/*.npz (motor order)
                                       ↓
                              convert_to_highdynamic.py
                                       ↓
                              motion_data/*_highdynamic.npz (policy order)
                                       ↓
                              官方 RL SDK 训练 → ONNX → 部署
```

### 当前状态

| 阶段 | 状态 |
|------|------|
| 动捕数据重定向 | ✅ 32 个 BVH 全部完成（29 个唯一动作） |
| high_dynamic 格式转换 | ✅ 29 个全部通过验证 |
| MuJoCo 运动学验证 | ✅ 29/29 全部通过（batch_verify_mujoco.py） |
| MuJoCo 物理跟踪验证 | ✅ 29 个动作 baseline 已建立（physics_track_verify.py） |
| 走路策略仿真 | ✅ 脚本就绪（run_walking_sim.py），待 GPU 服务器验证 |
| 等待训练代码 | ⏳ 需向项目组确认 high_dynamic 训练代码位置 |
| RL 训练 | ⏳ 待训练代码到位后开始 |
| 真机部署 | ⏳ 待训练完成后由公司部署 |

### 动作列表（29 个）

| 类别 | 动作 | 帧数 | 说明 |
|------|------|------|------|
| 跳高 | 跳高01-06 | 334-516 | 正面跳高，高度 0.7-1.1m |
| 翻箱 | 翻箱子01-14 | 641-1385 | 跨越不同高度箱子 |
| 楼梯 | 上/下弯楼梯、上/下楼梯、弯/直楼梯 | 440-603 | 各类楼梯动作 |
| 跑步 | 跑步01 | 525 | 平地跑步 |
| 跨栏 | 跨栏02-03 | 417-477 | 跨栏动作 |
| 匍匐 | 匍匐前进1 | 2623 | 匍匐前进（最长） |

---

## 项目结构

```text
omni_29dof_v260705/
├── readme.md                    # 本文档
├── GPU_SERVER_GUIDE.md          # GPU 服务器部署指南
├── setup_gpu_server.sh          # GPU 环境一键安装脚本
│
├── bvh_retarget.py              # BVH → 机器人关节角度重定向（v5，核心脚本）
├── convert_to_highdynamic.py    # retargeted NPZ → high_dynamic NPZ 转换
│
├── verify_highdynamic_mujoco.py # 单文件 MuJoCo 可视化验证
├── batch_verify_mujoco.py       # 批量运动学验证（29 个动作）
├── physics_track_verify.py      # 物理跟踪验证（PD 控制 + 重力）
├── run_walking_sim.py           # 走路策略仿真（ONNX + MuJoCo）
│
├── 数据1/                       # 原始动捕数据（23 个 BVH + CSV）
├── 第一组 跳高 翻箱/            # 原始动捕数据（9 个 BVH + CSV + FBX）
├── retargeted/                  # 重定向输出（29 个 NPZ，motor order）
├── motion_data/                 # high_dynamic 格式（29 个 NPZ，policy order）
│
├── assets/                      # 机器人模型资源
│   └── omni_29dof_nohead_noshoe/
│       ├── urdf/                # URDF 文件
│       └── meshes/              # 32 个 STL 网格
├── robots/                      # 机器人配置（ArticulationCfg）
├── actuators/                   # 电机模型（DelayedDCMotor）
│
├── omni_29dof_mjc/              # 官方 MuJoCo 模型
│   └── mjcf/omni_29dof.xml     # 29 执行器 MJCF
├── omni_rl_sdk/                 # 官方 RL SDK（部署框架）
│   ├── deploy_omni_sim_real/    # 仿真部署主程序
│   ├── policy/                  # 策略实现
│   │   ├── loco_mode/           # 走路策略（ONNX）
│   │   └── high_dynamic/        # 高动态策略（reference tracking）
│   └── FSM/                     # 状态机
│
├── frames/                      # MuJoCo 验证截图
└── _archive/                    # 归档（旧管线文件，已废弃）
```

---

## 快速开始

### 1. 重定向（BVH → NPZ）

```bash
# 单个文件
python bvh_retarget.py --input "数据1/跳高01_chr00.bvh" --output retargeted/跳高01_chr00.npz

# 批量处理整个目录
python bvh_retarget.py --input "数据1" --output retargeted/
python bvh_retarget.py --input "第一组 跳高 翻箱" --output retargeted/
```

### 2. 转换为 high_dynamic 格式

```bash
# 单个文件
python convert_to_highdynamic.py --input retargeted/跳高06_chr00.npz --output motion_data/ --fps 30

# 批量（用 Python）
python -c "
import sys; sys.path.insert(0, '.')
from convert_to_highdynamic import convert_npz
from pathlib import Path
for f in sorted(Path('retargeted').glob('*.npz')):
    convert_npz(str(f), f'motion_data/{f.stem}_highdynamic.npz', fps=30.0)
"
```

### 3. MuJoCo 可视化验证

```bash
pip install mujoco pillow
python verify_highdynamic_mujoco.py \
    --mjcf omni_29dof_mjc/mjcf/omni_29dof.xml \
    --npz motion_data/跳高06_chr00_highdynamic.npz \
    --retarget_npz retargeted/跳高06_chr00.npz \
    --save_frames --output_dir frames/
```

### 4. 批量验证（29 个动作）

```bash
# 运动学验证（禁用重力，直接设置姿态）
python batch_verify_mujoco.py

# 物理跟踪验证（开启重力，PD 控制）
python physics_track_verify.py
```

### 5. GPU 服务器部署

详见 [GPU_SERVER_GUIDE.md](GPU_SERVER_GUIDE.md)，包含：
- 环境搭建脚本（setup_gpu_server.sh）
- 走路策略仿真（run_walking_sim.py）
- 物理跟踪验证（physics_track_verify.py）
- 故障排查指南

---

## 关节顺序说明

重定向输出（motor order）和 high_dynamic 框架（policy order）的关节顺序不同，需要映射：

| 索引 | motor order | policy order |
|------|-------------|-------------|
| 0-5 | 左腿 6 关节 | hip_pitch L/R + waist_yaw |
| 6-11 | 右腿 6 关节 | hip_roll L/R + waist_roll |
| 12-14 | 腰 3 关节 | hip_yaw L/R + waist_pitch |
| 15-21 | 左臂 7 关节 | knee L/R + shoulder_pitch L/R + ... |
| 22-28 | 右臂 7 关节 | ... + wrist_roll L/R |

映射关系定义在 `convert_to_highdynamic.py` 的 `MOTOR_TO_POLICY_IDX` 数组中。

---

## 官方 RL SDK 说明

- **走路策略**（loco_mode）：`omni_7dof_63k_2file.onnx`，obs=90×10=900，action=29（实际 25 通过映射）
- **高动态策略**（high_dynamic）：reference tracking 框架，obs=149，action=29，NPZ 运动数据
- **部署流程**：训练 → ONNX 导出 → 放入 SDK → FSM 状态机切换 → 仿真验证 → 真机部署

---

# 以下为官方模型文档

---

# Omni 29-DOF 人形机器人模型（omni_29dof_nohead_noshoe）

本仓库提供了 Omni 29 自由度人形机器人在 **NVIDIA Isaac Lab / Isaac Sim** 中的完整模型配置，包含 URDF、网格资源、电机（执行器）参数辨识结果以及可直接导入使用的 `ArticulationCfg`。拿到本模型后，配合下文说明即可快速完成模型导入与强化学习训练。

---

## 1. 环境要求（最低版本）

| 组件 | 最低版本 |
| --- | --- |
| Isaac Lab | **2.2.0** 及以上 |
| Isaac Sim | **5.0.0** 及以上 |
| Python | 3.10+（随 Isaac Lab 提供） |
| PyTorch | 随 Isaac Lab 提供 |

> 说明：本模型直接通过 `sim_utils.UrdfFileCfg` 在运行时由 URDF 转换生成，并使用了 `UrdfConverterCfg.JointDriveCfg.PDGainsCfg`、`replace_cylinders_with_capsules` 等接口，这些 API 需要 Isaac Lab 2.2.0 / Isaac Sim 5.0.0 及以上版本，低于此版本会因接口不兼容而导入失败。

---

## 2. 机器人基础参数

| 参数 | 数值 |
| --- | --- |
| 总自由度（DOF） | **29** |
| 连杆（link）数量 | 30（含 `base_link`） |
| 整机质量 | 约 **38.87 kg** |
| 初始站立高度（base 高度） | 0.8 m |
| 浮动基座 | 是（`fix_base=False`） |
| 自碰撞 | 开启（`enabled_self_collisions=True`） |
| 接触传感器 | 开启（`activate_contact_sensors=True`） |

### 2.1 自由度分布

| 部位 | 关节 | 数量 |
| --- | --- | --- |
| 左/右腿 | hip_pitch、hip_roll、hip_yaw、knee_pitch、ankle_pitch、ankle_roll | 6 × 2 = 12 |
| 腰部 | waist_yaw、waist_roll、waist_pitch | 3 |
| 左/右臂 | shoulder_pitch、shoulder_roll、shoulder_yaw、elbow_pitch、elbow_yaw、wrist_pitch、wrist_roll | 7 × 2 = 14 |
| **合计** | | **29** |

> 头部（head_yaw / head_pitch）已作为固定视觉件合并进 `waist_pitch_link`，不占用自由度；踝部为并联结构，模型中按等效关节处理。

### 2.2 关节限位（来自 URDF，单位：rad / Nm / rad·s⁻¹）

| 关节 | 下限 | 上限 | effort | velocity |
| --- | --- | --- | --- | --- |
| hip_pitch_* | -2.6864 | 2.6864 | 140 | 20.9 |
| hip_roll_l / hip_roll_r | -0.52 / -2.96 | 2.96 / 0.52 | 140 | 20.9 |
| hip_yaw_* | -2.75 | 2.75 | 90 | 32.9 |
| knee_pitch_* | 0 | 2.87 | 140 | 20.9 |
| ankle_pitch_* | -0.87 | 0.52 | 50 | 31.4 |
| ankle_roll_* | -0.26 | 0.26 | 50 | 31.4 |
| waist_yaw | -2.7 | 2.7 | 90 | 32.9 |
| waist_roll / waist_pitch | -0.52 | 0.52 | 50 | 31.4 |
| shoulder_pitch_* | -3.14 | 2.7 | 25 | 31.4 |
| shoulder_roll_l / shoulder_roll_r | -0.52 / -2.355 | 2.355 / 0.52 | 25 | 31.4 |
| shoulder_yaw_* | -2.61 | 2.61 | 25 | 31.4 |
| elbow_pitch_* | -2.61 | 0.52 | 25 | 31.4 |
| elbow_yaw_* | -2.09 | 2.09 | 25 | 31.4 |
| wrist_pitch_* / wrist_roll_* | -1.57 | 1.57 | 10 | 7.8 |

### 2.3 默认初始关节角（`init_state`）

| 关节 | 角度 (rad) |
| --- | --- |
| hip_pitch_* | -0.26178 |
| knee_*  | 0.52356 |
| ankle_pitch_* | -0.26178 |
| elbow_pitch_* | -0.7 |
| shoulder_pitch_* | 0.3 |
| shoulder_roll_* | 0.0 |
| 其余关节 | 0.0 |

### 2.4 执行器（电机）配置

模型采用经参数辨识的直流电机模型 `DelayedDCMotor`（在 `DCMotor` 基础上增加 2~5 个物理步的随机指令延迟，用于 sim-to-real）。电机型号定义于 `robots/tiangong.py`：

| 电机组 | 覆盖关节 | 峰值力矩 (Nm) | 峰值转速 (rad/s) | 刚度 stiffness | 阻尼 damping |
| --- | --- | --- | --- | --- | --- |
| HRA88P_22_5 | hip_pitch / hip_roll / knee_pitch | 140 | 20.9 | 120 | 5 |
| HRA88P_14_3 | hip_yaw / waist_yaw | 90 | 32.9 | 100 | 5 |
| HRA58P_Parallel | ankle_*（并联，力矩×2） | 25×2 | 31.4 | 30 | 3 |
| HRA55P_Parallel | waist_roll / waist_pitch（并联，力矩×2） | 55×2 | 20.9 | 120 | 5 |
| HRA58P_Serial | shoulder_* / elbow_* | 25 | 31.4 | 50 | 2 |
| HTM4438_30 | wrist_pitch / wrist_roll | 10 | 7.85 | 5 | 1 |

> `armature`、`friction`、`viscous_friction`、`dynamic_friction` 等参数为下肢关节的辨识结果，详见 `robots/omni_29dof_nohead_noshoe_dcmotor_identified.py`。饱和力矩 `saturation_torque` 由额定/峰值功率自动推算（见 `robots/joint_cfg.py`）。

---

## 3. 文件结构

```text
omni_29dof_v260705/
├── readme.md                         # 本说明文档
├── assets/                           # 模型资源
│   ├── __init__.py                   # 定义 ASSET_DIR（资源根目录）
│   └── omni_29dof_nohead_noshoe/
│       ├── urdf/
│       │   └── omni_29dof_nohead_noshoe_merged_modify_feet.urdf  # 主 URDF（固定关节已合并，足部经修改）
│       └── meshes/                   # 32 个 STL 网格文件
├── robots/                           # 机器人 ArticulationCfg 配置
│   ├── joint_cfg.py                  # JointCfg 电机基类（含饱和力矩推算）
│   ├── tiangong.py                   # 各电机型号参数（HRA55P/HRA58P/HRA88P/HTM4438）
│   └── omni_29dof_nohead_noshoe_dcmotor_identified.py  # 主配置：OMNI_DCMOTOR_IDENTIFIED_CFG
└── actuators/                        # 自定义执行器模型
    ├── __init__.py
    └── actuators_pd.py               # DelayedDCMotor / DelayedDCMotorCfg（带延迟的直流电机）
```

### 关键对象

- `OMNI_DCMOTOR_IDENTIFIED_CFG`：完整的 `ArticulationCfg`，可直接用于场景中生成机器人。
- `OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE`：各关节动作缩放系数（默认 0.25），用于将策略网络输出映射到关节目标。
- `ASSET_DIR`：资源目录常量，配置中通过它定位 URDF/meshes。

---

## 4. 如何导入模型

### 4.1 前置条件

将本仓库根目录（`omni_29dof_v260705/`）加入 `PYTHONPATH`，以保证 `assets`、`robots`、`actuators` 三个包可被正确 import。例如：

```bash
export PYTHONPATH=<path to your assets>/omni_29dof_v260705:$PYTHONPATH
```

或在你的训练工程中，把这三个文件夹拷贝到工程包路径下并保持相对引用关系不变。

### 4.2 最简导入示例

```python
from robots.omni_29dof_nohead_noshoe_dcmotor_identified import (
    OMNI_DCMOTOR_IDENTIFIED_CFG,
    OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE,
)

# 在场景中生成机器人：将 prim_path 指向你的环境命名空间
robot_cfg = OMNI_DCMOTOR_IDENTIFIED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

### 4.3 在 InteractiveScene 中使用

```python
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from robots.omni_29dof_nohead_noshoe_dcmotor_identified import OMNI_DCMOTOR_IDENTIFIED_CFG


@configclass
class OmniSceneCfg(InteractiveSceneCfg):
    robot = OMNI_DCMOTOR_IDENTIFIED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # ... 地形、光照、传感器等
```

首次运行时，Isaac Lab 会根据 URDF 自动转换生成 USD 并缓存（`assets/omni_29dof_nohead_noshoe/.asset_hash` 用于判断资源是否变化）。转换过程会打印各电机注册信息（来自 `JointCfg`），属正常输出。

### 4.4 训练中使用动作缩放

`OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE` 提供了每个关节的动作缩放（默认 0.25），可在动作项配置中使用：

```python
from robots.omni_29dof_nohead_noshoe_dcmotor_identified import OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE

# 例如在 JointPositionActionCfg 中作为 scale 传入
# action = scale * policy_output + default_joint_pos
```

---

## 5. 常见问题（FAQ）

- **导入报错找不到 `assets` / `robots` / `actuators`**：确认已将仓库根目录加入 `PYTHONPATH`（见 4.1）。
- **`UrdfConverterCfg` / `PDGainsCfg` 相关 API 报错**：多为 Isaac Lab / Isaac Sim 版本过低，请升级到 2.2.0 / 5.0.0 及以上。
- **网格加载失败**：URDF 中 mesh 使用相对路径 `../meshes/*.STL`，请勿改变 `urdf/` 与 `meshes/` 的相对目录关系。
- **踝/腰关节力矩偏大**：并联结构关节的 `saturation_effort` / `effort_limit` 已按 ×2 处理，属预期设计。
