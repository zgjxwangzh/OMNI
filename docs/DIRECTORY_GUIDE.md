# 目录结构详解

> 每个目录和文件的作用说明 | 最后更新：2026-08-12

---

## 一、机器人模型层（底层基础）

这三个目录定义了机器人在仿真中的"物理存在"。

### `assets/` — 机器人 3D 模型资源

```
assets/
└── omni_29dof_nohead_noshoe/
    ├── urdf/
    │   └── omni_29dof_nohead_noshoe_merged_modify_feet.urdf  ← 主 URDF 文件
    ├── meshes/
    │   ├── base_link.STL          ← 躯干
    │   ├── hip_pitch_l_link.STL   ← 左髋
    │   ├── ... (共 32 个 STL 文件)
    └── .asset_hash                ← Isaac Lab 缓存（判断资源是否变化）
```

**作用**：定义机器人的几何形状和物理属性。
- **URDF**：描述机器人的连杆、关节、质量、惯性等
- **meshes/**：32 个 STL 网格文件，定义每个连杆的外观
- **.asset_hash**：Isaac Lab 首次运行时生成，用于判断 URDF 是否被修改过

**关键约束**：`urdf/` 和 `meshes/` 的相对路径不能改（URDF 中引用 `../meshes/*.STL`）。

---

### `robots/` — 机器人配置（ArticulationCfg）

```
robots/
├── joint_cfg.py                              ← 电机基类（JointCfg）
├── tiangong.py                               ← 6 种电机型号参数
└── omni_29dof_nohead_noshoe_dcmotor_identified.py  ← 主配置
```

| 文件 | 作用 |
|------|------|
| `joint_cfg.py` | `JointCfg` 类，定义电机基本参数（额定力矩、峰值力矩、转速等），自动计算饱和力矩 |
| `tiangong.py` | 6 种真实电机型号的参数：HRA88P_22_5、HRA88P_14_3、HRA58P、HRA55P、HTM4438_30 |
| `omni_29dof_nohead_noshoe_dcmotor_identified.py` | **核心文件**：`OMNI_DCMOTOR_IDENTIFIED_CFG`（完整机器人配置）+ `OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE`（动作缩放系数 0.25） |

**关键对象**：
- `OMNI_DCMOTOR_IDENTIFIED_CFG`：Isaac Lab 的 `ArticulationCfg`，可直接在场景中生成机器人
- `OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE`：每个关节的动作缩放系数（默认 0.25），将策略输出映射到关节目标

---

### `actuators/` — 自定义执行器模型

```
actuators/
├── __init__.py
└── actuators_pd.py    ← DelayedDCMotor 实现
```

**作用**：定义 `DelayedDCMotorCfg`（带延迟的直流电机模型）。

**为什么需要自定义**：Isaac Lab 自带的电机模型不够真实。`DelayedDCMotor` 增加了：
- 2~5 步的随机指令延迟（模拟真实通信延迟）
- 摩擦、惯性、粘性摩擦等辨识参数
- 力矩饱和、速度限制

**这是 sim-to-real 的关键**：仿真中用的电机模型越接近真实硬件，训练出的策略越容易部署到真机。

---

## 二、数据管线层（中层处理）

### `bvh_retarget.py` — BVH 重定向脚本

**作用**：将 BVH 动捕数据转换为机器人关节角度（NPZ 格式）。

**输入**：BVH 文件（真人动作）
**输出**：`retargeted/*.npz`（机器人关节角度，motor order）

**核心逻辑**：
1. 解析 BVH 骨架和动画数据
2. 将真人关节旋转映射到机器人关节
3. 处理关节顺序（motor order：按身体部位分组）
4. 输出 NPZ 文件（含 joint_pos、joint_vel、root_pos、root_rot 等）

---

### `retargeted/` — 重定向输出目录

```
retargeted/
├── 跳高01_chr00.npz    ← 334 帧
├── 跳高02_chr00.npz    ← 397 帧
├── ...
├── 跑步01_chr00.npz    ← 525 帧
├── 跨栏02_chr00.npz    ← 477 帧
├── 翻箱子01_chr00.npz  ← 1385 帧
├── 匍匐前进1_chr00.npz ← 2623 帧（最长）
└── ... (共 29 个 NPZ)
```

**格式**：每个 NPZ 包含：
- `fps`：帧率（标量）
- `joint_pos`：关节角度 `(T, 29)`
- `joint_vel`：关节速度 `(T, 29)`
- `root_positions`：根节点位置 `(T, 3)`
- `root_rotations`：根节点旋转 `(T, 4)` 四元数

---

### `omni_mimic/scripts/retargeted_npz_to_training_npz.py` — FK 转换脚本

**作用**：将 retargeted NPZ 转换为训练所需的 training NPZ。

**为什么需要这一步**：训练需要所有 30 个刚体的完整位姿（位置、旋转、线速度、角速度），而 retargeted NPZ 只有关节角度。FK 转换通过 Isaac Sim 的正运动学计算这些信息。

**输入**：`retargeted/*.npz`
**输出**：`training_data/*_training.npz`

**输出格式**：
- `fps`：帧率
- `joint_pos` / `joint_vel`：关节数据 `(T, 29)`
- `body_pos_w`：30 个刚体世界坐标位置 `(T, 30, 3)`
- `body_quat_w`：30 个刚体世界坐标旋转 `(T, 30, 4)`
- `body_lin_vel_w`：30 个刚体线速度 `(T, 30, 3)`
- `body_ang_vel_w`：30 个刚体角速度 `(T, 30, 3)`

---

## 三、训练框架层（核心引擎）

### `omni_mimic/` — 官方 mimic 训练框架

这是项目组提供的完整训练框架，基于 Isaac Lab + rsl_rl。

```
omni_mimic/
├── README.md                          ← 框架使用说明
├── source/
│   ├── whole_body_tracking/           ← 训练环境定义
│   │   ── whole_body_tracking/
│   │       ├── tasks/tracking/
│   │       │   ├── mdp/
│   │       │   │   ├── commands.py       ← 动作加载、自适应采样
│   │       │   │   ├── rewards.py        ← 奖励函数（14 刚体跟踪误差）
│   │       │   │   ├── observations.py   ← 观测空间定义（529 维）
│   │       │   │   ├── terminations.py   ← 终止条件
│   │       │   │   └── events.py         ← 域随机化
│   │       │   ├── tracking_env_omni_cfg.py  ← 环境总配置
│   │       │   └── config/omni/
│   │       │       ├── flat_env_cfg.py       ← Flat/Box 场景配置
│   │       │       └── agents/rsl_rl_ppo_cfg.py  ← PPO 超参数
│   │       ── utils/
│   │           └── my_on_policy_runner.py  ← 自定义训练循环
│   │
│   ── rsl_rl/                        ← PPO 训练器（项目定制版）
│       └── rsl_rl/
│           └── runners/
│               ── on_policy_runner.py  ← 训练循环（已修复 Isaac Lab 2.3.2 兼容）
│
└── scripts/
    ├── rsl_rl/
    │   ├── train.py                   ← 训练入口（已修复 dump_pickle）
    │   └── play.py                    ← 可视化入口
    ── retargeted_npz_to_training_npz.py  ← FK 转换脚本
```

**关键组件说明**：

| 组件 | 作用 |
|------|------|
| `rewards.py` | 计算跟踪误差奖励（14 个关键刚体的位置、旋转、速度误差） |
| `observations.py` | 定义 529 维观测空间（关节角度/速度 + 刚体位姿 + 历史 + 命令） |
| `commands.py` | 加载参考动作、计算命令、自适应采样（优先训练困难片段） |
| `terminations.py` | 定义何时终止 episode（anchor 偏差过大、末端位置偏差等） |
| `events.py` | 域随机化（质量、质心、摩擦、电机参数、外力推送等） |
| `on_policy_runner.py` | PPO 训练循环（已适配 Isaac Lab 2.3.2 的 TensorDict） |
| `train.py` | 训练入口，解析命令行参数，启动训练 |
| `play.py` | 加载训练好的模型，在仿真中播放 |

---

## 四、历史/参考层

### `TienKung-Lab/` — 旧版 AMP 训练框架

**状态**：已废弃（用于走路动作的 AMP 训练）

**为什么保留**：
- 走路动作仍可用此框架训练
- 包含历史实验数据和模型
- 作为参考实现

**注意**：跳高、舞蹈等高动态动作使用 `omni_mimic` 框架，不用这个。

---

### `omni_rl_sdk/` — 部署框架

**作用**：将训练好的策略部署到仿真/真机。

```
omni_rl_sdk/
├── deploy_omni_sim_real/    ← 仿真部署主程序
── policy/
│   ├── loco_mode/           ← 走路策略（ONNX）
│   ── high_dynamic/        ← 高动态策略（reference tracking）
└── FSM/                     ← 状态机（切换不同策略）
```

**部署流程**：
1. 训练完成 → 导出 ONNX 模型
2. 放入 `omni_rl_sdk/policy/high_dynamic/`
3. FSM 状态机根据场景切换策略（走路 ↔ 跳高 ↔ 其他）
4. 仿真验证 → 真机部署

---

## 五、文档层

| 文件 | 内容 |
|------|------|
| `readme.md` | 项目总览 + 机器人模型文档（原始） |
| `docs/PROJECT_OVERVIEW.md` | 项目总览（面向初学者） |
| `docs/DIRECTORY_GUIDE.md` | 本文件：目录详解 |
| `docs/PIPELINE_GUIDE.md` | 数据管线详解 |
| `docs/TRAINING_GUIDE.md` | 训练指南 |
| `docs/ENVIRONMENT_SETUP.md` | 环境搭建 |
| `docs/ROADMAP.md` | 路线图 |
| `ALIYUN_SETUP.md` | 阿里云 DSW 专用部署指南 |

---

## 六、其他文件

| 文件 | 作用 | 状态 |
|------|------|------|
| `bvh_retarget.py` | BVH 重定向脚本 | ✅ 在用 |
| `batch_verify_mujoco.py` | 批量 MuJoCo 运动学验证 | ✅ 已用 |
| `physics_track_verify.py` | 物理跟踪验证（PD 控制） | ✅ 已用 |
| `verify_highdynamic_mujoco.py` | 单文件 MuJoCo 可视化 | ✅ 已用 |
| `run_walking_sim.py` | 走路策略仿真 | ✅ 就绪 |
| `convert_to_highdynamic.py` | 旧版格式转换 | ⚠️ 已被 FK 转换替代 |
| `setup_gpu_server.sh` | GPU 服务器一键安装 | ⚠️ 参考用 |
| `frames/` | MuJoCo 验证截图 | 历史产物 |
| `_archive/` | 归档（旧管线文件） | 已废弃 |
| `数据1/` | 原始动捕数据（23 个 BVH） | 原始输入 |
| `第一组 跳高 翻箱/` | 原始动捕数据（9 个 BVH） | 原始输入 |
