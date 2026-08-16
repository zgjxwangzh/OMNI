# OMNI 29-DOF 项目总览

> 面向初学者的全方位项目介绍 | 最后更新：2026-08-12

---

## 一、这个项目是什么？

**一句话概括**：让 OMNI 29-DOF 人形机器人通过强化学习（RL）学会模仿人类动作（跳高、跑步、翻箱子等）。

**通俗解释**：
1. 先录制真人做动作的视频（动捕数据 BVH 格式）
2. 把真人动作"翻译"成机器人能理解的关节角度（重定向）
3. 用强化学习训练机器人"模仿"这些动作（训练策略网络）
4. 训练好的策略可以部署到真机上（sim-to-real）

---

## 二、核心概念（初学者必读）

### 2.1 什么是 OMNI 29-DOF？

OMNI 是一款人形机器人，"29-DOF"表示它有 **29 个自由度**（29 个可独立控制的关节）：

| 部位 | 关节 | 数量 |
|------|------|------|
| 左/右腿 | 髋关节(pitch/roll/yaw) + 膝盖 + 踝关节(pitch/roll) | 6 × 2 = 12 |
| 腰部 | yaw + roll + pitch | 3 |
| 左/右臂 | 肩(pitch/roll/yaw) + 肘(pitch/yaw) + 腕(pitch/roll) | 7 × 2 = 14 |
| **合计** | | **29** |

> 头部和鞋部已移除（固定件），不占用自由度。

### 2.2 什么是 BVH？

**BVH（Biovision Hierarchy）** 是一种动捕数据格式，记录了人体骨架每个关节的旋转和根节点位置。简单理解：**真人动作的数字化记录**。

### 2.3 什么是重定向（Retargeting）？

真人骨架和机器人骨架不一样（关节数量、长度、活动范围都不同）。**重定向**就是把真人动作"翻译"成机器人能执行的动作，同时尽量保持动作的自然性和可行性。

### 2.4 什么是 FK（正运动学）？

**Forward Kinematics（正运动学）**：已知关节角度，计算机器人每个部位（刚体）在世界坐标系中的位置和姿态。训练时需要所有 30 个刚体的完整位姿数据，而不仅仅是关节角度。

### 2.5 什么是强化学习（RL）训练？

让机器人在仿真环境中不断试错，通过奖励信号学习如何跟踪参考动作。核心要素：
- **状态（Observation）**：机器人当前感知到的信息（关节角度、速度、刚体位姿等）
- **动作（Action）**：机器人输出的关节目标位置
- **奖励（Reward）**：动作有多好（跟踪误差越小奖励越高）
- **策略（Policy）**：神经网络，输入状态 → 输出动作

### 2.6 什么是 sim-to-real？

在仿真（Simulation）中训练策略，然后部署到真实（Real）机器人上。关键挑战是仿真和现实之间的差异（domain gap），需要通过**域随机化（Domain Randomization）**来弥合。

---

## 三、技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据管线（Data Pipeline）                   │
│                                                                   │
│  BVH 动捕数据 → 重定向 → retargeted NPZ → FK 转换 → training NPZ  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
─────────────────────────────────────────────────────────────────┐
│                       训练框架（Training）                         │
│                                                                   │
│  omni_mimic (whole_body_tracking) + rsl_rl (PPO) + Isaac Lab    │
│                                                                   │
│  输入：training NPZ（参考动作轨迹）                                 │
│  输出：策略网络权重（.pt 文件）                                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       验证与部署（Deploy）                          │
│                                                                   │
│  play.py 可视化 → ONNX 导出 → omni_rl_sdk → 真机部署              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 仿真引擎 | NVIDIA Isaac Sim | 5.0 / 5.1 |
| RL 框架 | Isaac Lab | 2.3.2 |
| 训练算法 | PPO (rsl_rl) | 项目定制版 |
| 训练框架 | omni_mimic (whole_body_tracking) | 项目组提供 |
| 机器人模型 | URDF + DelayedDCMotor | 29-DOF |
| 动捕格式 | BVH → NPZ | 自研管线 |
| 验证工具 | MuJoCo | 运动学/物理验证 |
| 部署框架 | omni_rl_sdk | ONNX + FSM |

### 3.2 执行器模型

机器人使用 **DelayedDCMotor**（带延迟的直流电机模型），这是最接近真实硬件的模型：
- 模拟真实电机的力矩限制、速度限制
- 加入 2~5 步的随机指令延迟（模拟通信延迟）
- 包含摩擦、惯性等辨识参数
- 用于 sim-to-real 迁移

---

## 四、项目目录结构（概览）

```
omni_29dof_v260705/
├── 机器人模型层
│   ├── assets/          # URDF + 3D 网格
│   ├── robots/          # 机器人配置（ArticulationCfg）
│   ── actuators/       # 电机模型（DelayedDCMotor）
│
├── 数据管线层
│   ├── bvh_retarget.py              # BVH → 重定向 NPZ
│   ├── retargeted/                  # 重定向输出（29 个 NPZ）
│   └── omni_mimic/scripts/          # FK 转换 + 训练脚本
│
├── 训练框架层
│   └── omni_mimic/                  # 官方 mimic 训练框架
│       ├── source/whole_body_tracking/  # 环境、奖励、观测
│       ├── source/rsl_rl/               # PPO 训练器
│       └── scripts/rsl_rl/              # train.py / play.py
│
├── 历史/参考层
│   ├── TienKung-Lab/      # 旧版 AMP 训练框架（走路用）
│   └── omni_rl_sdk/       # 部署框架（ONNX + FSM）
│
└── 文档层
    ├── readme.md            # 本文档
    ├── PROJECT_OVERVIEW.md  # 项目总览（本文件）
    ├── DIRECTORY_GUIDE.md   # 目录详解
    ├── PIPELINE_GUIDE.md    # 数据管线详解
    ├── TRAINING_GUIDE.md    # 训练指南
    ├── ENVIRONMENT_SETUP.md # 环境搭建
    └── ROADMAP.md           # 路线图
```

> 每个目录的详细说明见 [DIRECTORY_GUIDE.md](DIRECTORY_GUIDE.md)

---

## 五、当前进展

| 阶段 | 状态 | 说明 |
|------|------|------|
| BVH 重定向 | ✅ 完成 | 29 个动作全部重定向 |
| MuJoCo 验证 | ✅ 完成 | 运动学 + 物理跟踪验证通过 |
| FK 数据转换 | ✅ 完成 | 29 个 training NPZ 生成 |
| omni_mimic 训练 | 🔄 进行中 | 跳高06 (AutoDL) + 跑步01 (阿里云) |
| 策略可视化 |  待验证 | play.py noVNC 可视化 |
| ONNX 导出 |  待训练完成 | 训练收敛后导出 |
| 真机部署 |  公司负责 | 训练完成后由公司部署 |

---

## 六、双机协作

| 服务器 | GPU | 用途 | Isaac Lab | Isaac Sim |
|--------|-----|------|-----------|-----------|
| **AutoDL** | RTX 4090 | 训练主力 | 2.3.2 | 5.0 |
| **阿里云 DSW** | A10 | 训练 + 可视化 | 2.3.2 | 5.1.0 |

- AutoDL：无 GUI，纯 headless 训练，TensorBoard 监控
- 阿里云 DSW：有 noVNC 桌面，可训练 + GUI 可视化验证

---

## 七、快速导航

| 我想了解... | 看这个文档 |
|------------|-----------|
| 每个目录/文件是干什么的 | [DIRECTORY_GUIDE.md](DIRECTORY_GUIDE.md) |
| 数据从 BVH 到训练怎么流转的 | [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) |
| 怎么搭建训练环境 | [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) |
| 怎么训练、怎么看指标 | [TRAINING_GUIDE.md](TRAINING_GUIDE.md) |
| 项目做到哪了、接下来做什么 | [ROADMAP.md](ROADMAP.md) |
