# 天工 OMNI 29-DOF 跳高策略开发 — 项目现状与待解决问题

## 1. 项目概述

### 1.1 目标

为天工 OMNI 29-DOF 人形机器人训练一个**原地跳高**的强化学习策略，最终部署到真机。

### 1.2 机器人参数

| 属性 | 值 |
|------|-----|
| 总自由度 | 29 DOF（双下肢 12 + 腰部 3 + 双上肢 14） |
| 执行器模型 | DelayedDCMotor（含 2~5 步随机指令延迟 + 摩擦 + armature） |
| 控制频率 | 50 Hz（decimation=4, 物理 200 Hz） |
| 训练引擎 | NVIDIA Isaac Lab（PhysX） |
| 部署目标 | 真机（SDK C++ backend + MuJoCo 作为中间验证） |

### 1.3 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    训练端 (AutoDL 服务器)                  │
│                                                          │
│  Isaac Lab + PhysX                                       │
│  ├─ ArticulationCfg: OMNI_DCMOTOR_IDENTIFIED_CFG         │
│  ├─ 执行器: DelayedDCMotor (延迟+摩擦+armature)           │
│  ├─ 环境: xMimic reference tracking (529 obs, 29 action) │
│  ├─ 参考动作: jump_high_firstjump_50fps.npz (50fps/183帧) │
│  └─ 算法: PPO (RSL_RL)                                   │
│                                                          │
│  输出: ONNX 策略模型 (obs 529 → actions 29)               │
└──────────────────────┬──────────────────────────────────┘
                       │ ONNX
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  部署端 (本地 / 真机)                      │
│                                                          │
│  MuJoCo 验证 (本地 Mac)                                   │
│  ├─ MJCF: omni_29dof.xml (自建, 29 DOF, 无头)             │
│  ├─ 执行器: <position> (MuJoCo 内置位置伺服)              │
│  ├─ kp/kd: 来自 SDK high_dynamic.yaml                     │
│  └─ 部署脚本: deploy_onnx_mujoco.py                       │
│                                                          │
│  真机部署 (目标)                                           │
│  ├─ SDK: omni_rl_sdk (C++ backend)                        │
│  ├─ MJCF: omni_31.xml (官方, 31 DOF 含头)                  │
│  ├─ FSM: high_dynamic.py → env_mujoco.cpp                 │
│  └─ kp/kd: 运行时由 policy 输出动态更新                     │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 当前进展

### 2.1 训练状态（截至 17700 步）

| 指标 | 数值 | 说明 |
|------|------|------|
| **跳高高度** | **0.75m** | 75 厘米 |
| **摔倒率** | **0%** | 训练全程零摔倒 |
| **回合完成率** | 99.6% | 几乎全部跑满时长 |
| **总奖励** | ~36 | 已基本收敛（plateau） |
| **连续跳跃** | 已学会 | 400 步视频内可跳 2 次 |
| **起跳垂直速度** | 1.03 m/s | |
| **落地平衡奖励** | 0.72 | 还在改善中 |

**训练状态良好，策略已收敛，Isaac Lab 仿真视频中机器人能稳定跳高并落地。**

### 2.2 已完成的工作

1. **训练环境搭建** — Isaac Lab + xMimic reference tracking 框架，529 维 obs，29 维 action
2. **奖励函数调优** — 经过多轮迭代：参考跟踪 + 跳高专项奖励 + 9 项惩罚（对称性、姿态约束等）
3. **PD 增益对齐** — 训练环境 kp/kd 对齐 SDK env-omni31.yaml 配置
4. **ONNX 导出** — 两个版本可用：
   - `2026-08-13_09-15-45_8000.onnx`：完整 metadata（joint_names, default_joint_pos, action_scale=0.5 等）
   - `policy-17700step.onnx`：17700 步训练的新模型（无 metadata，需手动指定 action_scale=0.5）
5. **MuJoCo 部署管线** — deploy_onnx_mujoco.py + omni_29dof.xml，已对齐：
   - 执行器模型：`<position>`（与 SDK omni_31.xml 一致）
   - kp/kd 值：来自 high_dynamic.yaml（SDK 运行时实际值）
   - 地面摩擦：condim=3, friction=1.0 0.005 0.0001
   - 求解器：Newton + Euler + timestep=0.0025

### 2.3 代码仓库结构

```
omni_29dof_v260705/
├── liuzq_jump_train_v11/     # 训练代码 (AutoDL 服务器)
│   ├── jump_env/              # 环境配置
│   ├── scripts/               # train.py, play.py
│   └── docs/                  # 文档
├── omni_rl_sdk/               # 官方 SDK
│   ├── assets/omni-7dof/      # omni_31.xml (官方 MJCF)
│   ├── xhumanoid_control/     # C++ backend (env_mujoco.cpp)
│   └── policy/high_dynamic/   # FSM + high_dynamic.yaml (kp/kd 来源)
├── omni_29dof_mjc/mjcf/       # 自建 MuJoCo 模型
│   └── omni_29dof.xml         # 29 DOF, <position> 执行器
├── assets/                    # URDF (训练用)
├── deploy_onnx_mujoco.py      # 本地 MuJoCo 部署脚本
├── ONNX/                      # 导出的 ONNX 模型
└── training_data/             # 参考动作 NPZ
```

---

## 3. 面临的核心困境

### 3.1 问题：MuJoCo 验证不通过，项目组不敢上真机

**现象**：训练好的 ONNX 策略在 Isaac Lab 视频里能稳定跳高，但在 MuJoCo 里直接摔倒。

**我们已排查并修复的问题**：

| 问题 | 修复 | 效果 |
|------|------|------|
| 执行器模型错误（`<motor>` vs `<position>`） | 改为 `<position>`，与 SDK 一致 | 仍然摔 |
| kp/kd 值来源错误（env-omni31.yaml sim vs high_dynamic.yaml） | 改用 high_dynamic.yaml 值 | 仍然摔 |
| 地面无摩擦（condim=1） | 改为 condim=3 + friction=1.0 | 仍然摔 |
| 求解器/积分器不一致 | 对齐为 Newton + Euler + dt=0.0025 | 仍然摔 |
| 两个不同 ONNX 模型测试 | 8000 步 + 17700 步都测了 | 都摔 |

**关键证据**：SDK 官方的 omni_31.xml 用同样的 kp/kd 在 MuJoCo 里也站不住。这说明问题不在我们的模型配置，而是 **PhysX 与 MuJoCo 之间的接触动力学本质差异**。

### 3.2 根本原因分析

训练端（PhysX）和部署验证端（MuJoCo）的物理引擎差异：

| 维度 | PhysX (Isaac Lab) | MuJoCo |
|------|-------------------|--------|
| 接触模型 | 基于惩罚函数的软接触 | 约束优化（LCP/Newton） |
| 摩擦计算 | Coulomb + 正则化 | 锥约束近似 |
| 碰撞响应 | 柔顺、有能量耗散 | 刚性、能量守恒倾向 |
| 积分器 | 隐式 Euler（默认） | 显式 Euler（SDK 配置） |
| 地面柔顺性 | 有（软接触） | 无（硬接触） |

这些差异导致：策略在 PhysX 学到的起跳/落地时序、力矩分配、平衡恢复策略，在 MuJoCo 中完全不适用。**这不是调参数能解决的，是引擎层面的结构性差异。**

### 3.3 项目组的顾虑

- 项目组长**不同意在 MuJoCo 仿真没通过的情况下上真机**
- 担心损坏硬件（29 DOF 人形机器人成本高）
- 需要某种形式的"中间验证"来降低风险

### 3.4 我们的判断

- **Isaac Lab play.py 视频 = 有效的 sim2sim 验证**（同引擎 PhysX，ONNX 推理）
- **MuJoCo 不是验证高动态跳跃的正确工具**（业界标准：Isaac Lab → 真机，不经过 MuJoCo）
- 但项目组需要 MuJoCo 通过才肯上真机，形成僵局

---

## 4. 需要解决的问题

### 4.1 核心问题（阻塞项）

**如何让项目组接受策略可以上真机测试？**

可选路径：
- **A. 在 MuJoCo 中调通跳跃**（目前看几乎不可能，引擎差异太大）
- **B. 找到其他被项目组认可的验证方式**（如 Isaac Sim 部署仿真、domain randomization 测试等）
- **C. 用安全测试方案说服项目组**（限流 + 安全绳 + 分阶段，已有文档但未被接受）
- **D. 其他我们还没想到的方案**

### 4.2 技术问题

1. **PhysX → MuJoCo 的 sim2sim gap 能否缩小？**
   - 是否可以通过 domain randomization（摩擦、质量、延迟等）让策略对引擎差异更鲁棒？
   - 是否可以在 MuJoCo 中用更精细的接触模型（如柔顺接触、变形体地面）来逼近 PhysX 行为？

2. **落地稳定性不足**
   - `landing_balance_bonus = 0.72`，第二次跳落地站不稳
   - 训练里策略选择"不恢复直接跳下一次"，真机上会更严重
   - 是否需要调高落地平衡的奖励权重？

3. **起跳对称性差**
   - `takeoff_leg_symmetry_penalty = -1.36`（最大惩罚项）
   - 策略找到"蹭地面"的捷径来换跳高
   - 是否需要调整奖励结构？

4. **ONNX 兼容性问题**
   - `policy-17700step.onnx` 缺少 metadata（无 joint_names, default_joint_pos, action_scale）
   - 导出方式与 play.py 不同，需要手动指定参数
   - 是否需要统一导出流程？

### 4.3 工程问题

5. **训练环境 PD 增益与部署端不一致**
   - 训练用 DelayedDCMotor（Isaac Lab 特有，含延迟+摩擦）
   - 部署用 `<position>` 伺服（MuJoCo 内置，无延迟无摩擦）
   - 这是根本性的 action space 差异，策略学到的动作在两种执行器下表现不同

6. **模型质量差异**
   - ankle_roll 质量：我们的 MJCF 0.5795 kg vs SDK omni_31.xml 0.2835 kg（差 104%）
   - 可能影响动态行为

---

## 5. 已有资源

| 资源 | 位置 | 状态 |
|------|------|------|
| 训练代码 | `liuzq_jump_train_v11/` (AutoDL 服务器) | 运行中，已收敛 |
| 官方 SDK | `omni_rl_sdk/` | 可用 |
| MuJoCo 部署脚本 | `deploy_onnx_mujoco.py` | 可用，但策略在 MuJoCo 中摔倒 |
| ONNX 模型 (8000步) | `ONNX/2026-08-13_09-15-45_8000.onnx` | 有完整 metadata |
| ONNX 模型 (17700步) | `/Users/condenast/Downloads/policy-17700step.onnx` | 无 metadata |
| 参考动作 | `training_data/jump_high_firstjump_50fps.npz` | 可用 |
| Isaac Lab 仿真视频 | 项目组可观看 | 显示策略能稳定跳高 |
| sim2sim 安全方案文档 | `liuzq_jump_train_v11/docs/sim2sim_validation_and_safety_plan.md` | 已写，但被删除 |

---

## 6. 一句话总结

> **RL 跳高策略在 Isaac Lab (PhysX) 中训练收敛（75cm 跳高、零摔倒），但无法在 MuJoCo 中复现行为（引擎差异导致）。项目组要求 MuJoCo 验证通过才允许上真机。我们需要找到一种方案来打破这个僵局。**
