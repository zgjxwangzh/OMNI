# OMNI 29-DOF 人形机器人训练环境技术说明

## 一、最终目标

让 **OMNI 29-DOF 人形机器人** 在物理仿真中学会自主走路/跑步，最终将训练好的策略部署到真机上。

核心技术路线：**强化学习 (RL) + 对抗性动作先验 (AMP)**

```
人体动捕数据 → 动作重定向 → 参考步态
                                ↓
物理仿真 (Isaac Sim) ← RL 训练 (PPO) → 策略网络 (.pt)
        ↑                                    ↓
   OMNI 机器人模型                     真机部署
```

---

## 二、技术栈全景

```
┌─────────────────────────────────────────────────────┐
│                  应用层 (我们写的)                     │
│  OMNI 机器人配置 / 环境注册 / 训练脚本 / 集成脚本      │
├─────────────────────────────────────────────────────┤
│              TienKung-Lab (训练框架)                  │
│  奖励函数 / AMP 动作先验 / 环境封装 / PPO 训练循环     │
├─────────────────────────────────────────────────────┤
│              Isaac Lab v2.3.0 (机器人框架)             │
│  场景管理 / 传感器 / 执行器 / 任务注册 / RL 接口       │
├─────────────────────────────────────────────────────┤
│              Isaac Sim 5.1.0 (仿真引擎)               │
│  PhysX 物理引擎 / RTX 渲染 / USD 场景 / GPU 并行      │
├─────────────────────────────────────────────────────┤
│              基础设施                                  │
│  Python 3.11 / PyTorch 2.7 / CUDA 12.6 / RTX 4090   │
└─────────────────────────────────────────────────────┘
```

---

## 三、各模块详细说明

### 1. Isaac Sim 5.1.0（NVIDIA 物理仿真引擎）

| 项目 | 说明 |
|------|------|
| 是什么 | NVIDIA 的机器人仿真平台，提供逼真的物理环境 |
| 核心能力 | PhysX 5 物理引擎（重力、碰撞、摩擦）、GPU 并行仿真（同时跑 4096 个机器人） |
| 安装方式 | `pip install isaacsim==5.1.0`（约 10GB） |
| 依赖 | Python 3.11（强制）、Vulkan（GPU 图形接口）、CUDA |
| 在本项目中的角色 | 提供“虚拟世界”——机器人在里面摔倒、站起来、学走路 |
| 来源 | NVIDIA PyPI: `https://pypi.nvidia.com`（官方文档: https://docs.isaacsim.omniverse.nvidia.com） |

### 2. Isaac Lab v2.3.0（机器人训练框架）

| 项目 | 说明 |
|------|------|
| 是什么 | 基于 Isaac Sim 的机器人学习框架（NVIDIA 官方） |
| 核心能力 | 标准化环境接口 (Gym API)、场景配置、传感器/执行器抽象、RL 训练集成 |
| 安装方式 | 源码安装 `./isaaclab.sh --install` |
| 依赖 | Isaac Sim 5.1.0 |
| 在本项目中的角色 | 提供“训练基础设施”——定义观测空间、动作空间、奖励接口 |
| 来源 | https://github.com/isaac-sim/IsaacLab （下载 v2.3.0 版本） |

### 3. TienKung-Lab（人形机器人 RL 训练框架）

| 项目 | 说明 |
|------|------|
| 是什么 | 北京人形机器人创新中心开源的人形机器人训练框架（比赛官方指定） |
| 核心能力 | 人形机器人专用奖励函数、AMP 对抗性动作先验、步态训练 pipeline |
| 安装方式 | 源码 `pip install -e .` |
| 依赖 | Isaac Lab ≥ 2.1、rsl_rl（定制版） |
| 在本项目中的角色 | 提供"训练方法论"——怎么定义走路奖励、怎么让步态自然 |
| 来源 | https://github.com/Open-X-Humanoid/TienKung-Lab |

#### TienKung-Lab 内部结构

```
TienKung-Lab/
├── legged_lab/
│   ├── envs/
│   │   ├── base/          ← 基础环境类 (BaseEnv)
│   │   ├── tienkung/      ← 天工机器人环境（模板）
│   │   └── omni/          ← OMNI 机器人环境（我们创建的）
│   │       ├── omni_env.py       ← 环境主逻辑
│   │       ├── walk_cfg.py       ← 走路任务配置（奖励权重等）
│   │       ├── run_cfg.py        ← 跑步任务配置
│   │       └── datasets/         ← AMP 动作数据
│   ├── assets/
│   │   └── omni_29dof/    ← OMNI 机器人模型
│   ├── scripts/
│   │   ├── train.py       ← 训练入口
│   │   └── play.py        ← 回放/评估
│   └── utils/
│       └── task_registry.py ← 任务注册系统
└── rsl_rl/                ← 定制版 RL 库（PPO + AMP）
```

### 4. rsl_rl（强化学习库）

| 项目 | 说明 |
|------|------|
| 是什么 | ETH 苏黎世 RSL 实验室的 RL 训练库（TienKung-Lab 含定制版） |
| 核心能力 | PPO 算法实现、OnPolicyRunner 训练循环、AMP Loader |
| 在本项目中的角色 | 实际执行"学习"——更新神经网络权重 |

### 5. OMNI 29-DOF 机器人模型

| 项目 | 说明 |
|------|------|
| 是什么 | 比赛用的人形机器人数字模型（29 个自由度） |
| 包含 | URDF（结构描述）、STL 网格（3D 外形）、电机参数（6 种型号） |
| 关节分布 | 左腿 6 + 右腿 6 + 腰 3 + 左臂 7 + 右臂 7 = 29 |
| 电机型号 | HRA88P_22_5（腿）、HRA88P_14_3（髋/腰）、HRA58P（踝/臂）、HRA55P（腰）、HTM4438（腕） |
| 特殊设计 | DelayedDCMotor（2-5 步随机延迟，模拟真实通信延迟，提升 sim-to-real） |

### 6. PyTorch 2.7.0 + CUDA

| 项目 | 说明 |
|------|------|
| 角色 | 深度学习框架，策略网络的前向/反向传播 |
| 版本 | 2.7.0+cu126（Isaac Sim 自带，无需单独安装） |
| GPU | NVIDIA RTX 4090 (24GB) |

---

## 四、依赖关系图

```
OMNI 机器人模型 (URDF + 电机参数)
        │
        ▼
TienKung-Lab ←── rsl_rl (PPO + AMP)
        │
        ▼
   Isaac Lab v2.3.0
        │
        ▼
   Isaac Sim 5.1.0 ←── Vulkan (GPU 图形)
        │
        ▼
   PyTorch 2.7 + CUDA 12.6
        │
        ▼
   Python 3.11 + RTX 4090
```

**版本对齐要求**（非常严格）：
- Python 必须 3.11（Isaac Sim 5.1 强制）
- Isaac Sim 5.1.0 ↔ Isaac Lab v2.3.0（官方对应）
- PyTorch 2.7.x + CUDA ≥ 12.6

---

## 五、训练流程

```
1. 加载 OMNI URDF → Isaac Sim 创建 4096 个并行机器人副本
2. 每个时间步：
   a. 读取机器人状态（关节角、速度、IMU、脚底力）→ 观测向量
   b. 策略网络 (MLP) 输入观测 → 输出 29 个关节目标角度
   c. PD 控制器执行力矩 → PhysX 物理仿真一步
   d. 计算奖励（高度保持 + 速度跟踪 + 节能 + 步态自然度...）
3. 每 24 步收集一批数据 → PPO 更新策略网络
4. 重复 50000 次迭代 → 策略收敛 → 保存 model_XXXX.pt
```

---

## 六、训练产出

| 文件 | 大小 | 说明 |
|------|------|------|
| `model_4900.pt` | ~26MB | 策略网络权重（3 层 MLP, 29 维输出） |
| `params/env.yaml` | 小 | 环境配置（观测/动作维度、奖励权重） |
| `params/agent.yaml` | 小 | 训练超参（学习率、batch size） |

**策略网络结构**：
- 输入：~100+ 维观测（关节状态 + IMU + 上一步动作 + 指令速度）
- 隐藏层：3 × 512 neurons (MLP)
- 输出：29 维（每个关节的目标角度偏移，× 0.25 缩放）

---

## 七、运行环境

| 项目 | 配置 |
|------|------|
| 云平台 | AutoDL 按量付费 GPU 实例 |
| GPU | NVIDIA RTX 4090 (24GB) |
| 系统 | Ubuntu 22.04 (容器) |
| 磁盘 | 系统盘 30GB + 数据盘 50GB + 共享存储 200GB |
| 训练耗时 | 约 7 小时 / 5000 迭代（4096 并行环境） |
| 特殊修复 | AutoDL 容器需手动创建 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`（Vulkan EGL 配置） |

---

## 八、关键命令

```bash
# 训练
python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=4096 --logger=tensorboard

# 监控
tensorboard --logdir=logs --host 0.0.0.0 --port 6006

# 从断点继续
python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=4096 \
    --load_run=<文件夹名> --checkpoint=model_XXXX.pt

# 评估
python legged_lab/scripts/play.py --task=omni_walk --headless --num_envs=1 \
    --load_run=<文件夹名> --checkpoint=model_XXXX.pt
```

---

## 九、当前训练成果

| 指标 | 训练前 | 训练后 (model_4900.pt) |
|------|:---:|:---:|
| mean_reward | -5 | +2（持续上升） |
| episode_length | 60 步 | **964 步** |
| 含义 | 走几步就倒 | 连续走 964 步不摔倒 |

---

## 十、后续工作

1. **动作重定向 (GMR)**：将人体动捕数据重定向到 OMNI 骨架，替换当前的 TienKung 临时数据，获得更自然的步态
2. **调参优化**：调整奖励权重，平衡速度/稳定性/能耗
3. **Sim-to-Real**：比赛方提供部署工具后，将 `.pt` 导出并部署到真机
4. **跑步训练**：`--task=omni_run`
