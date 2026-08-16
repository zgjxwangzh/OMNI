# 数据管线详解

> 从 BVH 动捕到训练完成的完整数据流转 | 最后更新：2026-08-12

---

## 一、管线总览

```
阶段 1: 数据准备          阶段 2: 训练              阶段 3: 部署
                                                                    
BVH → 重定向 → NPZ   →   FK 转换 → training NPZ  →  训练 → .pt  →  ONNX → 真机
  (本地)                  (GPU 服务器)               (GPU 服务器)    (公司)
```

---

## 二、阶段 1：数据准备（本地完成）

### 2.1 输入：BVH 动捕数据

**来源**：动作捕捉系统录制的真人动作。

**格式**：BVH 文件包含：
- **骨架定义**（HIERARCHY）：关节层级结构
- **动画数据**（MOTION）：每帧每个关节的旋转（欧拉角）+ 根节点位置

**本项目数据**：
- `数据1/`：23 个 BVH 文件
- `第一组 跳高 翻箱/`：9 个 BVH 文件
- 共 32 个文件，29 个唯一动作

### 2.2 处理：BVH 重定向（`bvh_retarget.py`）

**为什么需要重定向**：
- 真人有 ~60 个自由度，机器人只有 29 个
- 关节位置、长度、活动范围都不同
- 需要"翻译"成机器人能执行的动作

**重定向过程**：
```
BVH (真人骨架)
    ↓
1. 解析 BVH 骨架和动画
2. 提取根节点轨迹（位置 + 旋转）
3. 将真人关节旋转映射到机器人关节
4. 处理关节顺序（motor order）
5. 计算关节速度（数值微分）
    ↓
retargeted/*.npz (机器人关节角度)
```

**输出格式**（retargeted NPZ）：
```python
{
    'fps': 30,                          # 帧率
    'joint_pos': (T, 29),              # 关节角度（motor order）
    'joint_vel': (T, 29),              # 关节速度
    'root_positions': (T, 3),          # 根节点位置 (x, y, z)
    'root_rotations': (T, 4),          # 根节点旋转（四元数 xyzw）
}
```

**关节顺序（motor order）**：按身体部位分组
```
索引 0-5:   左腿 6 关节 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
索引 6-11:  右腿 6 关节
索引 12-14: 腰部 3 关节 (waist_yaw, waist_roll, waist_pitch)
索引 15-21: 左臂 7 关节
索引 22-28: 右臂 7 关节
```

### 2.3 验证：MuJoCo 验证

**目的**：确认重定向后的动作在物理上可行。

**两种验证**：
1. **运动学验证**（`batch_verify_mujoco.py`）：禁用重力，直接设置关节角度，看姿态是否正确
2. **物理跟踪验证**（`physics_track_verify.py`）：开启重力，用 PD 控制跟踪参考动作，看能否跟上

**验证通过标准**：
- 运动学：机器人姿态与参考动作一致
- 物理：PD 控制能大致跟踪动作（允许一定误差）

---

## 三、阶段 2：FK 数据转换（GPU 服务器）

### 3.1 为什么需要 FK 转换？

训练框架（omni_mimic）需要**所有 30 个刚体的完整位姿**作为参考：
- 位置（body_pos_w）
- 旋转（body_quat_w）
- 线速度（body_lin_vel_w）
- 角速度（body_ang_vel_w）

但 retargeted NPZ 只有**关节角度**，没有刚体位姿。

**FK（正运动学）**：已知关节角度 → 计算所有刚体位姿。

### 3.2 转换过程

```
retargeted/*.npz (关节角度)
    ↓
Isaac Sim 正运动学计算
    ↓
training_data/*_training.npz (完整刚体位姿)
```

**脚本**：`omni_mimic/scripts/retargeted_npz_to_training_npz.py`

**运行方式**（需要 GPU + Isaac Sim）：
```bash
isaaclab.sh -p omni_mimic/scripts/retargeted_npz_to_training_npz.py \
    --input_dir retargeted/ \
    --output_dir training_data/ \
    --input_fps 30 --headless
```

### 3.3 输出格式（training NPZ）

```python
{
    'fps': 30,                              # 帧率
    'joint_pos': (T, 29),                  # 关节角度
    'joint_vel': (T, 29),                  # 关节速度
    'body_pos_w': (T, 30, 3),             # 30 个刚体世界坐标位置
    'body_quat_w': (T, 30, 4),            # 30 个刚体世界坐标旋转（四元数）
    'body_lin_vel_w': (T, 30, 3),         # 30 个刚体线速度
    'body_ang_vel_w': (T, 30, 3),         # 30 个刚体角速度
}
```

**30 个刚体**：base_link + 29 个关节连杆

---

## 四、阶段 3：RL 训练（GPU 服务器）

### 4.1 训练框架：omni_mimic

**核心思想**：Mimic-style Reference Tracking
- 给定参考动作轨迹（training NPZ）
- 训练策略网络跟踪这个轨迹
- 奖励 = 跟踪误差的负值（误差越小奖励越高）

### 4.2 训练过程

```
training NPZ (参考动作)
    ↓
─────────────────────────────────────┐
│  训练循环（PPO 算法）                  │
│                                       │
│  1. 采样：从参考动作中采样初始状态      │
│  2.  rollout：策略与环境交互 N 步       │
│  3. 计算奖励：跟踪误差 + 正则化         │
│  4. 更新策略：PPO 优化                 │
│  5. 重复直到收敛                       │
└─────────────────────────────────────┘
    ↓
model_*.pt (训练好的策略网络)
```

**运行命令**：
```bash
isaaclab.sh -p omni_mimic/scripts/rsl_rl/train.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --headless --num_envs=4096 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --logger tensorboard --run_name jump06_test \
    --max_iterations 5000
```

### 4.3 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_envs` | 4096 | 并行环境数（GPU 显存决定） |
| `max_iterations` | 5000 | 训练迭代次数 |
| `task` | Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 | 任务配置 |
| `--headless` | - | 无 GUI 运行（服务器必需） |

**任务名解析**：
- `Tracking`：跟踪任务
- `Flat`：平地场景（还有 `Box` 场景）
- `Omni`：OMNI 机器人
- `Hist`：使用历史观测（5 步）
- `Delayed`：执行器延迟
- `DCMotor`：直流电机模型

### 4.4 观测空间（529 维）

```
观测 = [
    关节角度 (29) +
    关节速度 (29) +
    刚体位姿 (30×7=210) +  # 位置(3) + 旋转(4)
    刚体速度 (30×6=180) +  # 线速度(3) + 角速度(3)
    历史观测 (5步×29=145) +  # 关节角度历史
    命令 (36)              # 参考动作片段
]
```

### 4.5 奖励函数

```
总奖励 = 
    motion_global_anchor_pos × 1.0 +    # 锚点位置误差
    motion_global_anchor_ori × 0.5 +    # 锚点旋转误差
    motion_body_pos × 1.0 +             # 刚体位置误差
    motion_body_ori × 0.5 +             # 刚体旋转误差
    motion_body_lin_vel × 0.1 +         # 刚体线速度误差
    motion_body_ang_vel × 0.05 +        # 刚体角速度误差
    action_rate_l2 × -0.01 +            # 动作平滑性
    joint_limit × -0.1 +                # 关节限位惩罚
    undesired_contacts × -1.0           # 不希望的接触惩罚
```

### 4.6 终止条件

Episode 在以下情况终止：
- `anchor_pos`：锚点（通常是 pelvis）位置偏差 > 阈值
- `anchor_ori`：锚点旋转偏差 > 阈值
- `ee_body_pos`：末端执行器位置偏差 > 阈值
- `time_out`：达到最大步数

---

## 五、阶段 4：可视化验证（GPU 服务器 + GUI）

### 5.1 play.py 可视化

**目的**：在仿真中查看训练好的策略表现。

**命令**：
```bash
isaaclab.sh -p omni_mimic/scripts/rsl_rl/play.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-Play \
    --num_envs=1 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --load_run logs/rsl_rl/<最新目录名>
```

**注意**：
- 用 **Play** 版本的任务名（不是训练用的 v0 版本）
- 不加 `--headless` 才会弹出 GUI 窗口
- 在阿里云 DSW 的 noVNC 桌面中查看

### 5.2 评估指标

看什么：
- 机器人是否流畅地跟踪参考动作
- 有无异常抖动、摔倒
- 动作幅度是否与参考一致

---

## 六、阶段 5：部署（公司负责）

### 6.1 ONNX 导出

训练完成后，导出为 ONNX 格式：
```bash
isaaclab.sh -p omni_mimic/scripts/rsl_rl/play.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-Play \
    --num_envs=1 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --load_run logs/rsl_rl/<最新目录名> \
    --export_onnx
```

### 6.2 部署流程

```
model_*.pt → ONNX → omni_rl_sdk/policy/high_dynamic/ → FSM 状态机 → 真机
```

**FSM 状态机**：根据场景切换不同策略
- 平地走路 → loco_mode 策略
- 跳高/跨栏 → high_dynamic 策略
- 上下楼梯 → 对应策略

---

## 七、数据流转总结

```
┌──────────────────────────────────────────────────────────────────┐
│  本地机器（Mac）                                                    │
│                                                                    │
│  BVH 文件 → bvh_retarget.py → retargeted/*.npz                   │
│              ↓                                                     │
│         MuJoCo 验证（可选）                                         │
└──────────────────────────────────────────────────────────────────┘
                          ↓ scp 上传
┌──────────────────────────────────────────────────────────────────┐
│  GPU 服务器（AutoDL / 阿里云）                                      │
│                                                                    │
│  retargeted/*.npz → FK 转换 → training_data/*_training.npz       │
│                        ↓                                           │
│              omni_mimic 训练 → model_*.pt                          │
│                        ↓                                           │
│              play.py 可视化 / ONNX 导出                             │
└──────────────────────────────────────────────────────────────────┘
                          ↓ 下载
┌──────────────────────────────────────────────────────────────────┐
│  公司真机部署                                                       │
│                                                                    │
│  ONNX → omni_rl_sdk → FSM → 真机运行                               │
──────────────────────────────────────────────────────────────────┘
```
