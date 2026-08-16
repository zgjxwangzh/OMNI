# 训练指南

> 如何训练、调参、解读指标 | 最后更新：2026-08-12

---

## 一、训练前准备

### 1.1 确认数据就绪

```bash
# 检查 training_data 目录
ls training_data/

# 应该看到 29 个 _training.npz 文件
# 例如：跳高06_chr00_training.npz、跑步01_chr00_training.npz
```

### 1.2 选择动作

| 动作 | 文件 | 难度 | 推荐 |
|------|------|------|------|
| 跑步01 | 跑步01_chr00_training.npz | ⭐ 简单 | 首次训练推荐 |
| 跳高01-06 | 跳高0X_chr00_training.npz | ⭐⭐ 中等 | 核心动作 |
| 跨栏02-03 | 跨栏0X_chr00_training.npz | ⭐⭐ 中等 | - |
| 翻箱子01-14 | 翻箱子0X_chr00_training.npz | ⭐⭐⭐ 困难 | 需 Box 场景 |

### 1.3 选择场景

| 场景 | 任务名 | 适用动作 |
|------|--------|---------|
| **Flat（平地）** | `Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0` | 跳高、跑步、跨栏 |
| **Box（箱子）** | `Tracking-Box-Omni-Hist-Delayed-DCMotor-v0` | 翻箱子、后空翻 |

---

## 二、启动训练

### 2.1 基本命令

```bash
cd /root/autodl-tmp/omni_29dof_v260705  # AutoDL
# 或
cd /mnt/workspace/omni_29dof_v260705    # 阿里云

isaaclab.sh -p omni_mimic/scripts/rsl_rl/train.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --headless --num_envs=4096 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --logger tensorboard --run_name jump06_test \
    --max_iterations 5000
```

### 2.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | - | 任务配置名（见上表） |
| `--headless` | - | 无 GUI 运行（服务器必需） |
| `--num_envs` | 4096 | 并行环境数（显存不够就减到 2048） |
| `--motion_file` | - | 参考动作 NPZ 路径 |
| `--logger` | tensorboard | 日志工具（tensorboard / wandb） |
| `--run_name` | - | 运行名称（用于区分不同实验） |
| `--max_iterations` | 5000 | 最大训练迭代次数 |

### 2.3 批量训练（多个动作）

```bash
# 用文件夹作为输入，自动训练所有 NPZ
isaaclab.sh -p omni_mimic/scripts/rsl_rl/train.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --headless --num_envs=4096 \
    --motion_file training_data/ \
    --logger tensorboard --run_name batch_train \
    --max_iterations 5000
```

---

## 三、监控训练

### 3.1 TensorBoard

```bash
# AutoDL
tensorboard --logdir=/root/autodl-tmp/omni_29dof_v260705/logs/rsl_rl/ --port 6006 --bind_all

# 阿里云
/root/miniconda3/bin/tensorboard --logdir=/mnt/workspace/omni_29dof_v260705/logs/rsl_rl/ --port 6006 --bind_all
```

浏览器打开 `http://localhost:6006` 或通过服务器端口转发访问。

### 3.2 关键指标解读

#### Train/mean_reward（最重要）

| 阶段 | reward 范围 | 解读 |
|------|------------|------|
| 初期 (0-500 iter) | -1.0 ~ -0.1 | 策略在探索，正常 |
| 中期 (500-2000 iter) | -0.1 ~ 2.0 | 开始学会跟踪 |
| 后期 (2000-5000 iter) | 2.0 ~ 10+ | 收敛中 |

**趋势比绝对值重要**：持续上升 = 正常；长时间不动 = 可能卡住。

#### Train/mean_episode_length

| 趋势 | 解读 |
|------|------|
| 逐步增长 | ✅ 策略能跟踪更长时间 |
| 先升后降 | ⚠️ 可能遇到瓶颈，继续观察 |
| 持续很短 (<10) | ❌ 终止条件太严格或奖励有问题 |

#### Episode_Termination/*

| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `anchor_pos` | 锚点位置偏差终止 | < 1% |
| `anchor_ori` | 锚点旋转偏差终止 | < 1% |
| `ee_body_pos` | 末端位置偏差终止 | 初期可高，后期应 < 10% |
| `time_out` | 超时终止 | 越高越好（说明能完成整个动作） |

#### Metrics/motion/error_*

| 指标 | 含义 | 趋势 |
|------|------|------|
| `error_anchor_pos` | 锚点位置误差 | 应逐步下降 |
| `error_body_pos` | 刚体位置误差 | 应逐步下降 |
| `error_joint_pos` | 关节位置误差 | 应逐步下降 |
| `error_joint_vel` | 关节速度误差 | 应逐步下降 |

---

## 四、常见问题与调参

### 4.1 Reward 不上升

**可能原因**：
1. 动作太难（如匐前进 2623 帧）
2. 终止条件太严格
3. 学习率不合适

**解决方案**：
- 先试简单动作（跑步01）
- 检查 TensorBoard 的 termination 指标
- 调整 PPO 超参数（见 `omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/omni/agents/rsl_rl_ppo_cfg.py`）

### 4.2 Episode length 太短

**可能原因**：
- 终止阈值太严格
- 初始状态采样太困难

**解决方案**：
- 检查 `terminations.py` 中的阈值
- 调整 `commands.py` 中的自适应采样参数

### 4.3 OOM（显存不足）

```bash
# 减少并行环境数
--num_envs=2048  # 或 1024
```

### 4.4 训练速度太慢

| 因素 | 优化 |
|------|------|
| `num_envs` 太小 | 增加到 4096 或 8192（显存允许的话） |
| GPU 利用率低 | 检查是否有 CPU 瓶颈 |
| 动作太长 | 用 `--frame_range` 裁剪动作片段 |

---

## 五、训练完成后的操作

### 5.1 找到最佳模型

```bash
# 查看日志目录
ls logs/rsl_rl/<run_name>/

# 找到最新的 model_*.pt
ls -lt logs/rsl_rl/<run_name>/model_*.pt | head -5
```

### 5.2 可视化验证

```bash
isaaclab.sh -p omni_mimic/scripts/rsl_rl/play.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-Play \
    --num_envs=1 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --load_run logs/rsl_rl/<run_name>
```

### 5.3 导出 ONNX

```bash
isaaclab.sh -p omni_mimic/scripts/rsl_rl/play.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-Play \
    --num_envs=1 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --load_run logs/rsl_rl/<run_name> \
    --export_onnx
```

---

## 六、双机并行训练策略

| 机器 | 推荐动作 | 理由 |
|------|---------|------|
| AutoDL (RTX 4090) | 跳高06、跨栏 | 训练主力，跑复杂动作 |
| 阿里云 (A10) | 跑步01、跳高01-05 | 验证管线，跑简单动作 |

**避免重复**：两台机器不要训练同一个动作，浪费资源。

---

## 七、实验记录模板

每次训练记录以下信息：

```markdown
## 实验：跳高06 第一次训练

- **日期**：2026-08-11
- **机器**：AutoDL
- **动作**：跳高06_chr00_training.npz
- **任务**：Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0
- **num_envs**：4096
- **max_iterations**：5000
- **run_name**：jump06_test
- **结果**：
  - 最终 reward：~6.5 (2400 iter)
  - episode length：~120 帧
  - 主要终止：ee_body_pos
- **备注**：训练正常，reward 持续上升
```
