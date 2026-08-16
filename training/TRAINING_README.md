# Reference Tracking 训练指南

> OMNI 29-DOF 高动态动作（跳高、翻箱等）的 RL 训练流程

## 概述

本模块实现了基于 **reference tracking** 的训练环境，用于训练 OMNI 29-DOF 机器人的高动态动作策略。

与之前的 AMP 对抗学习不同，reference tracking 直接跟踪参考动作的每一帧，训练更稳定、更可控。

### 架构

```
训练侧（本模块）                    部署侧（omni_rl_sdk）
┌─────────────────────┐           ┌─────────────────────┐
│ Isaac Lab + rsl_rl  │           │ MuJoCo + ONNX       │
│                     │  导出     │                     │
│ env_reference.py    │ ──ONNX──▶ │ high_dynamic_policy │
│ (obs/reward/PD)     │           │ (obs 构建完全一致)   │
└─────────────────────┘           └─────────────────────┘
```

**关键设计**：训练环境的 obs 构建逻辑与推理侧 `high_dynamic_policy.py` 的 `_build_obs()` 完全对齐，确保 ONNX 可以直接部署。

---

## 快速开始

### 1. 环境检查

```bash
# 在 AutoDL 的 Isaac Lab conda 环境中
isaaclab.sh -p training/check_env.py
```

### 2. 开始训练

```bash
# 训练跳高动作（推荐先用单个动作验证）
isaaclab.sh -p training/train.py --headless --motion 跳高06 --num_envs 2048

# 训练全部 29 个动作
isaaclab.sh -p training/train.py --headless --num_envs 4096 --max_iter 20000

# 自定义参数
isaaclab.sh -p training/train.py --headless \
    --motion 跳高 \
    --num_envs 2048 \
    --max_iter 10000 \
    --lr 1e-4 \
    --experiment jump_v1
```

### 3. 监控训练

```bash
# TensorBoard（新开一个终端）
tensorboard --logdir logs/ref_tracking --port 6006
```

关注指标：
- `rew_pos`：关节位置跟踪奖励（越高越好，趋近 1.0）
- `rew_vel`：关节速度跟踪奖励
- `rew_ori`：base 朝向跟踪奖励
- `pos_error`：平均关节位置误差（越低越好）
- 总 reward 的 smoothed 趋势线

### 4. 导出 ONNX

```bash
# 训练完成后导出
python training/export_onnx.py \
    --checkpoint logs/ref_tracking/jump_v1/model_10000.pt \
    --output model/my_jump_policy.onnx
```

### 5. 部署验证

```bash
# 将 ONNX 放入 SDK
cp model/my_jump_policy.onnx omni_rl_sdk/policy/high_dynamic/model/

# 修改 high_dynamic.yaml 的 model.path
# 在 MuJoCo 中验证
python omni_rl_sdk/deploy_omni_sim_real/main.py
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `env_reference.py` | 核心环境类：obs 构建、reward 计算、PD 控制、运动数据加载 |
| `train_config.py` | Isaac Lab 环境配置 + rsl_rl PPO 超参 |
| `train.py` | 训练入口脚本 |
| `export_onnx.py` | PyTorch → ONNX 导出 |
| `check_env.py` | 环境检查脚本 |

---

## 观测空间（529 维）

与推理侧 `high_dynamic_policy.py` 完全对齐：

| 部分 | 维度 | 说明 |
|------|------|------|
| command | 58 | ref_joint_pos(29) + ref_joint_vel(29)，policy order |
| anchor_ori | 6 | base 旋转矩阵前两列展平 |
| history × 5 | 465 | 每帧 93 维 × 5 帧历史 |

每帧历史（93 维）：
- gravity(3)：body frame 重力方向
- ang_vel(3)：body frame 角速度
- joint_pos_err(29)：关节角度 - 默认角度（motor order）
- joint_vel(29)：关节角速度（motor order）
- last_action(29)：上一步动作输出

---

## 奖励函数

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| joint_pos_tracking | 1.0 | exp(-10 × MSE(pos_error)) |
| joint_vel_tracking | 0.5 | exp(-5 × MSE(vel_error)) |
| body_orientation | 0.5 | exp(-5 × (1 - |dot(q, q_ref)|)) |
| action_smoothness | -0.001 | -||a_t - a_{t-1}||² |
| energy | -0.0001 | -||tau||² |
| alive | 0.1 | 存活奖励 |

**调参建议**：
- 如果跟踪精度不够：增大 `rew_joint_pos` 或 `tracking_alpha_pos`
- 如果动作抖动：增大 `rew_action_smoothness` 的绝对值
- 如果训练太慢收敛：增大 `entropy_coef`

---

## 训练参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| num_envs | 2048-4096 | 并行环境数，越多越稳定 |
| max_iter | 10000-20000 | 单个动作 10k，全部 20k |
| learning_rate | 1e-4 | 自适应调度 |
| action_scale | 0.5 | 与 high_dynamic.yaml 一致 |

**预期训练时间**（A100 GPU）：
- 单个动作（2048 envs）：约 2-4 小时
- 全部 29 个动作（4096 envs）：约 8-12 小时

---

## 注意事项

1. **必须在 Isaac Lab 环境运行**：不是 `omni_gpu` 环境
2. **使用 `isaaclab.sh -p`**：不要直接 `python train.py`
3. **先用单动作验证**：`--motion 跳高06` 确认管线能跑通
4. **obs 维度必须匹配**：如果改了 obs 结构，ONNX 部署时也要同步改
5. **关节顺序**：运动数据是 policy order，仿真内部是 motor order，环境会自动转换

---

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| `Isaac Lab 未安装` | 确认在 Isaac Lab conda 环境中，用 `isaaclab.sh -p` |
| `CUDA 不可用` | 运行 `nvidia-smi`，如果 GPU 不可用则重启 AutoDL 实例 |
| `rsl_rl 未安装` | `pip install rsl_rl` |
| `未找到运动数据` | 检查 `motion_data/` 目录下有 `*_highdynamic.npz` 文件 |
| 训练 reward 不涨 | 检查 obs 是否正确（打印 obs_buf 统计量） |
| 训练后部署效果差 | 检查 obs 是否与推理侧完全对齐 |
