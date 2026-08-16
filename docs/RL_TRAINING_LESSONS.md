# RL 训练调优经验总结（2026-08-13 ~ 08-14）

## 一、背景

Omni 29-DOF 人形机器人跑步动作训练，使用 Isaac Lab 2.3.2 + omni_mimic 框架。
核心问题：**训练出的策略手臂完全伸展，不摆动**。

---

## 二、经历时间线

### 8月13日 21:27 — 原始训练启动
- 配置：`motion_body_pos` / `motion_body_ori` weight=1.0
- 结果：reward=7.5, episode=125，但**手臂仍然伸展**

### 8月14日 凌晨 — 权重调优尝试
| 时间 | 操作 | 结果 |
|------|------|------|
| 01:39 | weight=2.0 | 165步时 reward=-0.06, episode=1.7 → **被误判为崩溃** |
| 01:49 | weight=1.3 | 165步时 reward=-0.03, episode=1.9 → **被误判为崩溃** |
| 02:11 | weight=1.0（恢复） | 同样失败 → **被误判为崩溃** |
| 02:33 | resume from model_3000.pt | 50步时 reward=-0.08 → **被误判为崩溃** |
| 03:40 | weight=1.5 | 5000步时 reward=9-10, episode=125 ✅ |

### 8月14日 上午 — 关键发现
- 读取原始训练 TensorBoard 日志，发现**原始训练初期（0-500步）和新训练完全一样**
- 原始训练也是从 reward=-0.18, episode=2.5 开始，到 1000 步才飞跃到 reward=1.76, episode=93
- **结论：所有"崩溃"都是过早终止导致的误判**

### 8月14日 上午 — weight=1.5 视频验证
- 5000步视频：**手臂仍然伸展**，weight 调整无法解决根本问题

### 8月14日 上午 — 根因分析与修复
- **根因**：`motion_body_pos/ori` 跟踪的是刚体位置/朝向，手臂伸展时手腕位置变化小，策略找到了"偷懒"方案
- **修复**：新增 `joint_pos_tracking` 奖励项，直接跟踪全部 29 个关节角度
- **策略**：从 model_3000.pt resume，利用已有走路能力，只微调手臂

### 8月14日 09:49 — 新训练启动
- 配置：body_pos/ori=1.0 + joint_pos_tracking=0.5（29关节）
- Resume from model_3000.pt
- 恢复趋势：
  | 步数 | reward | episode |
  |------|--------|---------|
  | 3100 | 0.02 | 5.15 |
  | 3184 | 0.15 | 8.99 |
  | 3253 | 0.76 | 29.75 |

---

## 三、核心教训

### 教训 1：RL 训练初期"黑暗期"是正常的
- **现象**：前 500 步 reward 为负数、episode 极短（1-3步）
- **真相**：策略在随机探索，这是正常过程
- **教训**：**至少等到 1000 步再下结论**，不要 50-165 步就判定失败

### 教训 2：刚体位置跟踪 ≠ 关节角度跟踪
- **现象**：`motion_body_pos` weight 从 1.0 调到 2.0，手臂仍然伸展
- **根因**：手臂长，末端位置变化小，策略可以伸展手臂同时满足位置约束
- **教训**：需要**关节角度级别的显式约束**才能控制手臂姿态

### 教训 3：Resume 时新奖励配置会自动生效
- **原理**：checkpoint 只保存策略网络权重，环境配置从当前代码读取
- **启示**：可以 resume 已有策略 + 新奖励配置，**不需要从零训练**

### 教训 4：截图监控不精准
- **问题**：TensorBoard 截图只能估算数值，容易误判
- **改进**：用 Python 脚本直接读取 TensorBoard 事件文件，输出精确数值

---

## 四、最终配置优化

### 修改文件
`omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_omni_cfg.py`

### 修改内容
```python
# 新增：29 关节位置跟踪（第 411 行附近）
joint_pos_tracking = RewTerm(
    func=mdp.joint_pos_exp,
    weight=0.5,
    params={
        "command_name": "motion",
        "std": 0.3,
        "body_names": [
            # 左腿 6
            "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
            "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            # 右腿 6
            "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
            "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            # 腰部 3
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            # 左臂 7
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint",
            "wrist_pitch_l_joint", "wrist_roll_l_joint",
            # 右臂 7
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint",
            "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ],
    },
)
```

### 奖励配置总览
| 奖励项 | 权重 | 作用 |
|--------|------|------|
| motion_body_pos | 1.0 | 刚体位置跟踪（间接约束） |
| motion_body_ori | 1.0 | 刚体朝向跟踪（间接约束） |
| motion_body_lin_vel | 0.5 | 线速度跟踪 |
| motion_body_ang_vel | 0.5 | 角速度跟踪 |
| **joint_pos_tracking** | **0.5** | **29 关节角度直接约束（新增）** |
| action_rate_l2 | -0.1 | 动作平滑 |
| joint_limit | -10.0 | 关节限位惩罚 |
| undesired_contacts | -0.1 | 非预期接触惩罚 |

---

## 五、高效训练与互动建议

### 5.1 训练决策规则

| 场景 | 规则 |
|------|------|
| 从头训练 | **至少等到 1000 步**再判断是否成功 |
| Resume 训练 | **至少等到 200-500 步**再判断趋势 |
| 调整权重 | 每次只改一个参数，避免混淆归因 |
| 新奖励项 | 优先 resume 而非从零训练（节省 80% 时间） |

### 5.2 监控协议

**每次检查训练进度，在 AutoDL 上运行以下命令，把文本输出发给 AI：**

```bash
python3 -c "
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator('LOG_DIR')
ea.Reload()

for tag in ['Train/mean_reward', 'Train/mean_episode_length']:
    events = ea.Scalars(tag)
    print(f'=== {tag} ===')
    print(f'  总步数: {len(events)}')
    print(f'  最新: Step {events[-1].step}: {events[-1].value:.4f}')
    print(f'  最近5步:')
    for e in events[-5:]:
        print(f'    Step {e.step}: {e.value:.4f}')
    print()

for tag in ['Episode_Termination/ee_body_pos', 'Episode_Termination/time_out']:
    events = ea.Scalars(tag)
    print(f'=== {tag} ===')
    for e in events[-3:]:
        print(f'  Step {e.step}: {e.value:.4f}')
    print()
"
```

**将 `LOG_DIR` 替换为实际训练目录路径。**

### 5.3 视频生成时机

| 步数 | 操作 |
|------|------|
| 1000 | 首次检查趋势（文本数据） |
| 2000 | 生成视频，评估动作质量 |
| 5000 | 生成视频，确认改善效果 |
| 10000 | 最终评估，准备导出 |

### 5.4 导出命令模板

```bash
# 生成视频
/root/autodl-tmp/IsaacLab/isaaclab.sh -p omni_mimic/scripts/rsl_rl/play.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --num_envs=1 \
    --motion_file training_data/跑步01_chr00_training.npz \
    --load_run <RUN_NAME> \
    --checkpoint model_<STEP>.pt \
    --video --video_length 300

# 导出 ONNX（给组员 MuJoCo 测试）
/root/autodl-tmp/IsaacLab/isaaclab.sh -p omni_mimic/scripts/rsl_rl/play.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --num_envs=1 \
    --motion_file training_data/跑步01_chr00_training.npz \
    --load_run <RUN_NAME> \
    --checkpoint model_<STEP>.pt \
    --export_onnx
```

### 5.5 关键文件路径

| 文件 | 路径 |
|------|------|
| 奖励配置 | `omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_omni_cfg.py` |
| 训练日志 | `logs/rsl_rl/omni_flat/<TIMESTAMP>_<RUN_NAME>/` |
| Checkpoints | `logs/rsl_rl/omni_flat/<RUN_NAME>/model_<STEP>.pt` |
| 训练数据 | `training_data/跑步01_chr00_training.npz` |

---

## 六、待验证事项

- [ ] 4000 步视频：手臂是否有摆动
- [ ] 10000 步：reward 是否稳定收敛
- [ ] MuJoCo sim2sim：导出的 ONNX 策略在 MuJoCo 中的表现
- [ ] 如果手臂仍不理想：尝试提高 `joint_pos_tracking` weight 到 0.8-1.0
