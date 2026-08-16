# 跳高 06 优化训练 - 修改总结

## 问题诊断

**现象**：跳高训练视频显示起跳时和落地后膝盖离地太近（深蹲跳）

**根因**：
1. **参考动作膝盖角度过大**：原始 retarget NPZ 膝盖最大 136.8°（极限深蹲）
2. **关节限位太宽**：knee_pitch 限位 (0, 164°)，136° 在范围内未被裁剪
3. **缺少腿部关节跟踪**：训练配置中 `joint_pos_tracking_legs` 被注释
4. **终止条件太紧**：`ee_body_pos` 阈值 0.3m 对跳高动作过于严格

---

## 修改内容

### 1. bvh_retarget.py（本地已修改）

**文件**：`/Users/condenast/Downloads/omni_29dof_v260705/bvh_retarget.py`

**修改**：添加膝盖角度软限制（第 811-822 行）

```python
# 膝盖角度软限制：防止深蹲（110° = 1.92 rad）
# 机器人腿短，参考动作的深蹲会被放大，需要限制膝盖弯曲
KNEE_SOFT_LIMIT = 1.92  # 110°
knee_clipped = 0
for side in ['l', 'r']:
    idx = joint_idx_map[f'knee_pitch_{side}_joint']
    before = joint_angles[:, idx].copy()
    joint_angles[:, idx] = np.clip(joint_angles[:, idx], 0, KNEE_SOFT_LIMIT)
    knee_clipped += np.sum(before != joint_angles[:, idx])
if knee_clipped > 0:
    print(f"  [膝盖限制] {knee_clipped} 帧超过 {np.degrees(KNEE_SOFT_LIMIT):.0f}°，已裁剪")
```

**效果**：
- 膝盖角度：136.8° → **111.3°**（降低 25.5°）
- 裁剪帧数：148 帧（397 帧中的 37%）

---

### 2. tracking_env_omni_cfg.py（本地已修改）

**文件**：`/Users/condenast/Downloads/omni_29dof_v260705/omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_omni_cfg.py`

**修改 1**：启用腿部关节跟踪（第 479-491 行）

```python
# 腿部关节跟踪：防止深蹲，确保膝盖离地高度
joint_pos_tracking_legs = RewTerm(
    func=mdp.joint_pos_exp,
    weight=0.5,
    params={
        "command_name": "motion",
        "std": 0.3,
        "body_names": [
            "hip_pitch_l_joint", "hip_pitch_r_joint",
            "knee_pitch_l_joint", "knee_pitch_r_joint",
            "ankle_pitch_l_joint", "ankle_pitch_r_joint",
        ],
    },
)
```

**修改 2**：放宽终止条件（第 533 行）

```python
ee_body_pos = DoneTerm(
    func=mdp.bad_motion_body_pos_z_only,
    params={
        "command_name": "motion",
        "threshold": 0.5,  # 跳高需要更大容差，从 0.3 放宽到 0.5
        "body_names": [
            "ankle_roll_l_link",
            "ankle_roll_r_link",
            "wrist_roll_l_link",
            "wrist_roll_r_link",
        ],
    },
)
```

---

### 3. 重新生成跳高 06 NPZ（本地已完成）

**命令**：
```bash
python3 bvh_retarget.py --input "第一组 跳高 翻箱/跳高06_chr00.bvh" --output retargeted/跳高06_chr00.npz
```

**输出**：`retargeted/跳高06_chr00.npz`（107 KB, 397 帧×29 关节）

**验证**：
- 膝盖角度：max 111.3°（目标 ≤110°，低通滤波导致轻微 overshoot）
- 根高度：0.51-1.12m（合理）
- 接触率：左脚 84.9%，右脚 85.4%（正常）

---

## 部署步骤（服务器端）

### 步骤 1：上传文件（本地运行）

```bash
# 上传 retargeted NPZ
scp -P 33310 retargeted/跳高06_chr00.npz \
    root@connect.bjb2.seetacloud.com:/root/autodl-tmp/omni_29dof_v260705/retargeted/

# 上传训练配置
scp -P 33310 omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_omni_cfg.py \
    root@connect.bjb2.seetacloud.com:/root/autodl-tmp/omni_29dof_v260705/omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_omni_cfg.py
```

### 步骤 2：转换为 training NPZ（服务器运行）

```bash
/root/autodl-tmp/IsaacLab/isaaclab.sh -p omni_mimic/scripts/retargeted_npz_to_training_npz.py \
    --input_file retargeted/跳高06_chr00.npz \
    --output_file training_data/跳高06_chr00_training.npz \
    --input_fps 30 --headless
```

### 步骤 3：开始训练（服务器运行）

```bash
/root/autodl-tmp/IsaacLab/isaaclab.sh -p omni_mimic/scripts/rsl_rl/train.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --num_envs=4096 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --max_iterations=20000 \
    --run_name jump06_knee_fix \
    agent.save_interval=500
```

---

## 监控与验证

### TensorBoard 监控

```bash
tensorboard --logdir logs/rsl_rl/omni_flat/ --port 6006
```

### 关键指标检查点

| 步数 | 检查项 | 预期值 |
|------|--------|--------|
| 500 | reward/episode 趋势 | reward > 0, episode > 20 |
| 1000 | 走路恢复 | episode > 50 |
| 2000 | **生成视频** | 膝盖离地 > 0.3m |
| 5000 | 收敛确认 | reward 稳定，episode > 100 |
| 10000 | 最终评估 | 准备导出 ONNX |

### 视频生成命令

```bash
/root/autodl-tmp/IsaacLab/isaaclab.sh -p omni_mimic/scripts/rsl_rl/play.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --num_envs=1 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --load_run jump06_knee_fix \
    --checkpoint model_2000.pt \
    --video --video_length 300
```

---

## 预期效果

1. **膝盖高度改善**：从贴地深蹲 → 正常跳高姿态（膝盖离地 > 0.3m）
2. **起跳动作**：保持明显起跳，但膝盖弯曲 ≤ 110°
3. **落地缓冲**：膝盖弯曲适度，不会过度下蹲
4. **训练稳定性**：腿部关节跟踪奖励帮助策略学习正确的腿部姿态

---

## 回退方案

如果训练效果不理想：

1. **膝盖仍然太低**：降低 `KNEE_SOFT_LIMIT` 到 1.75 rad（100°）
2. **训练不稳定**：降低 `joint_pos_tracking_legs` weight 到 0.3
3. **频繁终止**：进一步放宽 `ee_body_pos` threshold 到 0.6

---

## 文件清单

| 文件 | 状态 | 位置 |
|------|------|------|
| `bvh_retarget.py` | ✅ 已修改 | 本地 |
| `tracking_env_omni_cfg.py` | ✅ 已修改 | 本地 |
| `retargeted/跳高06_chr00.npz` | ✅ 已生成 | 本地 |
| `deploy_jump06.sh` | ✅ 已创建 | 本地 |
| `training_data/跳高06_chr00_training.npz` | ⏳ 待转换 | 服务器 |

---

**下一步**：执行部署步骤，开始训练。
