# omni_mimic 跑步训练恢复计划

> 创建时间：2026-08-19
> 目的：恢复 8/13 FK fix 跑步训练，从 model_3000.pt resume 继续训练

---

## 1. 背景

### 1.1 训练历史

| 时间 | 训练目录 | 最终 checkpoint | 说明 |
|------|---------|----------------|------|
| 8/13 09:15 | `2026-08-13_09-15-45` | model_8000.pt | 原始跑步训练（无手臂惩罚），**FK 数据有 bug** |
| 8/13 21:27 | `2026-08-13_21-27-53_running01_fk_fix` | model_3000.pt | FK 修复后的跑步训练，**MuJoCo 验证通过** ✅ |
| 8/14 01:39~21:07 | 多个目录 | — | 手臂惩罚实验（w0.5~5.0），**全部失败** |
| 8/15+ | 多个目录 | — | 转向跳高训练 |

### 1.2 关键资产

| 资产 | 位置 | 状态 |
|------|------|------|
| `model_3000.pt` | `logs/rsl_rl/omni_flat/2026-08-13_21-27-53_running01_fk_fix/` | ✅ 最佳基线 |
| `2026-08-13_21-27-53_running01_fk_fix_3000.onnx` | `running_deploy_2k/` | ✅ MuJoCo 验证通过 |
| `tracking_env_omni_cfg.py.bak_original` | `omni_mimic/source/.../tasks/tracking/` | ✅ 原始跑步配置备份 |
| `跑步01_chr00_training.npz` | `running_deploy_2k/` | ✅ 跑步参考动作 |

### 1.3 为什么不用 model_8000.pt

`2026-08-13_09-15-45/model_8000.pt` 虽然训练步数更多（8000 vs 3000），但用的是**修复前的 FK 数据**。在 Isaac Lab 里看起来能跑（因为训练和验证用同一个 FK 管线），但导出的 ONNX 在 MuJoCo 里"躺在地上抽搐"。

`2026-08-13_21-27-53` 的 FK fix 修复了 `retargeted_npz_to_training_npz.py` 的数据转换问题，之后导出的 ONNX 通过了组员的 MuJoCo 验证。

### 1.4 为什么 model_3000.pt 是最佳基线

- FK 数据正确（修复后）
- MuJoCo 验证通过（组员确认）
- 3000 步可能还没完全收敛，策略还有提升空间
- 域随机化完整（7 类），保证 sim2sim 鲁棒性

---

## 2. 原始跑步配置详情

备份文件 `tracking_env_omni_cfg.py.bak_original` 记录了 21-27-53 训练时的确切配置。

### 2.1 奖励配置

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `motion_global_anchor_pos` | 0.5 | 基座位置跟踪（exp，std=0.3） |
| `motion_global_anchor_ori` | 0.5 | 基座朝向跟踪（exp，std=0.4） |
| `motion_body_pos` | 1.0 | 刚体位置跟踪（exp，std=0.3） |
| `motion_body_ori` | 1.0 | 刚体朝向跟踪（exp，std=0.4） |
| `motion_body_lin_vel` | 0.5 | 线速度跟踪（exp，std=1.0） |
| `motion_body_ang_vel` | 0.5 | 角速度跟踪（exp，std=3.14） |
| `action_rate_l2` | -0.1 | 动作平滑 |
| `joint_limit` | -10.0 | 关节限位 |
| `undesired_contacts` | -0.1 | 非预期接触 |

**无跳高奖励**（jump_height_bonus、tuck_bonus 等不存在）
**无手臂惩罚**（joint_pos_tracking 注释掉）

### 2.2 终止条件

| 终止项 | 阈值 | 说明 |
|--------|------|------|
| `anchor_pos` | 0.3 | 基座位置偏差 |
| `anchor_ori` | 0.9 | 基座朝向偏差 |
| `ee_body_pos` | 0.3 | 末端位置偏差（严格） |

注意：后来跳高训练时这些阈值被放宽到 0.5/1.2/1.0，备份里是原始严格值。

### 2.3 域随机化（7 类）

| 随机化项 | 参数 | 说明 |
|---------|------|------|
| 地面摩擦 | static 0.3~1.4, dynamic 0.2~1.2 | 大范围摩擦变化 |
| 关节默认位置 | ±0.01 rad | 小幅偏移 |
| 基座 COM | x±0.08, y±0.05, z±0.05 | 质心偏移 |
| 基座额外质量 | -2~+4 kg | 大范围质量变化 |
| 执行器刚度/阻尼 | 0.8~1.2 / 0.7~1.3 | 关键！让策略对 PD 增益变化鲁棒 |
| 连杆质量 | 0.9~1.1 | ±10% |
| 关节摩擦/armature | 0.4~1.6 / 0.8~1.2 | 大范围变化 |
| 定期外力推扰 | 1~3 秒间隔 | 抗扰动 |

### 2.4 观测配置

- 529 维 obs（含 5 帧历史）
- obs 噪声：`enable_corruption=True`
- 噪声范围：gravity ±0.05, ang_vel ±0.2, joint_pos ±0.01, joint_vel ±1.5

---

## 3. 恢复训练步骤

### 步骤 1：恢复配置

```bash
cd /root/autodl-tmp/omni_29dof_v260705
cp omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_omni_cfg.py.bak_original \
   omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_omni_cfg.py
```

### 步骤 2：从 model_3000.pt resume

```bash
isaaclab.sh -p omni_mimic/scripts/rsl_rl/train.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --num_envs=4096 \
    --max_iterations=10000 \
    --resume \
    --load_run 2026-08-13_21-27-53_running01_fk_fix \
    --checkpoint model_3000.pt \
    --run_name running_resume_from_3k
```

### 步骤 3：监控

```bash
tensorboard --logdir logs/rsl_rl/omni_flat/ --port 6006
```

关键指标：
- `Train/mean_reward`：应从 ~7.5 开始继续上升
- `Train/mean_episode_length`：应稳定在 100+（2 秒）
- `Episode_Termination/time_out`：应 > 0.9

### 步骤 4：导出 ONNX

```bash
isaaclab.sh -p omni_mimic/scripts/rsl_rl/play.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --num_envs=1 \
    --load_run running_resume_from_3k \
    --checkpoint model_8000.pt \
    --export_onnx
```

### 步骤 5：MuJoCo 验证

导出后给组员测试，确认 MuJoCo 仍能通过。

---

## 4. 后续优化方向（可选）

在确认跑步稳定后，可以尝试：

### 4.1 提升跑步稳定性

- 收紧终止条件（ee_body_pos 保持 0.3，不要放宽）
- 增大 action_rate_l2 权重（-0.1 → -0.3）
- 减小 push_robot 推力

### 4.2 放下手臂（难度高）

之前用 L2 惩罚全部失败（时间平均最小化陷阱）。新方案：
- 用 exp 惩罚代替 L2（liuzq V11 的 `arm_tracking_exp`）
- 恢复 body_pos/body_ori 的间接手臂约束
- 降低手臂跟踪的 std（0.3 → 0.2）

### 4.3 残差手臂策略

冻结下肢，只训练手臂 residual：
- `a_final = a_baseline + a_arm_residual`
- 保持 baseline 的跑步能力，只学手臂摆动

---

## 5. 注意事项

1. **不要修改域随机化** — 这是 MuJoCo 能过的根本原因
2. **不要修改 obs 噪声配置** — `enable_corruption=True` 是鲁棒性的关键
3. **resume 时环境配置从当前代码读取** — 确保 config 已恢复为备份版本
4. **训练步数建议 10000** — 从 3000 步继续，到 8000+ 步应该能完全收敛
5. **每次只改一个参数** — 避免混淆归因

---

## 6. 文件清单

| 文件 | 用途 |
|------|------|
| `tracking_env_omni_cfg.py.bak_original` | 原始跑步配置备份（需恢复） |
| `logs/rsl_rl/omni_flat/2026-08-13_21-27-53_running01_fk_fix/model_3000.pt` | 训练基线 |
| `running_deploy_2k/2026-08-13_21-27-53_running01_fk_fix_3000.onnx` | 已验证 ONNX |
| `running_deploy_2k/跑步01_chr00_training.npz` | 跑步参考动作 |
