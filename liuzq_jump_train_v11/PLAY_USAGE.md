# play.py 使用文档

回放已训练的 Omni 29-DOF 跳高策略，生成视频并导出模型。

## 基本用法

```bash
cd /root/autodl-tmp
isaaclab.sh -p liuzq_jump_train_v11/scripts/play.py --task Omni-Jump-v0 --headless --video
```

不加 `--checkpoint` 时自动加载最新 checkpoint。

## 参数说明

### 视频录制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--video` | flag | False | 录制视频 |
| `--video_length` | int | 400 | 视频长度（步数） |

### 相机视角

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `--camera_angle` | str | `side` | 见下表 | 相机视角 |

**可选视角：**

| 值 | 相机位置 (x, y, z) | 说明 |
|----|-------------------|------|
| `side` | (0.0, 5.0, 2.5) | 侧面（Y 轴方向），默认，适合观察跳高 |
| `front` | (5.0, 0.0, 2.5) | 正面（X 轴方向，机器人朝向前方） |
| `back` | (-5.0, 0.0, 2.5) | 背面 |
| `back_side` | (0.0, -5.0, 2.5) | 后侧面 |
| `eye_level` | (0.0, 5.0, 1.0) | 平视（腰部高度） |
| `overhead` | (0.0, 5.0, 4.0) | 俯瞰 |
| `diagonal` | (3.5, 3.5, 2.5) | 45° 斜角 |

> **注意**：机器人默认朝向 X 轴正方向（前方），所以：
> - 侧面 = Y 轴方向 (0, ±5, z)
> - 正面/背面 = X 轴方向 (±5, 0, z)

### 模型加载

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | 最新 | 指定 checkpoint 文件路径 |
| `--load_run` | str | 最新 run | 指定 run 目录名 |
| `--resume` | flag | False | 从 checkpoint 恢复（训练用） |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | `Omni-Jump-v0` | 任务名称 |
| `--num_envs` | int | None | 环境数量 |
| `--seed` | int | None | 随机种子 |
| `--motion_file` | str | None | 覆盖参考动作 npz 文件 |
| `--play_full_motion` | flag | False | 播放完整参考动作 |
| `--keep_running` | flag | False | 完整动作播完后继续运行 |
| `--real-time` | flag | False | 实时运行（如果可能） |
| `--headless` | flag | - | 无 GUI 模式（服务器必需） |

## 常用示例

### 1. 生成默认侧面视角视频

```bash
isaaclab.sh -p liuzq_jump_train_v11/scripts/play.py \
  --task Omni-Jump-v0 \
  --headless \
  --video \
  --video_length 400
```

### 2. 指定 checkpoint 和视角

```bash
isaaclab.sh -p liuzq_jump_train_v11/scripts/play.py \
  --task Omni-Jump-v0 \
  --headless \
  --video \
  --checkpoint liuzq_jump_train_v11/logs/rsl_rl/omni_jump/2026-08-16_21-37-31/model_3000.pt \
  --camera_angle front
```

### 3. 正面视角

```bash
isaaclab.sh -p liuzq_jump_train_v11/scripts/play.py \
  --task Omni-Jump-v0 \
  --headless \
  --video \
  --camera_angle front
```

### 4. 平视视角（观察腿部动作）

```bash
isaaclab.sh -p liuzq_jump_train_v11/scripts/play.py \
  --task Omni-Jump-v0 \
  --headless \
  --video \
  --camera_angle eye_level
```

### 5. 俯瞰视角（观察整体轨迹）

```bash
isaaclab.sh -p liuzq_jump_train_v11/scripts/play.py \
  --task Omni-Jump-v0 \
  --headless \
  --video \
  --camera_angle overhead
```

### 6. 播放完整参考动作

```bash
isaaclab.sh -p liuzq_jump_train_v11/scripts/play.py \
  --task Omni-Jump-v0 \
  --headless \
  --play_full_motion
```

### 7. 实时播放（本地调试）

```bash
python liuzq_jump_train_v11/scripts/play.py \
  --task Omni-Jump-v0 \
  --real-time \
  --camera_angle side
```

## 输出文件

- **视频**：`logs/rsl_rl/omni_jump/<run_dir>/videos/play/`
- **导出模型**：`logs/rsl_rl/omni_jump/<run_dir>/exported/policy.pt` 和 `policy.onnx`

## 注意事项

1. 服务器上必须加 `--headless` 参数
2. `--video` 会自动启用相机（`enable_cameras=True`）
3. 不加 `--checkpoint` 时自动加载最新 checkpoint
4. 相机视角只影响视频录制，不影响训练
