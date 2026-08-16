# 阿里云 DSW 训练环境搭建指南

> 目标：在阿里云 DSW 上搭建 omni_mimic 训练环境，与 AutoDL 并行训练

## 环境信息

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA A10 × 1 |
| Isaac Lab | 2.3.2 |
| Isaac Sim | 5.1.0 |
| Python | 3.11 |
| conda 环境 | `base`（默认激活） |
| isaaclab.sh | `/workspace/isaaclab/isaaclab.sh` |
| 工作目录 | `/mnt/workspace/` |

---

## Step 1：上传代码（本地已修复好的版本）

本地 Mac 打包：

```bash
cd /Users/condenast/Downloads
tar czf omni_project_v4.tar.gz \
    -C omni_29dof_v260705 \
    --exclude='__pycache__' \
    --exclude='.qoder' \
    --exclude='.DS_Store' \
    --exclude='_archive' \
    --exclude='frames*' \
    --exclude='数据1' \
    --exclude='第一组 跳高 翻箱' \
    --exclude='motion_data' \
    --exclude='training' \
    --exclude='xMimic_extracted' \
    --exclude='._*' \
    .

scp omni_project_v4.tar.gz root@<阿里云地址>:/mnt/workspace/
```

阿里云解压：

```bash
cd /mnt/workspace
tar xzf omni_project_v4.tar.gz
ls omni_29dof_v260705/omni_mimic/   # 确认存在
```

> 本地文件已包含所有 Isaac Lab 2.3.2 兼容性修复，无需额外修改。

---

## Step 2：安装依赖

```bash
# 阿里云 DSW 默认已激活 base 环境，无需额外 conda activate
# 如果不在 base 环境，执行：conda activate base

# 安装 rsl_rl
cd /mnt/workspace/omni_29dof_v260705/omni_mimic/source/rsl_rl
pip install -e .

# 安装 whole_body_tracking
cd ../whole_body_tracking
pip install -e .

# 验证
python -c "import rsl_rl; print('OK')"
```

---

## Step 3：FK 数据转换

```bash
cd /mnt/workspace/omni_29dof_v260705

# 删除 macOS 垃圾文件
find retargeted/ -name '._*' -delete

# 批量 FK 转换
/workspace/isaaclab/isaaclab.sh -p omni_mimic/scripts/retargeted_npz_to_training_npz.py \
    --input_dir retargeted/ \
    --output_dir training_data/ \
    --input_fps 30 --headless
```

---

## Step 4：开始训练

```bash
cd /mnt/workspace/omni_29dof_v260705

# 跑步01（推荐先试，最简单）
/workspace/isaaclab/isaaclab.sh -p omni_mimic/scripts/rsl_rl/train.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --headless --num_envs=4096 \
    --motion_file training_data/跑步01_chr00_training.npz \
    --logger tensorboard --run_name run01_test \
    --max_iterations 5000

# 监控
tensorboard --logdir=/mnt/workspace/omni_29dof_v260705/logs/rsl_rl/ --port 6006 --bind_all
```

## Step 5：训练完成后可视化验证

训练完成后，在阿里云 DSW 终端执行 play.py，通过 noVNC 桌面查看仿真效果。

```bash
cd /mnt/workspace/omni_29dof_v260705

# 查看训练日志目录，找到最新的 run
ls logs/rsl_rl/

# 可视化播放（不加 --headless，会弹出 Isaac Sim 窗口，在 noVNC 中查看）
/workspace/isaaclab/isaaclab.sh -p omni_mimic/scripts/rsl_rl/play.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-Play \
    --num_envs=1 \
    --motion_file training_data/跑步01_chr00_training.npz \
    --load_run logs/rsl_rl/<最新目录名>
```

### Play 任务名对应关系

| 训练任务 | Play 任务 |
|----------|-----------|
| `Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0` | `Tracking-Flat-Omni-Hist-Delayed-DCMotor-Play` |
| `Tracking-Box-Omni-Hist-Delayed-DCMotor-v0` | `Tracking-Box-Omni-Hist-Delayed-DCMotor-Play` |

### noVNC 访问方式

1. 在 DSW 终端执行上述 play 命令
2. 打开阿里云 DSW 的 **noVNC 桌面**（浏览器访问 DSW 提供的桌面入口）
3. 在 noVNC 桌面中可以看到 Isaac Sim 仿真窗口，观察机器人动作跟踪效果

### 注意事项

- `--num_envs=1` 只看 1 个机器人，方便观察
- `--load_run` 指向训练日志目录（里面要有 `model_*.pt` 和 `params/`）
- 不加 `--headless` 才会弹出 GUI 窗口
- 如果 Isaac Sim GUI 启动失败，检查 DISPLAY 环境变量：`echo $DISPLAY`，为空则 `export DISPLAY=:1`

---

## 可用动作列表

| 动作 | 文件 | 场景 | 帧数 | 推荐 |
|------|------|------|------|------|
| 跑步01 | 跑步01_chr00_training.npz | Flat | 525 | ⭐⭐ 先试 |
| 跨栏02-03 | 跨栏0X_chr00_training.npz | Flat | 417-477 | ⭐⭐ |
| 跳高01-05 | 跳高0X_chr00_training.npz | Flat | 334-516 | ⭐ |
| 翻箱子01-14 | 翻箱子0X_chr00_training.npz | Box | 641-1385 | 需Box场景 |

---

## 常见问题

| 问题 | 解决 |
|------|------|
| `isaaclab.sh` 路径不对 | 阿里云 DSW 固定为 `/workspace/isaaclab/isaaclab.sh` |
| conda 环境名不同 | `conda env list` 查看 |
| Isaac Sim 5.1.0 报错 | 把错误发给我 |
| OOM | 减小 `--num_envs` 到 2048 |
