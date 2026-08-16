# 环境搭建指南

> AutoDL + 阿里云 DSW 完整部署流程 | 最后更新：2026-08-12

---

## 一、环境要求

### 1.1 硬件

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | NVIDIA GPU (CUDA 12.4+) | RTX 4090 / A10 |
| 显存 | 16 GB | 24 GB+ |
| 内存 | 32 GB | 64 GB |
| 存储 | 100 GB | 200 GB+ |

### 1.2 软件

| 组件 | 版本 |
|------|------|
| Isaac Lab | 2.3.2 |
| Isaac Sim | 5.0 或 5.1 |
| Python | 3.11 |
| CUDA | 12.4+ |

---

## 二、AutoDL 部署

### 2.1 环境信息

| 项目 | 值 |
|------|-----|
| 工作目录 | `/root/autodl-tmp/omni_29dof_v260705/` |
| Isaac Lab | `/root/autodl-tmp/IsaacLab/` |
| conda 环境 | `env_isaaclab` |
| isaaclab.sh | `/root/autodl-tmp/IsaacLab/isaaclab.sh` |

### 2.2 部署步骤

#### Step 1：上传代码

```bash
# 本地 Mac 打包
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

# 上传到 AutoDL
scp omni_project_v4.tar.gz root@<AutoDL地址>:/root/autodl-tmp/
```

#### Step 2：解压

```bash
cd /root/autodl-tmp
tar xzf omni_project_v4.tar.gz
ls omni_29dof_v260705/omni_mimic/   # 确认存在
```

#### Step 3：安装依赖

```bash
# 激活 conda 环境
conda activate env_isaaclab

# 安装 rsl_rl
cd /root/autodl-tmp/omni_29dof_v260705/omni_mimic/source/rsl_rl
pip install -e .

# 安装 whole_body_tracking
cd ../whole_body_tracking
pip install -e .

# 验证
python -c "import rsl_rl; print('OK')"
```

#### Step 4：FK 数据转换

```bash
cd /root/autodl-tmp/omni_29dof_v260705

# 删除 macOS 垃圾文件
find retargeted/ -name '._*' -delete

# 批量 FK 转换
/root/autodl-tmp/IsaacLab/isaaclab.sh -p omni_mimic/scripts/retargeted_npz_to_training_npz.py \
    --input_dir retargeted/ \
    --output_dir training_data/ \
    --input_fps 30 --headless
```

#### Step 5：开始训练

```bash
cd /root/autodl-tmp/omni_29dof_v260705

/root/autodl-tmp/IsaacLab/isaaclab.sh -p omni_mimic/scripts/rsl_rl/train.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --headless --num_envs=4096 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --logger tensorboard --run_name jump06_test \
    --max_iterations 5000
```

#### Step 6：监控

```bash
tensorboard --logdir=/root/autodl-tmp/omni_29dof_v260705/logs/rsl_rl/ --port 6006 --bind_all
```

---

## 三、阿里云 DSW 部署

### 3.1 环境信息

| 项目 | 值 |
|------|-----|
| 工作目录 | `/mnt/workspace/omni_29dof_v260705/` |
| Isaac Lab | `/workspace/isaaclab/` |
| conda 环境 | `base`（默认激活） |
| isaaclab.sh | `/workspace/isaaclab/isaaclab.sh` |
| GPU | NVIDIA A10 |
| Isaac Sim | 5.1.0 |

### 3.2 部署步骤

#### Step 1：上传代码

同 AutoDL Step 1，上传到 `/mnt/workspace/`。

#### Step 2：解压

```bash
cd /mnt/workspace
tar xzf omni_project_v4.tar.gz
ls omni_29dof_v260705/omni_mimic/
```

#### Step 3：安装依赖

```bash
# 阿里云 DSW 默认已激活 base 环境
# 如果不在 base 环境，执行：conda activate base

cd /mnt/workspace/omni_29dof_v260705/omni_mimic/source/rsl_rl
pip install -e .

cd ../whole_body_tracking
pip install -e .

python -c "import rsl_rl; print('OK')"
```

#### Step 4：FK 数据转换

```bash
cd /mnt/workspace/omni_29dof_v260705

find retargeted/ -name '._*' -delete

/workspace/isaaclab/isaaclab.sh -p omni_mimic/scripts/retargeted_npz_to_training_npz.py \
    --input_dir retargeted/ \
    --output_dir training_data/ \
    --input_fps 30 --headless
```

> **注意**：如果报 `ModuleNotFoundError: No module named 'isaaclab'`，先执行 `conda deactivate` 退出 conda 环境再跑。

#### Step 5：开始训练

```bash
cd /mnt/workspace/omni_29dof_v260705

/workspace/isaaclab/isaaclab.sh -p omni_mimic/scripts/rsl_rl/train.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --headless --num_envs=4096 \
    --motion_file training_data/跑步01_chr00_training.npz \
    --logger tensorboard --run_name run01_test \
    --max_iterations 5000
```

#### Step 6：可视化验证（阿里云特有）

训练完成后，在 noVNC 桌面查看：

```bash
cd /mnt/workspace/omni_29dof_v260705

/workspace/isaaclab/isaaclab.sh -p omni_mimic/scripts/rsl_rl/play.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-Play \
    --num_envs=1 \
    --motion_file training_data/跑步01_chr00_training.npz \
    --load_run logs/rsl_rl/<最新目录名>
```

在 noVNC 桌面中可以看到 Isaac Sim 仿真窗口。

---

## 四、Isaac Lab 2.3.2 兼容性修复

> **重要**：本地代码已包含所有修复，无需在服务器上再跑修复脚本。

修复内容（已包含在上传的代码中）：

| 文件 | 问题 | 修复 |
|------|------|------|
| `train.py` | `dump_pickle` 被移除 | 用本地 `pickle.dump` 替代 |
| `on_policy_runner.py` | `get_observations()` 返回 TensorDict | 提取 `['policy']` 和 `['critic']` |
| `on_policy_runner.py` | `step()` 返回 TensorDict | 同样提取 tensor |
| `retargeted_npz_to_training_npz.py` | macOS `._` 文件干扰 | 添加过滤 |

---

## 五、常见问题

### 5.1 `ModuleNotFoundError: No module named 'isaaclab'`

**原因**：conda 环境干扰了 `isaaclab.sh -p` 的 Python 选择。

**解决**：
```bash
conda deactivate
# 然后重新跑命令
```

### 5.2 `ValueError: too many values to unpack (expected 2)`

**原因**：加载了旧版 rsl_rl（如 TienKung-Lab 的）。

**解决**：
```bash
# 重命名旧版 rsl_rl
mv /mnt/workspace/TienKung-Lab/rsl_rl /mnt/workspace/TienKung-Lab/rsl_rl.bak

# 重新安装 omni_mimic 的 rsl_rl
cd /mnt/workspace/omni_29dof_v260705/omni_mimic/source/rsl_rl
pip install -e .
```

### 5.3 OOM（显存不足）

```bash
# 减少并行环境数
--num_envs=2048  # 或 1024
```

### 5.4 TensorBoard 看不到数据

```bash
# 用绝对路径
tensorboard --logdir=/mnt/workspace/omni_29dof_v260705/logs/rsl_rl/ --port 6006 --bind_all
```

### 5.5 `._` 文件导致 FK 转换失败

```bash
# 删除 macOS 资源分支文件
find retargeted/ -name '._*' -delete
```

---

## 六、验证清单

部署完成后，逐项检查：

- [ ] `omni_mimic/` 目录存在
- [ ] `retargeted/` 目录有 29 个 NPZ 文件
- [ ] `training_data/` 目录有 29 个 `_training.npz` 文件
- [ ] `python -c "import rsl_rl; print('OK')"` 输出 OK
- [ ] FK 转换脚本能正常运行
- [ ] 训练能启动并输出 TensorBoard 日志
- [ ] TensorBoard 能看到 reward 曲线
