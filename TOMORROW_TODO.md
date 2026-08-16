# 今晚操作指南 — omni_mimic 训练管线（精简版）

> AutoDL 环境：Isaac Lab 在 `/root/autodl-tmp/IsaacLab/`，conda 在 `/root/autodl-tmp/conda_envs/env_isaaclab`

---

## 第 1 步：上传代码（无 GPU 可做）

本地 Mac 执行：
```bash
cd /Users/condenast/Downloads
tar czf omni_project_v3.tar.gz \
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

scp omni_project_v3.tar.gz root@<AutoDL地址>:/root/autodl-tmp/
```

AutoDL 执行：
```bash
cd /root/autodl-tmp/omni_29dof_v260705
tar xzf ../omni_project_v3.tar.gz
ls omni_mimic/ && ls retargeted/   # 确认两个目录都在
```

---

## 第 2 步：安装依赖（无 GPU 可做）

```bash
source /root/autodl-tmp/conda_envs/env_isaaclab/bin/activate

# 安装 rsl_rl
cd /root/autodl-tmp/omni_29dof_v260705/omni_mimic/source/rsl_rl
pip install -e .

# 安装 whole_body_tracking
cd ../whole_body_tracking
pip install -e .

# 验证
python -c "import rsl_rl; print('✓ rsl_rl')"
python -c "import whole_body_tracking; print('✓ wbt')"
```

**装完可以关机（无 GPU 模式省钱）**

---

## 第 3 步：FK 数据转换 + 训练（需要 GPU！）

开机选 GPU 实例后：
```bash
source /root/autodl-tmp/conda_envs/env_isaaclab/bin/activate
nvidia-smi   # 确认 GPU 可用
cd /root/autodl-tmp/omni_29dof_v260705

# ① FK 转换（先用跳高06 单文件测试）
/root/autodl-tmp/IsaacLab/isaaclab.sh -p omni_mimic/scripts/retargeted_npz_to_training_npz.py \
    --input_file retargeted/跳高06_chr00.npz \
    --output_file training_data/跳高06_chr00_training.npz \
    --input_fps 30 --headless

# ② 验证数据
python3 -c "
import numpy as np
d = np.load('training_data/跳高06_chr00_training.npz')
print('Keys:', list(d.keys()))
print('joint_pos:', d['joint_pos'].shape)
print('body_pos_w:', d['body_pos_w'].shape)
print('fps:', d['fps'])
"

# ③ 开始训练
/root/autodl-tmp/IsaacLab/isaaclab.sh -p omni_mimic/scripts/rsl_rl/train.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --headless --num_envs=4096 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --logger tensorboard --run_name jump06_test \
    --max_iterations 5000
```

---

## 不用管的东西（避免混淆）

| 文件/目录 | 说明 |
|----------|------|
| `xMimic.tar.gz` | 开源版，给 dex_evt 用的，**不用** |
| `omni_29dof_gpu/` | MuJoCo 验证目录，**现在不用** |
| `TienKung-Lab/` | 旧的 AMP 走路训练，**现在不用** |
| `omni_rl_sdk/` | 部署 SDK，**训练完再说** |

---

## 遇到报错？

**直接把完整错误信息发给我。**
