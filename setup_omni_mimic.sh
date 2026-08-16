#!/bin/bash
# ============================================================
# omni_mimic 训练环境安装 + 数据转换 + 训练启动
# 在 AutoDL 上执行，需要先激活 Isaac Lab conda 环境
# ============================================================
set -e

PROJECT_DIR="/root/autodl-tmp/omni_29dof_v260705"  # 项目目录
ISAACLAB="/root/autodl-tmp/IsaacLab"               # Isaac Lab 安装目录
MIMIC_DIR="${PROJECT_DIR}/omni_mimic"

echo "============================================================"
echo "  Step 1: 安装 omni_mimic 依赖"
echo "============================================================"

# 安装 rsl_rl（omni_mimic 自带版本）
cd "${MIMIC_DIR}/source/rsl_rl"
pip install -e .
echo "✓ rsl_rl 安装完成"

# 安装 whole_body_tracking
cd "${MIMIC_DIR}/source/whole_body_tracking"
pip install -e .
echo "✓ whole_body_tracking 安装完成"

# 验证安装
python -c "import whole_body_tracking; print('✓ whole_body_tracking import OK')"
python -c "import rsl_rl; print('✓ rsl_rl import OK')"

echo ""
echo "============================================================"
echo "  Step 2: 数据转换（retargeted NPZ → training NPZ）"
echo "  需要 Isaac Sim，使用 isaaclab.sh -p 启动"
echo "============================================================"

cd "${PROJECT_DIR}"

${ISAACLAB}/isaaclab.sh -p omni_mimic/scripts/retargeted_npz_to_training_npz.py \
    --input_dir retargeted/ \
    --output_dir training_data/ \
    --input_fps 30 \
    --headless

echo ""
echo "✓ 数据转换完成！"
echo "  输出目录: training_data/"
ls -la training_data/

echo ""
echo "============================================================"
echo "  Step 3: 验证训练数据格式"
echo "============================================================"

python3 -c "
import numpy as np
import glob

files = sorted(glob.glob('training_data/*_training.npz'))
print(f'找到 {len(files)} 个训练数据文件')

for f in files[:3]:  # 只显示前3个
    data = np.load(f)
    name = f.split('/')[-1]
    T = data['joint_pos'].shape[0]
    N_bodies = data['body_pos_w'].shape[1]
    fps = data['fps'][0]
    print(f'  {name}: {T}帧, {N_bodies}刚体, {fps}fps')
    print(f'    joint_pos: {data[\"joint_pos\"].shape}')
    print(f'    body_pos_w: {data[\"body_pos_w\"].shape}')

if len(files) > 3:
    print(f'  ... 还有 {len(files)-3} 个文件')
"

echo ""
echo "============================================================"
echo "  Step 4: 开始训练（跳高06 验证管线）"
echo "============================================================"

cd "${PROJECT_DIR}"

${ISAACLAB}/isaaclab.sh -p omni_mimic/scripts/rsl_rl/train.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --headless \
    --num_envs=4096 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --logger tensorboard \
    --run_name jump06_test \
    --max_iterations 5000

echo ""
echo "✓ 训练完成！"
echo "  日志目录: logs/rsl_rl/omni_flat/"
echo "  查看 TensorBoard: tensorboard --logdir logs/rsl_rl/"
