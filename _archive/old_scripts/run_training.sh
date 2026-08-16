#!/bin/bash
# ============================================================
# OMNI 29-DOF 训练启动脚本 (基于 TienKung-Lab)
# 用法: bash run_training.sh [模式]
# 模式: 验证 / 走路 / 跑步 / 原版走路 / tensorboard
# ============================================================

CONDA_ENV_NAME="env_isaaclab"
DATA_DIR="/root/autodl-tmp"
TKL_DIR="${DATA_DIR}/TienKung-Lab"
export CONDA_ENVS_PATH="${DATA_DIR}/conda_envs"
export CONDA_PKGS_DIRS="${DATA_DIR}/conda_pkgs"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

cd ${TKL_DIR}

MODE=${1:-"帮助"}

# -------------------------------------------------------
# OMNI 训练 (集成后使用)
# -------------------------------------------------------
if [[ "$MODE" == "验证" || "$MODE" == "test" ]]; then
    echo "============================================"
    echo "  OMNI 验证模式: 64 环境 × 100 步"
    echo "  目的: 快速确认环境能跑通"
    echo "============================================"
    python legged_lab/scripts/train.py \
        --task=omni_walk \
        --headless \
        --logger=tensorboard \
        --num_envs=64 \
        --max_iterations=100

elif [[ "$MODE" == "走路" || "$MODE" == "walk" ]]; then
    echo "============================================"
    echo "  OMNI 走路训练: 4096 环境"
    echo "  RTX 4090 24GB 推荐配置"
    echo "============================================"
    python legged_lab/scripts/train.py \
        --task=omni_walk \
        --headless \
        --logger=tensorboard \
        --num_envs=4096

elif [[ "$MODE" == "跑步" || "$MODE" == "run" ]]; then
    echo "============================================"
    echo "  OMNI 跑步训练: 4096 环境"
    echo "============================================"
    python legged_lab/scripts/train.py \
        --task=omni_run \
        --headless \
        --logger=tensorboard \
        --num_envs=4096

# -------------------------------------------------------
# 原版 TienKung 训练 (用于对比/验证框架本身没问题)
# -------------------------------------------------------
elif [[ "$MODE" == "原版走路" || "$MODE" == "tk_walk" ]]; then
    echo "============================================"
    echo "  TienKung 原版走路 (验证框架本身)"
    echo "============================================"
    python legged_lab/scripts/train.py \
        --task=walk \
        --headless \
        --logger=tensorboard \
        --num_envs=4096

elif [[ "$MODE" == "原版跑步" || "$MODE" == "tk_run" ]]; then
    echo "============================================"
    echo "  TienKung 原版跑步"
    echo "============================================"
    python legged_lab/scripts/train.py \
        --task=run \
        --headless \
        --logger=tensorboard \
        --num_envs=4096

# -------------------------------------------------------
# 推理/回放
# -------------------------------------------------------
elif [[ "$MODE" == "回放" || "$MODE" == "play" ]]; then
    echo "============================================"
    echo "  OMNI 策略回放 (需要 GUI 或 headless 导出)"
    echo "============================================"
    echo "  用法: bash run_training.sh 回放 <run文件夹> <checkpoint文件>"
    echo ""
    if [ -n "$2" ] && [ -n "$3" ]; then
        python legged_lab/scripts/play.py \
            --task=omni_walk \
            --num_envs=1 \
            --load_run="$2" \
            --checkpoint="$3"
    else
        echo "  示例: bash run_training.sh 回放 logs/omni_walk/2026-07-27_12-00-00 model_5000.pt"
        echo ""
        echo "  可用的训练记录:"
        ls -lt ${TKL_DIR}/logs/ 2>/dev/null | head -10
    fi

# -------------------------------------------------------
# Sim2Sim (MuJoCo 交叉验证)
# -------------------------------------------------------
elif [[ "$MODE" == "sim2sim" || "$MODE" == "mujoco" ]]; then
    echo "============================================"
    echo "  Sim2Sim: 导出策略到 MuJoCo 验证"
    echo "============================================"
    if [ -n "$2" ]; then
        python legged_lab/scripts/sim2sim.py \
            --task omni_walk \
            --policy "$2" \
            --duration 100
    else
        echo "  用法: bash run_training.sh sim2sim <policy.pt路径>"
        echo "  示例: bash run_training.sh sim2sim logs/omni_walk/xxx/exported/policy.pt"
    fi

# -------------------------------------------------------
# TensorBoard
# -------------------------------------------------------
elif [[ "$MODE" == "tensorboard" || "$MODE" == "tb" ]]; then
    echo "============================================"
    echo "  启动 TensorBoard (端口 6006)"
    echo "============================================"
    echo ""
    echo "  本地电脑执行 SSH 隧道:"
    echo "    ssh -L 6006:localhost:6006 -p <SSH端口> root@<SSH地址>"
    echo "  然后浏览器打开: http://localhost:6006"
    echo ""
    echo "  (AutoDL 也支持在网页端直接映射端口)"
    echo ""
    tensorboard --logdir=logs --host 0.0.0.0 --port 6006

# -------------------------------------------------------
# 帮助
# -------------------------------------------------------
else
    echo "============================================"
    echo "  OMNI 29-DOF 训练脚本 (TienKung-Lab)"
    echo "============================================"
    echo ""
    echo "用法: bash run_training.sh <模式>"
    echo ""
    echo "OMNI 训练:"
    echo "  验证       64环境×100步，快速确认环境正常"
    echo "  走路       4096环境，正式训练走路策略"
    echo "  跑步       4096环境，正式训练跑步策略"
    echo ""
    echo "原版 TienKung (验证框架):"
    echo "  原版走路   用 TienKung 自带机器人训练"
    echo "  原版跑步   用 TienKung 自带机器人训练"
    echo ""
    echo "其他:"
    echo "  回放       加载训练好的策略回放"
    echo "  sim2sim    导出到 MuJoCo 交叉验证"
    echo "  tensorboard  启动 TensorBoard 看曲线"
    echo ""
    echo "典型流程:"
    echo "  1. bash run_training.sh 验证       # 确认能跑"
    echo "  2. bash run_training.sh 走路       # 正式训练"
    echo "  3. bash run_training.sh tensorboard  # 看曲线"
    echo "  4. bash run_training.sh 回放 ...   # 看效果"
fi
