#!/bin/bash
# ============================================================================
# Omni 29-DOF 跳高训练 - 一键启动脚本
#
# 用法（迁移到服务器后，在工程根目录）：
#   bash run_train.sh --headless --num_envs 4096
#   bash run_train.sh --headless --num_envs 4096 --max_iterations 10000
#
# 行为：
#   1) 自动做环境自检（首次会自动安装 rsl-rl-lib 等依赖）
#   2) 自动检测 GPU 数量，选择单卡 / 双卡并行（torchrun + --distributed）
#   3) 双卡模式下，自动把 --num_envs 拆成每卡一半（不传则每卡用环境默认 2048）
#
# 可选环境变量：
#   TRAIN_MODE=auto|dual|single   训练模式（默认 auto：>=2 卡走双卡，否则单卡）
#   TRAIN_GPU_COUNT=2             双卡模式的进程数（默认 2）
# ============================================================================
set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
source "$PROJECT_ROOT/lib_env.sh"

echo "=========================================================="
echo "  Omni 29-DOF 跳高训练 - 一键启动"
echo "=========================================================="

# ---- 0. 环境自检（幂等，首次会自动安装依赖）----
if [ -z "${SKIP_SETUP:-}" ]; then
    echo "[0/4] 环境自检 ..."
    bash "$PROJECT_ROOT/setup_environment.sh"
else
    echo "[0/4] SKIP_SETUP=1，跳过环境自检"
fi

# ---- 1. Python ----
PYTHON="$(detect_python)"
echo "[1/4] Python: $PYTHON"

# ---- 2. 检测 GPU ----
NUM_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    echo "!! 未检测到 GPU（nvidia-smi 无输出）。请确认容器已用 --gpus all 启动。"
    exit 1
fi
echo "[2/4] 检测到 GPU 数量: $NUM_GPUS"

# ---- 3. 模式选择 ----
MODE="${TRAIN_MODE:-auto}"
if [ "$MODE" = "auto" ]; then
    if [ "$NUM_GPUS" -ge 2 ]; then MODE="dual"; else MODE="single"; fi
fi
NGPU="${TRAIN_GPU_COUNT:-2}"
if [ "$MODE" = "dual" ] && [ "$NUM_GPUS" -lt "$NGPU" ]; then
    echo "!! 双卡模式需要 ${NGPU} 张卡，但只检测到 ${NUM_GPUS} 张。已降级为单卡。"
    MODE="single"
fi
echo "[3/4] 训练模式: $MODE"

# ---- 4. 组装训练命令 ----
# 收集用户传入的 train.py 参数
ARGS=("$@")

if [ "$MODE" = "dual" ]; then
    # 双卡：拆分 --num_envs 为每卡一半
    for ((i = 0; i < ${#ARGS[@]}; i++)); do
        if [ "${ARGS[$i]}" = "--num_envs" ]; then
            TOTAL_ENVS="${ARGS[$((i + 1))]}"
            PER_RANK=$((TOTAL_ENVS / NGPU))
            echo "   双卡: 总环境 ${TOTAL_ENVS} → 每卡 ${PER_RANK}（${NGPU} 卡并行）"
            ARGS[$((i + 1))]="$PER_RANK"
            break
        fi
    done
    if [ -z "$TOTAL_ENVS" ]; then
        echo "   双卡: 未指定 --num_envs，每卡使用环境默认值（2×2048=4096）"
    fi

    echo "[4/4] 启动 ${NGPU} 卡并行训练（torchrun + --distributed）..."
    exec "$PYTHON" -m torch.distributed.run \
        --standalone \
        --nproc_per_node="$NGPU" \
        "$PROJECT_ROOT/scripts/train.py" \
        --distributed \
        "${ARGS[@]}"
else
    echo "[4/4] 启动单卡训练 ..."
    exec "$PYTHON" "$PROJECT_ROOT/scripts/train.py" "${ARGS[@]}"
fi
