#!/bin/bash
# ============================================================
# OMNI 29-DOF GPU 服务器环境搭建脚本
# 适用于：有 NVIDIA GPU 的 Linux 服务器（AutoDL / 自建等）
# ============================================================
set -e

echo "=========================================="
echo " OMNI 29-DOF 环境搭建"
echo "=========================================="

# ── 1. Python 环境 ──
echo ""
echo "[1/5] 检查 Python 环境..."

# 检测是否在 conda 环境中
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "  ⚠ 检测到 conda 环境: $CONDA_DEFAULT_ENV"
    echo ""
    echo "  选择："
    echo "    1. 在当前 conda 环境安装（可能冲突）"
    echo "    2. 创建新的 conda 环境 omni_gpu（推荐）"
    echo "    3. 退出 conda，用系统 Python"
    echo ""
    read -p "  请输入选项 [1/2/3] (默认 2): " choice
    choice=${choice:-2}
    
    case $choice in
        1)
            echo "  → 在当前环境安装"
            ;;
        2)
            echo "  → 创建新环境 omni_gpu..."
            conda create -n omni_gpu python=3.10 -y
            # 初始化 conda（如果还没初始化）
            eval "$(conda shell.bash hook 2>/dev/null)" || conda init bash
            conda activate omni_gpu
            ;;
        3)
            echo "  → 退出 conda..."
            conda deactivate
            echo "  当前 Python: $(which python3)"
            ;;
        *)
            echo "  无效选项，退出"
            exit 1
            ;;
    esac
fi

# 检查 Python 版本
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "  ✓ python3: $PY_VERSION ($(which python3))"
    
    # 检查版本是否 >= 3.10
    MAJOR=$(echo $PY_VERSION | cut -d. -f1)
    MINOR=$(echo $PY_VERSION | cut -d. -f2)
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
        echo "  ✗ Python 版本过低，需要 3.10+"
        exit 1
    fi
else
    echo "  ✗ 未找到 Python3，请先安装"
    exit 1
fi

# ── 2. 核心依赖 ──
echo ""
echo "[2/5] 安装核心依赖..."
pip install --upgrade pip
pip install mujoco numpy pyyaml pillow onnxruntime-gpu

# ── 3. 验证安装 ──
echo ""
echo "[3/5] 验证安装..."
python3 -c "
import mujoco; print(f'  ✓ mujoco {mujoco.__version__}')
import numpy; print(f'  ✓ numpy {numpy.__version__}')
import yaml; print(f'  ✓ pyyaml {yaml.__version__}')
import onnxruntime as ort
print(f'  ✓ onnxruntime {ort.__version__}')
print(f'    providers: {ort.get_available_providers()}')
if 'CUDAExecutionProvider' in ort.get_available_providers():
    print('    ✓ CUDA 可用')
else:
    print('    ⚠ CUDA 不可用，将使用 CPU')
"

# ── 4. 检查 GPU ──
echo ""
echo "[4/5] 检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "  ✓ GPU 检测成功"
else
    echo "  ⚠ nvidia-smi 不可用"
fi

# ── 5. 检查项目文件 ──
echo ""
echo "[5/5] 检查项目文件..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

check_file() {
    if [ -f "$SCRIPT_DIR/$1" ]; then
        echo "  ✓ $1"
    else
        echo "  ✗ $1 缺失！请上传项目文件"
    fi
}

check_file "omni_29dof_mjc/mjcf/omni_29dof.xml"
check_file "omni_rl_sdk/policy/loco_mode/model/omni_7dof_63k_2file.onnx"
check_file "omni_rl_sdk/policy/loco_mode/config/LocoMode.yaml"
check_file "retargeted/跳高06_chr00.npz"
check_file "motion_data/跳高06_chr00_highdynamic.npz"

echo ""
echo "=========================================="
echo " 环境搭建完成！"
echo ""
echo " 下一步："
echo "   1. 跑走路仿真: python3 run_walking_sim.py"
echo "   2. 跑物理跟踪验证: python3 physics_track_verify.py"
echo "=========================================="
