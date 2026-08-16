#!/bin/bash
# ============================================================
# TienKung-Lab + OMNI 环境验证脚本
# 逐项检查: Python / PyTorch / Isaac Sim / Isaac Lab /
#           TienKung-Lab / OMNI 模型 / 集成状态
# ============================================================

CONDA_ENV_NAME="env_isaaclab"
DATA_DIR="/root/autodl-tmp"
export CONDA_ENVS_PATH="${DATA_DIR}/conda_envs"
export CONDA_PKGS_DIRS="${DATA_DIR}/conda_pkgs"

PASS=0
FAIL=0
WARN=0

pass() { echo "  [✓ 通过] $1"; ((PASS++)); }
fail() { echo "  [✗ 失败] $1"; ((FAIL++)); }
warn() { echo "  [! 警告] $1"; ((WARN++)); }

echo "============================================"
echo "  TienKung-Lab + OMNI 环境验证"
echo "============================================"

# 激活 conda
eval "$(conda shell.bash hook)"
if ! conda activate ${CONDA_ENV_NAME} 2>/dev/null; then
    fail "conda 环境 '${CONDA_ENV_NAME}' 不存在"
    echo ""
    echo "请先运行: bash setup_tienkung.sh"
    exit 1
fi

# -------------------------------------------------------
echo ""
echo "[1/7] Python 版本"
PYTHON_VER=$(python --version 2>&1)
if [[ "$PYTHON_VER" == *"3.11"* ]]; then
    pass "$PYTHON_VER"
else
    fail "需要 3.11，当前 $PYTHON_VER"
fi

# -------------------------------------------------------
echo ""
echo "[2/7] PyTorch + CUDA"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
cuda_ok = torch.cuda.is_available()
print(f'  CUDA 可用: {cuda_ok}')
if cuda_ok:
    print(f'  CUDA 版本: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'  显存: {vram:.1f} GB')
    if vram < 20:
        print('  [提示] 显存 < 24GB，训练时降低 --num_envs')
else:
    print('  [提示] 无 GPU - 确认实例是否有 GPU')
exit(0 if cuda_ok else 1)
" && pass "PyTorch + CUDA" || warn "CUDA 不可用 (无GPU实例?)"

# -------------------------------------------------------
echo ""
echo "[3/7] Isaac Sim"
if python -c "import isaacsim; print(f'  版本: {isaacsim.__version__}')" 2>/dev/null; then
    pass "Isaac Sim 已安装"
else
    fail "Isaac Sim 未安装或导入失败"
    echo "    修复: pip install \"isaacsim[all,extscache]==5.1.0\" --extra-index-url https://pypi.nvidia.com"
fi

# -------------------------------------------------------
echo ""
echo "[4/7] Isaac Lab"
if [ -d "${DATA_DIR}/IsaacLab" ]; then
    pass "IsaacLab 目录存在"
else
    fail "IsaacLab 目录不存在: ${DATA_DIR}/IsaacLab"
fi

if python -c "import isaaclab; print(f'  isaaclab 可导入')" 2>/dev/null; then
    pass "isaaclab Python 包可用"
else
    fail "isaaclab 导入失败"
    echo "    修复: cd ${DATA_DIR}/IsaacLab && ./isaaclab.sh --install rsl_rl"
fi

# 检查 rsl_rl
if python -c "import rsl_rl; print(f'  rsl_rl 可导入')" 2>/dev/null; then
    pass "rsl_rl 可用"
else
    warn "rsl_rl 导入失败 (TienKung-Lab 自带版本可能覆盖)"
fi

# -------------------------------------------------------
echo ""
echo "[5/7] TienKung-Lab"
TKL_DIR="${DATA_DIR}/TienKung-Lab"
if [ -d "$TKL_DIR" ]; then
    pass "TienKung-Lab 目录存在"
else
    fail "TienKung-Lab 不存在: $TKL_DIR"
    echo "    修复: git clone https://github.com/Open-X-Humanoid/TienKung-Lab.git $TKL_DIR"
fi

if python -c "import legged_lab" 2>/dev/null; then
    pass "legged_lab 包可导入"
else
    warn "legged_lab 导入失败"
    echo "    修复: cd $TKL_DIR && pip install -e ."
fi

# 检查训练脚本
if [ -f "${TKL_DIR}/legged_lab/scripts/train.py" ]; then
    pass "train.py 存在"
else
    fail "train.py 不存在"
fi

# -------------------------------------------------------
echo ""
echo "[6/7] OMNI 29-DOF 模型"
OMNI_DIR="${TKL_DIR}/legged_lab/assets/omni_29dof"
if [ -d "$OMNI_DIR" ]; then
    pass "OMNI 模型已部署到 TienKung-Lab"
    # 检查关键文件
    if [ -f "${OMNI_DIR}/omni_29dof_nohead_noshoe/urdf/omni_29dof_nohead_noshoe_merged_modify_feet.urdf" ]; then
        pass "URDF 文件存在"
    else
        fail "URDF 文件缺失"
    fi
    MESH_COUNT=$(find "${OMNI_DIR}/omni_29dof_nohead_noshoe/meshes" -name "*.STL" 2>/dev/null | wc -l)
    if [ "$MESH_COUNT" -ge 30 ]; then
        pass "STL 网格文件: ${MESH_COUNT} 个"
    else
        warn "STL 网格文件只有 ${MESH_COUNT} 个 (预期 32)"
    fi
    if [ -f "${OMNI_DIR}/robots/omni_29dof_nohead_noshoe_dcmotor_identified.py" ]; then
        pass "机器人配置文件存在"
    else
        fail "机器人配置文件缺失"
    fi
else
    fail "OMNI 模型未部署"
    echo "    修复: 重新运行 setup_tienkung.sh 或手动复制模型文件"
fi

# -------------------------------------------------------
echo ""
echo "[7/7] OMNI 环境集成"
OMNI_ENV_DIR="${TKL_DIR}/legged_lab/envs/omni"
if [ -d "$OMNI_ENV_DIR" ]; then
    pass "OMNI 环境目录存在"
    if [ -f "${OMNI_ENV_DIR}/walk_cfg.py" ]; then
        pass "walk_cfg.py 存在"
    else
        warn "walk_cfg.py 缺失"
    fi
else
    warn "OMNI 环境未集成 (需要运行 integrate_omni.py)"
    echo "    修复: cd $TKL_DIR && python integrate_omni.py"
fi

# -------------------------------------------------------
echo ""
echo "============================================"
echo "  验证结果: ${PASS} 通过 / ${WARN} 警告 / ${FAIL} 失败"
echo "============================================"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "  环境就绪！可以开始训练:"
    echo "    bash run_training.sh 验证    # 快速跑通测试"
    echo "    bash run_training.sh 走路    # 正式训练走路"
    echo ""
else
    echo ""
    echo "  有 ${FAIL} 项失败，请按上面的修复提示处理后重新验证"
    echo ""
fi
