#!/bin/bash
# ============================================================
# TienKung-Lab + OMNI 29-DOF 环境一键安装脚本
# 目标平台: AutoDL Ubuntu 22.04 + RTX 4090
# 版本对齐: Python 3.11 + Isaac Sim 5.1.0 + Isaac Lab v2.3.0
# ============================================================
# 架构说明:
#   Isaac Sim 5.1  = NVIDIA 物理仿真引擎
#   Isaac Lab 2.3  = 机器人训练框架 (基于 Isaac Sim)
#   TienKung-Lab   = 人形机器人 RL 训练框架 (基于 Isaac Lab)
#   OMNI 29-DOF    = 你的机器人模型 (塞进 TienKung-Lab 训练)
# ============================================================
# 已知坑及解决方案:
# 1. IsaacSim 5.1.0 依赖含 pywin32 (Linux 不兼容) -> pip 会自动跳过
# 2. isaaclab.sh --install 末尾 VSCode 报错 -> 不影响功能，忽略
# 3. PyTorch 版本冲突 -> 先装 Isaac Sim，再强制覆盖 torch
# 4. GLIBC 版本要求 -> 必须 Ubuntu 22.04
# 5. TienKung-Lab 官方基于 IsaacSim 4.5/IsaacLab 2.1，但 OMNI 模型
#    需要 IsaacLab >= 2.2 的新 API，所以用 5.1/2.3，可能需小幅适配
# ============================================================
set -e

# ===================== 配置区 =====================
CONDA_ENV_NAME="env_isaaclab"
ISAACLAB_BRANCH="v2.3.0"
DATA_DIR="/root/autodl-tmp"           # AutoDL 数据盘 (大容量、高IO)
OMNI_MODEL_DIR="omni_29dof_v260705"   # OMNI 模型文件夹名
# conda 环境装数据盘，节省系统盘空间 (系统盘通常只有 30GB)
export CONDA_ENVS_PATH="${DATA_DIR}/conda_envs"
export CONDA_PKGS_DIRS="${DATA_DIR}/conda_pkgs"
# ================================================

echo "============================================"
echo "  TienKung-Lab + OMNI 环境安装脚本"
echo "  Python 3.11 + Isaac Sim 5.1 + Isaac Lab 2.3"
echo "============================================"
echo ""
echo "  数据盘: ${DATA_DIR}"
echo "  conda 环境: ${DATA_DIR}/conda_envs/${CONDA_ENV_NAME}"
echo ""

# -------------------------------------------------------
# 第一步: 检查系统前置条件
# -------------------------------------------------------
echo "[1/8] 检查系统前置条件..."

# GLIBC >= 2.35 (Ubuntu 22.04 自带)
GLIBC_VERSION=$(ldd --version 2>&1 | head -n1 | grep -oP '\d+\.\d+$' || echo "unknown")
echo "  GLIBC 版本: $GLIBC_VERSION"

# NVIDIA 驱动
if command -v nvidia-smi &> /dev/null; then
    NVIDIA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  NVIDIA 驱动: $NVIDIA_DRIVER"
    echo "  GPU: $GPU_NAME ($GPU_MEM)"
else
    echo "  [提示] nvidia-smi 未找到 - 安装阶段不需要 GPU，训练时再确认"
fi

# Ubuntu 版本
UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "unknown")
echo "  Ubuntu: $UBUNTU_VERSION"
if [[ "$UBUNTU_VERSION" != "22.04" ]]; then
    echo "  [警告] 推荐 Ubuntu 22.04，当前版本可能有 GLIBC 兼容问题"
fi

# 磁盘空间检查
AVAIL_SPACE=$(df -BG ${DATA_DIR} 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
if [ -n "$AVAIL_SPACE" ] && [ "$AVAIL_SPACE" -lt 50 ]; then
    echo "  [警告] 数据盘剩余 ${AVAIL_SPACE}GB，建议至少 50GB"
fi

echo "  前置检查完成 ✓"

# -------------------------------------------------------
# 第 1.5 步: 修复 AutoDL 容器 Vulkan 问题
# -------------------------------------------------------
# AutoDL 容器缺少 NVIDIA EGL 配置文件，导致 Vulkan 无法识别 GPU
# 这是 Isaac Sim 运行的必要条件
echo ""
echo "[1.5/8] 配置 Vulkan (AutoDL 容器修复)..."

mkdir -p /usr/share/glvnd/egl_vendor.d
cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json << 'VEOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
VEOF

mkdir -p /etc/vulkan/icd.d
cat > /etc/vulkan/icd.d/nvidia_icd.json << 'VEOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
        "api_version" : "1.3.277"
    }
}
VEOF

# 安装必要系统库
apt install -y libegl1 libglvnd0 libgl1 libxt6 libglu1-mesa libvulkan1 2>/dev/null

echo "  Vulkan 配置完成 ✓"

# -------------------------------------------------------
# 第二步: 创建 conda 环境 (Python 3.11)
# -------------------------------------------------------
echo ""
echo "[2/8] 创建 conda 环境 (Python 3.11)..."

# 确保数据盘目录存在
mkdir -p ${DATA_DIR}
mkdir -p ${DATA_DIR}/conda_envs
mkdir -p ${DATA_DIR}/conda_pkgs

if ! command -v conda &> /dev/null; then
    echo "  conda 未找到，安装 Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda3
    eval "$(/root/miniconda3/bin/conda shell.bash hook)"
    rm /tmp/miniconda.sh
fi

if conda env list | grep -q "${CONDA_ENV_NAME}"; then
    echo "  环境 '${CONDA_ENV_NAME}' 已存在，跳过创建"
else
    conda create -n ${CONDA_ENV_NAME} python=3.11 -y
fi

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

# 验证 Python 版本
PYTHON_VER=$(python --version 2>&1)
echo "  Python: $PYTHON_VER"
if [[ "$PYTHON_VER" != *"3.11"* ]]; then
    echo "  [错误] Python 版本不是 3.11！Isaac Sim 5.1 强制要求 3.11"
    exit 1
fi

pip install --upgrade pip -q
echo "  conda 环境就绪 ✓"

# -------------------------------------------------------
# 第三步: 安装 Isaac Sim 5.1.0 (pip 方式，约 10GB)
# -------------------------------------------------------
echo ""
echo "[3/8] 安装 Isaac Sim 5.1.0 (约 10GB，AutoDL 内网约 10-20 分钟)..."

if python -c "import isaacsim" 2>/dev/null; then
    echo "  Isaac Sim 已安装，跳过"
else
    # 先装核心包
    pip install isaacsim==5.1.0 --extra-index-url https://pypi.nvidia.com
    # 再装完整扩展包
    pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com || {
        echo "  [提示] 完整包有冲突，分包安装..."
        pip install isaacsim-apps==5.1.0 isaacsim-replicator==5.1.0 \
            isaacsim-extscache-physics==5.1.0 isaacsim-extscache-kit==5.1.0 \
            isaacsim-extscache-kit-sdk==5.1.0 \
            --extra-index-url https://pypi.nvidia.com
    }
    echo "  Isaac Sim 5.1.0 安装完成 ✓"
fi

# -------------------------------------------------------
# 第四步: 检查 PyTorch (Isaac Sim 已自带，通常不需重装)
# -------------------------------------------------------
echo ""
echo "[4/8] 检查 PyTorch..."

TORCH_OK=$(python -c "
import torch
if torch.__version__.startswith('2.7') and torch.cuda.is_available():
    print('yes')
else:
    print('no')
" 2>/dev/null)

if [ "$TORCH_OK" = "yes" ]; then
    python -c "import torch; print(f'  PyTorch {torch.__version__} + CUDA {torch.version.cuda} 已就绪，跳过重装')"
else
    echo "  需要安装 PyTorch 2.7.0..."
    pip install --force-reinstall torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128 -q
fi
echo "  PyTorch 就绪 ✓"

# -------------------------------------------------------
# 第五步: 安装 Isaac Lab v2.3.0
# -------------------------------------------------------
echo ""
echo "[5/8] 安装 Isaac Lab ${ISAACLAB_BRANCH}..."
echo "  位置: ${DATA_DIR}/IsaacLab"

cd ${DATA_DIR}
if [ -d "IsaacLab" ]; then
    echo "  IsaacLab 目录已存在，跳过"
else
    # 优先查找本地上传的压缩包 (解决 GitHub clone 慢的问题)
    ISAACLAB_ARCHIVE=""
    for f in IsaacLab*.zip IsaacLab*.tar.gz IsaacLab*.tar; do
        if [ -f "${DATA_DIR}/$f" ]; then
            ISAACLAB_ARCHIVE="${DATA_DIR}/$f"
            break
        fi
    done

    if [ -n "$ISAACLAB_ARCHIVE" ]; then
        echo "  发现本地压缩包: $(basename $ISAACLAB_ARCHIVE)"
        echo "  正在解压到 IsaacLab/ ..."
        mkdir -p IsaacLab
        case "$ISAACLAB_ARCHIVE" in
            *.zip) unzip -q -o "$ISAACLAB_ARCHIVE" -d IsaacLab ;;
            *.tar.gz) tar xzf "$ISAACLAB_ARCHIVE" -C IsaacLab ;;
            *.tar) tar xf "$ISAACLAB_ARCHIVE" -C IsaacLab ;;
        esac
        # 如果解压后里面只有一个顶层目录 (如 IsaacLab-2.3.0/)，把内容提上来
        SUBDIRS=($(ls IsaacLab))
        if [ ${#SUBDIRS[@]} -eq 1 ] && [ -d "IsaacLab/${SUBDIRS[0]}" ]; then
            echo "  检测到嵌套目录 ${SUBDIRS[0]}，展平中..."
            mv IsaacLab/${SUBDIRS[0]}/* IsaacLab/ 2>/dev/null
            mv IsaacLab/${SUBDIRS[0]}/.* IsaacLab/ 2>/dev/null
            rmdir IsaacLab/${SUBDIRS[0]} 2>/dev/null
        fi
        echo "  解压完成 ✓"
    else
        echo "  未找到本地压缩包，尝试 git clone (可能较慢)..."
        echo "  [提示] 如果太慢，可以 Ctrl+C 取消，然后:"
        echo "    1. 本地电脑下载: https://github.com/isaac-sim/IsaacLab/archive/refs/tags/v2.3.0.zip"
        echo "    2. 上传到 ${DATA_DIR}/IsaacLab-2.3.0.zip"
        echo "    3. 重新运行本脚本"
        echo ""
        git clone https://github.com/isaac-sim/IsaacLab.git --branch ${ISAACLAB_BRANCH} --depth 1
    fi
fi

cd IsaacLab

# 系统依赖
apt install cmake build-essential -y 2>/dev/null || sudo apt install cmake build-essential -y 2>/dev/null

# 安装 Isaac Lab + rsl_rl (TienKung-Lab 需要 rsl_rl)
chmod +x isaaclab.sh
echo "  正在安装 Isaac Lab 扩展 (可能需要 10-20 分钟)..."
./isaaclab.sh --install rsl_rl 2>&1 | tail -5

# 验证
if python -c "import isaaclab" 2>/dev/null; then
    echo "  Isaac Lab 安装成功 ✓"
else
    echo "  [警告] isaaclab 导入失败，查看上方日志"
    echo "  如果只是 VSCode settings 报错则可忽略"
fi

# -------------------------------------------------------
# 第六步: 克隆并安装 TienKung-Lab
# -------------------------------------------------------
echo ""
echo "[6/8] 安装 TienKung-Lab..."
echo "  位置: ${DATA_DIR}/TienKung-Lab"

cd ${DATA_DIR}
if [ -d "TienKung-Lab" ]; then
    echo "  TienKung-Lab 目录已存在，跳过"
else
    # 同样优先查找本地压缩包
    TKL_ARCHIVE=""
    for f in TienKung-Lab*.zip TienKung-Lab*.tar.gz TienKung-Lab*.tar; do
        if [ -f "${DATA_DIR}/$f" ]; then
            TKL_ARCHIVE="${DATA_DIR}/$f"
            break
        fi
    done

    if [ -n "$TKL_ARCHIVE" ]; then
        echo "  发现本地压缩包: $(basename $TKL_ARCHIVE)"
        echo "  正在解压到 TienKung-Lab/ ..."
        mkdir -p TienKung-Lab
        case "$TKL_ARCHIVE" in
            *.zip) unzip -q -o "$TKL_ARCHIVE" -d TienKung-Lab ;;
            *.tar.gz) tar xzf "$TKL_ARCHIVE" -C TienKung-Lab ;;
            *.tar) tar xf "$TKL_ARCHIVE" -C TienKung-Lab ;;
        esac
        # 如果解压后里面只有一个顶层目录 (如 TienKung-Lab-main/)，把内容提上来
        SUBDIRS=($(ls TienKung-Lab))
        if [ ${#SUBDIRS[@]} -eq 1 ] && [ -d "TienKung-Lab/${SUBDIRS[0]}" ]; then
            echo "  检测到嵌套目录 ${SUBDIRS[0]}，展平中..."
            mv TienKung-Lab/${SUBDIRS[0]}/* TienKung-Lab/ 2>/dev/null
            mv TienKung-Lab/${SUBDIRS[0]}/.* TienKung-Lab/ 2>/dev/null
            rmdir TienKung-Lab/${SUBDIRS[0]} 2>/dev/null
        fi
        echo "  解压完成 ✓"
    else
        echo "  未找到本地压缩包，尝试 git clone..."
        echo "  [提示] 如果太慢: https://github.com/Open-X-Humanoid/TienKung-Lab/archive/refs/heads/main.zip"
        git clone https://github.com/Open-X-Humanoid/TienKung-Lab.git
    fi
fi

cd TienKung-Lab

# 修复: flatdict 等包需要 setuptools (Python 3.11 默认不带 pkg_resources)
pip install setuptools --upgrade -q

# 安装 TienKung-Lab 主包
pip install -e . -q
echo "  TienKung-Lab 主包安装完成"

# 安装自带的 rsl_rl (可能是定制版)
if [ -d "rsl_rl" ]; then
    cd rsl_rl
    pip install -e . -q
    cd ..
    echo "  rsl_rl (TienKung-Lab 版) 安装完成"
fi

echo "  TienKung-Lab 安装完成 ✓"

# -------------------------------------------------------
# 第七步: 部署 OMNI 29-DOF 模型
# -------------------------------------------------------
echo ""
echo "[7/8] 部署 OMNI 29-DOF 机器人模型..."

# OMNI 模型应该已经上传到 AutoDL (通过 SCP 或 AutoDL 文件管理器)
# 检查模型是否在预期位置
OMNI_SOURCE=""
for candidate in \
    "${DATA_DIR}/${OMNI_MODEL_DIR}" \
    "/root/${OMNI_MODEL_DIR}" \
    "$(dirname "$0")"; do
    if [ -f "${candidate}/robots/omni_29dof_nohead_noshoe_dcmotor_identified.py" ]; then
        OMNI_SOURCE="${candidate}"
        break
    fi
done

if [ -z "$OMNI_SOURCE" ]; then
    echo "  [提示] 未找到 OMNI 模型文件。请手动上传:"
    echo "    将 omni_29dof_v260705/ 文件夹上传到 ${DATA_DIR}/"
    echo "    然后重新运行本脚本，或手动执行第八步"
else
    echo "  找到 OMNI 模型: ${OMNI_SOURCE}"
    # 复制到 TienKung-Lab 的 assets 目录
    OMNI_DEST="${DATA_DIR}/TienKung-Lab/legged_lab/assets/omni_29dof"
    if [ -d "$OMNI_DEST" ]; then
        echo "  OMNI 模型已部署到 TienKung-Lab，跳过"
    else
        mkdir -p "$OMNI_DEST"
        cp -r "${OMNI_SOURCE}/assets/omni_29dof_nohead_noshoe" "$OMNI_DEST/"
        cp -r "${OMNI_SOURCE}/robots" "$OMNI_DEST/"
        cp -r "${OMNI_SOURCE}/actuators" "$OMNI_DEST/"
        echo "  OMNI 模型已复制到: ${OMNI_DEST}"
    fi
    echo "  OMNI 模型部署完成 ✓"
fi

# -------------------------------------------------------
# 第八步: 运行 OMNI 集成脚本
# -------------------------------------------------------
echo ""
echo "[8/8] 集成 OMNI 到 TienKung-Lab 训练框架..."

INTEGRATE_SCRIPT="${DATA_DIR}/TienKung-Lab/integrate_omni.py"
if [ -f "$INTEGRATE_SCRIPT" ]; then
    cd ${DATA_DIR}/TienKung-Lab
    python integrate_omni.py
    echo "  OMNI 集成完成 ✓"
else
    echo "  [提示] integrate_omni.py 未找到"
    echo "  请将 integrate_omni.py 上传到 ${DATA_DIR}/TienKung-Lab/"
    echo "  然后运行: cd ${DATA_DIR}/TienKung-Lab && python integrate_omni.py"
fi

# -------------------------------------------------------
# 完成
# -------------------------------------------------------
echo ""
echo "============================================"
echo "  安装完成！"
echo "============================================"
echo ""
echo "后续步骤:"
echo "  1. 激活环境:    conda activate ${CONDA_ENV_NAME}"
echo "  2. 验证环境:    bash verify_env.sh"
echo "  3. 验证训练:    bash run_training.sh 验证"
echo "  4. 完整训练:    bash run_training.sh 走路"
echo ""
echo "如果遇到版本兼容问题 (TienKung-Lab 基于 IsaacLab 2.1，"
echo "我们装的是 2.3)，查看 integrate_omni.py 的输出提示。"
echo ""
echo "TensorBoard 查看训练曲线:"
echo "  bash run_training.sh tensorboard"
echo "  本地 SSH 隧道: ssh -L 6006:localhost:6006 -p <端口> root@<地址>"
