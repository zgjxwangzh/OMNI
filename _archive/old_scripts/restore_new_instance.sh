#!/bin/bash
# ============================================================
# 新实例快速恢复脚本
# 适用场景: 从已保存的镜像新建实例，autodl-fs 已自动挂载
# 前提: /root/autodl-fs/omni_project/ 里有关键文件
# 耗时: 约 30-40 分钟 (主要是 Isaac Sim pip 下载)
# ============================================================
set -e

# ===================== 配置区 =====================
FS_DIR="/root/autodl-fs/omni_project"     # 共享文件存储 (跨实例持久)
DATA_DIR="/root/autodl-tmp"               # 数据盘 (高速，但不随镜像)
CONDA_ENV_NAME="env_isaaclab"
# ================================================

echo "============================================"
echo "  新实例恢复脚本"
echo "  从 autodl-fs 恢复 OMNI 训练环境"
echo "============================================"
echo ""

# -------------------------------------------------------
# 第一步: 检查 autodl-fs 文件是否齐全
# -------------------------------------------------------
echo "[1/6] 检查共享存储文件..."

MISSING=0
for f in "TienKung-Lab" "omni_29dof_v260705"; do
    if [ ! -d "${FS_DIR}/$f" ]; then
        echo "  [缺失] ${FS_DIR}/$f"
        MISSING=1
    else
        echo "  [✓] $f"
    fi
done

# 安装包 (zip/tar.gz) 检查
ISAACLAB_ARCHIVE=""
for f in "${FS_DIR}"/IsaacLab*.tar.gz "${FS_DIR}"/IsaacLab*.zip; do
    if [ -f "$f" ]; then
        ISAACLAB_ARCHIVE="$f"
        echo "  [✓] $(basename $f)"
        break
    fi
done
if [ -z "$ISAACLAB_ARCHIVE" ]; then
    echo "  [缺失] IsaacLab.tar.gz 或 IsaacLab.zip"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "  [错误] 共享存储文件不完整！请确认 ${FS_DIR}/ 下有以下内容:"
    echo "    - TienKung-Lab/          (训练框架代码)"
    echo "    - omni_29dof_v260705/    (OMNI 机器人模型)"
    echo "    - IsaacLab.tar.gz        (Isaac Lab 源码)"
    exit 1
fi

echo "  文件检查通过 ✓"

# -------------------------------------------------------
# 第二步: 复制文件到数据盘 (高速 IO)
# -------------------------------------------------------
echo ""
echo "[2/6] 复制文件到数据盘 (autodl-fs → autodl-tmp)..."

mkdir -p ${DATA_DIR}

# TienKung-Lab
if [ ! -d "${DATA_DIR}/TienKung-Lab" ]; then
    echo "  复制 TienKung-Lab..."
    cp -r "${FS_DIR}/TienKung-Lab" "${DATA_DIR}/"
fi
echo "  [✓] TienKung-Lab"

# OMNI 模型
if [ ! -d "${DATA_DIR}/omni_29dof_v260705" ]; then
    echo "  复制 omni_29dof_v260705..."
    cp -r "${FS_DIR}/omni_29dof_v260705" "${DATA_DIR}/"
fi
echo "  [✓] omni_29dof_v260705"

# IsaacLab 压缩包 (后面解压)
if [ ! -d "${DATA_DIR}/IsaacLab" ]; then
    echo "  复制 IsaacLab 压缩包..."
    cp "$ISAACLAB_ARCHIVE" "${DATA_DIR}/"
fi
echo "  [✓] IsaacLab 压缩包"

echo "  文件复制完成 ✓"

# -------------------------------------------------------
# 第三步: 创建 conda 环境
# -------------------------------------------------------
echo ""
echo "[3/6] 创建 conda 环境 (Python 3.11)..."

export CONDA_ENVS_PATH="${DATA_DIR}/conda_envs"
export CONDA_PKGS_DIRS="${DATA_DIR}/conda_pkgs"
mkdir -p ${DATA_DIR}/conda_envs ${DATA_DIR}/conda_pkgs

if ! command -v conda &> /dev/null; then
    echo "  安装 Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda3
    eval "$(/root/miniconda3/bin/conda shell.bash hook)"
    rm /tmp/miniconda.sh
fi

eval "$(conda shell.bash hook)"

if conda env list | grep -q "${CONDA_ENV_NAME}"; then
    echo "  环境已存在，跳过"
else
    conda create -n ${CONDA_ENV_NAME} python=3.11 -y
fi

conda activate ${CONDA_ENV_NAME}
pip install --upgrade pip setuptools -q
echo "  conda 环境就绪 ✓"

# -------------------------------------------------------
# 第四步: 安装 Isaac Sim + PyTorch + Isaac Lab
# -------------------------------------------------------
echo ""
echo "[4/6] 安装 Isaac Sim 5.1.0 (约 10GB，AutoDL 内网 10-20 分钟)..."

if python -c "import isaacsim" 2>/dev/null; then
    echo "  Isaac Sim 已安装，跳过"
else
    pip install isaacsim==5.1.0 --extra-index-url https://pypi.nvidia.com
    pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com || {
        pip install isaacsim-apps==5.1.0 isaacsim-replicator==5.1.0 \
            isaacsim-extscache-physics==5.1.0 isaacsim-extscache-kit==5.1.0 \
            isaacsim-extscache-kit-sdk==5.1.0 \
            --extra-index-url https://pypi.nvidia.com
    }
fi
echo "  Isaac Sim 就绪 ✓"

# PyTorch (Isaac Sim 通常已自带)
TORCH_OK=$(python -c "import torch; print('yes' if torch.__version__.startswith('2.7') and torch.cuda.is_available() else 'no')" 2>/dev/null)
if [ "$TORCH_OK" != "yes" ]; then
    echo "  安装 PyTorch..."
    pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 -q
fi
echo "  PyTorch 就绪 ✓"

# Isaac Lab
echo "  安装 Isaac Lab..."
cd ${DATA_DIR}
if [ ! -d "IsaacLab" ]; then
    mkdir -p IsaacLab
    ARCHIVE_BASE=$(basename "$ISAACLAB_ARCHIVE")
    case "$ARCHIVE_BASE" in
        *.zip) unzip -q -o "${DATA_DIR}/${ARCHIVE_BASE}" -d IsaacLab ;;
        *.tar.gz) tar xzf "${DATA_DIR}/${ARCHIVE_BASE}" -C IsaacLab ;;
    esac
    # 展平嵌套目录
    SUBDIRS=($(ls IsaacLab))
    if [ ${#SUBDIRS[@]} -eq 1 ] && [ -d "IsaacLab/${SUBDIRS[0]}" ]; then
        mv IsaacLab/${SUBDIRS[0]}/* IsaacLab/ 2>/dev/null
        mv IsaacLab/${SUBDIRS[0]}/.* IsaacLab/ 2>/dev/null
        rmdir IsaacLab/${SUBDIRS[0]} 2>/dev/null
    fi
fi

cd IsaacLab
chmod +x isaaclab.sh
./isaaclab.sh --install rsl_rl 2>&1 | tail -3
echo "  Isaac Lab 就绪 ✓"

# -------------------------------------------------------
# 第五步: 安装 TienKung-Lab + 集成 OMNI
# -------------------------------------------------------
echo ""
echo "[5/6] 安装 TienKung-Lab + 集成 OMNI..."

cd ${DATA_DIR}/TienKung-Lab
pip install -e . -q

if [ -d "rsl_rl" ]; then
    cd rsl_rl && pip install -e . -q && cd ..
fi

# 运行 OMNI 集成 (修复路径、注册任务、复制动作数据等)
if [ -f "integrate_omni.py" ]; then
    python integrate_omni.py
else
    # 从 OMNI 模型目录找
    if [ -f "${DATA_DIR}/omni_29dof_v260705/integrate_omni.py" ]; then
        cp "${DATA_DIR}/omni_29dof_v260705/integrate_omni.py" .
        python integrate_omni.py
    fi
fi
echo "  TienKung-Lab + OMNI 集成完成 ✓"

# -------------------------------------------------------
# 第六步: 验证
# -------------------------------------------------------
echo ""
echo "[6/6] 快速验证..."

cd ${DATA_DIR}/TienKung-Lab
VERIFY_OK=$(python -c "
from legged_lab.utils.task_registry import task_registry
from legged_lab.envs import *
if 'omni_walk' in task_registry.task_classes:
    print('yes')
else:
    print('no')
" 2>/dev/null)

if [ "$VERIFY_OK" = "yes" ]; then
    echo "  [✓] omni_walk 任务注册成功"
else
    echo "  [!] 任务注册验证失败，请手动检查"
fi

echo ""
echo "============================================"
echo "  恢复完成！可以开始训练了"
echo "============================================"
echo ""
echo "  训练命令:"
echo "    cd ${DATA_DIR}/TienKung-Lab"
echo "    python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=4096 --logger=tensorboard"
echo ""
echo "  TensorBoard:"
echo "    tensorboard --logdir=logs --host 0.0.0.0 --port 6006"
echo ""
