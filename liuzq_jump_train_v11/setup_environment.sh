#!/bin/bash
# ============================================================================
# Omni 29-DOF 跳高训练 - 服务器 Docker 内环境自检 / 依赖安装
#
# 用法：在服务器 Docker 容器内执行：
#   bash setup_environment.sh
#
# 它会自动：
#   1) 定位 Python（优先环境变量 PYTHON，其次 conda 的 isaaclab env，最后默认 python3）
#   2) 校验 Isaac Lab 是否可导入（>= 2.2.0 / Isaac Sim >= 5.0.0）
#   3) 安装 RSL-RL（>= 4.0，支持多卡 DDP）及运行依赖
#   4) 校验 isaaclab_rl / isaaclab_tasks
# ============================================================================
set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
source "$PROJECT_ROOT/lib_env.sh"

echo "=========================================================="
echo "  Omni 29-DOF 跳高训练 - 环境自检 / 安装脚本"
echo "=========================================================="

# ---------- 1. 选择 python ----------
PYTHON="$(detect_python)"
echo "[1/4] 使用 Python: $PYTHON"

# ---------- 2. 校验 isaaclab ----------
echo "[2/4] 校验 Isaac Lab ..."
if ! "$PYTHON" -c "import isaaclab" >/dev/null 2>&1; then
    echo "!! 当前 Python 环境里没有 Isaac Lab，请先激活正确的环境再运行本脚本。"
    echo "   常见做法（根据你 Docker 的实际情况选一种）："
    echo '     conda activate isaaclab'
    echo '     或 export PYTHON=/path/to/isaaclab/python 后重新运行本脚本'
    echo '     或用 install_isaaclab.sh 在本机/容器里创建 isaaclab 环境'
    exit 1
fi
"$PYTHON" - <<'EOF'
import importlib.metadata as md
import isaaclab
import os

print("   Isaac Lab 路径:", isaaclab.__file__)
ver = getattr(isaaclab, "__version__", None)
if not ver:
    try:
        ver = md.version("isaaclab")
    except Exception:
        ver = None
if not ver:  # 兜底：读仓库根 VERSION 文件
    p = os.path.dirname(isaaclab.__file__)
    for _ in range(4):
        p = os.path.dirname(p)
        vf = os.path.join(p, "VERSION")
        if os.path.exists(vf):
            ver = open(vf).read().strip()
            break
print("   Isaac Lab 版本:", ver if ver else "未知（旧版，无 __version__）")
EOF

# ---------- 3. 安装训练依赖 ----------
echo "[3/4] 安装/升级 RSL-RL 与运行依赖 ..."
# rsl-rl-lib 版本必须与 Isaac Lab 匹配:
#   - 服务器为旧版 Isaac Lab（RslRlPpoActorCriticCfg 无 actor_obs_normalization 字段，
#     isaaclab 无 __version__ 属性）→ 用 rsl-rl-lib ==2.3.3（官方 pin；
#     2.3.3 自带多卡分布式，支持 run_train.sh 的双卡模式）
#   - 若部署到 Isaac Lab 2.3+/main，才需要 4.x/5.x（配合 scripts/rsl_rl_compat.py 迁移）
"$PYTHON" -m pip install "rsl-rl-lib==2.3.3" pyyaml packaging

# ---------- 4. 校验 RL 框架与任务依赖 ----------
echo "[4/4] 校验 RSL-RL / isaaclab_rl / isaaclab_tasks ..."
"$PYTHON" - <<'EOF'
import importlib.metadata as md
import rsl_rl  # noqa: F401
import isaaclab_rl  # noqa: F401
import isaaclab_tasks  # noqa: F401
print("   rsl-rl-lib      :", md.version("rsl-rl-lib"))
print("   isaaclab_rl     :", isaaclab_rl.__file__)
print("   isaaclab_tasks  :", isaaclab_tasks.__file__)
try:  # 诊断 rsl-rl >=4.0 所需的 policy->actor/critic 迁移来源
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    print("   迁移方式        : 内置 handle_deprecated_rsl_rl_cfg（Isaac Lab main）")
except ImportError:
    print("   迁移方式        : 无内置迁移（旧版 Isaac Lab，由 scripts/rsl_rl_compat.py 兜底）")
print("   依赖检查通过 OK")
EOF

echo
echo "=========================================================="
echo " 环境就绪！接下来用一键训练脚本："
echo "   bash run_train.sh --headless --num_envs 4096"
echo " （run_train.sh 会自动检测 GPU 并选择单卡/双卡并行）"
echo "=========================================================="
