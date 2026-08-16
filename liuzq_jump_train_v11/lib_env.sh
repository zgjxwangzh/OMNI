#!/bin/bash
# ============================================================================
# 公共函数：定位 Isaac Lab 的 Python 解释器
# 被 setup_environment.sh / run_train.sh 复用。
#
# 查找顺序：
#   1) 环境变量 PYTHON（手动指定）
#   2) conda 的 isaaclab 环境
#   3) 系统 python3
# ============================================================================

detect_python() {
    local PYTHON="${PYTHON:-}"

    # 2) conda 的 isaaclab 环境
    if [ -z "$PYTHON" ] && command -v conda >/dev/null 2>&1; then
        local env_path
        env_path="$(conda env list 2>/dev/null | awk '$1=="isaaclab"{print $NF; exit}')"
        if [ -n "$env_path" ] && [ -x "$env_path/bin/python" ]; then
            PYTHON="$env_path/bin/python"
        fi
    fi

    # 3) 系统 python3
    if [ -z "$PYTHON" ]; then
        PYTHON="$(command -v python3 || command -v python || true)"
    fi

    if [ -z "$PYTHON" ]; then
        echo "[ERROR] 未找到 python3，请先安装/激活 Python 环境。" >&2
        return 1
    fi

    echo "$PYTHON"
}
