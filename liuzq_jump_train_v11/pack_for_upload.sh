#!/bin/bash
# ============================================================================
# 把整个跳高训练工程打包成 tar.gz，便于上传到服务器。
# 排除：__pycache__ / *.pyc / logs / .git
#
# 用法：bash pack_for_upload.sh
# 产物：<my_omni_jump_train_v2 上一级目录>/omni_jump_<时间戳>.tar.gz
# ============================================================================
set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
PROJ_NAME="$(basename "$PROJECT_ROOT")"
OUT="${PROJECT_ROOT}/../omni_jump_${STAMP}.tar.gz"

tar --exclude='__pycache__' --exclude='*.pyc' --exclude='logs' --exclude='.git' \
    -czf "$OUT" -C "$(dirname "$PROJECT_ROOT")" "$PROJ_NAME"

echo "打包完成: $OUT"
echo
echo "上传到服务器（示例）："
echo "  scp ${OUT} user@server:/path/to/   # 或 rsync"
echo
echo "服务器上解压："
echo "  cd /path/to && tar -xzf omni_jump_${STAMP}.tar.gz && cd ${PROJ_NAME}"
