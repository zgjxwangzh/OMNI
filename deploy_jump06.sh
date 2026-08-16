#!/bin/bash
# 跳高 06 优化训练 - 一步到位部署脚本
# 
# 修改内容：
# 1. bvh_retarget.py: 添加膝盖角度软限制（110°）
# 2. tracking_env_omni_cfg.py: 启用腿部关节跟踪 + 放宽终止条件
# 3. 重新生成跳高 06 NPZ（膝盖 max 136.8° → 111.3°）
#
# 用法：在 AutoDL 服务器上运行
#   bash deploy_jump06.sh

set -e

echo "=========================================="
echo "跳高 06 优化训练部署"
echo "=========================================="

# 1. 上传 retargeted NPZ（从本地）
echo ""
echo "[1/4] 上传 retargeted NPZ..."

# 在本地运行：
# scp -P 37219 retargeted/跳高06_chr00.npz root@connect.bjb1.seetacloud.com:/root/autodl-tmp/omni_29dof_v260705/retargeted/
scp -P 37219 retargeted/跳高06_chr00.npz root@connect.bjb1.seetacloud.com:/root/autodl-tmp/omni_29dof_v260705/retargeted/

scp -P 37219 omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_omni_cfg.py root@connect.bjb1.seetacloud.com:/root/autodl-tmp/omni_29dof_v260705/omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_omni_cfg.py

# 2. 转换为 training NPZ
echo ""
echo "[2/4] 转换为 training NPZ..."
/root/autodl-tmp/IsaacLab/isaaclab.sh -p omni_mimic/scripts/retargeted_npz_to_training_npz.py \
    --input_file retargeted/跳高06_chr00.npz \
    --output_file training_data/跳高06_chr00_training.npz \
    --input_fps 30 --headless

# 3. 上传修改后的训练配置（从本地）
echo ""
echo "[3/4] 上传训练配置..."
# 在本地运行：
# scp -P 33310 omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_omni_cfg.py \
#     root@connect.bjb2.seetacloud.com:/root/autodl-tmp/omni_29dof_v260705/omni_mimic/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_omni_cfg.py

# 4. 开始训练
echo ""
echo "[4/4] 开始训练..."
/root/autodl-tmp/IsaacLab/isaaclab.sh -p omni_mimic/scripts/rsl_rl/train.py \
    --task=Tracking-Flat-Omni-Hist-Delayed-DCMotor-v0 \
    --num_envs=4096 \
    --motion_file training_data/跳高06_chr00_training.npz \
    --max_iterations=20000 \
    --run_name jump06_knee_fix \
    agent.save_interval=500

echo ""
echo "=========================================="
echo "训练已启动！"
echo "=========================================="
echo ""
echo "监控命令："
echo "  tensorboard --logdir logs/rsl_rl/omni_flat/ --port 6006"
echo ""
echo "检查点："
echo "  - 1000 步：检查 reward/episode 趋势"
echo "  - 2000 步：生成视频，检查膝盖高度"
echo "  - 5000 步：确认收敛"
echo ""
