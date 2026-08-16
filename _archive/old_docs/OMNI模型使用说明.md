# OMNI 29-DOF 机器人模型使用说明

## 一、文件清单

压缩包内包含两个训练目录：

```
omni_base_model.tar.gz          ← 基础走路模型（5万步训练）
omni_stair_model.tar.gz         ← 爬楼梯微调模型（在基础模型上微调）
```

每个目录包含：
- `model_xxxx.pt` — 模型权重文件
- `params/` — 网络结构配置（必须与 .pt 文件一起使用）

---

## 二、环境要求

### 硬件
- NVIDIA GPU（建议 RTX 2060 及以上，显存 ≥ 8GB）
- 有显示环境（桌面系统，非纯命令行服务器）

### 软件
- Ubuntu 20.04/22.04 或 Windows + WSL2
- NVIDIA 驱动 + CUDA 11.8+
- TienKung-Lab 代码仓库（与训练环境版本一致）
- Isaac Lab 环境（`env_isaaclab` conda 环境）

### 代码版本
确保你的 TienKung-Lab 仓库包含 OMNI 29-DOF 集成：
```bash
# 检查 OMNI 环境是否存在
ls legged_lab/envs/omni/
# 应该看到 omni_env.py, walk_cfg.py 等文件
```

---

## 三、解压模型

```bash
# 进入 TienKung-Lab 目录
cd ~/TienKung-Lab  # 或你的实际路径

# 解压基础模型
cd logs/omni_walk/
tar xzf /path/to/omni_base_model.tar.gz

# 解压爬楼梯模型
tar xzf /path/to/omni_stair_model.tar.gz
```

解压后目录结构：
```
logs/omni_walk/
├── 2026-07-28_23-43-16/        ← 基础走路模型
│   ├── model_4900.pt
│   └── params/
── 2026-08-08_11-41-45/        ← 爬楼梯微调模型
    ├── model_4800.pt
    └── params/
```

---

## 四、查看仿真效果

### 4.1 激活环境

```bash
conda activate env_isaaclab  # 或你的 Isaac Lab 环境名
```

### 4.2 运行基础走路模型

```bash
cd ~/TienKung-Lab  # 确保在仓库根目录

python legged_lab/scripts/play.py --task=omni_walk --num_envs=1 \
    --load_run=2026-07-28_23-43-16 --checkpoint=model_4900.pt
```

### 4.3 运行爬楼梯模型

```bash
python legged_lab/scripts/play.py --task=omni_walk --num_envs=1 \
    --load_run=2026-08-08_11-41-45 --checkpoint=model_4800.pt
```

### 4.4 预期效果

- 会弹出 **Isaac Sim 3D 窗口**
- 看到 OMNI 机器人在仿真环境中行走/爬楼梯
- 按 `Ctrl+C` 或关闭窗口退出

### 4.5 常见问题

**问题 1：报错 `ModuleNotFoundError: No module named 'legged_lab'`**
```bash
# 确保在仓库根目录运行，或设置 PYTHONPATH
export PYTHONPATH=/path/to/TienKung-Lab:$PYTHONPATH
```

**问题 2：报错 `IsADirectoryError`**
```bash
# 检查 params/ 目录是否完整
ls logs/omni_walk/2026-07-28_23-43-16/params/
# 应该有 env.yaml, agent.yaml 等文件
```

**问题 3：Isaac Sim 窗口闪退**
- 检查 NVIDIA 驱动版本（建议 525+）
- 检查 Vulkan 支持：`vulkaninfo | head`
- 尝试降低渲染质量：加 `--rendering_mode performance` 参数

**问题 4：机器人动作异常（抖动、摔倒）**
- 确认代码版本与训练环境一致
- 确认 URDF 文件未被修改
- 确认 `params/` 目录完整

---

## 五、继续微调训练

如果想用这些模型作为起点继续训练新动作（如跳高）：

### 5.1 准备动捕数据

将动捕数据转换为 AMP 格式（需要 `.npz` → `.txt` 转换脚本）。

### 5.2 修改配置

```bash
# 编辑 walk_cfg.py，指向新的动捕数据
nano legged_lab/envs/omni/walk_cfg.py

# 修改以下两行：
amp_motion_files_display = ["legged_lab/envs/omni/datasets/motion_visualization/你的文件.txt"]
amp_motion_files = ["legged_lab/envs/omni/datasets/motion_amp_expert/你的文件.txt"]
```

### 5.3 启动训练

```bash
# 从基础模型出发
python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=4096 \
    --load_run=2026-07-28_23-43-16 --checkpoint=model_4900.pt \
    --logger=tensorboard

# 或从爬楼梯模型出发
python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=4096 \
    --load_run=2026-08-08_11-41-45 --checkpoint=model_4800.pt \
    --logger=tensorboard
```

### 5.4 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir=logs --host 0.0.0.0 --port 6006

# 浏览器打开
# http://localhost:6006  （本地）
# http://服务器IP:6006   （远程服务器）
```

关注指标：
- **Train/mean_reward**：越高越好，爬楼梯模型达到 12-14
- **Train/mean_episode_length**：越长越好，爬楼梯模型达到 450-550

---

## 六、模型性能参考

| 模型 | 训练步数 | Reward | Episode Length | 说明 |
|---|---|---|---|---|
| 基础走路 | 50,000 | +2 | 964 | 通用行走能力 |
| 爬楼梯微调 | ~20,000 | +13 | 550 | 在基础模型上微调 |

**注意**：Reward 和 Episode Length 的绝对值取决于奖励函数设计，不同任务之间不能直接比较。

---

## 七、文件打包/分享

如果你想把自己的训练成果分享给其他人：

```bash
# 只打包最新模型 + params（不要打包所有中间检查点）
cd logs/omni_walk/你的训练目录/
mkdir -p tmp/你的训练目录
cp model_最新.pt tmp/你的训练目录/
cp -r params/ tmp/你的训练目录/
cd tmp
tar czf ../模型名称.tar.gz 你的训练目录/
cd ..
rm -rf tmp/
```

---

## 八、技术支持

遇到问题时提供以下信息：
1. 完整的错误日志
2. `nvidia-smi` 输出
3. TienKung-Lab 的 git commit hash：`git log --oneline -1`
4. Python 环境：`conda list | grep -E "torch|isaac"`
