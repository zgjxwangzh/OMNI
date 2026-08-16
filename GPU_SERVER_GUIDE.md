# GPU 服务器部署指南

本文档说明如何在 GPU 服务器上搭建 OMNI 29-DOF 仿真环境，验证部署管线。

## 目录

1. [环境搭建](#1-环境搭建)
2. [文件上传](#2-文件上传)
3. [走路仿真验证](#3-走路仿真验证)
4. [物理跟踪验证](#4-物理跟踪验证)
5. [预期结果](#5-预期结果)
6. [故障排查](#6-故障排查)

---

## 1. 环境搭建

### 1.1 系统要求

- Python 3.10+（推荐 3.10 或 3.11）
- NVIDIA GPU + CUDA 11.8+ 或 12.x
- Linux（Ubuntu 20.04/22.04 推荐）

### 1.2 一键安装

```bash
# 上传项目文件后，进入项目目录
cd /root/omni_29dof_v260705_gpu

# 运行安装脚本（会自动检测 conda 并提示选择）
bash setup_gpu_server.sh
```

**AutoDL 用户注意**：如果默认 conda 环境被占用，脚本会提示你选择：
- 选项 2（推荐）：创建新的 `omni_gpu` 环境
- 选项 3：退出 conda，用系统 Python

### 1.3 手动安装（如果脚本失败）

```bash
# 1. 创建 conda 环境（推荐）
conda create -n omni python=3.10 -y
conda activate omni

# 2. 安装核心依赖
pip install mujoco numpy pyyaml pillow
pip install onnxruntime-gpu  # GPU 推理

# 3. 验证安装
python3 -c "
import mujoco; print(f'✓ mujoco {mujoco.__version__}')
import onnxruntime as ort
print(f'✓ onnxruntime {ort.__version__}')
print(f'  providers: {ort.get_available_providers()}')
"
```

### 1.4 验证 GPU

```bash
nvidia-smi
```

应显示 GPU 型号、驱动版本、显存。

---

## 2. 文件上传

### 2.1 需要上传的文件

从本地 Mac 上传整个项目目录：

```bash
# 在本地 Mac 执行
cd /Users/condenast/Downloads

# 打包（只包含 GPU 服务器需要的文件，输出到上级目录）
tar czf omni_project.tar.gz \
    -C omni_29dof_v260705 \
    --exclude='__pycache__' \
    --exclude='.qoder' \
    --exclude='.DS_Store' \
    --exclude='_archive' \
    --exclude='frames*' \
    --exclude='数据1' \
    --exclude='第一组 跳高 翻箱' \
    --exclude='assets' \
    --exclude='robots' \
    --exclude='actuators' \
    --exclude='*.pyc' \
    --exclude='MUJOCO_LOG.TXT' \
    --exclude='*_results.txt' \
    --exclude='omni_project.tar.gz' \
    --exclude='._*' \
    .

# 上传到 GPU 服务器
scp omni_project.tar.gz user@gpu-server:/path/to/
```

### 2.2 在 GPU 服务器解压

```bash
# 在 GPU 服务器执行

# 方式 1：解压到新目录（推荐，避免覆盖已有项目）
mkdir -p /root/omni_29dof_v260705_gpu
cd /root/omni_29dof_v260705_gpu
tar xzf /path/to/omni_project.tar.gz

# 方式 2：如果已有旧目录，先备份再覆盖
cd /root
mv omni_29dof_v260705 omni_29dof_v260705_backup  # 备份旧目录
mkdir omni_29dof_v260705
cd omni_29dof_v260705
tar xzf /path/to/omni_project.tar.gz
```

### 2.3 验证文件完整性

```bash
# 检查关键文件
ls -la omni_29dof_mjc/mjcf/omni_29dof.xml
ls -la omni_rl_sdk/policy/loco_mode/model/*.onnx
ls -la motion_data/*_highdynamic.npz | wc -l  # 应显示 29
```

---

## 3. 走路仿真验证

**方案 1**：用 SDK 自带的走路 ONNX 验证部署管线。

### 3.1 运行

```bash
cd /path/to/omni_29dof_v260705

# 默认：跑 4000 步（10 秒）
python3 run_walking_sim.py

# 用 GPU 推理
python3 run_walking_sim.py --device cuda

# 跑更长时间
python3 run_walking_sim.py --steps 8000
```

### 3.2 预期输出

```
✓ MuJoCo 模型加载: 36 qpos, 29 actuators
  重力: [ 0.  0. -9.81]
✓ ONNX 模型加载: .../omni_7dof_63k_2file.onnx
  provider: CUDAExecutionProvider  # 或 CPUExecutionProvider
✓ 配置加载:
  obs=90, history=10, decimation=4
  action_scale=0.25, policy_joints=25

═══ 开始仿真 ═══
  dt=0.0025s, control_dt=0.01s
  步数: 4000 (10.0s)
  速度指令: vx=0.5 m/s

  t=1.0s  h=0.820m  pos=[0.05, 0.00, 0.82]
  t=2.0s  h=0.818m  pos=[0.10, 0.00, 0.82]
  ...

═══ 仿真结果 ═══
  总步数: 4000
  仿真时间: 10.0s
  初始高度: 0.820m
  最终高度: 0.815m
  最低高度: 0.810m
  最高高度: 0.825m
  X 位移: 0.500m
  Y 位移: 0.010m

  ✓ 机器人保持直立，走路仿真正常
  ✓ 无 NaN
```

### 3.3 成功标准

- 机器人保持直立（高度 0.7-0.9m）
- X 方向有位移（向前走）
- 无 NaN、无摔倒

---

## 4. 物理跟踪验证

**方案 2**：用 PD 控制 + 重力验证 29 个动作的可跟踪性。

### 4.1 运行

```bash
# 验证所有 29 个动作
python3 physics_track_verify.py

# 只验证某个动作
python3 physics_track_verify.py --motion 跳高06

# 调整 PD 增益
python3 physics_track_verify.py --kp_scale 1.5
```

### 4.2 预期输出

```
✓ 模型加载: 36 qpos, 29 actuators
  重力: [ 0.  0. -9.81]
  kp_scale=1.0, kd_scale=1.0
  sim_dt=0.002s

找到 29 个动作

动作                      帧数     误差均值     误差最大         高度范围     最大扭矩     摔倒   状态
-------------------------------------------------------------------------------------
跳高06_chr00             397    1.215    2.740    0.41→1.12    140.0      否  ⚠ 跟踪差
翻箱子01_chr00          1385    1.276    9.817    0.51→1.48    140.0      否  ⚠ 跟踪差
...

良好: 0, 一般: 0, 摔倒/跟踪差: 29
```

### 4.3 结果解读

**所有动作都会显示"跟踪差"**——这是预期的！

| 现象 | 解释 |
|------|------|
| 跟踪误差 1-2 rad | PD 控制器带宽有限，跟不上快速动态 |
| 部分动作误差爆炸 | 数值不稳定，动作太激进 |
| 摔倒（如果 base 不固定） | 需要平衡控制，PD 做不到 |

**这个脚本的价值**：
- 验证仿真环境能跑通
- 给出 PD 控制的 baseline
- RL 训练后应该比这更好

---

## 5. 预期结果

### 5.1 环境验证清单

- [ ] `nvidia-smi` 显示 GPU
- [ ] `python3 -c "import mujoco"` 成功
- [ ] `python3 -c "import onnxruntime"` 成功
- [ ] ONNX provider 包含 `CUDAExecutionProvider`（如果用 GPU）
- [ ] 走路仿真能跑完，机器人不摔倒
- [ ] 物理跟踪验证能跑完（即使结果"差"也正常）

### 5.2 下一步

环境验证通过后：

1. **等训练代码**：向项目组要 high_dynamic 的训练代码
2. **准备训练数据**：29 个 high_dynamic NPZ 已就绪
3. **开始训练**：用 GPU 服务器训练 RL 策略

---

## 6. 故障排查

### 6.1 MuJoCo 安装失败

```bash
# 如果 pip install mujoco 失败
pip install --upgrade pip
pip install mujoco --no-cache-dir
```

### 6.2 onnxruntime-gpu 找不到

```bash
# 检查 CUDA 版本
nvcc --version

# 安装对应版本
# CUDA 11.8: pip install onnxruntime-gpu==1.16.0
# CUDA 12.x: pip install onnxruntime-gpu  # 最新版
```

### 6.3 走路仿真报错 "ONNX 文件不存在"

```bash
# 检查文件路径
ls omni_rl_sdk/policy/loco_mode/model/

# 如果文件缺失，重新上传项目文件
```

### 6.4 仿真中遇到 NaN

```bash
# 通常是数值不稳定，尝试：
# 1. 减小仿真步长
python3 run_walking_sim.py --steps 2000  # 先跑短时间看是否稳定

# 2. 检查 ONNX 模型是否损坏
python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('omni_rl_sdk/policy/loco_mode/model/omni_7dof_63k_2file.onnx')
print('✓ ONNX 模型正常')
"
```

### 6.5 GPU 服务器无显示（headless）

走路仿真默认 headless，不需要显示器。如果报错：

```bash
# 确保没有调用渲染
# run_walking_sim.py 默认 --headless，无需额外参数
```

---

## 附录：脚本说明

| 脚本 | 用途 | 依赖 |
|------|------|------|
| `setup_gpu_server.sh` | 环境安装 | bash |
| `run_walking_sim.py` | 走路 ONNX 仿真 | mujoco, onnxruntime, pyyaml |
| `physics_track_verify.py` | 物理跟踪验证 | mujoco, numpy |
| `batch_verify_mujoco.py` | 运动学验证（无需 GPU） | mujoco, numpy |

---

**文档版本**：v1.0  
**更新日期**：2026-08-10  
**维护者**：项目团队
