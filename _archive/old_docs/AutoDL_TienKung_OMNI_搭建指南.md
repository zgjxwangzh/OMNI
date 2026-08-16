# AutoDL 搭建 TienKung-Lab + OMNI 29-DOF 训练环境

## 整体架构（先搞懂再动手）

```
┌─────────────────────────────────────────────────┐
│  Isaac Sim 5.1 (NVIDIA 物理仿真引擎)             │
│  ┌───────────────────────────────────────────┐  │
│  │  Isaac Lab 2.3 (机器人训练框架)             │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │  TienKung-Lab (人形机器人 RL 框架)    │  │  │
│  │  │  ┌───────────────────────────────┐  │  │  │
│  │  │  │  OMNI 29-DOF (你的机器人模型)  │  │  │  │
│  │  │  └───────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

训练方法: **RL + AMP** (强化学习 + 对抗性动作先验)，让机器人模仿人类走路/跑步。

---

## 版本对齐表（核心，不能搞混）

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | **3.11** | Isaac Sim 5.x 强制要求 |
| Isaac Sim | **5.1.0** | pip 安装 |
| Isaac Lab | **v2.3.0** | 源码安装，OMNI 模型需要 ≥2.2 的 API |
| PyTorch | **2.7.0** + CUDA 12.8 | 官方指定 |
| TienKung-Lab | **main 分支** | 训练框架 |
| Ubuntu | **22.04** | GLIBC 2.35+ |
| GPU | **RTX 4090 24GB** | 推荐，4096 并行环境 |

> 注意: TienKung-Lab 官方基于 IsaacSim 4.5 + IsaacLab 2.1，但 OMNI 模型用了
> IsaacLab 2.2+ 的新 API (`UrdfConverterCfg.JointDriveCfg.PDGainsCfg` 等)，
> 所以我们装 5.1 + 2.3。TienKung-Lab 代码可能需要小幅适配。

---

## 第一步：创建 AutoDL 实例

### 1.1 选镜像
- **推荐**: AutoDL 官方镜像 → `PyTorch 2.x` + `Ubuntu 22.04` + `CUDA 12.x`
- 或搜索社区镜像中预装 Isaac 的（能省安装时间）

### 1.2 选 GPU
- **RTX 4090 (24GB)** — 性价比最高，4096 环境没问题
- 显存不够就降 `--num_envs`

### 1.3 磁盘
- 系统盘: 默认 30GB 够用（我们不往系统盘装东西）
- 数据盘: **≥ 50GB**（Isaac Sim 约 10GB + Isaac Lab 3GB + 其他）

---

## 第二步：下载仓库并上传到 AutoDL

> **重要**: AutoDL 访问 GitHub 很慢，强烈建议在本地电脑先下载好再上传！

### 2.1 本地电脑下载两个仓库

浏览器打开以下链接直接下载 zip：

| 仓库 | 下载链接 |
|------|----------|
| Isaac Lab v2.3.0 | https://github.com/isaac-sim/IsaacLab/archive/refs/tags/v2.3.0.zip |
| TienKung-Lab | https://github.com/Open-X-Humanoid/TienKung-Lab/archive/refs/heads/main.zip |

下载后你会得到:
- `IsaacLab-2.3.0.zip` (约 100-200MB)
- `TienKung-Lab-main.zip` (约 50-100MB)

### 2.2 上传到 AutoDL 数据盘

把以下文件全部上传到 `/root/autodl-tmp/`：

```bash
# 方法1: SCP (本地电脑执行)
scp -P <SSH端口> IsaacLab-2.3.0.zip TienKung-Lab-main.zip root@<SSH地址>:/root/autodl-tmp/
scp -r -P <SSH端口> omni_29dof_v260705/ root@<SSH地址>:/root/autodl-tmp/

# 方法2: AutoDL 网页端 → 文件管理 → 上传
```

上传后 `/root/autodl-tmp/` 应该有:
```
/root/autodl-tmp/
├── IsaacLab-2.3.0.zip        ← 不用手动解压，脚本会自动处理
├── TienKung-Lab-main.zip     ← 同上
└── omni_29dof_v260705/       ← 本文件夹 (脚本+模型)
```

> 脚本会自动检测这些 zip 文件并解压，不需要你手动解压。
> 如果你已经提前解压好了（目录名是 `IsaacLab` 和 `TienKung-Lab`），脚本也会自动跳过。

---

## 第三步：一键安装

```bash
# SSH 登录 AutoDL 后
cd /root/autodl-tmp/omni_29dof_v260705
chmod +x setup_tienkung.sh
bash setup_tienkung.sh
```

脚本会自动完成 (约 40-60 分钟):
1. ✅ 检查系统 (GLIBC / GPU / Ubuntu 版本)
2. ✅ 创建 conda 环境 (Python 3.11，装在数据盘)
3. ✅ 安装 Isaac Sim 5.1.0 (pip，约 10GB)
4. ✅ 安装 PyTorch 2.7.0 + CUDA 12.8
5. ✅ 克隆并安装 Isaac Lab v2.3.0 + rsl_rl
6. ✅ 克隆并安装 TienKung-Lab
7. ✅ 部署 OMNI 模型到 TienKung-Lab
8. ✅ 运行 OMNI 集成脚本

---

## 第四步：验证环境

```bash
bash verify_env.sh
```

全部 ✓ 就可以训练了。

---

## 第五步：开始训练

```bash
# 1. 快速验证 (64 环境 × 100 步，几分钟)
bash run_training.sh 验证

# 2. 正式训练走路 (4096 环境，几小时到一天)
bash run_training.sh 走路

# 3. 看训练曲线
bash run_training.sh tensorboard
# 本地 SSH 隧道: ssh -L 6006:localhost:6006 -p <端口> root@<地址>
# 浏览器打开 http://localhost:6006
```

---

## 第六步：查看/使用训练结果

```bash
# 回放训练好的策略
bash run_training.sh 回放 <run文件夹> <checkpoint文件>

# 导出到 MuJoCo 交叉验证
bash run_training.sh sim2sim <policy.pt路径>
```

训练输出在: `/root/autodl-tmp/TienKung-Lab/logs/`

---

## 训练任务说明

| 任务名 | 说明 | 命令 |
|--------|------|------|
| `omni_walk` | OMNI 走路 (RL + AMP) | `bash run_training.sh 走路` |
| `omni_run` | OMNI 跑步 | `bash run_training.sh 跑步` |
| `walk` | TienKung 原版走路 (验证框架) | `bash run_training.sh 原版走路` |

---

## 关于动作数据 (AMP)

TienKung-Lab 用 **AMP (Adversarial Motion Priors)** 让机器人模仿人类动作。
需要动作数据放在 `legged_lab/envs/omni/datasets/motion_amp_expert/`。

### 获取动作数据的方式:

**方式 A: 用 GMR 重定向 (推荐)**
```bash
# 1. 准备 SMPLX 格式的人体动捕数据 (AMASS 数据集)
# 2. 重定向到 OMNI 骨架
python scripts/smplx_to_robot.py --smplx_file <数据路径> --robot omni --save_path output.pkl

# 3. 格式转换
python legged_lab/scripts/gmr_data_conversion.py --input_pkl output.pkl \
    --output_txt legged_lab/envs/omni/datasets/motion_visualization/motion.txt

# 4. 生成 AMP 专家数据
python legged_lab/scripts/play_amp_animation.py --task=omni_walk --num_envs=1 \
    --save_path legged_lab/envs/omni/datasets/motion_amp_expert/motion.txt --fps 30.0
```

**方式 B: 暂时不用 AMP (先跑通基础 RL)**
在 `walk_cfg.py` 中把 AMP 相关权重设为 0，只用基础奖励 (速度跟踪、平衡等)。

---

## 常见问题排查

| 报错 | 原因 | 解决 |
|------|------|------|
| `pip install isaacsim` 找不到 | Python 不是 3.11 | 确认 conda 环境 |
| `GLIBC_2.35 not found` | 不是 Ubuntu 22.04 | 换镜像 |
| `task omni_walk not found` | 集成脚本没跑/注册失败 | `python integrate_omni.py` |
| 关节数量不匹配 | TienKung 和 OMNI DOF 不同 | 修改 cfg 中 num_actions=29 |
| link 名不存在 | OMNI 的 link 名和 TienKung 不同 | 修改奖励中的 body_names |
| API 参数报错 | Isaac Lab 2.1→2.3 接口变化 | 按报错信息调整参数 |
| 显存不足 OOM | num_envs 太大 | 降到 2048 或 1024 |
| `rsl_rl` 冲突 | TienKung-Lab 自带版本 vs Isaac Lab 版本 | 以 TienKung-Lab 的为准 |

---

## 文件清单

```
omni_29dof_v260705/
├── setup_tienkung.sh      ← 一键安装 (先跑这个)
├── verify_env.sh          ← 环境验证
├── run_training.sh        ← 训练/回放/TensorBoard
├── integrate_omni.py      ← OMNI 集成到 TienKung-Lab
├── AutoDL_TienKung_OMNI_搭建指南.md  ← 本文档
├── assets/                ← OMNI 模型 (URDF + 32个STL)
├── robots/                ← 机器人配置 (电机参数等)
└── actuators/             ← 自定义执行器 (DelayedDCMotor)
```

---

## 时间预估

| 步骤 | 耗时 |
|------|------|
| AutoDL 创建实例 | 5 分钟 |
| 上传文件 | 2-5 分钟 |
| setup_tienkung.sh 安装 | 40-60 分钟 |
| verify_env.sh 验证 | 1 分钟 |
| 验证训练 (64 env × 100 iter) | 5-10 分钟 |
| 正式训练 (4096 env × 数万 iter) | 数小时~1天 |
