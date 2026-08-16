# Omni 29-DOF 人形机器人 · 跳高（High Jump）强化学习训练工程（V11 · xMimic 路线 + V7 算法）

基于 **NVIDIA Isaac Lab / Isaac Sim** 的**跳高**运动模仿训练工程，适用于 **Omni 29-DOF 人形机器人**（`omni_29dof_v260705` 模型包）。本工程是 **jump_high 项目**的服务器迁移版：自定义环境（`jump_env/`）继承 **xMimic `whole_body_tracking`** 框架，策略 obs 用 **529 维**（`obs529.py`，逐字节复刻公司 SDK `high_dynamic` 布局，含 yaw 校准 / warmup / 5 帧历史），参考动作 `motion/jump_high_firstjump_50fps.npz`（50fps），RL 框架为 **RSL-RL (PPO)**，支持单卡 / 双卡并行训练。

> 本工程为**自包含**工程：把整个 `my_omni_jump_train_v11/` 目录拷到服务器即可训练，`whole_body_tracking` / `omni_29dof_v260705` 均已内嵌，无需改动 Isaac Lab 本体。

---

## 目录

- [1. 工程结构](#1-工程结构)
- [2. 版本要求](#2-版本要求)
- [3. 训练数据说明](#3-训练数据说明)
- [4. 本地打包与上传到服务器](#4-本地打包与上传到服务器)
- [5. 进入 Docker 容器与确认环境](#5-进入-docker-容器与确认环境)
- [6. 搭建训练环境](#6-搭建训练环境)
- [7. 开始训练](#7-开始训练)
- [8. 监控训练](#8-监控训练)
- [9. 评估跳高效果](#9-评估跳高效果)
- [10. 回放与导出策略](#10-回放与导出策略)
- [11. 续训（断点恢复）](#11-续训断点恢复)
- [12. 常见问题排查](#12-常见问题排查)
- [13. 调参入口速查](#13-调参入口速查)

---

## 1. 工程结构

```
my_omni_jump_train_v11/
├── jump_env/                        # 训练任务包（gym 环境注册 + MDP + 配置）
│   ├── __init__.py                  #   注册任务 Omni-Jump-v0（见 §13）
│   ├── omni_jump_env_cfg.py         #   环境配置（场景/动作/观测/命令/奖励/事件/终止）
│   ├── omni_jump_env.py             #   OmniJumpEnv(ManagerBasedRLEnv) + JumpScoreTracker
│   ├── score_metric.py              #   跳高成绩代理指标（纯 torch）
│   ├── jump_phase_rewards.py        #   跳高专项奖励（站立/起跳/腾空收腿，纯 torch）
│   ├── mdp/
│   │   ├── obs529.py                #   ★ 529 维 obs，逐字节复刻 SDK high_dynamic
│   │   ├── commands.py              #   JumpMotionCommand（每回合从站立帧 0 开始）
│   │   └── jump_rewards.py          #   对称/腾空高度等奖励
│   └── agents/rsl_rl_ppo_cfg.py     #   PPO 超参数（[512,256,128]，50Hz 缩放）
├── whole_body_tracking/             # xMimic 基底（已剪枝：TrackingEnvCfg + MotionCommand + mdp）
├── omni_29dof_v260705/              # 机器人模型包（URDF / STL / 电机辨识参数）
├── motion/
│   └── jump_high_firstjump_50fps.npz  # 训练参考动作（50fps / 183 帧 / xMimic 格式）
├── scripts/
│   ├── train.py                     # 训练入口（支持 --distributed 多卡）
│   ├── play.py                      # 回放 + 导出 policy.pt / policy.onnx
│   ├── smoke_obs529.py              # 32-env 冒烟：验证 529 obs 集成正确
│   ├── validate_motion_npz.py       # 换动作数据时校验 npz（keys/shapes/fps/body 序）
│   ├── rsl_rl_compat.py             # RSL-RL 双版本兼容层（policy → actor/critic）
│   └── cli_args.py                  # RSL-RL 命令行参数
├── run_train.sh                     # ★ 一键训练：自检环境 + 自动单卡/双卡并行
├── lib_env.sh                       # Python 环境检测公共函数
├── setup_environment.sh             # 服务器环境自检 / 依赖安装
├── pack_for_upload.sh               # 打包上传脚本
└── README.md
```

---

## 2. 版本要求

| 组件 | 要求 |
| --- | --- |
| Isaac Lab | **2.2.x 或 2.3.x / main**（两种都已兼容） |
| Isaac Sim | **5.0.x（配 Isaac Lab 2.2）** 或 **5.1.x（配 2.3+）** |
| rsl-rl-lib | **与 Isaac Lab 匹配**：旧版(2.2.x 接口) → **==2.3.3**；2.3+/main → **4.x/5.x**（`setup_environment.sh` 默认装 2.3.3，本服务器为旧版接口） |
| GPU | 2×4090（单卡也支持，`--num_envs` 调小即可） |
| 容器 | 以 `--gpus all` 启动（否则容器里看不到 GPU） |

> **重要（版本匹配）**：rsl-rl-lib 版本必须与 Isaac Lab 匹配。
> - **本服务器 `/opt/isaaclab` 是旧版 Isaac Lab（2.2.x 接口）**。判断依据：`RslRlPpoActorCriticCfg` 没有 `actor_obs_normalization` 字段、`isaaclab` 没有 `__version__` 属性。→ 必须用 **rsl-rl-lib ==2.3.3**（`setup_environment.sh` 默认装；2.3.3 自带多卡分布式，支持 run_train.sh 双卡）。不要升级到 4.x/5.x。
> - 若部署到 Isaac Lab 2.3+/main，才用 rsl-rl-lib 4.x/5.x；rsl-rl >= 4.0 需要 `actor`/`critic` 配置，`scripts/rsl_rl_compat.py` 会自动把 `policy` 迁过去。
> - 配置里**不要**给 `RslRlPpoActorCriticCfg` 传 `actor_obs_normalization`/`critic_obs_normalization`（旧版没有这两个字段，传了报 TypeError）；观测归一化由 `empirical_normalization=True` 控制。

---

## 3. 训练数据说明

`motion/jump_high_firstjump_50fps.npz` 是跳高参考动作的 **xMimic 格式**（50fps，隔帧抽取自 100fps 母本，与部署 SDK HighDynamic 50Hz 同频）：

| key | shape | 含义 |
| --- | --- | --- |
| `joint_pos` | (183, 29) | 29 关节角（**POLICY 序** = build_first_jump 运行时序 = obs529 期望序） |
| `joint_vel` | (183, 29) | 关节角速度 |
| `body_pos_w` | (183, 30, 3) | 30 body 世界位置（OMNI_BODY_NAMES 序，base_link 索引 0） |
| `body_quat_w` | (183, 30, 4) | 30 body 世界四元数（wxyz） |
| `body_lin_vel_w` | (183, 30, 3) | body 世界线速度 |
| `body_ang_vel_w` | (183, 30, 3) | body 世界角速度 |
| `fps` | — | 50 |

**关键设计**：
- **obs 529 维**（`obs529.py`）= 58 参考（joint_pos+joint_vel）+ 6 anchor_ori + 5 帧 × 93 历史（gravity + ang_vel + joint_pos_rel + joint_vel + action），逐字节复刻公司 SDK `high_dynamic` 布局，含 yaw 校准 / warmup。
- **critic 430 维**（特权）：基类 PrivilegedCfg（command + body_pos/ori + base_vel + joint + actions）。
- **每回合从参考站立帧 0 开始**（`JumpMotionCommand` 锁相位 0），时刻 t 逐帧对齐参考帧 t。
- **控制 50Hz**（dt 0.005 × decimation 4）+ **参考 50fps** 逐帧对齐。
- **action scale 0.5**（SDK high_dynamic.yaml），目标 = `default_pos + 0.5 × action`。

---

## 4. 本地打包与上传到服务器

### 4.1 本机打包

```bash
cd /home/liuziqi/my_omni_jump_train_v11
bash pack_for_upload.sh
```

产物：`/home/liuziqi/omni_jump_<时间戳>.tar.gz`（约 18MB，不含日志/缓存）。

### 4.2 上传到服务器

```bash
# 方式一：scp（普通端口 22）
scp /home/liuziqi/omni_jump_<时间戳>.tar.gz 用户名@服务器IP:/home/用户名/

# 方式二：scp 非标准端口
scp -P 2222 /home/liuziqi/omni_jump_<时间戳>.tar.gz 用户名@服务器IP:/home/用户名/

# 方式三：rsync（传大文件推荐）
rsync -avzP /home/liuziqi/omni_jump_<时间戳>.tar.gz 用户名@服务器IP:/home/用户名/
```

### 4.3 服务器上解压

```bash
ssh 用户名@服务器IP
cd /home/用户名
tar -xzf omni_jump_<时间戳>.tar.gz
cd my_omni_jump_train_v11
ls     # 应看到 README.md jump_env/ whole_body_tracking/ omni_29dof_v260705/ motion/ scripts/ run_train.sh ...
```

---

## 5. 进入 Docker 容器与确认环境

### 5.1 进入容器

```bash
docker ps          # 找到 isaacsim/isaaclab 容器
docker exec -it <容器名或ID> bash
```

### 5.2 确认 GPU（关键）

```bash
nvidia-smi         # 必须能看到 2 张 4090
```

看不到 GPU → 容器没带 GPU 启动，需要管理员用 `docker run ... --gpus all ...` 重启容器。

### 5.3 确认 Python 与 Isaac Lab

```bash
conda env list
~/miniconda3/envs/isaaclab/bin/python -c "import isaaclab; print(isaaclab.__version__)" 2>&1 | head -3
```

记录能 import isaaclab 的 python 绝对路径（脚本也能自动找）。

---

## 6. 搭建训练环境

### 6.1 一键自检（最常见情况）

```bash
cd /path/to/my_omni_jump_train_v11
bash setup_environment.sh
```

预期输出：

```
[1/4] 使用 Python: /path/to/.../python
[2/4] 校验 Isaac Lab ...  Isaac Lab 版本: 2.3.x
[3/4] 安装/升级 RSL-RL 与运行依赖 ...      ← 首次会联网装 rsl-rl-lib
[4/4] 校验 RSL-RL / isaaclab_rl / isaaclab_tasks ...  依赖检查通过 OK
```

看到 `依赖检查通过 OK` 即环境就绪。

> 若 `[2/4]` 报"没有 Isaac Lab"，手动指定 python 后重跑：`export PYTHON=/path/to/isaaclab/python && bash setup_environment.sh`。

---

## 7. 开始训练

### 7.1 一键训练（2×4090 双卡，总 4096 环境）

```bash
cd /path/to/my_omni_jump_train_v11

# ★ 最简用法
bash run_train.sh --headless --num_envs 4096
```

脚本会自动：自检环境 → 检测 2 卡 → 双卡并行（每卡 2048 env）→ `torchrun` 启动。

### 7.2 首次运行会发生什么

- 首次做 **URDF→USD 转换**，会停顿几分钟（属正常，转换结果已缓存，之后秒启）。
- 每轮打印一行：`iter / Mean reward / Mean episode length / entropy` 等。
- 双卡时 `nvidia-smi` 应看到 2 个 python 进程，各占 ~20+GB 显存。

### 7.3 训练参数

| 参数 | 说明 | 建议值 |
| --- | --- | --- |
| `--headless` | 无渲染窗口，服务器训练必需 | 必加 |
| `--num_envs` | **总**环境数，双卡自动 ÷2 | 4096（可 8192） |
| `--task` | 任务 ID | 默认 `Omni-Jump-v0` |
| `--max_iterations` | 训练迭代上限 | 默认 20000 |
| `--motion_file` | 换动作数据 | 可选 |
| `--run_name xxx` | 本次 run 后缀 | 可选 |
| `--video --video_interval 1000` | 训练中录 mp4 | 排查动作时用 |

### 7.4 长时间训练的保活（重要）

SSH 断开会终止前台进程。用 **tmux**：

```bash
# 容器内
tmux new -s train
bash run_train.sh --headless --num_envs 4096
# Ctrl+B 然后 D 脱离（训练继续）；重新挂回：tmux attach -t train
```

没有 tmux：`apt-get install -y tmux`。或后台方式：

```bash
nohup bash run_train.sh --headless --num_envs 4096 > train.log 2>&1 &
tail -f train.log
```

---

## 8. 监控训练

### 8.1 TensorBoard

```bash
# 服务器上 `python` 不存在，用 isaac-sim 的 python；tensorboard 需先装一次：
#   /isaac-sim/python.sh -m pip install tensorboard
/isaac-sim/python.sh -m tensorboard.main --logdir logs/rsl_rl/omni_jump --port 6006
# 本机浏览器访问（SSH 转发）：ssh -L 6006:localhost:6006 用户名@服务器IP
```

### 8.2 如何判断训练正常

| 指标 | 正常信号 |
| --- | --- |
| `Mean reward` | 持续上升并稳定 |
| `Mean episode length` | 从低升到 ~200（一集 = 完整 4.0s 序列）；**必须 >1** |
| `entropy` | 为正，缓慢下降 |
| `Jump/*` | `score_apex_lowest_link` 上升（成绩代理，参考满分 ~0.45）、`airborne_leg_tuck` >0（真腾空收腿） |
| `time_out` | 占比高（回合活到超时 = 没被终止） |

**大概节奏**：数百迭代学会站立/下蹲，1000+ 迭代出现起跳，2000+ 迭代起跳逐渐稳定，腾空收腿是后期难点。

---

## 9. 评估跳高效果

> 评估脚本（`eval_takeoff.py` / `eval_policy.py`）源自 jump_high，需按自包含工程调整 sys.path 后使用，或直接看训练 TensorBoard 的 `Jump/*` 指标。

训练到有 checkpoint 后，最直接的方式是回放肉眼确认（见 §10）。也可用 TensorBoard 的 `Jump/score_apex_lowest_link`（成绩代理，参考满分 ~0.452）判断收敛。

---

## 10. 回放与导出策略

### 10.1 回放（服务器无显示器，用 --headless 录视频）

```bash
# 完整序列回放（从相位 0 播一整遍）并录 mp4
python scripts/play.py --task Omni-Jump-v0 --headless --video \
    --play_full_motion --load_run <run目录名>

# 默认回放（不指定 load_run 时自动取最新 checkpoint）
python scripts/play.py --headless --video --load_run <run目录名>

# 视频位置
ls logs/rsl_rl/omni_jump/<run目录名>/videos/play/
# 拉回本机：scp 用户名@服务器IP:/path/to/.../play/*.mp4 ./
```

### 10.2 导出部署用策略

`play.py` 运行时会自动导出：

```
logs/rsl_rl/omni_jump/<run目录名>/exported/policy.pt   (TorchScript/JIT)
logs/rsl_rl/omni_jump/<run目录名>/exported/policy.onnx (ONNX)
```

`policy.onnx` 可用于真机/嵌入式部署（配合动作缩放 0.5 与 obs529 字节对齐）。

---

## 11. 续训（断点恢复）

```bash
# 查看已有 run 和 checkpoint
ls logs/rsl_rl/omni_jump/

# 续训（--load_run 可省略，自动取最新）
bash run_train.sh --headless --num_envs 4096 --resume --load_run <run目录名>
```

注意：续训会新建一个时间戳目录，但会从 checkpoint 加载模型权重和训练轮次继续。

---

## 12. 常见问题排查

| # | 现象 | 排查与解决 |
| --- | --- | --- |
| 1 | `rsl_rl` 找不到 / 版本过低 | `python -m pip install "rsl-rl-lib==2.3.3"` 后重跑 `bash setup_environment.sh` |
| 1b | 报 `cannot import name 'handle_deprecated_rsl_rl_cfg'` | 正常现象（旧版 Isaac Lab 无此内置迁移函数），`scripts/rsl_rl_compat.py` 已兜底，无需处理 |
| 1c | 报 `TypeError: RslRlPpoActorCriticCfg.__init__() got an unexpected keyword argument 'actor_obs_normalization'` | cfg 里给旧版类传了 2.3+ 才有的字段。去掉 `actor_obs_normalization`/`critic_obs_normalization`（见 §2，归一化用 `empirical_normalization`） |
| 1d | `import whole_body_tracking` 报 `RslRlMLPModelCfg` 不存在 | 本工程已剪枝（删 config/dex_evt），若仍报说明拷入了未剪枝版本；确保 `whole_body_tracking/tasks/tracking/` 下无 `config/` |
| 2 | 报错找不到 `robots`/`assets`/`actuators` | `scripts/train.py` 已自动加路径；自定义脚本需 `export PYTHONPATH=/path/to/my_omni_jump_train_v11/omni_29dof_v260705:$PYTHONPATH` |
| 3 | 报 `UrdfConverterCfg`/`PDGainsCfg` 等 API 不存在 | Isaac Lab/Sim 版本太低，需 >= 2.2.0 / >= 5.0.0 |
| 4 | 启动后卡住几分钟 | 首次 URDF→USD 转换，属正常；改模型后删 `omni_29dof_v260705/assets/omni_29dof_nohead_noshoe/.asset_hash` 可强制重转 |
| 5 | 显存不足崩溃 | 减小 `--num_envs`（单卡 1024/512）；或改 `jump_env/omni_jump_env_cfg.py` 里 `scene.num_envs` 默认值 |
| 6 | 机器人一直不跳 | 正常，前几百迭代在探索。看 `Mean episode length` 是否上升、`Jump/*` 指标是否 >0 |
| 7 | 报显示器/EGL 错误 | 确认容器 `--gpus all`；必要时 `--width 640 --height 480` |
| 8 | 双卡起不来 / 只用了 1 卡 | 确认 rsl-rl-lib 为 2.3.3（自带分布式）；`nvidia-smi` 见 2 卡；容器 `--gpus all`；先单卡验证：`TRAIN_MODE=single bash run_train.sh --headless --num_envs 2048` |
| 9 | SSH 断开训练就没了 | 用 tmux / nohup（见 §7.4） |
| 10 | 想清空重训 | `rm -rf logs/rsl_rl/omni_jump/`（只删日志，不删代码） |
| 11 | 服务器没网装不了 rsl-rl | 需先在内网/离线源备好 rsl-rl-lib 的 wheel，或让管理员放行 pip 源 |
| 12 | 换了一组动作 npz | 用 `python scripts/validate_motion_npz.py <新npz>` 校验（keys/shapes/fps=50/body 序/帧0站立），再改 `jump_env/omni_jump_env_cfg.py` 的 `MOTION_FILE`（或 `JUMP_HIGH_MOTION_FILE` 环境变量）；**注意**：`jump_phase_rewards.py` 里按参考帧标定的相位窗口（站立/腾空/推蹬帧号、ref_len、max_tuck）需按新 npz 重算 |

---

## 13. 调参入口速查

| 想改什么 | 改哪里 |
| --- | --- |
| 起跳完成奖励 | `jump_env/omni_jump_env_cfg.py` → `takeoff_completion_bonus.weight` |
| 腾空收腿奖励 | `airborne_leg_tuck.weight` + `jump_phase_rewards.py` 的 `max_tuck` |
| 站立脚不动惩罚 | `standing_fidget_penalty.weight` / `standing_symmetry_penalty.weight` |
| 站立判无效硬终止 | `standing_invalid`（slip_thresh/lift_thresh） |
| 腾空收腿轨迹 | `airborne_tuck_tracking_penalty.weight` |
| 推蹬段腿部跟踪 | `leg_tracking_penalty.weight` |
| 身体前倾惩罚 | `body_lean_penalty.weight` |
| 动作对称惩罚 | `action_symmetry_penalty.weight` |
| 参考动作 / 换 npz | `MOTION_FILE`（或 `JUMP_HIGH_MOTION_FILE` 环境变量） |
| 环境并行数 / 物理步长 | `jump_env/omni_jump_env_cfg.py` `scene.num_envs` / `self.decimation` |
| 网络结构 / 学习率 / 迭代数 | `jump_env/agents/rsl_rl_ppo_cfg.py` |
| 单卡/双卡 | 不改代码，用 `run_train.sh` 的 `TRAIN_MODE` / `TRAIN_GPU_COUNT` |

---

## 14. V11 算法：V7 奖励栈（2026-08-13）

V11 把 **V7 的 26 项奖励**移植到 jump_high（xMimic）框架，替换 jump_high 原自定义奖励（motion_*/takeoff_completion/standing_*/airborne_* 全部移除）。

### 移植的 V7 奖励（`jump_env/mdp/v7_rewards.py`）

| 类别 | 奖励项 |
| --- | --- |
| 模仿跟踪 | `track_dof_pos`(+3.0)、`track_root_ori`(+2.0)、`track_yaw`(+2.5)、`arm_tracking`(+2.0) |
| 腰约束 | `waist_yaw_penalty`(-0.5)、`waist_roll_penalty`(-1.5) |
| 跳高塑造 | `jump_height_bonus`(+5.0, thr 0.85)、`takeoff_vertical_vel`(+3.0)、`premature_jump_penalty`(-1.0)、`tuck_bonus`(+5.0)、`flight_yaw_penalty`(-0.15) |
| 姿态约束 | `elbow_bend`(-1.5)、`hip_spread`(-8.0)、`torso_backward_lean`(-6.0)、`torso_roll`(-2.0)、`arm_back`(-3.0) |
| 左右对称 | `feet_force_symmetry`(-0.1)、`feet_contact_symmetry`(-0.1)、`leg_symmetry`(-1.0) |
| 常规 | `termination_penalty`(-200)、`boundary_penalty`(-1.0) |
| 保留通用项 | `action_rate_l2`(×0.5)、`joint_torque_l2`、`joint_pos_limits`、`joint_vel_limits`、`undesired_contacts` |

### 关键适配
- **jump_mask / first_jump_frame**：V7 的 `takeoff_vertical_vel`/`premature_jump_penalty`/`torso_*`/`flight_yaw` 依赖它们。`JumpMotionCommand.__init__` 从参考膝角推断（knee>1.2），落地 133 帧截断，`first_jump_frame=57`。
- **帧号重标**：V7 335帧 → jump_high 183帧，按相位实测重导（tuck start 115、flight cutoff 132），不按比例缩放。
- **高度阈值重标**：jump_high 参考 base 峰值 0.961m，arm_back airborne 1.0→0.90、jump_height thr 0.9→0.85。
- **终止换 V7**：bad_anchor_ori(1.2)/fell(0.25)/out_of_bounds(1.5)，移除依赖 body 跟踪的 ee_body_pos/anchor_pos（防回合被误杀）。
- **修正 `_quat_yaw`**：V7 版 wxyz 约定错误（纯 yaw 恒返 0），V11 用正确 wxyz 公式。

### 从 model_49999 续训（跨版本）

model_49999 是 rsl-rl 5.x 格式（jump_high 本机训练），服务器是 2.3.3，需先转换：

```bash
# 本机: 转换 5.x → 2.3.3
python scripts/convert_rsl5_to_233.py /home/liuziqi/model_49999.pt
# 产物: /home/liuziqi/model_49999_233.pt

# 上传到服务器工程 logs/ 目录, 然后续训:
bash run_train.sh --headless --num_envs 4096 --resume \
    --load_run <run目录> --checkpoint model_49999_233.pt
```

转换脚本已验证：actor/critic 权重、std、obs_norm 统计与原始 5.x checkpoint 数值一致。

---

## 任务 ID 速查

| 任务 ID | 用途 |
| --- | --- |
| `Omni-Jump-v0` | 跳高训练 / 回放 / 导出 |

---

## 一句话开始

```bash
# 服务器容器内
cd /path/to/my_omni_jump_train_v11
bash setup_environment.sh                     # 一次性
tmux new -s train
bash run_train.sh --headless --num_envs 4096  # 双卡训练
```
