# OMNI 29-DOF 跳高策略 MuJoCo 仿真验证指南

## 概述

本文档指导项目组成员在 MuJoCo 中加载跳高 ONNX 策略进行 sim2sim 验证。

**重要：** 跳高策略使用 `high_dynamic` 框架，与走路策略（`loco_mode`）完全不同，不能混用。

---

## 1. 环境准备

### 1.1 Python 环境

```bash
# 创建独立环境（避免与 Isaac Lab 训练环境冲突）
conda create -n omni_sim python=3.10 -y
conda activate omni_sim
```

### 1.2 安装依赖

```bash
pip install mujoco onnxruntime numpy pyyaml
```

**注意：** CPU 版 onnxruntime 即可，不需要 GPU 版本。

---

## 2. 文件准备

### 2.1 必需文件清单

从 AutoDL 下载以下文件到本地工作目录：

```
omni_29dof_v260705/
├── omni_29dof_mjc/mjcf/omni_29dof.xml          # MuJoCo 模型
├── omni_rl_sdk/policy/high_dynamic/            # 策略框架
│   ├── config/high_dynamic.yaml                # 配置文件（需修改）
│   ├── high_dynamic_policy.py                  # 策略逻辑
│   ├── HighDynamicContext.py                   # 上下文管理
│   ── high_dynamic.py                         # FSM 状态
├── omni_rl_sdk/policy/base_policy/             # 基础策略类
── omni_rl_sdk/common/                         # 通用工具
├── jump06_onnx_14998.tar.gz                    # 跳高 ONNX 模型
└── jump06_npz.tar.gz                           # 跳高参考动作 NPZ
```

### 2.2 解压文件

```bash
# 解压 ONNX 和 NPZ
tar xzf jump06_onnx_14998.tar.gz
tar xzf jump06_npz.tar.gz

# 文件位置：
# ONNX: 2026-08-12_16-57-55_14998.onnx
# NPZ:  跳高06_chr00_training.npz
```

---

## 3. 配置文件修改

### 3.1 修改 `high_dynamic.yaml`

编辑 `omni_rl_sdk/policy/high_dynamic/config/high_dynamic.yaml`：

```yaml
model:
  path: "../../../2026-08-12_16-57-55_14998.onnx"  # 指向你的 ONNX 文件
  device: "cpu"
  inference_backend: "onnxruntime"

observation:
  history_length: 5

decimation: 8

motion:
  auto_switch: false

networks:
  - name: "jump06"
    model_path: "../../../2026-08-12_16-57-55_14998.onnx"  # 同上
    motion_path: "../../../跳高06_chr00_training.npz"       # 指向 NPZ 文件
```

**关键修改：**
- `model.path` → ONNX 文件路径
- `networks[0].model_path` → ONNX 文件路径
- `networks[0].motion_path` → NPZ 文件路径

---

## 4. 运行仿真

### 4.1 创建仿真脚本

创建 `run_jump_sim.py`：

```python
#!/usr/bin/env python3
"""
跳高策略 MuJoCo 物理仿真

使用 high_dynamic 框架加载跳高 ONNX 策略：
  - 需要参考动作 NPZ 文件
  - 使用 reference tracking 模式
  - 验证 sim2sim 部署管线

使用方法：
    python3 run_jump_sim.py
    python3 run_jump_sim.py --steps 2000    # 跑 2000 步
"""
import argparse
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="跳高策略 MuJoCo 物理仿真")
    parser.add_argument("--model", default="omni_29dof_mjc/mjcf/omni_29dof.xml")
    parser.add_argument("--config", default="omni_rl_sdk/policy/high_dynamic/config/high_dynamic.yaml")
    parser.add_argument("--steps", type=int, default=2000, help="仿真步数 (默认 2000 = 5s@400Hz)")
    args = parser.parse_args()

    try:
        import mujoco
    except ImportError:
        print("✗ MuJoCo 未安装，请 pip install mujoco")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    # ─ 1. 加载 MuJoCo 模型 ──
    model_path = os.path.join(script_dir, args.model)
    if not os.path.isfile(model_path):
        print(f"✗ 模型文件不存在：{model_path}")
        sys.exit(1)

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    print(f"✓ MuJoCo 模型加载：{mj_model.nq} qpos, {mj_model.nu} actuators")
    print(f"  重力：{mj_model.opt.gravity}")

    # ── 2. 加载 high_dynamic 策略 ──
    from omni_rl_sdk.policy.high_dynamic.high_dynamic_policy import HighDynamicPolicy

    config_path = os.path.join(script_dir, args.config)
    if not os.path.isfile(config_path):
        print(f"✗ 配置文件不存在：{config_path}")
        sys.exit(1)

    # 从配置读取 motion 信息
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    motion_cfg = cfg.get('networks', [{}])[0]
    motions = [{
        'name': motion_cfg.get('name', 'jump06'),
        'motion_path': motion_cfg.get('motion_path'),
    }]

    policy = HighDynamicPolicy(
        config_path=config_path,
        device='cpu',
        model_path=None,  # 从 config 读取
        motions=motions,
    )

    print(f"✓ 策略加载完成")
    print(f"  num_obs={policy.num_obs}, num_actions={policy.num_actions}")
    print(f"  history_length={policy.history_length}")

    # ── 3. 初始化状态 ──
    # 设置初始姿态（站立姿势）
    mj_data.qpos[0:3] = [0, 0, 0.82]  # base position
    mj_data.qpos[3:7] = [1, 0, 0, 0]  # base quaternion (w,x,y,z)
    mj_data.qpos[7:7+29] = policy.default_dof_pos  # joint angles

    mujoco.mj_forward(mj_model, mj_data)

    # ── 4. 仿真循环 ──
    dt = mj_model.opt.timestep
    control_dt = dt * policy.decimation
    print(f"\n═══ 开始仿真 ═══")
    print(f"  dt={dt:.4f}s, control_dt={control_dt:.4f}s")
    print(f"  步数：{args.steps} ({args.steps * dt:.1f}s)\n")

    heights = []
    base_positions = []
    step = 0

    # 创建 state_cmd 对象（模拟 StateAndCmd）
    class StateCmd:
        def __init__(self):
            self.q = np.zeros(mj_model.nq, dtype=np.float32)
            self.dq = np.zeros(mj_model.nv, dtype=np.float32)
            self.gravity_ori = np.array([0, 0, -1], dtype=np.float32)
            self.ang_vel = np.zeros(3, dtype=np.float32)
            self.base_quat = np.array([1, 0, 0, 0], dtype=np.float32)

    state_cmd = StateCmd()

    # PD gains
    kp = np.array(cfg['dof']['kp'], dtype=np.float32)
    kd = np.array(cfg['dof']['kd'], dtype=np.float32)

    policy.reset()

    for sim_step in range(args.steps):
        # 每 decimation 步推理一次
        if sim_step % policy.decimation == 0:
            # 更新 state_cmd
            state_cmd.q[:] = mj_data.qpos.astype(np.float32)
            state_cmd.dq[:] = mj_data.qvel.astype(np.float32)

            quat = mj_data.qpos[3:7].astype(np.float32)
            qw, qx, qy, qz = quat
            state_cmd.gravity_ori = np.array([
                2 * (-qz * qx + qw * qy),
                -2 * (qz * qy + qw * qx),
                1 - 2 * (qw * qw + qz * qz),
            ], dtype=np.float32)
            state_cmd.ang_vel = mj_data.qvel[3:6].astype(np.float32)
            state_cmd.base_quat = quat.copy()

            # 获取观察和动作
            obs = policy.get_observation(state_cmd)
            action_motor = policy.get_action(obs)

            # 目标位置 = default + action
            target_pos = policy.default_dof_pos + action_motor

        # PD 控制
        q_current = mj_data.qpos[7:7+29].astype(np.float64)
        dq_current = mj_data.qvel[6:6+29].astype(np.float64)
        q_error = target_pos - q_current

        tau = kp * q_error - kd * dq_current
        mj_data.ctrl[:] = tau

        # 步进仿真
        mujoco.mj_step(mj_model, mj_data)

        # 记录
        heights.append(mj_data.qpos[2])
        base_positions.append(mj_data.qpos[0:3].copy())
        step = sim_step

        # 每 200 步打印一次状态
        if (sim_step + 1) % 200 == 0:
            t_now = (sim_step + 1) * dt
            h = mj_data.qpos[2]
            pos = mj_data.qpos[0:3]
            print(f"  t={t_now:.1f}s  h={h:.3f}m  pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    # ── 5. 结果统计 ─
    print(f"\n═══ 仿真结果 ═══")
    print(f"  总步数：{step + 1}")
    print(f"  仿真时间：{(step + 1) * dt:.1f}s")
    print(f"  初始高度：{heights[0]:.3f}m")
    print(f"  最终高度：{heights[-1]:.3f}m")
    print(f"  最低高度：{min(heights):.3f}m")
    print(f"  最高高度：{max(heights):.3f}m")
    print(f"  X 位移：{base_positions[-1][0] - base_positions[0][0]:.3f}m")
    print(f"  Y 位移：{base_positions[-1][1] - base_positions[0][1]:.3f}m")

    # 检查是否摔倒
    if min(heights) < 0.3:
        print(f"\n  ⚠ 机器人摔倒了！最低高度 {min(heights):.3f}m < 0.3m")
    elif max(heights) > 2.0:
        print(f"\n   机器人飞起来了！最高高度 {max(heights):.3f}m > 2.0m")
    else:
        print(f"\n  ✓ 机器人保持直立，跳高仿真正常")

    # NaN 检查
    if np.any(np.isnan(mj_data.qpos)):
        print(f"  ✗ 检测到 NaN！")
    else:
        print(f"  ✓ 无 NaN")


if __name__ == "__main__":
    main()
```

### 4.2 运行

```bash
cd omni_29dof_v260705
python3 run_jump_sim.py
```

**预期输出：**
```
✓ MuJoCo 模型加载：36 qpos, 29 actuators
  重力：[0. 0. -9.81]
[HighDynamic] Loaded motion 'jump06': ... frames=446, bodies=1, max_play=446
[HighDynamic] initialized (history=5, num_obs=544, motion_frames=446, max_play=446)
✓ 策略加载完成
  num_obs=544, num_actions=29
  history_length=5

═══ 开始仿真 ═══
  dt=0.0025s, control_dt=0.0200s
  步数：2000 (5.0s)

  t=0.5s  h=0.820m  pos=[0.00, 0.00, 0.82]
  t=1.0s  h=0.850m  pos=[0.00, 0.00, 0.85]
  ...
```

---

## 5. 常见问题

### Q1: `ModuleNotFoundError: No module named 'omni_rl_sdk'`

**解决：** 确保在项目根目录运行，或设置 PYTHONPATH：
```bash
export PYTHONPATH=/path/to/omni_29dof_v260705:$PYTHONPATH
```

### Q2: `ONNX obs size XXX does not match computed num_obs 544`

**原因：** ONNX 模型与配置不匹配。

**解决：** 确认 `high_dynamic.yaml` 中的 `history_length=5` 和 `num_actions=29`。

### Q3: `Motion file not found`

**原因：** NPZ 文件路径错误。

**解决：** 检查 `high_dynamic.yaml` 中 `motion_path` 是否指向正确的 NPZ 文件。

### Q4: 机器人立即摔倒

**可能原因：**
1. 初始姿态不对（应使用 `default_pos`）
2. PD gains 不合适
3. ONNX 模型质量差

**调试：** 降低 `--steps` 到 500，观察前 1 秒的行为。

---

## 6. 与 Isaac Sim 可视化的区别

| | MuJoCo 仿真 | Isaac Sim 可视化 |
|---|---|---|
| 目的 | sim2sim 验证，轻量级 | 完整物理仿真，视觉验证 |
| 依赖 | mujoco + onnxruntime | Isaac Sim 5.1.0 + Isaac Lab |
| 速度 | 快（CPU 即可） | 慢（需要 GPU） |
| 渲染 | 无（或简单可视化） | 完整 3D 渲染 |
| 适用场景 | 快速验证策略有效性 | 最终效果展示 |

---

## 7. 下一步

验证通过后：
1. 将 ONNX 部署到实机
2. 或继续训练其他动作（跨栏、楼梯等）
