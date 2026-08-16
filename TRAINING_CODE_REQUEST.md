# 训练代码需求清单

> 致：项目组长  
> 自：[你的名字]  
> 日期：2026-08-11  
> 主题：请求 high_dynamic 训练代码

---

## 背景

我们已完成以下工作，准备进入 RL 训练阶段：

1. ✅ **动捕数据重定向**：32 个 BVH → 29 个 high_dynamic NPZ（全部通过 MuJoCo 验证）
2. ✅ **SDK 部署验证**：走路 ONNX 在 MuJoCo 中正常运行，机器人保持直立
3. ✅ **物理跟踪 baseline**：29 个动作的 PD 控制验证完成
4. ✅ **训练环境准备**：AutoDL GPU 服务器环境已搭建（conda omni_gpu）

---

## 需要的训练代码

### 1. 核心训练框架

| 需求 | 说明 |
|------|------|
| **训练入口脚本** | 类似 `train.py`，启动 high_dynamic 策略训练 |
| **环境实现** | high_dynamic 的 Isaac Lab 环境类（参考 `high_dynamic.py`） |
| **奖励函数** | reference tracking 的奖励定义（joint pos/vel tracking, etc.） |
| **课程学习** | 如果有 curriculum 或 automatic sim-to-real 策略 |

### 2. 配置文件

| 需求 | 说明 |
|------|------|
| **训练超参** | learning rate, batch size, discount factor, etc. |
| **网络结构** | actor/critic 的 MLP 层数、隐藏单元数 |
| **环境配置** | sim params, domain randomization, noise |
| **动作配置** | 每个动作的 NPZ 路径、帧率、循环方式 |

### 3. 模型导出

| 需求 | 说明 |
|------|------|
| **ONNX 导出脚本** | 将训练好的 policy 导出为 ONNX（部署用） |
| **checkpoint 格式** | PyTorch .pt 还是其他格式 |
| **推理代码示例** | 如何用导出的 ONNX 做推理（验证导出正确性） |

### 4. 文档/示例

| 需求 | 说明 |
|------|------|
| **训练命令示例** | 如何启动训练（命令行参数） |
| **预期训练曲线** | 大概多少 step 收敛，reward 范围 |
| **已知问题** | 训练中的坑、注意事项 |

---

## 当前数据准备

我们已准备好 29 个 high_dynamic NPZ 文件：

```
motion_data/
├── 跳高01-06_chr00_highdynamic.npz    # 6 个跳高
├── 翻箱子01-14_chr00_highdynamic.npz  # 14 个翻箱
├── 上/下/弯/直楼梯_chr00_highdynamic.npz  # 6 个楼梯
├── 跑步01_chr00_highdynamic.npz       # 1 个跑步
├── 跨栏02-03_chr00_highdynamic.npz    # 2 个跨栏
└── 匍匐前进1_chr00_highdynamic.npz    # 1 个匍匐
```

**格式**：每个 NPZ 包含 `joint_pos` (T,29), `joint_vel` (T,29), `body_quat_w` (T,1,4)

**建议优先训练**：跳高动作（最稳定，高度变化 0.55-0.76m）

---

## 训练环境

- **服务器**：AutoDL GPU（NVIDIA A100/RTX 系列）
- **Python**：3.10（conda 环境 `omni_gpu`）
- **已安装**：mujoco, onnxruntime-gpu, numpy, pyyaml, pillow
- **待安装**：训练框架依赖（rsl_rl / stable-baselines3 / 其他？）

---

## 问题

1. **训练框架**：用 rsl_rl、stable-baselines3 还是自研？
2. **Isaac Lab 版本**：需要哪个版本？是否和现有 SDK 兼容？
3. **训练时长**：单个动作大概训练多久（小时/step）？
4. **多任务训练**：能否同时训练多个动作，还是每个动作单独训练？
5. **sim2real**：是否有 domain randomization 或 system identification 流程？

---

## 联系方式

如有问题，随时沟通。期待训练代码到位后开始训练！

谢谢！
