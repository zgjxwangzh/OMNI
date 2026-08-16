# Omni 29-DOF 跑步训练：下一阶段解决方案

你现在不要继续盲目尝试 `joint_pos_l2_penalty` 的不同权重，也不要继续做类似 w0.5 / w1 / w2 / w5 的遍历。

根据当前训练结果，我们已经基本确认：

1. 原始模型 `model_8000.pt` 可以稳定跑步，但手臂伸展。
2. 增加手臂 position L2 penalty 后，策略倾向于让手臂固定/下垂，而不是产生周期性摆臂。
3. 两阶段 resume + penalty 也导致 locomotion 退化。
4. 因此当前问题不是简单的“继续调 penalty weight”，而更可能是 reward formulation 和训练方式的问题。
5. 同时，当前跑步 ONNX 在 MuJoCo 中无法正常运行，这是一个独立的 P0 问题，必须先定位。

## 第一优先级：先解决 ONNX 部署

不要训练新模型，先拿目前唯一正常跑步的：

`logs/rsl_rl/omni_flat/2026-08-13_09-15-45/model_8000.pt`

以及对应 ONNX，建立完整的验证链：

PyTorch → ONNX Runtime → omni_rl_sdk → MuJoCo

重点排查：

* ONNX `action_scale=0.5` 与 SDK action scale 是否匹配
* ONNX exporter 是否已经完成 action scaling，SDK 是否又重复 scaling
* PyTorch 和 ONNX Runtime 对完全相同 observation 的 action 输出是否一致
* SDK 的 observation 构建是否与 omni_mimic 训练时完全一致
* `time_step` 是否正确传入
* reference motion 在 ONNX 内部和 SDK 外部的处理方式是否一致
* 29 个 joint 的 order / sign / default position 是否一致
* control frequency / action latency 是否一致
* 当前 SDK 使用的 Policy class 是否真正支持 omni_mimic ONNX

### 重要要求

不要凭猜测修改参数。

先建立一个 deterministic test：

给 PyTorch 和 ONNX Runtime 完全相同的：

* obs
* time_step

然后比较：

* action
* 每个 joint 的 action difference

如果 PyTorch ≈ ONNX，再继续查 SDK。

如果 PyTorch != ONNX，就优先查 exporter / normalization / embedded reference motion。

请把每一步验证结果记录下来。

---

# 第二优先级：不要再继续调 arm L2 penalty

当前已有实验已经说明：

`joint_pos_l2_penalty` 会导致手臂趋向于时间平均位置，而不是学习摆臂。

因此下一步不要继续搜索 penalty weight。

我们需要改变 formulation。

---

# 第三优先级：优先尝试 Residual Arm Policy

以已经稳定跑步的 `model_8000.pt` 作为 baseline / teacher。

不要重新训练整个机器人。

目标是：

> 保持原来的腿和身体跑步能力，只学习一个不会破坏 locomotion 的手臂控制 residual。

概念上：

`a_final = a_baseline + a_arm_residual`

其中 residual 第一阶段只允许修改手臂相关 DOF。

也就是说：

* lower body：尽量保持 baseline
* torso：尽量保持 baseline
* arm：允许新 policy 学习

第一阶段先冻结 lower body，只训练 arm。

如果框架不方便直接实现 residual policy，请先研究 omni_mimic 当前代码结构，找到实现这种训练最小改动的方式，不要立即大规模修改框架。

---

# 第四优先级：手臂 reward 不再只使用绝对位置 L2

优先研究以下方向：

### 1. Gait phase conditioned arm movement

跑步是周期性运动。

尝试建立 gait phase，例如：

`phase → sin(phase), cos(phase)`

让手臂动作与 gait phase 相关，而不是简单要求：

`arm_position ≈ reference_position`

### 2. Arm velocity / movement

增加对合理手臂运动的约束，避免 policy 通过“手臂不动”获得较低 reward。

但不要简单奖励“手臂动得越快越好”，避免产生疯狂甩臂。

### 3. Arm-leg coordination

研究：

* 左臂 ↔ 右腿
* 右臂 ↔ 左腿

之间的 phase relationship。

目标不是单纯让手臂移动，而是让手臂运动与 running gait 协调。

### 4. Teacher / baseline constraint

新 policy 可以改变手臂，但不要轻易破坏 baseline 的 lower-body running behavior。

可以研究对 lower-body action / state 加 teacher consistency constraint。

---

# 第五优先级：采用渐进式训练，不要一次性全身重新训练

建议研究：

Stage 0：

已有稳定 running policy。

Stage 1：

lower body frozen，只训练 arm。

Stage 2：

arm 已经产生稳定摆动后，允许少量 lower-body fine-tuning。

Stage 3：

必要时再逐渐增加全身可训练范围。

每个阶段都必须验证：

* episode length
* running stability
* body velocity
* body orientation
* arm ROM
* arm velocity
* arm-leg coordination
* 视频中的实际动作

---

# 第六优先级：如果 RL residual 方案复杂，先做一个最小可行实验

不要一开始就构建复杂算法。

可以先研究一个非常简单的 baseline：

现有稳定 running policy

*

一个简单的周期性 arm trajectory

例如基于 gait phase 的 sinusoidal arm movement。

目的不是直接作为最终模型，而是回答一个关键问题：

> “在当前已经稳定的 running gait 上，增加合理幅度的周期性摆臂，是否会破坏 locomotion？”

如果这个实验都无法稳定，那么应该先研究动作幅度、phase、joint selection 和控制方式。

如果这个实验可以稳定，再进一步考虑 imitation / residual RL。

---

# 实验原则

从现在开始，每一个实验必须有明确 hypothesis。

不要再：

“感觉这个 weight 可能有用，所以训练几个小时看看。”

而应该：

实验 A：

“冻结 lower body，只训练 arm，验证 arm 是否可以在不破坏 running 的情况下学习周期运动。”

实验 B：

“加入 gait phase，验证 arm 是否能够形成与 gait frequency 一致的周期运动。”

实验 C：

“加入 teacher constraint，验证 lower-body stability 是否得到保护。”

每个实验完成后必须回答：

1. 做了什么？
2. 为什么这么做？
3. 修改了哪些代码 / 配置？
4. 训练了多久？
5. 结果是什么？
6. 哪些指标改善？
7. 视频行为是否真的改善？
8. 是否值得进入下一轮？

---

# 非常重要

你现在是这个项目的技术负责人。

不要为了“继续训练”而训练。

如果一个实验失败，要分析 failure mode，并据此改变下一次实验，而不是简单调整一个数字继续跑。

如果发现当前框架不适合实现上述方案，先告诉我：

* 当前框架限制是什么
* 哪个文件/模块限制了实现
* 最小修改方案是什么
* 修改风险是什么

再开始改代码。

最终目标不是得到一个 TensorBoard 上 reward 更高的模型，而是：

**稳定跑步 + 可见自然摆臂 + 能够通过 ONNX → MuJoCo → 真机部署。**
