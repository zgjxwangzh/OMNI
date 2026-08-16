"""跳高专项奖赏(纯 torch, 无 isaaclab 依赖, 可独立单测)。

背景: 阶段一纯模仿的模仿项都是 `exp(-error/σ²)`, 误差大时饱和到 0 且梯度≈0;
叠加终止阈值放宽(1.0m), 一个"站着不蹲"的策略不会被判死、能刷低分 → 存在
卡在"只站稳、不学蹲跳"局部盆地的风险。以下是直接补丁:

1) `takeoff_completion_bonus`(起跳完成奖): 参考腾空窗口(实测双脚离地 115~132
   帧)内, 机器人双脚都离地 -> 逐帧 +1。二元硬信号, 站着绝对拿不到; 窗口锁在
   参考腾空段, 防止"原地蹦"刷分。

2) `standing_too_long_penalty`(站着过久惩罚): 参考已进入下蹲/推蹬(15~132 帧),
   机器人 root 还高出参考 root 超过 margin 的部分 -> 线性惩罚。线性(非 exp)
   所以不饱和, 在"站着没蹲"的远距区也保留梯度——正好补上模仿项饱和的盲区。
   完美跟踪时 lag<=0 -> 0, 不惩罚; 参考腾空段参考 root 高于机器人, 自然为 0。

3) `airborne_leg_tuck`(腾空收腿奖): 参考腾空窗口(115~132)+ 双脚离地时, 奖励
   机器人当前**最低 link 的世界高度**(对应比赛成绩定义: 最高点时身体最低部位
   到地面距离)。参考最高点收腿到踝 0.462m(成绩 0.464m), 所以完美收腿 → 奖励
   趋近 ~0.46; 跳了不收腿(踝还贴低位) → 奖励小; 站着 → 窗口外/没离地 = 0。
   与之前删掉的 `jump_height` 关键区别: 窗口锁死腾空段且要求双脚离地, 站着
   永远拿不到, 不存在"深蹲段奖励站着不动"的漏洞。单帧加满 = 0.46×weight×dt,
   和起跳奖同级量级。

4) `leg_tracking_penalty`(推蹬段腿部跟踪惩罚): 深蹲底到落地前(95~132)9 个腿部
   关节(hip_pitch L/R、knee L/R、ankle_pitch L/R、ankle_roll L/R、waist_pitch)
   对参考的**线性** MAE。model_9300 实测: 深蹲(15~95)跟得很好, 但推蹬瞬间
   (100~118)模仿项 exp(-error/σ²)饱和 → 髋/膝/踝无梯度 → 不从深蹲向上延伸、
   反而前扑(俯仰冲到 +0.55~+0.59, 65% 回合死于 anchor_ori), 前扑时脚步乱动 =
   腿在抓地。线性(不饱和)在推蹬窗内把腿往参考延伸轨迹拉, 补上模仿项饱和的
   盲区; ankle_roll 参考锁 0 → 脚掌不左右乱晃。窗口外(站立/深蹲/落地)不生效。

5) `standing_fidget_penalty`(站立段"脚不能动"惩罚): 站立窗口(0~15, 下蹲前)内
   机器人"脚乱动"的惩罚(原线性, 2026-08-12 平方化 + 新增站立对称, 见 7), 度量按
   规则 1.3.7 原义:**只罚脚, 不罚关节**:
     - 脚水平滑动(脚在移动) = 脚(ankle_roll)世界水平速度超 margin 部分
     - 脚离地/抬起 = 脚世界高度高于参考脚高超 margin 部分
   规则原义是"起跳前双脚均不得离开地面或移动…方视为有效成绩", 站立段脚滑/脚
   抬在真机上 = 该次试跳判**无效**。参考站立帧 0~15 已实测完全静止(脚固定、
   root 固定 0.782m、pitch=0), 所以站立窗内脚的任何滑动/抬起都是多余动作。
   **为什么不用关节速度**(教训 15): 规则只禁"脚"离开地面或移动, 脚钉地但
   腿/腰/臂在微调(平衡/预备)是**合法**的; 用关节速度把合法也罚了 → 惩罚有
   物理下限永远到不了 0(实测 model_29299 卡在 -1.3 平台), 且和回合开始打架 →
   起跳退化。复位本就走参考站立帧 0(hip -0.068/knee 0.290, 非深蹲), 站立窗无
   "起身", 脚度量抓住的是真"脚蹭地/滑动"。模仿项 exp(-err/σ²)奖励瞬时
   位姿不罚速率, action_rate 又减半 → 脚附近快速小抖动没有梯度; 这里用线性
   (不饱和)惩罚补上, 死区 margin 吸收 PD/接触噪声。窗口外(下蹲/推蹬/腾空)
   不生效, 不干扰已学好的起跳。

6) `airborne_tuck_tracking_penalty`(腾空收腿轨迹惩罚): 参考腾空窗(115~132)内,
   膝 L/R + 踝 pitch L/R 对参考**钟形收腿轨迹**的**线性** MAE。model_29299 实测
   腾空窗膝 0.291 vs 参考 1.07 —— 腿完全伸直没蜷, 踝(成绩最低部位)抬不起来。
   为什么 leg_tracking 没拉住: 它是 9 关节均值([95,132]), 膝的偏差被稀释、且
   窗口覆盖推蹬伸腿(和收腿反向), 收腿梯度挤不过伸腿; 模仿项 body_pos 是 30
   body 平均 + exp 饱和, 对"脚踝抬高"无梯度; airborne_leg_tuck 窗口窄+要求
   离地+cap, 信号稀疏。这里聚焦 4 个"抬脚直驱关节"在纯腾空窗, 线性不饱和、
   不稀释、不跟伸腿打架。窗口外(站立/深蹲/推蹬/落地) = 0。完美跟踪 = 0。

7) `standing_fidget_penalty` **平方化** + `standing_symmetry_penalty`(站立对称,
   2026-08-12, model_49999 后, 用户选"非线性强罚+加固对称"): 旧线性 fidget 全程
   只占回合奖励 ~3%(weight -4, 实测 -0.46/总~14), 策略无视 → 站不稳(0.38/帧)不
   归零。两处修复:
   - fidget 改 **平方**: `mean((slip/scale)² + (lift/scale)², feet)`, scale=0.1,
     weight -1。margin 内 PD/接触噪声(0.02~0.05 m/s)≈免费; 真滑动 0.38 m/s →
     12.9/帧 → ~-3.9/回合(≈28%), 站立段压过模仿项成为主目标; 0.1→0.64、0.5→23,
     大滑动代价暴涨 —— 软"判无效"(平滑梯度, 无终止二进制墙、不误杀整回合)。
   - 新增站立窗对称罚(weight -6): 站不稳根因 = 动作左右不对称(单脚先动/先抬 →
     脚滑+离地, 规则判无效)。与 `action_symmetry_penalty`(mdp/jump_rewards.py)
     同款 13 成对不对称量, 只锁站立窗(0~15)直接给"动作不对称"梯度, 不给起跳段
     加约束。参考站立帧严格镜像 → 完美站立 asym→0, 无罚不掉下限。

8) `standing_invalid_termination`(站立窗口"判无效"**硬终止**, 2026-08-12, model_49999
   二次轮跑后, 用户选策略 A"治本"): 软罚(fidget 平方 + 站立对称)给了梯度但压不掉
   左右不对称结构(实测二次轮 fidget 平台 ~-0.4、站立对称 -0.72/全局 -0.78 全卡死,
   站立实滑仍 ~0.22 m/s)。这里把规则 1.3.7 升级为**硬判无效**: 站立窗(0~15)内任一
   脚 slip > slip_thresh 或 lift > lift_thresh → 整回合终止, 后续跳高奖励全部丢失。
   - 与软罚互补: 软罚给平滑梯度(先把量压小), 硬终止给"必须静止"的硬约束(把结构
     压死) —— 站不稳不再只是"扣点分", 而是"整跳白做"。
   - 阈值取在接触/PD 噪声(≤0.05 m/s)之上、当前策略实滑(~0.22 m/s)之下: 噪声帧
     不会误杀, 真滑动立即判死。`grace` 跳过复位前 2 帧接触瞬态(接触求解刚建立的
     速度尖峰, 防 100% 误杀)。
   - 窗口外(下蹲/推蹬/腾空/落地) = 不判死, 不干扰起跳。返回布尔 (N,)。

窗口帧号来源: `motion/jump_high_firstjump_50fps.npz` 实测(双脚离地 115~132, 下蹲起
15, 腾空最高点 ~123 帧踝抬至 0.462, 站立段 0~15 完全静止)。参考在回合内循环
(183 帧), 故用 `time_steps % ref_len` 定位相位。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


def _both_feet_airborne(env, sensor_cfg, force_threshold: float) -> torch.Tensor:
    """双脚是否都离地(布尔张量)。与 isaaclab 内置 undesired_contacts 同款:
    用历史力(history_length 帧)取最大, 抗单帧噪声; 双脚各自的历史最大法向力
    都低于阈值 = 全程无接触 = 离地。"""
    contact = env.scene.sensors[sensor_cfg.name]
    forces = contact.data.net_forces_w_history[:, :, sensor_cfg.body_ids]  # (N, H, 2脚, 3)
    max_per_foot = torch.norm(forces, dim=-1).max(dim=1).values  # (N, 2脚)
    return (max_per_foot < force_threshold).all(dim=-1)


def takeoff_completion_bonus(
    env: "ManagerBasedRLEnv",
    sensor_cfg: "SceneEntityCfg",
    command_name: str = "motion",
    t_to: int = 115,
    t_land: int = 132,
    ref_len: int = 183,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """起跳完成奖: 参考腾空窗口内双脚离地 -> 1, 否则 0。

    返回正值, 由正权重产生奖赏。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    t = command.time_steps % ref_len
    in_window = (t >= t_to) & (t <= t_land)
    both_air = _both_feet_airborne(env, sensor_cfg, force_threshold)
    return (both_air & in_window).float()


def airborne_leg_tuck(
    env: "ManagerBasedRLEnv",
    sensor_cfg: "SceneEntityCfg",
    command_name: str = "motion",
    t_to: int = 115,
    t_land: int = 132,
    ref_len: int = 183,
    force_threshold: float = 1.0,
    max_tuck: float = 0.462,
) -> torch.Tensor:
    """腾空收腿奖: 参考腾空窗口 + 双脚离地时, 奖励最低 link 世界高度(封顶)。

    返回正值(米), 由正权重产生奖赏。**cap 在 `max_tuck`(参考最高点收腿踝高
    0.462m)**: 达到参考即满分, 超过不再多给 —— 这样与模仿项(motion_body_pos/
    _ori 要求"正好等于参考姿态")目标一致, 只提供"往参考收腿方向"的梯度,
    不会像"无上限越高越好"那样和模仿项互相拉扯、把策略带偏。完美收腿 → 0.462。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    t = command.time_steps % ref_len
    in_window = (t >= t_to) & (t <= t_land)
    both_air = _both_feet_airborne(env, sensor_cfg, force_threshold)
    min_link_h = robot.data.body_pos_w[:, :, 2].min(dim=1).values.clamp(max=max_tuck)
    return torch.where(both_air & in_window, min_link_h, torch.zeros_like(min_link_h))


def standing_too_long_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    margin: float = 0.06,
    t_crouch_start: int = 15,
    t_land: int = 132,
    ref_len: int = 183,
) -> torch.Tensor:
    """站着过久惩罚: 参考已进入下蹲/推蹬期, 机器人还高出参考 -> 线性惩罚。

    返回正值(高出量, 米), 由负权重产生惩罚。线性因此不饱和, 远距仍有梯度。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    t = command.time_steps % ref_len
    in_phase = (t >= t_crouch_start) & (t <= t_land)
    ref_root_h = command.anchor_pos_w[:, 2]
    robot_root_h = robot.data.root_pos_w[:, 2]
    lag = (robot_root_h - ref_root_h - margin).clamp(min=0.0)
    return torch.where(in_phase, lag, torch.zeros_like(lag))


# 手臂 14 关节在【运行时序】(BFS, 与 eval_policy.py joint_names / OMNI_BODY_NAMES 同序)
# 的列号。参考 npz 列序 = 运行时序, robot.data.joint_pos 也是运行时序, 所以列号两边通用。
ARM_JOINT_COLS = [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

# 参考中"锁死"的关节(BFS 列号): 参考全程为 0, 应纹丝不动。
# waist_yaw(2) / waist_roll(5) / ankle_roll_L(17) / ankle_roll_R(18)。
# 注意: 腕关节也锁 0, 但已被 ARM_JOINT_COLS 覆盖, 这里只补 arm 惩罚漏掉的 4 个。
LOCKED_JOINT_COLS = [2, 5, 17, 18]

# 推蹬/腾空段腿部 9 关节(BFS 列号): 深蹲底→伸直→收腿全过程的驱动关节。
# hip_pitch L/R(0,1) / knee L/R(9,10) / ankle_pitch L/R(13,14) / ankle_roll L/R(17,18)
# / waist_pitch(8)。与 arm/locked/body_lean 同根因: 模仿项 exp 在快速推蹬时饱和,
# 腿变"自由关节" → 前扑/脚步乱动。
LEG_TRACK_COLS = [0, 1, 9, 10, 13, 14, 17, 18, 8]


def arm_deviation_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
) -> torch.Tensor:
    """手臂偏差惩罚: 14 个手臂关节对参考对应帧的**线性**平均绝对偏差(rad)。

    背景: 模仿项 `exp(-error/σ²)` 在跳跃时腿/躯干偏差一大就饱和到 0, 手臂贡献的
    梯度≈0, 手臂变成"自由关节" -> 乱转(model_34299 实测 sho_roll 甩到 1.68 rad,
    是参考的 ~7 倍)。这里用线性(不饱和)惩罚, 远距区保留梯度, 且只罚手臂关节:
    完美跟踪(手臂贴合参考)时 = 0, 不干扰腿/躯干的模仿主项。

    返回正值(平均绝对偏差, rad), 由负权重惩罚。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    ref_arm = command.joint_pos[:, ARM_JOINT_COLS]     # (N, 14) 参考手臂角
    act_arm = robot.data.joint_pos[:, ARM_JOINT_COLS]  # (N, 14) 实际手臂角
    dev = (act_arm - ref_arm).abs().mean(dim=-1)       # (N,) 平均绝对偏差
    return dev


def locked_joint_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
) -> torch.Tensor:
    """锁死关节惩罚: waist_yaw/waist_roll/ankle_roll 对参考对应帧的**线性**偏差。

    参考里这 4 个关节全程锁 0(build_first_jump 镜像对称化时归零), 但模仿项
    `exp(-error/σ²)` 在跳跃时饱和后, 它们会像手臂一样变"自由关节"乱动
    (后仰/不自然的来源之一)。arm_deviation_penalty 只覆盖 14 个手臂关节,
    不覆盖这里 —— 补线性(不饱和)惩罚: 完美锁 0 -> 0, 不干扰腿/躯干主项。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    ref_locked = command.joint_pos[:, LOCKED_JOINT_COLS]     # (N, 4) 参考锁死角(0)
    act_locked = robot.data.joint_pos[:, LOCKED_JOINT_COLS]  # (N, 4) 实际角
    dev = (act_locked - ref_locked).abs().mean(dim=-1)       # (N,) 平均绝对偏差
    return dev


def leg_tracking_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    t_to: int = 95,
    t_land: int = 132,
    ref_len: int = 183,
) -> torch.Tensor:
    """推蹬段腿部跟踪惩罚: 深蹲底到落地前(95~132)9 个腿部关节对参考的**线性** MAE。

    背景: model_9300 实测, 深蹲(15~95)跟得很好(俯仰 +0.22 vs 参考 +0.23、root 0.670
    vs 0.660), 但推蹬瞬间(100~118)模仿项 `exp(-error/σ²)` 饱和 → 髋/膝/踝贡献的
    梯度≈0 → 机器人不从深蹲向上延伸、反而前扑(俯仰冲到 +0.55~+0.59, 参考 +0.34),
    65% 回合死于 anchor_ori; 前扑时脚步乱动 = 腿在抓地。线性(不饱和)惩罚在推蹬窗
    内把 9 个腿关节往参考延伸轨迹拉, 补上模仿项饱和的盲区; ankle_roll 参考锁 0 →
    脚掌不左右乱晃。窗口外(站立/深蹲/落地)不生效, 不干扰已学好的深蹲。
    完美跟踪 = 0。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    t = command.time_steps % ref_len
    in_window = (t >= t_to) & (t <= t_land)
    ref_legs = command.joint_pos[:, LEG_TRACK_COLS]     # (N, 9) 参考腿部角
    act_legs = robot.data.joint_pos[:, LEG_TRACK_COLS]  # (N, 9) 实际腿部角
    dev = (act_legs - ref_legs).abs().mean(dim=-1)      # (N,) 平均绝对偏差
    return torch.where(in_window, dev, torch.zeros_like(dev))


# 腾空收腿 4 个"抬脚直驱关节"(BFS 列号): knee L/R(9,10) / ankle_pitch L/R(13,14)。
# 参考腾空窗(115~132)膝快起快落(0.16→1.89→0.27 钟形, 峰值帧124)、踝 pitch 同步
# 0→0.49, 把踝抬到最高点 0.462m(成绩 0.452 的关键驱动)。hip_pitch 收腿也有份, 但
# 已被 leg_tracking_penalty([95,132]) + motion_body_pos 覆盖, 这里只补膝/踝, 避免
# 和推蹬伸腿梯度打架、也避免 9 关节均值把信号稀释(见函数 docstring)。
TUCK_TRACK_COLS = [9, 10, 13, 14]


def airborne_tuck_tracking_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    t_to: int = 115,
    t_land: int = 132,
    ref_len: int = 183,
) -> torch.Tensor:
    """腾空收腿轨迹惩罚: 膝/踝对参考腾空窗钟形收腿轨迹的**线性** MAE。

    背景: model_29299 实测腾空窗膝 0.291 vs 参考 1.07 —— 腿完全伸直没蜷, 踝
    (成绩最低部位)抬不起来, score 卡 0.135(参考 0.452)。三个根因:
      - 模仿项 body_pos 是 30 body 平均 + exp 饱和 → 对"脚踝抬高"无梯度;
      - airborne_leg_tuck 窗口窄(17/183 帧)+ 要求双脚离地(只有 70% 帧离地)+ cap
        → 信号稀疏, 推不动"空中伸腿"这种大尺度行为;
      - leg_tracking_penalty 是 9 关节均值([95,132]) → 膝偏差被稀释, 且窗口覆盖
        推蹬伸腿(95~115, 和收腿反向), 收腿梯度挤不过伸腿。
    这里聚焦腾空窗(115~132)4 个"抬脚直驱关节", 对参考钟形收腿轨迹做**线性**
    (不饱和)平均绝对偏差 —— 远距仍有梯度、不稀释、不跟伸腿打架。当前不收腿时
    每步均值偏差 ~0.05(weight -4.0 → Episode_Reward ~-0.2, 最强惩罚档), 贴参考
    收腿 → 0。窗口外(站立/深蹲/推蹬/落地) = 0, 不干扰已学好的起跳/推蹬。
    完美跟踪 = 0。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    t = command.time_steps % ref_len
    in_window = (t >= t_to) & (t <= t_land)
    ref_tuck = command.joint_pos[:, TUCK_TRACK_COLS]     # (N, 4) 参考膝/踝收腿轨迹
    act_tuck = robot.data.joint_pos[:, TUCK_TRACK_COLS]  # (N, 4) 实际膝/踝
    dev = (act_tuck - ref_tuck).abs().mean(dim=-1)       # (N,) 平均绝对偏差
    return torch.where(in_window, dev, torch.zeros_like(dev))


def _quat_pitch(q: torch.Tensor) -> torch.Tensor:
    """世界系四元数 (N,4) (w,x,y,z) -> base_link 俯仰角 (N,) rad。

    公式: pitch = atan2(2(wy - zx), 2(w²+z²)-1)。已用参考 firstjump npz 验证符号:
    站立 0° / 下蹲前倾 +0.50 rad / 腾空 0° / 落地回 +0.28。前倾为正, 后仰为负。
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * y - z * x), 2 * (w * w + z * z) - 1)


def body_lean_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    t_to: int = 95,
    t_land: int = 132,
    ref_len: int = 183,
) -> torch.Tensor:
    """推蹬/腾空段身体前倾惩罚: base_link pitch 对参考的**线性**绝对偏差。

    参考里 base_link pitch: 站立 0°、下蹲前倾 +0.50rad、推蹬回 0°、腾空 0°、落地
    缓冲 +0.28°。`motion_global_anchor_orientation_error_exp` 是 exp(-err/σ²),
    推蹬/跳跃时饱和到 0 → 该段朝向偏离没有梯度, 前扑/后仰修不回来(和手臂/锁死
    关节/腿部同根因)。这里用线性(不饱和)偏差, 窗口 95~132: 从深蹲底开始, 覆盖
    推蹬伸直(95~115, 治"前扑不伸直")和腾空(115~132, 治"后仰/没直起来"); 不碰
    下蹲段合理的 +0.50rad 前倾, 也不罚落地段缓冲前倾。完美跟踪 = 0。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    t = command.time_steps % ref_len
    in_window = (t >= t_to) & (t <= t_land)
    ref_pitch = _quat_pitch(command.anchor_quat_w)          # (N,) 参考 base_link 俯仰
    act_pitch = _quat_pitch(command.robot_anchor_quat_w)    # (N,) 实际 base_link 俯仰
    dev = (act_pitch - ref_pitch).abs()
    return torch.where(in_window, dev, torch.zeros_like(dev))


# 站立段"脚不能动"的脚部 body 列号(OMNI_BODY_NAMES 序): ankle_roll_l_link=18,
# ankle_roll_r_link=19, 是末端"脚"(nohead_noshoe, 无独立脚/鞋体)。规则 1.3.7
# 只限"起跳前双脚不得离开地面或移动", 所以度量用**脚**本身:
#   - 脚水平速度 -> 脚在移动(滑/蹭)
#   - 脚高度 vs 参考脚高 -> 脚离地/抬起
# 不用关节速度(教训 15): 规则只禁"脚", 脚钉地但关节微调合法; 关节速度有罚不掉的
# 下限(实测 model_29299 卡 -1.3 平台)。复位走参考站立帧 0, 站立窗无"起身"。
FOOT_COLS = [18, 19]


def standing_fidget_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    t_stand: int = 15,
    ref_len: int = 183,
    slip_margin: float = 0.02,
    lift_margin: float = 0.02,
    scale: float = 0.1,
) -> torch.Tensor:
    """站立窗口(0~15, 下蹲前)内机器人"脚乱动"的**平方**惩罚(规则 1.3.7 原义)。

    规则: "起跳前双脚均不得离开地面或移动…方视为有效成绩" —— 站立段脚离地/脚
    滑动在真机上 = 该次试跳判**无效**(无扣分制, 成绩直接作废)。参考站立帧 0~15
    已实测完全静止(脚固定、root 固定 0.782m、pitch=0), 所以站立窗内脚的任何
    滑动/抬起都是多余动作。模仿项 exp(-err/σ²)奖励瞬时位姿不罚速率, action_rate
    又减半 → 脚附近快速小抖动没有梯度。度量补上规则禁止的两个量(两个脚取平均):
      slip = 脚(ankle_roll)世界**水平**速度超 slip_margin 的部分  (脚在移动)
      lift = 脚世界高度高于参考脚高超 lift_margin 的部分        (脚离地/抬起)
    惩罚 = mean((slip/scale)² + (lift/scale)², feet) —— **平方, 不是线性**:
      旧线性(weight -4)全程只占回合奖励 ~3%(-0.46/总~14), 策略无视 → 站不稳不归
      零。平方后(weight -1, scale 0.1):
        - margin 内 PD/接触噪声(0.02~0.05 m/s) → (·/0.1)² ≤ 0.25, ≈免费;
        - 真滑动 0.38 m/s → ((0.36)/0.1)² ≈ 12.9/帧 → ~-3.9/回合(≈28%), 站立段
          压过模仿项成为主目标; 0.1→0.64、0.5→23 → 大滑动代价暴涨。
      软"判无效": 平滑梯度(不饱和), 无终止的二进制墙、不会因阈值误杀整回合。
    **不用关节速度**(教训 15): 规则只禁"脚"离开地面或移动, 脚钉地但腿/腰/臂在
    微调(平衡/预备)是**合法**的; 关节速度把合法也罚了, 有下限到不了 0(实测
    model_29299 卡在 -1.3 平台, 且起跳退化)。复位本就走参考站立帧 0(hip
    -0.068/knee 0.290, 非深蹲), 站立窗无"起身", 脚度量抓住的是真"脚蹭地/滑动"。
    窗口外(下蹲/推蹬/腾空) = 0, 不干扰起跳。

    返回正值((m/s)² 量级), 由负权重("脚乱动扣大分")产生惩罚。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    t = command.time_steps % ref_len
    in_standing = t < t_stand
    v = robot.data.body_lin_vel_w[:, FOOT_COLS, :]                # (N,2,3) 脚世界速度
    slip = (v[..., :2].norm(dim=-1) - slip_margin).clamp(min=0.0)  # (N,2) 脚水平滑动
    ref_foot_z = command.body_pos_w[:, FOOT_COLS, 2]              # (N,2) 参考脚世界高度
    act_foot_z = robot.data.body_pos_w[:, FOOT_COLS, 2]           # (N,2) 实际脚世界高度
    lift = (act_foot_z - ref_foot_z - lift_margin).clamp(min=0.0)  # (N,2) 脚离地/抬起
    fidget = ((slip / scale) ** 2 + (lift / scale) ** 2).mean(dim=-1)  # (N,)
    return torch.where(in_standing, fidget, torch.zeros_like(fidget))


# 站立窗左右对称: 与 mdp/jump_rewards.py `action_symmetry_penalty` 同款成对列表
# (pitch 相等对 6 组 + mirror 镜像对 7 组, 运行时序 BFS 列号), 仅窗口锁站立段。
# 参考站立帧严格左右镜像 → 完美站立 asym→0。model_29299 实测站立窗动作 L/R 差
# ankle_pitch 1.69 / hip_pitch 0.47 / ankle_roll 0.47 rad —— 单脚先动/先抬 → 脚滑
# +离地(fidget 平台根因)。直接给"动作不对称"梯度(前瞻: 脚还没滑之前就推对称)。
STAND_SYM_PITCH_PAIRS = [(0, 1), (9, 10), (13, 14), (11, 12), (21, 22), (25, 26)]
STAND_SYM_MIRROR_PAIRS = [(3, 4), (6, 7), (17, 18), (15, 16), (19, 20), (23, 24), (27, 28)]


def standing_symmetry_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    t_stand: int = 15,
    ref_len: int = 183,
    scale: float = 0.3,
) -> torch.Tensor:
    """站立窗口(0~15)内动作左右不对称的**平方**惩罚(根因定向, 规则 1.3.7)。

    背景: 站立段"脚乱动"(fidget 平台 ~0.4/帧)的根因是动作左右不对称 —— 单脚
    先动/先抬 → 脚滑+离地(规则判无效)。全局 `action_symmetry_penalty`(weight -1)
    在站立窗归一化 asym≈0.5 压不住, 且它全局生效会碰起跳/腾空段。这里把同一
    成对不对称量**只锁站立窗**(0~15)用更高权重罚:
      - 直接给"动作不对称"梯度(前瞻: 脚还没滑之前就推对称, 比 fidget 间接度量
        更早生效);
      - 不给起跳/腾空段加约束(该段全局 -1.0 已够, 专注治"站不稳"根因);
      - 与全局对称罚不冲突(参考严格镜像, 完美跟随两侧都≈0)。
    **平方, 不是 exp(教训: exp 饱和掐死梯度)**: 原 `1-exp(-asym/0.15)` 在
    asym≈0.5 时惩罚顶到 0.96 但梯度只剩 ~0.24, 策略收到"反正罚满, 改不改没
    区别" → 对称永远学不会(实测站立 -0.72 / 全局 -0.79 平台卡死)。平方
    `(asym/scale)²` 不封顶: 不对称越大梯度越大(0.5 → 罚 2.8、梯度 ~11), 全程
    有学习信号, 降到对称时梯度也平滑归零。参考站立帧严格左右镜像 → 完美站立
    asym→0, 无罚不掉下限。返回正值(无量纲平方), 由负权重("不对称扣分")产生
    惩罚。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    t = command.time_steps % ref_len
    in_standing = t < t_stand
    actions = env.action_manager.action  # (N, 29, 运行时序 BFS)
    asym = torch.zeros(actions.shape[0], device=actions.device)
    for i, j in STAND_SYM_PITCH_PAIRS:
        asym += torch.abs(actions[:, i] - actions[:, j])
    for i, j in STAND_SYM_MIRROR_PAIRS:
        asym += torch.abs(actions[:, i] + actions[:, j])
    asym = asym / (len(STAND_SYM_PITCH_PAIRS) + len(STAND_SYM_MIRROR_PAIRS))
    asym = (asym / scale) ** 2
    return torch.where(in_standing, asym, torch.zeros_like(asym))


def standing_invalid_termination(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    t_stand: int = 15,
    ref_len: int = 183,
    slip_thresh: float = 0.10,
    lift_thresh: float = 0.05,
    grace: int = 2,
) -> torch.Tensor:
    """站立窗口(0~15)内任一脚滑动/离地超阈值 → 判**无效**(规则 1.3.7 硬终止)。

    与 fidget 平方罚 + 站立对称罚互补: 软罚给平滑梯度(压"量"), 这里是**硬约束**
    (压"结构")。规则 1.3.7 原义是起跳前脚动 = 该次试跳判无效、成绩作废 —— 二次轮
    实测软罚只能把站立实滑压到 ~0.22 m/s 平台(fidget -0.4 / 站立对称 -0.72 全卡死),
    压不掉"脚还在动"这个根因。这里把规则变成终止: 站立窗内**任一脚** slip >
    slip_thresh 或 lift > lift_thresh → 整回合判无效终止, 后续跳高奖励全部丢失
    —— 站不稳不再是"扣点分", 而是"整跳白做"。

    阈值取在接触/PD 噪声(≤0.05 m/s)之上、当前策略实滑(~0.22 m/s)之下:
      - 噪声帧不会误杀(死区);
      - 真滑动立即判死(硬终止)。
    `grace` 跳过复位后前 2 帧接触瞬态(接触求解刚建立时脚速度可能有尖峰, 防 100%
    误杀整回合)。窗口外(下蹲/推蹬/腾空/落地) = 不判死, 不干扰起跳。

    返回布尔 (N,): True = 该回合判无效, 立即终止。
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    t = command.time_steps % ref_len
    in_standing = (t < t_stand) & (t >= grace)
    v = robot.data.body_lin_vel_w[:, FOOT_COLS, :]                 # (N,2,3) 脚世界速度
    slip = v[..., :2].norm(dim=-1)                                 # (N,2) 脚水平滑动(绝对速度)
    ref_foot_z = command.body_pos_w[:, FOOT_COLS, 2]               # (N,2) 参考脚世界高度
    act_foot_z = robot.data.body_pos_w[:, FOOT_COLS, 2]            # (N,2) 实际脚世界高度
    lift = (act_foot_z - ref_foot_z).clamp(min=0.0)                # (N,2) 脚离地/抬起(绝对)
    bad = (slip > slip_thresh) | (lift > lift_thresh)              # (N,2) 任一脚违规
    return in_standing & bad.any(dim=-1)                           # (N,) bool
