#!/usr/bin/env python3
"""
从 TensorBoard events 文件中分析所有 checkpoint，找出最优模型。

用法:
  python3 find_best_checkpoint.py --log_dir logs/rsl_rl/<your_run_dir>

评分标准 (权重可调):
  - error_joint_pos: 越低越好 (目标 < 1.5)
  - mean_episode_length: 越高越好 (目标 > 400)
  - anchor_pos 终止率: 越低越好 (目标 < 5%)
  - mean_reward: 越高越好
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("需要安装 tensorboard: pip install tensorboard")
    sys.exit(1)


# 评分权重 (可根据需求调整)
SCORE_WEIGHTS = {
    "error_joint_pos": -3.0,       # 越低越好，负权重
    "mean_episode_length": 0.01,    # 越高越好
    "anchor_pos_term": -2.0,        # 越低越好 (终止率)
    "mean_reward": 0.1,             # 越高越好
}

# 目标阈值
TARGETS = {
    "error_joint_pos": 1.5,
    "mean_episode_length": 400,
    "anchor_pos_term": 0.05,  # 5%
}


def find_events_files(log_dir):
    """找到所有 events 文件"""
    events_files = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.startswith("events.out"):
                events_files.append(os.path.join(root, f))
    return events_files


def parse_events(events_file):
    """解析 events 文件，提取关键指标"""
    ea = EventAccumulator(events_file)
    ea.Reload()

    # 我们关心的指标
    target_tags = [
        "Metrics/motion/error_joint_pos",
        "Train/mean_episode_length",
        "Episode_Termination/anchor_pos",
        "Train/mean_reward",
        "Metrics/motion/error_body_pos",
        "Metrics/motion/error_body_rot",
        "Metrics/motion/error_joint_vel",
        "Episode_Reward/joint_pos_penalty",
        "Episode_Reward/track_joint_pos",
        "Episode_Reward/termination_penalty",
        "Policy/mean_noise_std",
    ]

    data = {}
    for tag in target_tags:
        try:
            events = ea.Scalars(tag)
            data[tag] = [(e.step, e.value) for e in events]
        except KeyError:
            data[tag] = []

    return data


def find_checkpoints(log_dir):
    """找到所有 checkpoint pt 文件"""
    checkpoints = []
    for f in Path(log_dir).rglob("model_*.pt"):
        # 提取步数
        stem = f.stem  # e.g., "model_19900"
        try:
            step = int(stem.split("_")[1])
            checkpoints.append((step, str(f)))
        except (IndexError, ValueError):
            continue
    checkpoints.sort()
    return checkpoints


def get_metrics_at_step(data, step, window=50):
    """获取指定步数附近的指标平均值 (平滑)"""
    result = {}
    for tag, events in data.items():
        if not events:
            result[tag] = None
            continue
        # 找到 step 附近的 events
        nearby = [v for s, v in events if abs(s - step) <= window]
        if nearby:
            result[tag] = sum(nearby) / len(nearby)
        else:
            # 找最近的
            closest = min(events, key=lambda x: abs(x[0] - step))
            result[tag] = closest[1]
    return result


def score_checkpoint(metrics):
    """综合评分"""
    score = 0.0
    details = {}

    # error_joint_pos
    ejp = metrics.get("Metrics/motion/error_joint_pos")
    if ejp is not None:
        score += SCORE_WEIGHTS["error_joint_pos"] * ejp
        details["error_joint_pos"] = f"{ejp:.4f} (目标<{TARGETS['error_joint_pos']})"

    # mean_episode_length
    mel = metrics.get("Train/mean_episode_length")
    if mel is not None:
        score += SCORE_WEIGHTS["mean_episode_length"] * mel
        details["episode_length"] = f"{mel:.1f} (目标>{TARGETS['mean_episode_length']})"

    # anchor_pos 终止率
    apt = metrics.get("Episode_Termination/anchor_pos")
    if apt is not None:
        score += SCORE_WEIGHTS["anchor_pos_term"] * apt
        details["anchor_pos_term"] = f"{apt:.4f} (目标<{TARGETS['anchor_pos_term']})"

    # mean_reward
    mr = metrics.get("Train/mean_reward")
    if mr is not None:
        score += SCORE_WEIGHTS["mean_reward"] * mr
        details["mean_reward"] = f"{mr:.4f}"

    return score, details


def main():
    parser = argparse.ArgumentParser(description="找出最优 checkpoint")
    parser.add_argument("--log_dir", type=str, required=True, help="日志目录")
    parser.add_argument("--top", type=int, default=10, help="显示前 N 个结果")
    parser.add_argument("--min_step", type=int, default=0, help="最小步数")
    parser.add_argument("--max_step", type=int, default=999999, help="最大步数")
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        print(f"错误: 目录不存在: {log_dir}")
        sys.exit(1)

    print(f"扫描目录: {log_dir}")
    print()

    # 找到所有 events 文件
    events_files = find_events_files(log_dir)
    if not events_files:
        print("未找到 events 文件")
        sys.exit(1)

    print(f"找到 {len(events_files)} 个 events 文件")

    # 合并所有 events 数据
    all_data = {}
    for ef in events_files:
        data = parse_events(ef)
        for tag, events in data.items():
            if tag not in all_data:
                all_data[tag] = []
            all_data[tag].extend(events)

    # 按 step 排序
    for tag in all_data:
        all_data[tag].sort()

    print(f"指标范围: step {min(v[0][0] for v in all_data.values() if v)} - {max(v[-1][0] for v in all_data.values() if v)}")
    print()

    # 找到所有 checkpoints
    checkpoints = find_checkpoints(log_dir)
    if not checkpoints:
        print("未找到 checkpoint 文件")
        sys.exit(1)

    print(f"找到 {len(checkpoints)} 个 checkpoints")
    print()

    # 评估每个 checkpoint
    results = []
    for step, path in checkpoints:
        if step < args.min_step or step > args.max_step:
            continue
        metrics = get_metrics_at_step(all_data, step)
        score, details = score_checkpoint(metrics)
        results.append((step, path, score, metrics, details))

    # 按评分排序
    results.sort(key=lambda x: x[2], reverse=True)

    # 输出结果
    print(f"{'='*80}")
    print(f"TOP {args.top} 最优 Checkpoints")
    print(f"{'='*80}")
    print()

    for i, (step, path, score, metrics, details) in enumerate(results[:args.top]):
        print(f"#{i+1} Step={step:6d}  Score={score:+.4f}")
        print(f"   文件: {path}")
        for k, v in details.items():
            marker = " ✅" if (
                (k == "error_joint_pos" and float(v.split()[0]) < TARGETS["error_joint_pos"]) or
                (k == "episode_length" and float(v.split()[0]) > TARGETS["mean_episode_length"]) or
                (k == "anchor_pos_term" and float(v.split()[0]) < TARGETS["anchor_pos_term"])
            ) else ""
            print(f"   {k}: {v}{marker}")
        print()

    # 特别标注满足所有目标的 checkpoint
    print(f"{'='*80}")
    print("满足所有目标的 Checkpoints:")
    print(f"{'='*80}")
    print()

    qualified = []
    for step, path, score, metrics, details in results:
        ejp = metrics.get("Metrics/motion/error_joint_pos", 999)
        mel = metrics.get("Train/mean_episode_length", 0)
        apt = metrics.get("Episode_Termination/anchor_pos", 1.0)

        if ejp < TARGETS["error_joint_pos"] and mel > TARGETS["mean_episode_length"] and apt < TARGETS["anchor_pos_term"]:
            qualified.append((step, path, score, metrics))

    if qualified:
        for step, path, score, metrics in qualified[:10]:
            ejp = metrics["Metrics/motion/error_joint_pos"]
            mel = metrics["Train/mean_episode_length"]
            apt = metrics["Episode_Termination/anchor_pos"]
            print(f"  Step={step:6d}  error_joint_pos={ejp:.4f}  episode_len={mel:.0f}  anchor_term={apt:.4f}")
            print(f"  文件: {path}")
            print()
    else:
        print("  暂无 checkpoint 同时满足所有目标")
        print()
        print("  最接近目标的:")
        for step, path, score, metrics, details in results[:5]:
            ejp = metrics.get("Metrics/motion/error_joint_pos", 999)
            mel = metrics.get("Train/mean_episode_length", 0)
            apt = metrics.get("Episode_Termination/anchor_pos", 1.0)
            print(f"  Step={step:6d}  error_joint_pos={ejp:.4f}  episode_len={mel:.0f}  anchor_term={apt:.4f}")


if __name__ == "__main__":
    main()
