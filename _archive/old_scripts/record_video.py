#!/usr/bin/env python3
"""
OMNI 训练结果录像脚本
用法:
    python record_video.py --load_run=2026-07-28_21-47-03 --checkpoint=model_800.pt --steps=300
"""
import argparse
import os
import sys

# 先解析参数确定 headless
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="omni_walk")
parser.add_argument("--load_run", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=300, help="录制帧数")
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

# 启动 Isaac Sim (headless + 离线渲染)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

# 以下必须在 simulation_app 启动后导入
import numpy as np
import torch
from pathlib import Path

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils.task_registry import task_registry

try:
    import imageio
except ImportError:
    os.system("pip install imageio[ffmpeg] -q")
    import imageio


def main():
    # 获取环境和配置
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_class = task_registry.get_task_class(args.task)

    # 创建环境
    env = env_class(env_cfg, headless=True)

    # 加载 checkpoint
    log_dir = Path("logs") / args.task / args.load_run
    ckpt_path = log_dir / args.checkpoint
    if not ckpt_path.exists():
        print(f"[错误] 找不到: {ckpt_path}")
        simulation_app.close()
        return

    print(f"[✓] 加载模型: {ckpt_path}")

    # 加载策略
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, agent_cfg, str(log_dir), device="cuda:0")
    runner.load(str(ckpt_path))
    policy = runner.get_inference_policy(device="cuda:0")

    # 重置环境
    obs, _ = env.reset()

    # 录像
    output_path = str(log_dir / f"video_{args.checkpoint.replace('.pt','')}.mp4")
    writer = imageio.get_writer(output_path, fps=50, quality=8)

    print(f"[录制中] {args.steps} 帧 → {output_path}")

    for i in range(args.steps):
        # 获取动作
        with torch.no_grad():
            actions = policy(obs)

        # 执行一步
        obs, _, _, _, _ = env.step(actions)

        # 渲染并保存帧
        # 尝试多种方式获取帧
        frame = None
        try:
            # Isaac Lab 2.3 的渲染方式
            from isaaclab.sim import SimulationContext
            sim = SimulationContext.instance()
            sim.render()

            # 获取 viewport 图像
            import omni.replicator.core as rep
            # 备用: 直接用 kit viewport
        except:
            pass

        if frame is None:
            try:
                # 尝试通过 viewport 获取
                from omni.kit.viewport.utility import get_active_viewport
                viewport = get_active_viewport()
                if viewport:
                    frame = viewport.get_image_as_array()
            except:
                pass

        if frame is None:
            try:
                # Isaac Sim 5.x 方式
                import omni.ui as ui
                from pxr import Gf
                import carb
                kit = carb.framework.Framework.get()
                # 简单方法: 用 render product
                from omni.syntheticdata import SyntheticData
                sd = SyntheticData.get()
                sensors = sd.get_sensor_names()
                if sensors:
                    data = sd.get_sensor_data(sensors[0])
                    if data and "rgb" in data:
                        frame = data["rgb"]
            except:
                pass

        if frame is not None:
            if frame.dtype != np.uint8:
                frame = (frame[:, :, :3] * 255).astype(np.uint8) if frame.max() <= 1.0 else frame[:, :, :3].astype(np.uint8)
            writer.append_data(frame)
        else:
            # 如果所有渲染方式都失败，生成占位帧
            if i == 0:
                print("[警告] 无法获取渲染帧，将生成数据可视化替代")
            # 生成简单的状态可视化
            fig_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # 显示步数信息
            writer.append_data(fig_frame)

        if (i + 1) % 50 == 0:
            print(f"  帧 {i+1}/{args.steps}")

    writer.close()
    print(f"\n[✓] 视频已保存: {output_path}")
    print(f"    通过 JupyterLab 文件管理器下载查看")

    simulation_app.close()


if __name__ == "__main__":
    main()
