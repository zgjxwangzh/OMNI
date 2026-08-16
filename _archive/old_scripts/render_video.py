#!/usr/bin/env python3
"""
用 Isaac Lab 相机传感器录制 OMNI 训练视频
比 viewport 渲染更轻量，不容易崩溃
用法: python render_video.py
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="omni_walk")
parser.add_argument("--load_run", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

# 启动 Isaac Sim (headless + 相机)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
)
simulation_app = app_launcher.app

import numpy as np
import torch
from pathlib import Path

# 导入环境
from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils.task_registry import task_registry

# 安装 imageio
try:
    import imageio
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'imageio[ffmpeg]', '-q'])
    import imageio


def main():
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_class = task_registry.get_task_class(args.task)

    # 尝试给环境添加相机
    try:
        from isaaclab.sensors import CameraCfg, Camera
        from isaaclab.utils import configclass

        # 添加一个跟随相机到场景配置
        if not hasattr(env_cfg.scene, 'camera'):
            env_cfg.scene.camera = CameraCfg(
                prim_path="/World/envs/env_.*/Robot/base_link",
                update_period=0.02,
                height=480,
                width=640,
                data_types=["rgb"],
                spawn_kwargs={"clipping_range": (0.1, 100.0)},
            )
            print("[✓] 已添加相机配置")
    except Exception as e:
        print(f"[!] 添加相机失败: {e}")
        print("  将使用备用方案")
        env_cfg.scene.camera = None

    # 创建环境
    env = env_class(env_cfg, headless=True)

    # 加载模型
    log_dir = Path("logs") / args.task / args.load_run
    ckpt_path = log_dir / args.checkpoint
    print(f"[✓] 加载: {ckpt_path}")

    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, agent_cfg, str(log_dir), device="cuda:0")
    runner.load(str(ckpt_path))
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()

    # 输出路径
    output_path = str(log_dir / f"video_{args.checkpoint.replace('.pt','')}.mp4")
    writer = imageio.get_writer(output_path, fps=50, quality=8)

    print(f"[录制中] {args.steps} 帧 → {output_path}")

    camera_available = False
    frame_count = 0

    for i in range(args.steps):
        with torch.no_grad():
            actions = policy(obs)

        obs, _, _, _, _ = env.step(actions)

        # 尝试从相机获取帧
        frame = None
        if not camera_available:
            try:
                # Isaac Lab 的相机传感器 API
                if hasattr(env.scene, 'sensors') and 'camera' in env.scene.sensors:
                    cam = env.scene.sensors['camera']
                    cam.update(dt=env.step_dt)
                    rgb_data = cam.data.output.get("rgb", None)
                    if rgb_data is not None:
                        frame = rgb_data[0].cpu().numpy()  # 第一个环境
                        if frame.shape[-1] == 4:
                            frame = frame[:, :, :3]  # RGBA → RGB
                        camera_available = True
                        print(f"[✓] 相机可用，帧大小: {frame.shape}")
            except Exception as e:
                if i == 0:
                    print(f"[!] 相机获取失败: {e}")

        if frame is not None:
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            writer.append_data(frame)
            frame_count += 1
        else:
            # 备用: 用 matplotlib 生成关节状态可视化
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(3, 1, figsize=(8, 6))
                fig.suptitle(f'OMNI Walk - Step {i+1}/{args.steps}', fontsize=12)

                # 关节角度
                if hasattr(obs, 'cpu'):
                    obs_np = obs[0].cpu().numpy()
                else:
                    obs_np = obs[0] if len(obs.shape) > 1 else obs

                # 简化可视化 - 显示观测向量的分段
                n = min(len(obs_np), 87)
                axes[0].bar(range(n), obs_np[:n], color='steelblue', width=0.8)
                axes[0].set_title('Observations')
                axes[0].set_xlabel('Feature Index')

                # 动作
                act_np = actions[0].cpu().numpy() if hasattr(actions, 'cpu') else actions[0]
                axes[1].bar(range(len(act_np)), act_np, color='coral', width=0.8)
                axes[1].set_title('Actions (Joint Targets)')
                axes[1].set_xlabel('Joint Index')
                axes[1].set_ylim(-1, 1)

                # 步数进度
                axes[2].bar(['Progress'], [i+1], color='green', width=0.5)
                axes[2].set_ylim(0, args.steps)
                axes[2].set_title('Step Progress')

                plt.tight_layout()

                # 转成图像
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                frame = np.asarray(buf)[:, :, :3]
                writer.append_data(frame)
                frame_count += 1
                plt.close(fig)
            except:
                # 最后的备用: 纯黑帧
                writer.append_data(np.zeros((480, 640, 3), dtype=np.uint8))
                frame_count += 1

        if (i + 1) % 50 == 0:
            mode = "3D渲染" if camera_available else "数据可视化"
            print(f"  {mode} 帧 {i+1}/{args.steps}")

    writer.close()
    print(f"\n[✓] 视频已保存: {output_path} ({frame_count} 帧)")
    print(f"    通过 JupyterLab 文件管理器下载查看")

    simulation_app.close()


if __name__ == "__main__":
    main()
