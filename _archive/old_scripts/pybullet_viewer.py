#!/usr/bin/env python3
"""
PyBullet 回放 OMNI 训练结果（完全不依赖 Isaac Sim）
从保存的 params/*.yaml 读取配置，直接构建策略网络
用法: python pybullet_viewer.py --load_run=xxx --checkpoint=model_4900.pt
"""
import argparse
import os
import sys
import subprocess

def ensure_deps():
    deps = ['pybullet', 'imageio[ffmpeg]', 'pyyaml']
    for pkg in deps:
        try:
            if pkg == 'pyyaml':
                __import__('yaml')
            elif pkg.startswith('imageio'):
                __import__('imageio')
            else:
                __import__(pkg)
        except ImportError:
            print(f"[安装] {pkg}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

ensure_deps()

import numpy as np
import pybullet as p
import pybullet_data
import imageio
import torch
import yaml
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="omni_walk")
parser.add_argument("--load_run", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--zero_action", action="store_true", help="不跑策略，关节保持零位（看默认姿势）")
args = parser.parse_args()


def find_urdf():
    candidates = [
        "assets/omni_29dof_nohead_noshoe/urdf/omni_29dof_nohead_noshoe_merged_modify_feet.urdf",
        "../omni_29dof_v260705/assets/omni_29dof_nohead_noshoe/urdf/omni_29dof_nohead_noshoe_merged_modify_feet.urdf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    import glob
    results = glob.glob("**/omni_29dof*.urdf", recursive=True)
    if results:
        return os.path.abspath(results[0])
    return None


def get_joint_names(urdf_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    for joint in root.findall('.//joint'):
        jtype = joint.get('type')
        if jtype in ('revolute', 'continuous'):
            joints.append(joint.get('name'))
    return joints


class SimpleActorCritic(torch.nn.Module):
    """简化版 ActorCritic，只实现 act() 用于推理"""
    def __init__(self, obs_dim, num_actions, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Actor network
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ELU())
            prev = h
        layers.append(torch.nn.Linear(prev, num_actions))
        self.actor = torch.nn.Sequential(*layers)

        # init
        for m in self.actor:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=1.0)
                torch.nn.init.zeros_(m.bias)
        # last layer small init
        torch.nn.init.uniform_(self.actor[-1].weight, -0.01, 0.01)

    def act(self, obs):
        return self.actor(obs)

    def load_state_dict(self, state_dict, strict=False):
        # 支持多种命名: actor.*, actor_mean.*
        actor_state = {}
        for k, v in state_dict.items():
            for prefix in ('actor.', 'actor_mean.'):
                if k.startswith(prefix):
                    actor_state[k.replace(prefix, '', 1)] = v
                    break
        if actor_state:
            self.actor.load_state_dict(actor_state, strict=False)
        else:
            super().load_state_dict(state_dict, strict=False)


def main():
    log_dir = Path("logs") / args.task / args.load_run

    # 1. 从 params YAML 读取配置
    env_yaml_path = log_dir / "params" / "env.yaml"
    agent_yaml_path = log_dir / "params" / "agent.yaml"

    # 自定义 YAML loader 支持 Python tuple 等标签
    class CustomLoader(yaml.SafeLoader):
        pass
    def _tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))
    def _object_constructor(loader, suffix, node):
        return loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode) else loader.construct_mapping(node)
    CustomLoader.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor)
    CustomLoader.add_multi_constructor('tag:yaml.org,2002:python/', _object_constructor)

    if env_yaml_path.exists():
        with open(env_yaml_path) as f:
            env_cfg = yaml.load(f, Loader=CustomLoader)
        obs_dim = env_cfg.get('num_observations', 106)
        act_dim = env_cfg.get('num_actions', 29)
        print(f"[✓] env.yaml: obs={obs_dim}, actions={act_dim}")
    else:
        # 默认值
        obs_dim = 106
        act_dim = 29
        print(f"[!] 找不到 env.yaml，使用默认: obs={obs_dim}, actions={act_dim}")

    if agent_yaml_path.exists():
        with open(agent_yaml_path) as f:
            agent_cfg = yaml.load(f, Loader=CustomLoader)
        policy_cfg = agent_cfg.get('policy', {})
        hidden_dims = policy_cfg.get('hidden_dims', [512, 256, 128])
        print(f"[✓] agent.yaml: hidden_dims={hidden_dims}")
    else:
        hidden_dims = [512, 256, 128]
        print(f"[!] 找不到 agent.yaml，使用默认 hidden_dims={hidden_dims}")

    # 3. 加载 checkpoint，从中推断真实 obs_dim
    ckpt_path = log_dir / args.checkpoint
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        model_state = state["model_state_dict"]
    elif "model" in state:
        model_state = state["model"]
    else:
        model_state = state

    # 从 actor 第一层权重推断真实 obs_dim
    first_layer_key = None
    for k in model_state:
        if 'actor.0.weight' in k or 'actor_mean.0.weight' in k:
            first_layer_key = k
            break
    if first_layer_key:
        real_obs_dim = model_state[first_layer_key].shape[1]
        print(f"[✓] 从 checkpoint 推断: 真实 obs_dim = {real_obs_dim}")
        obs_dim = real_obs_dim

    # 2. 构建策略网络 (使用真实 obs_dim)
    policy = SimpleActorCritic(obs_dim, act_dim, hidden_dims)

    policy.load_state_dict(model_state, strict=False)
    policy.eval()
    print(f"[✓] 策略加载: {ckpt_path}")

    # 4. 启动 PyBullet
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # 5. 加载 OMNI URDF
    urdf_path = find_urdf()
    if urdf_path is None:
        print("[错误] 找不到 OMNI URDF!")
        sys.exit(1)
    print(f"[✓] URDF: {urdf_path}")

    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 1.2], useFixedBase=False)
    joint_names = get_joint_names(urdf_path)
    n_joints = p.getNumJoints(robot_id)
    print(f"[✓] 机器人: {n_joints} 个关节, {len(joint_names)} 个可动关节")

    # 关节映射
    joint_map = {}
    for i in range(n_joints):
        info = p.getJointInfo(robot_id, i)
        joint_map[info[1].decode('utf-8')] = i

    # 6. 录制设置
    width, height = 640, 480
    output_path = str(log_dir / f"pybullet_video_{args.checkpoint.replace('.pt','')}.mp4")
    writer = imageio.get_writer(output_path, fps=50, quality=8)
    print(f"[录制] {args.steps} 帧 → {output_path}")

    # 初始化观测
    obs = torch.zeros(1, obs_dim)

    for step in range(args.steps):
        # 策略推理 or 零动作
        with torch.no_grad():
            if args.zero_action:
                actions = torch.zeros(1, act_dim)
            else:
                actions = policy.act(obs)

        # 动作 → 关节角度 (action_scale=0.25)
        joint_targets = actions[0].numpy() * 0.25
        joint_targets = np.clip(joint_targets, -3.14, 3.14)

        # 设置关节
        for idx, target in enumerate(joint_targets):
            if idx < len(joint_names) and joint_names[idx] in joint_map:
                p.resetJointState(robot_id, joint_map[joint_names[idx]], targetValue=target)

        # 物理步进
        for _ in range(4):
            p.stepSimulation()

        # 相机跟随 + 缓慢旋转
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        yaw = 45 + step * 0.7
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=pos,
            distance=3.0,
            yaw=yaw,
            pitch=-25,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(60, width / height, 0.1, 100)

        # 渲染
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        frame = np.array(rgb)[:, :, :3]
        writer.append_data(frame)

        # 更新观测（简化版：关节角度 + 速度 + 上一动作 + 指令速度）
        obs_list = []
        for jname in joint_names[:act_dim]:
            if jname in joint_map:
                st = p.getJointState(robot_id, joint_map[jname])
                obs_list.extend([st[0], st[1]])  # 角度, 角速度
        obs_list.extend(joint_targets[:act_dim].tolist())  # 上一动作
        obs_list.extend([1.0, 0.0, 0.0])  # 指令速度 (forward)
        # 补齐
        while len(obs_list) < obs_dim:
            obs_list.append(0.0)
        obs = torch.tensor([obs_list[:obs_dim]], dtype=torch.float32)

        if (step + 1) % 100 == 0:
            print(f"  帧 {step+1}/{args.steps} | 位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    writer.close()
    p.disconnect()
    print(f"\n[✓] 视频已保存: {output_path}")
    print(f"    大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
