#!/usr/bin/env python3
"""
将训练好的 PyTorch 策略导出为 ONNX 格式
==========================================

导出的 ONNX 可直接放入 omni_rl_sdk 的 high_dynamic 框架部署。

输入：rsl_rl 的 checkpoint（.pt 文件）
输出：ONNX 文件（obs → actions）

使用方法：
    python training/export_onnx.py \
        --checkpoint logs/ref_tracking/ref_tracking_jump/model_10000.pt \
        --output model/ref_tracking_jump.onnx

    # 验证导出正确性
    python training/export_onnx.py \
        --checkpoint logs/ref_tracking/ref_tracking_jump/model_10000.pt \
        --output model/ref_tracking_jump.onnx \
        --verify
"""

import argparse
import os
import sys
import numpy as np
import torch


def export_to_onnx(checkpoint_path, output_path, num_obs=529, num_actions=29):
    """将 PyTorch ActorCritic 策略导出为 ONNX"""

    # ── 1. 加载 checkpoint ──
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # rsl_rl checkpoint 结构
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "policy" in checkpoint:
        state_dict = checkpoint["policy"]
    else:
        state_dict = checkpoint
    print(f"  ✓ 加载成功，keys: {list(state_dict.keys())[:5]}...")

    # ── 2. 构建 Actor 网络 ──
    # rsl_rl 的 ActorCritic 结构：
    #   actor: MLP(obs → action_mean)
    #   结构：Linear(obs_dim, hidden1) → ELU → Linear(hidden1, hidden2) → ELU → Linear(hidden2, act_dim)

    # 从 checkpoint 推断网络结构
    # 通常 key 格式：actor.0.weight, actor.0.bias, actor.2.weight, ...
    actor_keys = [k for k in state_dict.keys() if k.startswith("actor.")]
    if not actor_keys:
        print("  ✗ checkpoint 中未找到 actor 权重！")
        print(f"    可用 keys: {list(state_dict.keys())}")
        return False

    # 推断隐藏层维度
    hidden_dims = []
    for k in sorted(actor_keys):
        if "weight" in k:
            dim = state_dict[k].shape[1]  # 输入维度
            if dim == num_obs:
                continue  # 第一层，跳过
            hidden_dims.append(dim)

    print(f"  推断网络结构: {num_obs} → {hidden_dims} → {num_actions}")

    # 构建导出模型
    class ExportablePolicy(torch.nn.Module):
        def __init__(self, actor_state_dict, obs_dim, act_dim, hidden_dims):
            super().__init__()
            layers = []
            in_dim = obs_dim
            for i, h_dim in enumerate(hidden_dims):
                layers.append(torch.nn.Linear(in_dim, h_dim))
                layers.append(torch.nn.ELU())
                in_dim = h_dim
            layers.append(torch.nn.Linear(in_dim, act_dim))
            self.actor = torch.nn.Sequential(*layers)

            # 加载权重
            self.actor.load_state_dict({
                k.replace("actor.", ""): v
                for k, v in actor_state_dict.items()
                if k.startswith("actor.")
            })

        def forward(self, obs):
            return self.actor(obs)

    policy = ExportablePolicy(state_dict, num_obs, num_actions, hidden_dims)
    policy.eval()

    # ── 3. 导出 ONNX ──
    dummy_input = torch.randn(1, num_obs)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"\n导出 ONNX: {output_path}")
    torch.onnx.export(
        policy,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        },
    )
    print(f"  ✓ ONNX 导出成功")

    # ── 4. 验证 ──
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        result = session.run(None, {"obs": dummy_input.numpy()})
        print(f"  ✓ ONNX 验证通过：output shape = {result[0].shape}")
    except ImportError:
        print(f"  ⚠ onnxruntime 未安装，跳过验证")
    except Exception as e:
        print(f"  ✗ ONNX 验证失败：{e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="导出训练策略为 ONNX")
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint 路径")
    parser.add_argument("--output", required=True, help="输出 ONNX 文件路径")
    parser.add_argument("--num_obs", type=int, default=529, help="观测维度")
    parser.add_argument("--num_actions", type=int, default=29, help="动作维度")
    parser.add_argument("--verify", action="store_true", help="验证导出正确性")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"✗ checkpoint 不存在: {args.checkpoint}")
        sys.exit(1)

    success = export_to_onnx(
        args.checkpoint, args.output,
        num_obs=args.num_obs, num_actions=args.num_actions,
    )

    if success:
        print(f"\n✓ 导出完成！")
        print(f"  部署方法：将 {args.output} 放入 omni_rl_sdk/policy/high_dynamic/model/")
        print(f"  修改 high_dynamic.yaml 的 model.path 指向新 ONNX")
    else:
        print(f"\n✗ 导出失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
