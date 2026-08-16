#!/usr/bin/env python
"""把 rsl-rl 5.x 的 checkpoint(actor_state_dict/critic_state_dict)转成服务器
rsl-rl 2.3.3 格式(model_state_dict)。

2026-08-13 V11: 用户要从 model_49999(rsl-rl 5.x, jump_high 本机训练)在服务器
(rsl-rl 2.3.3)续训。两格式不兼容, 需转换。

格式映射:
  5.x actor_state_dict.mlp.N.*  →  2.3.3 model_state_dict.actor.N.*
  5.x critic_state_dict.mlp.N.*  →  2.3.3 model_state_dict.critic.N.*
  5.x actor.distribution.std_param → 2.3.3 model_state_dict.std
  5.x actor.obs_normalizer.*     →  2.3.3 obs_norm_state_dict(顶层)
  5.x critic.obs_normalizer.*    →  2.3.3 privileged_obs_norm_state_dict(顶层)

用法:
  python scripts/convert_rsl5_to_233.py /home/liuziqi/model_49999.pt
  # 输出: /home/liuziqi/model_49999_233.pt (服务器 2.3.3 格式)
"""

import argparse
import os

import torch


def _map_mlp(mlp_state: dict, prefix: str) -> dict:
    """把 5.x 的 mlp.N.* 映射成 2.3.3 的 {prefix}.N.*。"""
    out = {}
    for k, v in mlp_state.items():
        if k.startswith("mlp."):
            out[prefix + k[len("mlp"):]] = v
    return out


def convert(src_path: str, dst_path: str | None = None) -> str:
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    assert "actor_state_dict" in ckpt and "critic_state_dict" in ckpt, (
        f"不是 rsl-rl 5.x 格式(缺 actor_state_dict/critic_state_dict): {src_path}"
    )
    actor5 = ckpt["actor_state_dict"]
    critic5 = ckpt["critic_state_dict"]

    model_state_dict = {}
    model_state_dict.update(_map_mlp(actor5, "actor"))
    model_state_dict.update(_map_mlp(critic5, "critic"))
    # std 参数
    if "distribution.std_param" in actor5:
        model_state_dict["std"] = actor5["distribution.std_param"]
    else:
        raise ValueError("5.x actor_state_dict 缺 distribution.std_param")

    # obs_norm 统计
    obs_norm_state_dict = {
        k.replace("obs_normalizer.", ""): v
        for k, v in actor5.items()
        if k.startswith("obs_normalizer.")
    }
    privileged_obs_norm_state_dict = {
        k.replace("obs_normalizer.", ""): v
        for k, v in critic5.items()
        if k.startswith("obs_normalizer.")
    }

    # 组装 2.3.3 格式(optimizer 无法复用, 置空)
    out = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "iter": ckpt.get("iter", 0),
        "infos": ckpt.get("infos", {}),
        "obs_norm_state_dict": obs_norm_state_dict,
        "privileged_obs_norm_state_dict": privileged_obs_norm_state_dict,
    }

    if dst_path is None:
        base, ext = os.path.splitext(src_path)
        dst_path = f"{base}_233{ext}"
    torch.save(out, dst_path)
    return dst_path


def main():
    ap = argparse.ArgumentParser(description="Convert rsl-rl 5.x checkpoint to 2.3.3 format.")
    ap.add_argument("src", help="5.x checkpoint (.pt)")
    ap.add_argument("--dst", default=None, help="输出路径(默认 <src>_233.pt)")
    args = ap.parse_args()
    dst = convert(args.src, args.dst)
    print(f"[saved] {dst}")
    print("上传到服务器后续训: --resume --checkpoint <dst 文件名>")


if __name__ == "__main__":
    main()
