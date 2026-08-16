#!/usr/bin/env python3
"""
训练环境检查脚本
================

在 AutoDL 上运行此脚本，验证训练环境是否就绪。

使用方法：
    # 在 Isaac Lab conda 环境中运行
    isaaclab.sh -p training/check_env.py

    # 或直接在 conda 环境中
    python training/check_env.py
"""

import sys
import os


def check(name, fn):
    """运行检查项并打印结果"""
    try:
        result = fn()
        print(f"  ✓ {name}: {result}")
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False


def main():
    print("=" * 60)
    print("  OMNI 29-DOF 训练环境检查")
    print("=" * 60)

    all_ok = True

    # ── 1. Python ──
    print(f"\n[1/7] Python 环境")
    all_ok &= check("Python 版本", lambda: f"{sys.version}")
    all_ok &= check("Python 路径", lambda: sys.executable)

    # ── 2. PyTorch + CUDA ──
    print(f"\n[2/7] PyTorch + CUDA")
    import torch
    all_ok &= check("PyTorch", lambda: torch.__version__)
    all_ok &= check("CUDA 可用", lambda: f"{'是' if torch.cuda.is_available() else '否'}")
    if torch.cuda.is_available():
        all_ok &= check("GPU", lambda: torch.cuda.get_device_name(0))
        all_ok &= check("显存", lambda: f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("  ⚠ CUDA 不可用！请检查 nvidia-smi 并重启 AutoDL 实例")
        all_ok = False

    # ── 3. Isaac Lab ──
    print(f"\n[3/7] Isaac Lab")
    try:
        import isaaclab
        all_ok &= check("Isaac Lab", lambda: isaaclab.__version__)
    except ImportError:
        print("  ✗ Isaac Lab 未安装！")
        print("    请确认在 Isaac Lab 的 conda 环境中运行")
        print("    或使用: isaaclab.sh -p training/check_env.py")
        all_ok = False

    # ── 4. rsl_rl ──
    print(f"\n[4/7] rsl_rl（训练框架）")
    try:
        from rsl_rl.runners import OnPolicyRunner
        from rsl_rl.modules import ActorCritic
        all_ok &= check("rsl_rl", lambda: "OnPolicyRunner + ActorCritic 可用")
    except ImportError:
        print("  ✗ rsl_rl 未安装！")
        print("    安装: pip install rsl_rl")
        all_ok = False

    # ── 5. 项目文件 ──
    print(f"\n[5/7] 项目文件")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    files_to_check = [
        ("URDF 模型", "assets/omni_29dof_nohead_noshoe/urdf/omni_29dof_nohead_noshoe_merged_modify_feet.urdf"),
        ("机器人配置", "robots/omni_29dof_nohead_noshoe_dcmotor_identified.py"),
        ("电机模型", "actuators/actuators_pd.py"),
        ("训练环境", "training/env_reference.py"),
        ("训练配置", "training/train_config.py"),
        ("训练入口", "training/train.py"),
        ("ONNX 导出", "training/export_onnx.py"),
    ]
    for name, path in files_to_check:
        full_path = os.path.join(project_dir, path)
        exists = os.path.isfile(full_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_ok = False

    # ── 6. 训练数据 ──
    print(f"\n[6/7] 训练数据（motion_data/）")
    motion_dir = os.path.join(project_dir, "motion_data")
    if os.path.isdir(motion_dir):
        npz_files = [f for f in os.listdir(motion_dir) if f.endswith("_highdynamic.npz")]
        all_ok &= check("NPZ 文件数", lambda: f"{len(npz_files)} 个")
        if len(npz_files) > 0:
            # 验证第一个文件
            import numpy as np
            first = os.path.join(motion_dir, npz_files[0])
            data = np.load(first)
            keys = list(data.keys())
            shapes = {k: data[k].shape for k in keys}
            all_ok &= check("数据格式", lambda: f"keys={keys}, shapes={shapes}")
    else:
        print(f"  ✗ motion_data/ 目录不存在！")
        all_ok = False

    # ── 7. 可选依赖 ──
    print(f"\n[7/7] 可选依赖")
    for pkg in ["onnxruntime", "mujoco", "tensorboard"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"  ✓ {pkg}: {ver}")
        except ImportError:
            print(f"  ⚠ {pkg}: 未安装（可选）")

    # ── 总结 ──
    print(f"\n{'=' * 60}")
    if all_ok:
        print(f"  ✓ 所有检查通过！可以开始训练。")
        print(f"\n  下一步：")
        print(f"    isaaclab.sh -p training/train.py --headless --motion 跳高06")
    else:
        print(f"  ✗ 有检查项未通过，请修复后再开始训练。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
