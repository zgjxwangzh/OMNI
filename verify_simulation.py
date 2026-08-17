#!/usr/bin/env python3
"""
Omni RL SDK — MuJoCo 仿真验证脚本（Smoke Test）

用途：同事拿到 SDK 后，先跑这个脚本，快速验证环境是否配置正确、
ONNX/NPZ 是否能正常加载推理、MuJoCo 仿真能否启动。

运行方式：
    cd omni_rl_sdk
    python3 verify_simulation.py

每一项都会打印 ✅ 或 ❌，全部 ✅ 说明环境就绪，可以跑主程序。
"""
import sys
import os

# 把 SDK 根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✅"
FAIL = "❌"


def check(name, ok, detail=""):
    if ok:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}  —— {detail}")
    return ok


def main():
    print("=" * 60)
    print("  Omni RL SDK 仿真验证")
    print("=" * 60)
    all_ok = True

    # ---------- 1. Python 依赖 ----------
    print("\n【1】Python 依赖检查")
    for pkg in ["numpy", "onnxruntime", "yaml"]:
        try:
            __import__(pkg)
            ok = True
        except ImportError:
            ok = False
        all_ok &= check(pkg, ok, "缺失，请 pip install " + pkg)

    # ---------- 2. 编译产物 ----------
    print("\n【2】编译产物检查")
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "install", "lib")
    so_needed = ["env_interface_py.so", "libenv_mujoco.so", "libmujoco.so.3.9.0"]
    for so in so_needed:
        exists = os.path.isfile(os.path.join(lib_dir, so))
        all_ok &= check(so, exists, "缺失，需要编译 SDK")

    # ---------- 3. ONNX 加载 + 推理 ----------
    print("\n【3】ONNX 策略加载 + 推理")
    try:
        import numpy as np
        import onnxruntime as ort
        onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "policy/high_dynamic/config/model/policy.onnx")
        if not os.path.isfile(onnx_path):
            all_ok &= check("ONNX 文件", False, "policy.onnx 不存在")
        else:
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            inputs = sess.get_inputs()
            outputs = sess.get_outputs()
            obs_dim = inputs[0].shape[1] if len(inputs[0].shape) > 1 else "?"
            act_dim = outputs[0].shape[1] if len(outputs[0].shape) > 1 else "?"
            all_ok &= check("ONNX 加载", True, f"输入 obs[{obs_dim}] 输出 actions[{act_dim}]")

            # 构造一个站立观测做推理
            obs = np.zeros((1, int(obs_dim)), dtype=np.float32)
            # 参考朝向 rot6d identity（不同策略布局可能不同，这里做安全尝试）
            try:
                actions = sess.run(None, {inputs[0].name: obs})[0][0]
                amax = float(np.abs(actions).max())
                all_ok &= check("ONNX 推理", True, f"动作幅度 max={amax:.2f}")
            except Exception as e:
                all_ok &= check("ONNX 推理", False, str(e))
    except Exception as e:
        all_ok &= check("ONNX 加载", False, str(e))

    # ---------- 4. NPZ 格式检查 ----------
    print("\n【4】NPZ 参考运动检查")
    npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "policy/high_dynamic/config/motion/jump_high_firstjump_50fps.npz")
    if not os.path.isfile(npz_path):
        all_ok &= check("NPZ 文件", False, "jump_high_firstjump_50fps.npz 不存在")
    else:
        import numpy as np
        d = np.load(npz_path)
        for key, shape in [("joint_pos", (None, 29)), ("joint_vel", (None, 29)), ("body_quat_w", (None, None, 4))]:
            if key in d:
                ok = d[key].ndim == len(shape)
                all_ok &= check(f"NPZ {key}", ok, f"shape={d[key].shape}")
            else:
                all_ok &= check(f"NPZ {key}", False, "缺失 key")

    # ---------- 5. MuJoCo 环境初始化（可选，可能因无显示失败）----------
    print("\n【5】MuJoCo 环境初始化")
    try:
        from deploy_omni_sim_real import create_optimized_interface
        env = create_optimized_interface(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "deploy_omni_sim_real/config/env-omni31.yaml"))
        rd = env.get_robot_data()
        all_ok &= check("MuJoCo 初始化", True, f"关节数={len(rd.q_a_) - 6}")
    except Exception as e:
        all_ok &= check("MuJoCo 初始化", False, str(e))

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ 全部通过，环境就绪！可运行：python3 deploy_omni_sim_real/main.py")
    else:
        print("  ❌ 有检查项未通过，请根据上面的提示修复")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
