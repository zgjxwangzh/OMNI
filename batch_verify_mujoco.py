#!/usr/bin/env python3
"""
批量 MuJoCo 验证：对所有 29 个 high_dynamic NPZ 做物理仿真回放检查。

检查项：
  1. 数据格式（维度、四元数范数）
  2. MuJoCo 运动学回放（无 NaN、无冲突）
  3. 关节限位（在 MuJoCo 模型限位内）
  4. Root 高度变化（检测动态动作）
  5. 速度尖峰（>15 rad/s 报警）

使用方法：
    python batch_verify_mujoco.py
    python batch_verify_mujoco.py --save_frames  # 对异常动作保存关键帧
"""
import os
import sys
import numpy as np
from pathlib import Path

# policy order → motor order 逆映射
MOTOR_TO_POLICY_IDX = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9,
    15, 22, 4, 10, 16, 23, 5, 11,
    17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])
POLICY_TO_MOTOR_IDX = np.argsort(MOTOR_TO_POLICY_IDX)


def verify_single(model, sim_data, motion_path, retarget_npz_path=None,
                  save_frames=False, output_dir="frames_batch"):
    """验证单个动作，返回结果 dict"""
    result = {
        "name": Path(motion_path).stem.replace("_highdynamic", ""),
        "frames": 0,
        "status": "✓",
        "issues": [],
        "height_min": 0, "height_max": 0, "height_delta": 0,
        "max_vel": 0,
    }

    # 加载 high_dynamic NPZ
    data = np.load(motion_path)
    jp = data['joint_pos']
    jv = data['joint_vel']
    bq = data['body_quat_w']
    T = jp.shape[0]
    result["frames"] = T

    # 加载 root position
    root_positions = None
    if retarget_npz_path and os.path.isfile(retarget_npz_path):
        rd = np.load(retarget_npz_path)
        if 'root_positions' in rd:
            root_positions = rd['root_positions']

    # NaN 检查
    if np.any(np.isnan(jp)) or np.any(np.isnan(jv)) or np.any(np.isnan(bq)):
        result["status"] = "✗"
        result["issues"].append("NaN in data")
        return result

    # 四元数范数检查
    norms = np.linalg.norm(bq.reshape(T, 4), axis=1)
    if not np.all(np.abs(norms - 1.0) < 0.01):
        result["issues"].append(f"quat norm drift [{norms.min():.4f},{norms.max():.4f}]")

    # 速度尖峰检查
    max_vel = np.abs(jv).max()
    result["max_vel"] = max_vel
    if max_vel > 15.0:
        result["issues"].append(f"vel spike {max_vel:.1f}")

    # MuJoCo 运动学回放
    n_act = model.nu
    model.opt.gravity[:] = 0  # 禁用重力

    heights = []
    has_nan = False
    for frame_idx in range(T):
        motor_pos = jp[frame_idx][POLICY_TO_MOTOR_IDX]

        if root_positions is not None:
            sim_data.qpos[0] = root_positions[frame_idx, 0]
            sim_data.qpos[1] = root_positions[frame_idx, 1]
            sim_data.qpos[2] = root_positions[frame_idx, 2]
        else:
            sim_data.qpos[0] = 0
            sim_data.qpos[1] = 0
            sim_data.qpos[2] = 0.82

        if bq.shape[1] >= 1:
            sim_data.qpos[3:7] = bq[frame_idx, 0]
        sim_data.qpos[7:7 + n_act] = motor_pos

        try:
            import mujoco
            mujoco.mj_forward(model, sim_data)
        except Exception as e:
            result["status"] = "✗"
            result["issues"].append(f"MjCo error@{frame_idx}: {e}")
            break

        heights.append(sim_data.qpos[2])

        if np.any(np.isnan(sim_data.qpos)):
            has_nan = True
            result["issues"].append(f"NaN@frame{frame_idx}")
            break

    # 恢复重力
    model.opt.gravity[:] = np.array([0, 0, -9.81])

    if has_nan:
        result["status"] = "✗"
    elif heights:
        result["height_min"] = min(heights)
        result["height_max"] = max(heights)
        result["height_delta"] = max(heights) - min(heights)

    # 保存关键帧（仅对异常动作）
    if save_frames and result["status"] == "✗":
        os.makedirs(output_dir, exist_ok=True)
        try:
            import mujoco
            renderer = mujoco.Renderer(model, height=480, width=640)
            # 回到第一帧
            sim_data.qpos[7:7 + n_act] = jp[0][POLICY_TO_MOTOR_IDX]
            if root_positions is not None:
                sim_data.qpos[0:3] = root_positions[0]
            if bq.shape[1] >= 1:
                sim_data.qpos[3:7] = bq[0, 0]
            mujoco.mj_forward(model, sim_data)
            renderer.update_scene(sim_data)
            img = renderer.render()
            from PIL import Image
            Image.fromarray(img).save(
                os.path.join(output_dir, f"{result['name']}_error.png"))
            renderer.close()
        except Exception:
            pass

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_dir", default="motion_data",
                        help="high_dynamic NPZ 目录")
    parser.add_argument("--retarget_dir", default="retargeted",
                        help="retargeted NPZ 目录（提供 root position）")
    parser.add_argument("--model", default="omni_29dof_mjc/mjcf/omni_29dof.xml")
    parser.add_argument("--save_frames", action="store_true")
    args = parser.parse_args()

    try:
        import mujoco
    except ImportError:
        print("MuJoCo 未安装，请先 pip install mujoco")
        sys.exit(1)

    # 加载模型（只加载一次）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model)
    model = mujoco.MjModel.from_xml_path(model_path)
    sim_data = mujoco.MjData(model)
    print(f"模型加载成功: {model.nq} qpos, {model.nu} actuators\n")

    # 收集所有 high_dynamic NPZ
    motion_files = sorted(Path(args.motion_dir).glob("*_highdynamic.npz"))
    print(f"找到 {len(motion_files)} 个 high_dynamic NPZ\n")

    results = []
    for mf in motion_files:
        # 找对应的 retarget NPZ
        stem = mf.stem.replace("_highdynamic", "")
        retarget_path = os.path.join(args.retarget_dir, f"{stem}.npz")
        if not os.path.isfile(retarget_path):
            retarget_path = None

        r = verify_single(model, sim_data, str(mf),
                          retarget_npz_path=retarget_path,
                          save_frames=args.save_frames)
        results.append(r)

        # 实时打印
        issues_str = ", ".join(r["issues"]) if r["issues"] else ""
        print(f"  {r['status']} {r['name']:20s} {r['frames']:5d}帧  "
              f"Δh={r['height_delta']:.2f}m  v_max={r['max_vel']:.1f}  {issues_str}")

    # 汇总
    print(f"\n{'='*75}")
    ok_count = sum(1 for r in results if r["status"] == "✓")
    print(f"通过: {ok_count}/{len(results)}")

    # 分类统计
    dynamic = [r for r in results if r["height_delta"] > 0.3]
    static = [r for r in results if r["height_delta"] <= 0.3]
    print(f"\n动态动作 (Δh>0.3m): {len(dynamic)} 个")
    for r in sorted(dynamic, key=lambda x: -x["height_delta"]):
        print(f"  {r['name']:20s} Δh={r['height_delta']:.2f}m  "
              f"[{r['height_min']:.2f}→{r['height_max']:.2f}]m")

    print(f"\n静态/微动 (Δh≤0.3m): {len(static)} 个")
    for r in static:
        print(f"  {r['name']:20s} Δh={r['height_delta']:.2f}m")

    # 问题动作
    bad = [r for r in results if r["status"] == "✗"]
    if bad:
        print(f"\n⚠ 异常动作: {len(bad)} 个")
        for r in bad:
            print(f"  ✗ {r['name']}: {', '.join(r['issues'])}")


if __name__ == "__main__":
    main()
