#!/usr/bin/env python3
"""Verify a TienKung-Lab RSL-RL checkpoint (.pt) in MuJoCo.

Usage (headless):
    python3.11 verify_pt_mujoco.py \
        --ckpt "/Users/grace/Downloads/2026-07-28_23-43-16 3/model_4900.pt" \
        --duration 10 --command 0.5

Usage (GUI, macOS):
    mjpython verify_pt_mujoco.py \
        --ckpt "/Users/grace/Downloads/2026-07-28_23-43-16 3/model_4900.pt" \
        --duration 10 --command 0.5 --gui
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. Extract actor from RSL-RL checkpoint and export to TorchScript
# ---------------------------------------------------------------------------

def extract_actor_from_checkpoint(ckpt_path: str) -> torch.nn.Module:
    """Load an RSL-RL checkpoint and return a standalone actor TorchScript module."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Invalid RSL-RL checkpoint: {ckpt_path}")

    state = ckpt["model_state_dict"]
    actor_state = {k.replace("actor.", ""): v for k, v in state.items() if k.startswith("actor.")}

    # Infer architecture from weight shapes
    input_dim = actor_state["0.weight"].shape[1]   # e.g. 1020
    h1 = actor_state["0.weight"].shape[0]           # e.g. 512
    h2 = actor_state["2.weight"].shape[0]           # e.g. 256
    h3 = actor_state["4.weight"].shape[0]           # e.g. 128
    output_dim = actor_state["6.weight"].shape[0]   # e.g. 29

    print(f"[INFO] Actor architecture: {input_dim} -> {h1} -> {h2} -> {h3} -> {output_dim}")
    print(f"[INFO] Checkpoint iteration: {ckpt.get('iter', '?')}")

    # Build matching actor MLP
    actor = torch.nn.Sequential(
        torch.nn.Linear(input_dim, h1),
        torch.nn.ELU(),
        torch.nn.Linear(h1, h2),
        torch.nn.ELU(),
        torch.nn.Linear(h2, h3),
        torch.nn.ELU(),
        torch.nn.Linear(h3, output_dim),
    )
    actor.load_state_dict(actor_state)
    actor.eval()

    # Wrap in a TorchScript-compatible module
    class ActorModule(torch.nn.Module):
        def __init__(self, model: torch.nn.Sequential):
            super().__init__()
            self.model = model

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.model(obs)

    module = ActorModule(actor)
    module.eval()
    scripted = torch.jit.script(module)

    # Verify export
    dummy = torch.zeros(1, input_dim)
    out = scripted(dummy)
    assert out.shape == (1, output_dim), f"Actor output shape mismatch: {out.shape}"
    print(f"[INFO] Actor exported to TorchScript successfully (output dim = {output_dim})")
    return scripted


# ---------------------------------------------------------------------------
# 2. MuJoCo verification
# ---------------------------------------------------------------------------

def run_verification(
    policy: torch.nn.Module,
    duration_s: float = 10.0,
    command_vx: float = 0.5,
    command_vy: float = 0.0,
    command_wz: float = 0.0,
    gui: bool = False,
    policy_dt: float = 0.020,
    base_height: float = 0.8,
):
    """Run the policy in MuJoCo and print metrics."""
    # Import here so headless mode doesn't need the viewer
    import importlib.util
    import mujoco
    if gui:
        import mujoco.viewer

    # Load omni_contract first (dependency of omni_mujoco)
    project_root = Path(__file__).resolve().parent
    
    # Search for TienKung-Lab in multiple locations
    tienkung_candidates = [
        project_root / "TienKung-Lab",
        project_root.parent / "Omni_100m" / "TienKung-Lab",
        Path("/Users/grace/Downloads/Omni_100m/TienKung-Lab"),
    ]
    
    tienkung_dir = None
    for candidate in tienkung_candidates:
        if (candidate / "legged_lab" / "omni_contract.py").exists():
            tienkung_dir = candidate
            break
    
    if tienkung_dir is None:
        raise FileNotFoundError(f"TienKung-Lab not found in any of: {tienkung_candidates}")
    
    contract_path = tienkung_dir / "legged_lab" / "omni_contract.py"
    mujoco_runner_path = tienkung_dir / "legged_lab" / "omni_mujoco.py"

    # Make legged_lab importable for omni_mujoco's internal import
    if str(tienkung_dir) not in sys.path:
        sys.path.insert(0, str(tienkung_dir))

    # Load omni_contract as legged_lab.omni_contract
    spec_contract = importlib.util.spec_from_file_location("legged_lab.omni_contract", contract_path)
    contract_mod = importlib.util.module_from_spec(spec_contract)
    sys.modules["legged_lab.omni_contract"] = contract_mod
    spec_contract.loader.exec_module(contract_mod)

    # Load omni_mujoco
    spec_runner = importlib.util.spec_from_file_location("legged_lab.omni_mujoco", mujoco_runner_path)
    runner_mod = importlib.util.module_from_spec(spec_runner)
    sys.modules["legged_lab.omni_mujoco"] = runner_mod
    spec_runner.loader.exec_module(runner_mod)

    OmniMujocoRunner = runner_mod.OmniMujocoRunner

    # Use the project's MuJoCo XML (TienKung-Lab's default path may not exist)
    project_root = Path(__file__).resolve().parent
    default_xml = project_root / "omni_29dof_mjc" / "mjcf" / "omni_29dof.xml"
    if not default_xml.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {default_xml}")

    runner = OmniMujocoRunner(model_path=default_xml, policy_dt=policy_dt, base_height=base_height)
    runner.set_command((command_vx, command_vy, command_wz))

    steps = max(1, round(duration_s / policy_dt))
    print(f"\n[INFO] Running {steps} policy steps ({steps * policy_dt:.1f}s) "
          f"with command vx={command_vx}, vy={command_vy}, wz={command_wz}")

    viewer = None
    if gui:
        viewer = mujoco.viewer.launch_passive(runner.model, runner.data)

    t0 = time.time()
    max_torque = 0.0
    heights = []

    for step_i in range(steps):
        obs = runner.observation()
        with torch.inference_mode():
            action = policy(torch.from_numpy(obs).unsqueeze(0))
            if isinstance(action, tuple):
                action = action[0]
            action = action.squeeze(0).detach().cpu().numpy()

        runner.step_policy_action(action)
        max_torque = max(max_torque, float(np.max(np.abs(runner.data.ctrl))))
        heights.append(float(runner.data.qpos[2]))

        if gui and viewer is not None:
            viewer.sync()

    elapsed = time.time() - t0
    metrics = runner.metrics()

    result = {
        "duration_s": float(runner.data.time),
        "policy_steps": steps,
        "wall_time_s": round(elapsed, 2),
        "command_vx_mps": command_vx,
        "distance_x_m": round(metrics["distance_m"], 3),
        "final_base_height_m": round(metrics["base_height_m"], 3),
        "min_base_height_m": round(min(heights), 3),
        "max_base_height_m": round(max(heights), 3),
        "max_abs_torque_nm": round(max_torque, 2),
        "foot_contact_count": metrics["foot_contact_count"],
        "undesired_contact": metrics["undesired_contact"],
        "joint_limit_fraction": round(metrics["joint_limit_fraction"], 4),
        "torque_saturation_fraction": round(metrics["torque_saturation_fraction"], 4),
        "max_joint_velocity_rads": round(metrics["max_abs_joint_velocity"], 2),
    }

    print("\n" + "=" * 50)
    print("MuJoCo Verification Results")
    print("=" * 50)
    for k, v in result.items():
        print(f"  {k:30s}: {v}")
    print("=" * 50)

    if viewer is not None:
        print("\n[INFO] GUI closed. Press Enter to exit.")
        input()

    return result


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify TienKung-Lab PT checkpoint in MuJoCo")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model_XXXX.pt checkpoint")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration in seconds")
    parser.add_argument("--command", type=float, default=0.5, help="Forward velocity command (m/s)")
    parser.add_argument("--command_y", type=float, default=0.0, help="Lateral velocity command (m/s)")
    parser.add_argument("--command_wz", type=float, default=0.0, help="Yaw rate command (rad/s)")
    parser.add_argument("--gui", action="store_true", help="Show MuJoCo GUI viewer")
    parser.add_argument("--policy_dt", type=float, default=0.020, help="Policy control dt")
    parser.add_argument("--base_height", type=float, default=0.8, help="Initial base height (m)")
    parser.add_argument("--export_only", action="store_true", help="Only export TorchScript, don't run")
    parser.add_argument("--output_ts", type=str, default=None, help="Path to save TorchScript (.pt)")
    args = parser.parse_args()

    # Step 1: Extract actor and export to TorchScript
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    policy = extract_actor_from_checkpoint(args.ckpt)

    # Optionally save TorchScript
    if args.output_ts:
        torch.jit.save(policy, args.output_ts)
        print(f"[INFO] TorchScript saved to: {args.output_ts}")

    if args.export_only:
        print("[INFO] Export-only mode, skipping MuJoCo evaluation.")
        return

    # Step 2: Run MuJoCo verification
    run_verification(
        policy=policy,
        duration_s=args.duration,
        command_vx=args.command,
        command_vy=args.command_y,
        command_wz=args.command_wz,
        gui=args.gui,
        policy_dt=args.policy_dt,
        base_height=args.base_height,
    )


if __name__ == "__main__":
    main()
