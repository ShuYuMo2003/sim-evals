"""
Smoke test for the local DROID IsaacLab environment.

This script intentionally does not connect to any policy server. It only checks
that IsaacLab can launch, the DROID environment can be created, and reset
returns the expected observation structure.
"""

import argparse
import os
import time

import gymnasium as gym
import torch


def _sanitize_isaac_runtime_env() -> None:
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        filtered_parts = []
        for part in ld_library_path.split(":"):
            if not part:
                continue
            if part.startswith("/usr/local/cuda-") or part == "/usr/local/cuda/lib64":
                continue
            filtered_parts.append(part)
        os.environ["LD_LIBRARY_PATH"] = ":".join(filtered_parts)

    for key in ("CUDA_HOME", "CUDA_PATH"):
        if key in os.environ:
            os.environ.pop(key)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for the DROID IsaacLab environment.")
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=10.0)
    args, _ = parser.parse_known_args()
    return args


def main(
    scene: int = 1,
    headless: bool = False,
    hold_seconds: float = 10.0,
) -> None:
    _sanitize_isaac_runtime_env()

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Smoke test for the DROID IsaacLab environment.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()

    policy_obs = obs["policy"]
    print("Smoke test passed.")
    print(f"scene={scene}")
    print(f"policy keys={sorted(policy_obs.keys())}")
    print(f"external_cam shape={tuple(policy_obs['external_cam'].shape)}")
    print(f"wrist_cam shape={tuple(policy_obs['wrist_cam'].shape)}")
    print(f"arm_joint_pos shape={tuple(policy_obs['arm_joint_pos'].shape)}")
    print(f"gripper_pos shape={tuple(policy_obs['gripper_pos'].shape)}")

    if not headless:
        print(f"Running random actions for {hold_seconds:.1f} seconds...")
        end_time = time.time() + hold_seconds
        while time.time() < end_time:
            action = env.action_space.sample()
            action = gym.spaces.utils.flatten(env.action_space, action)
            action = torch.as_tensor(action, device=env.unwrapped.device, dtype=torch.float32).reshape(1, -1)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args = _parse_args()
    main(
        scene=args.scene,
        headless=args.headless,
        hold_seconds=args.hold_seconds,
    )
