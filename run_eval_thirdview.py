"""
Run DROID evaluation with the 3-view joint-position client.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import mediapy
import torch
import tyro
from tqdm import tqdm

from sim_evals.inference.droid_jointpos_thirdview import Client as DroidJointPosThirdViewClient


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


def main(
    episodes: int = 10,
    headless: bool = False,
    scene: int = 1,
    remote_host: str = "localhost",
    remote_port: int = 8000,
    open_loop_horizon: int = 8,
):
    _sanitize_isaac_runtime_env()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Run DROID evaluation with StarVLA third-view joint-position client.")
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

    if scene == 1:
        instruction = "put the cube in the bowl"
    elif scene == 2:
        instruction = "put the can in the mug"
    elif scene == 3:
        instruction = "put banana in the bin"
    else:
        raise ValueError(f"Scene {scene} not supported")

    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()
    client = DroidJointPosThirdViewClient(
        remote_host=remote_host,
        remote_port=remote_port,
        open_loop_horizon=open_loop_horizon,
    )

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    max_steps = env.env.max_episode_length
    with torch.no_grad():
        for ep in range(episodes):
            for _ in tqdm(range(max_steps), desc=f"Episode {ep + 1}/{episodes}"):
                ret = client.infer(obs, instruction)
                if not headless:
                    cv2.imshow("DROID Third-View Client", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            client.reset()
            mediapy.write_video(
                video_dir / f"episode_{ep}.mp4",
                video,
                fps=15,
            )
            video = []

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
