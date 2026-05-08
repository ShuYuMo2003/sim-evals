import tyro
import numpy as np
from PIL import Image
from openpi_client import websocket_client_policy, image_tools

from .abstract_client import InferenceClient


class Client(InferenceClient):
    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = 8,
    ) -> None:
        self.open_loop_horizon = open_loop_horizon
        self.client = websocket_client_policy.WebsocketClientPolicy(remote_host, remote_port)

        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.control_step = 0
        self.history_steps = []
        self.history_frames = []

    def visualize(self, request: dict):
        curr_obs = self._extract_observation(request)
        right_img = image_tools.resize_with_pad(curr_obs["right_image"], 180, 320)
        left_img = image_tools.resize_with_pad(curr_obs["left_image"], 180, 320)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 180, 320)
        combined = np.concatenate([right_img, left_img, wrist_img], axis=1)
        return combined

    def reset(self):
        self.client.infer({"reset": True})
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.control_step = 0
        self.history_steps = []
        self.history_frames = []

    def infer(self, obs: dict, instruction: str) -> dict:
        curr_obs = self._extract_observation(obs)
        right_img = image_tools.resize_with_pad(curr_obs["right_image"], 180, 320)
        left_img = image_tools.resize_with_pad(curr_obs["left_image"], 180, 320)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 180, 320)
        stitched_frame = np.concatenate([right_img, left_img, wrist_img], axis=0)
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            executed_count = self.actions_from_chunk_completed
            self.actions_from_chunk_completed = 0
            request_data = {
                "observation/exterior_image_0_left": right_img,
                "observation/exterior_image_1_left": left_img,
                "observation/wrist_image_left": wrist_img,
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
                "control_step": self.control_step,
                "history/executed_action_count": executed_count,
                "history/step_indices": np.asarray(self.history_steps, dtype=np.int64),
                "history/stitched_frames": list(self.history_frames),
            }
            self.pred_action_chunk = self.client.infer(request_data)["actions"]

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1
        self.history_steps.append(self.control_step)
        self.history_frames.append(stitched_frame)
        if len(self.history_steps) > 256:
            self.history_steps = self.history_steps[-256:]
            self.history_frames = self.history_frames[-256:]
        self.control_step += 1

        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        viz = np.concatenate([right_img, left_img, wrist_img], axis=1)

        return {"action": action, "viz": viz}

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        left_image = obs_dict["policy"]["external_cam_2"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, left_image, wrist_image], axis=1)
            Image.fromarray(combined_image).save("robot_camera_views.png")

        return {
            "left_image": left_image,
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }


if __name__ == "__main__":
    import torch

    args = tyro.cli(Client)
    client = Client(args)
    fake_obs = {
        "policy": {
            "external_cam": np.zeros((1, 180, 320, 3), dtype=np.uint8),
            "external_cam_2": np.zeros((1, 180, 320, 3), dtype=np.uint8),
            "wrist_cam": np.zeros((1, 180, 320, 3), dtype=np.uint8),
            "arm_joint_pos": torch.zeros((7,), dtype=torch.float32),
            "gripper_pos": torch.zeros((1,), dtype=torch.float32),
        },
    }
    fake_instruction = "pick up the object"

    import time

    start = time.time()
    client.infer(fake_obs, fake_instruction)
    num = 20
    for _ in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
