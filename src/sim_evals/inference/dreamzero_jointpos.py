import uuid

import numpy as np
from openpi_client import image_tools

from .abstract_client import InferenceClient
from .dreamzero_websocket_client import WebsocketClientPolicy


class Client(InferenceClient):
    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 6000,
        open_loop_horizon: int = 8,
    ) -> None:
        self.client = WebsocketClientPolicy(remote_host, remote_port)
        self.open_loop_horizon = open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())

    def visualize(self, request: dict):
        curr_obs = self._extract_observation(request)
        right_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        left_img = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        return np.concatenate([right_img, wrist_img, left_img], axis=1)

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())

    def infer(self, obs: dict, instruction: str) -> dict:
        curr_obs = self._extract_observation(obs)
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            request_data = {
                "observation/exterior_image_0_left": image_tools.resize_with_pad(
                    curr_obs["right_image"], 180, 320
                ),
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    curr_obs["left_image"], 180, 320
                ),
                "observation/wrist_image_left": image_tools.resize_with_pad(
                    curr_obs["wrist_image"], 180, 320
                ),
                "observation/joint_position": curr_obs["joint_position"].astype(np.float64),
                "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                "observation/gripper_position": curr_obs["gripper_position"].astype(np.float64),
                "prompt": instruction,
                "session_id": self.session_id,
            }
            self.pred_action_chunk = self.client.infer(request_data)

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        right_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        left_img = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        viz = np.concatenate([right_img, wrist_img, left_img], axis=1)
        return {"action": action, "viz": viz}

    def _extract_observation(self, obs_dict):
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        left_image = obs_dict["policy"]["external_cam_2"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        return {
            "right_image": right_image,
            "left_image": left_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
