import time

import numpy as np
import torch
from PIL import Image
from openpi_client import websocket_client_policy, image_tools
from typing import List

from .abstract_client import InferenceClient
from .ikfk_utils import DroidFrankaIKFK, _load_pk

def compute_abs_eef_position(current_pose: np.ndarray, following_poses: List):
    pk = _load_pk()

    def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = (
            pk.euler_angles_to_matrix(torch.as_tensor(pose[3:], dtype=torch.float64), "XYZ")
            .detach()
            .cpu()
            .numpy()
        )
        transform[:3, 3] = pose[:3]
        return transform

    def matrix_to_pose(transform: np.ndarray) -> np.ndarray:
        rot = torch.as_tensor(transform[:3, :3], dtype=torch.float64).unsqueeze(0)
        euler = pk.matrix_to_euler_angles(rot, "XYZ")[0].detach().cpu().numpy()
        pos = transform[:3, 3]
        return np.concatenate([pos, euler], axis=0)

    current_transform = pose_to_matrix(current_pose)
    absolute_poses = []

    for following_pose in following_poses:
        following_pose = np.asarray(following_pose, dtype=np.float64)
        delta_transform = pose_to_matrix(following_pose)
        current_transform = current_transform @ delta_transform
        absolute_pose = matrix_to_pose(current_transform).astype(np.float32)
        absolute_poses.append(absolute_pose)

    return list(absolute_poses)


class Client(InferenceClient):
    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = 8,
        pose_encoding: str = "euler_xyz",
        ik_device: str = "cpu",
        max_ik_position_error_m: float = 0.05,
        max_ik_rotation_error_rad: float = 0.5,
    ) -> None:
        self.open_loop_horizon = open_loop_horizon
        self.client = websocket_client_policy.WebsocketClientPolicy(
            remote_host, remote_port
        )
        print("server metadata", self.client.get_server_metadata())
        self.kinematics = DroidFrankaIKFK(
            pose_encoding=pose_encoding,
            device=ik_device,
        )
        self.max_ik_position_error_m = max_ik_position_error_m
        self.max_ik_rotation_error_rad = max_ik_rotation_error_rad

        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
    
    def _clean_up_instruction(self, x: str) -> str:
        x = x.capitalize()
        return x

    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the policy in a server-client setup
        """
        instruction = self._clean_up_instruction(instruction)
        curr_obs = self._extract_observation(obs)
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            request_data = {
                "observation/exterior_image_0_left": curr_obs["left_image"],
                "observation/exterior_image_1_left": curr_obs["right_image"],
                "observation/wrist_image_left": curr_obs["wrist_image"],
                "observation/cartesian_position": curr_obs["cartesian_position"],
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            t0 = time.time()
            print("[CLIENT] waiting for remote inference.")
            pred_delta_action_chunk = self.client.infer(request_data)["actions"]
            print(f"[CLIENT] new action chunk received, time used {time.time() - t0:.2f}")

            cartesian_position = curr_obs['cartesian_position']
            pred_transform_action_chunk = compute_abs_eef_position(
                cartesian_position, 
                list(map(lambda x : x[:6], pred_delta_action_chunk))
            )
            self.pred_action_chunk = [
                torch.cat(pred_transform_action_chunk[i], pred_delta_action_chunk[-1:])
                for i in range(len(self.pred_action_chunk))
            ]

        policy_action = np.asarray(self.pred_action_chunk[self.actions_from_chunk_completed])
        self.actions_from_chunk_completed += 1

        if policy_action.shape[-1] != 7:
            raise ValueError(
                "Expected a 7D DROID Cartesian action chunk `[x, y, z, rx, ry, rz, gripper]`, "
                f"but got shape {policy_action.shape}."
            )

        target_cartesian = policy_action[:-1]
        ik_result = self.kinematics.ik(
            target_cartesian,
            seed_joint_position=curr_obs["joint_position"],
        )
        if (
            not ik_result.converged
            and (
                ik_result.err_pos > self.max_ik_position_error_m
                or ik_result.err_rot > self.max_ik_rotation_error_rad
            )
        ):
            arm_action = curr_obs["joint_position"]
        else:
            arm_action = ik_result.joint_position

        if policy_action[-1].item() > 0.5:
            gripper_action = np.ones((1,), dtype=np.float32)
        else:
            gripper_action = np.zeros((1,), dtype=np.float32)
        action = np.concatenate([arm_action, gripper_action]).astype(np.float32)

        both = np.concatenate([curr_obs["left_image"], curr_obs["wrist_image"], curr_obs["right_image"]], axis=1)
        return {"action": action, "viz": both}

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        # Assign images
        left_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        right_image = obs_dict["policy"]["external_cam_2"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()
        cartesian_position = self.kinematics.fk(joint_position)

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "left_image": left_image,
            "wrist_image": wrist_image,
            "cartesian_position": cartesian_position,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
