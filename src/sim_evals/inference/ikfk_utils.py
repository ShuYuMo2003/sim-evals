from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

PoseEncoding = Literal["axis_angle", "euler_xyz"]

_DROID_FRANKA_URDF = (
    Path(__file__).resolve().parents[3] / "assets/robots/droid/franka_panda/panda_arm.urdf"
)
_DROID_FRANKA_END_LINK = "panda_link8"
_DROID_FRANKA_HOME = np.array(
    [0.0, -np.pi / 5.0, 0.0, -4.0 * np.pi / 5.0, 0.0, 3.0 * np.pi / 5.0, 0.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class IKResult:
    joint_position: np.ndarray
    converged: bool
    err_pos: float
    err_rot: float


def _load_pk():
    try:
        import pytorch_kinematics as pk
    except ImportError as exc:
        raise ImportError(
            "pytorch-kinematics is required for DROID IK/FK. Run `uv sync` first."
        ) from exc
    return pk


class DroidFrankaIKFK:
    """Thin FK/IK wrapper for the DROID Franka arm.

    We use the local Panda URDF shipped with the sim assets and expose a 6D
    Cartesian pose vector as `[x, y, z, rx, ry, rz]`.
    By default the rotation part is axis-angle, matching the most common
    end-effector state/action encoding used in DROID-style pipelines.
    """

    def __init__(
        self,
        *,
        urdf_path: str | Path = _DROID_FRANKA_URDF,
        end_link_name: str = _DROID_FRANKA_END_LINK,
        pose_encoding: PoseEncoding = "axis_angle",
        tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        tcp_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        ik_max_iterations: int = 60,
        ik_pos_tolerance: float = 1e-4,
        ik_rot_tolerance: float = 2e-3,
        ik_lr: float = 0.25,
    ) -> None:
        self.pk = _load_pk()
        self.pose_encoding = pose_encoding
        self.device = torch.device(device)
        self.dtype = dtype
        self.ik_max_iterations = ik_max_iterations
        self.ik_pos_tolerance = ik_pos_tolerance
        self.ik_rot_tolerance = ik_rot_tolerance
        self.ik_lr = ik_lr

        urdf_path = Path(urdf_path)
        urdf_text = urdf_path.read_bytes()
        self.chain = self.pk.build_serial_chain_from_urdf(urdf_text, end_link_name=end_link_name).to(
            device=self.device, dtype=self.dtype
        )
        self.joint_limits = (
            torch.as_tensor(self.chain.get_joint_limits(), dtype=self.dtype, device=self.device).T
        )
        self.home_joint_position = torch.as_tensor(
            _DROID_FRANKA_HOME, dtype=self.dtype, device=self.device
        )

        tcp_rot = self.pk.euler_angles_to_matrix(
            torch.as_tensor(tcp_rpy, dtype=self.dtype, device=self.device), "XYZ"
        )
        self._tcp_transform = self.pk.Transform3d(
            pos=torch.as_tensor(tcp_offset, dtype=self.dtype, device=self.device),
            rot=tcp_rot,
            device=self.device,
            dtype=self.dtype,
        )
        self._tcp_inverse = self._tcp_transform.inverse()

    def fk(self, joint_position: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.forward_kinematics(joint_position)

    def ik(
        self,
        target_pose: np.ndarray | torch.Tensor,
        *,
        seed_joint_position: np.ndarray | torch.Tensor,
    ) -> IKResult:
        return self.inverse_kinematics(target_pose, seed_joint_position=seed_joint_position)

    def forward_kinematics(self, joint_position: np.ndarray | torch.Tensor) -> np.ndarray:
        joint_tensor = self._as_joint_tensor(joint_position)
        target_tf = self.chain.forward_kinematics(joint_tensor).compose(self._tcp_transform)
        pose = self._matrix_to_pose(target_tf.get_matrix())
        return self._maybe_squeeze_numpy(pose, joint_position)

    def inverse_kinematics(
        self,
        target_pose: np.ndarray | torch.Tensor,
        *,
        seed_joint_position: np.ndarray | torch.Tensor,
    ) -> IKResult:
        seed_joint_tensor = self._as_joint_tensor(seed_joint_position)
        if seed_joint_tensor.shape[0] != 1:
            raise ValueError("IK currently expects a single seed joint configuration.")

        goal_pose = self._pose_to_transform(target_pose).compose(self._tcp_inverse)
        retry_configs = self._make_retry_configs(seed_joint_tensor[0])
        solver = self.pk.PseudoInverseIK(
            self.chain,
            retry_configs=retry_configs,
            joint_limits=self.joint_limits,
            pos_tolerance=self.ik_pos_tolerance,
            rot_tolerance=self.ik_rot_tolerance,
            max_iterations=self.ik_max_iterations,
            lr=self.ik_lr,
            early_stopping_any_converged=True,
            early_stopping_no_improvement="all",
            clamp_to_limits=True,
            debug=False,
        )
        solution = solver.solve(goal_pose)
        best_retry = self._select_best_retry(solution)
        return IKResult(
            joint_position=solution.solutions[0, best_retry].detach().cpu().numpy().astype(np.float32),
            converged=bool(solution.converged[0, best_retry].item()),
            err_pos=float(solution.err_pos[0, best_retry].item()),
            err_rot=float(solution.err_rot[0, best_retry].item()),
        )

    def _make_retry_configs(self, seed_joint_position: torch.Tensor) -> torch.Tensor:
        center = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2.0
        retries = torch.stack(
            [
                seed_joint_position,
                self.home_joint_position,
                0.5 * (seed_joint_position + self.home_joint_position),
                0.75 * seed_joint_position + 0.25 * self.home_joint_position,
                center,
            ],
            dim=0,
        )
        return torch.clamp(retries, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def _select_best_retry(self, solution) -> int:
        score = solution.err_pos[0] / self.ik_pos_tolerance + solution.err_rot[0] / self.ik_rot_tolerance
        if solution.converged[0].any():
            converged_indices = torch.nonzero(solution.converged[0], as_tuple=False).flatten()
            best_local = torch.argmin(score[converged_indices]).item()
            return int(converged_indices[best_local].item())
        return int(torch.argmin(score).item())

    def _pose_to_transform(self, pose: np.ndarray | torch.Tensor):
        pose_tensor = self._as_pose_tensor(pose)
        rot_vec = pose_tensor[:, 3:]
        if self.pose_encoding == "axis_angle":
            rot = self.pk.quaternion_to_matrix(self.pk.axis_angle_to_quaternion(rot_vec))
        elif self.pose_encoding == "euler_xyz":
            rot = self.pk.euler_angles_to_matrix(rot_vec, "XYZ")
        else:
            raise ValueError(f"Unsupported pose encoding: {self.pose_encoding}")
        return self.pk.Transform3d(
            pos=pose_tensor[:, :3],
            rot=rot,
            device=self.device,
            dtype=self.dtype,
        )

    def _matrix_to_pose(self, matrix: torch.Tensor) -> torch.Tensor:
        pos = matrix[:, :3, 3]
        rot = matrix[:, :3, :3]
        if self.pose_encoding == "axis_angle":
            rot_vec = self.pk.quaternion_to_axis_angle(self.pk.matrix_to_quaternion(rot))
        elif self.pose_encoding == "euler_xyz":
            rot_vec = self.pk.matrix_to_euler_angles(rot, "XYZ")
        else:
            raise ValueError(f"Unsupported pose encoding: {self.pose_encoding}")
        return torch.cat([pos, rot_vec], dim=-1)

    def _as_joint_tensor(self, joint_position: np.ndarray | torch.Tensor) -> torch.Tensor:
        joint_tensor = torch.as_tensor(joint_position, dtype=self.dtype, device=self.device)
        if joint_tensor.ndim == 1:
            joint_tensor = joint_tensor.unsqueeze(0)
        if joint_tensor.shape[-1] != 7:
            raise ValueError(f"Expected 7 joint values, got shape {tuple(joint_tensor.shape)}")
        return joint_tensor

    def _as_pose_tensor(self, pose: np.ndarray | torch.Tensor) -> torch.Tensor:
        pose_tensor = torch.as_tensor(pose, dtype=self.dtype, device=self.device)
        if pose_tensor.ndim == 1:
            pose_tensor = pose_tensor.unsqueeze(0)
        if pose_tensor.shape[-1] != 6:
            raise ValueError(f"Expected 6D Cartesian pose, got shape {tuple(pose_tensor.shape)}")
        return pose_tensor

    @staticmethod
    def _maybe_squeeze_numpy(value: torch.Tensor, reference: np.ndarray | torch.Tensor) -> np.ndarray:
        array = value.detach().cpu().numpy().astype(np.float32)
        if np.asarray(reference).ndim == 1:
            return array[0]
        return array
