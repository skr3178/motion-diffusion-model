"""
SMPL utilities for pose visualization.
Adapted from external SMPL visualization code.
"""

import os
from pathlib import Path

import numpy as np
import torch

try:
    import dotenv
    import roma
    import smplx
    from smplx.utils import SMPLHOutput
    SMPL_AVAILABLE = True
except ImportError:
    SMPL_AVAILABLE = False
    SMPLHOutput = None  # Placeholder when not available
    print("Warning: SMPL dependencies not available. Install smplx, roma for SMPL visualization.")

# Try to load SMPL model path from environment
if SMPL_AVAILABLE:
    try:
        if os.getenv("SMPL_MODEL_PATH") is None:
            dotenv.load_dotenv()
        SMPL_BASE_DIR = Path(os.getenv("SMPL_MODEL_PATH", "./smpl_models"))
    except:
        SMPL_BASE_DIR = Path("./smpl_models")
else:
    SMPL_BASE_DIR = None

_BODY_MODEL = {
    "male": None,
    "female": None,
    "neutral": None,
}

JOINT_NAMES = [
    "pelvis",  # 0
    "left_hip",  # 1
    "right_hip",  # 2
    "spine1",  # 3
    "left_knee",  # 4
    "right_knee",  # 5
    "spine2",  # 6
    "left_ankle",  # 7
    "right_ankle",  # 8
    "spine3",  # 9
    "left_foot",  # 10
    "right_foot",  # 11
    "neck",  # 12
    "left_collar",  # 13
    "right_collar",  # 14
    "head",  # 15
    "left_shoulder",  # 16
    "right_shoulder",  # 17
    "left_elbow",  # 18
    "right_elbow",  # 19
    "left_wrist",  # 20
    "right_wrist",  # 21
]

# Primal root position when betas are all 0
BASE_ROOT_POSITION_MALE = np.array([-0.00217368, -0.24078918, 0.02858379])  # when using SMPL-H
BASE_ROOT_POSITION_FEMALE = np.array([-0.00087631, -0.21141872, 0.02782112])  # when using SMPL-H


class BodyModelWrapper:
    def __init__(self, num_betas=10, gender="neutral", device="cuda") -> None:
        if not SMPL_AVAILABLE:
            raise ImportError("SMPL dependencies not available")

        self.bm = smplx.create(
            SMPL_BASE_DIR,
            model_type="smplh",
            num_betas=num_betas,
            gender=gender,
            use_pca=False,
        ).to(device)
        self.device = device
        self.required_keys = [
            "transl",
            "global_orient",
            "body_pose",
            "left_hand_pose",
            "right_hand_pose",
            "betas",
        ]
        self.body_param_dims = {
            "transl": 3,
            "global_orient": 3,
            "body_pose": 63,
            "left_hand_pose": 45,
            "right_hand_pose": 45,
            "betas": num_betas,
        }

    def __call__(self, body_params=None, **body_params_kwargs) -> SMPLHOutput:
        if body_params is None:
            body_params = body_params_kwargs
        elif isinstance(body_params, torch.Tensor):
            body_params = {
                "transl": body_params[:, :3],
                "global_orient": roma.unitquat_to_rotvec(body_params[:, 3:7]),
                "body_pose": roma.unitquat_to_rotvec(body_params[:, 7:].reshape(-1, 21, 4)).reshape(-1, 63),
            }

        key = next(iter(body_params))  # any existing key in the body_params
        num_frames = body_params[key].shape[0]
        for key in self.required_keys:
            if key not in body_params:
                body_params[key] = torch.zeros(
                    (num_frames, self.body_param_dims[key]),
                    dtype=torch.float32,
                    device=self.device,
                )

        output: SMPLHOutput = self.bm(**body_params)
        return output

    def to(self, device):
        self.bm = self.bm.to(device)
        self.device = device
        return self

    @property
    def faces(self):
        return self.bm.faces


def load_body_model(num_betas=10, gender="neutral", device="cuda") -> BodyModelWrapper:
    """Load SMPL body model."""
    global _BODY_MODEL
    if _BODY_MODEL[gender] is None:
        _BODY_MODEL[gender] = BodyModelWrapper(num_betas, gender, device)
    return _BODY_MODEL[gender]


def convert_pose_format(poses):
    """
    Convert our 159-dim pose format to SMPL-compatible format.
    Our format: [156 pose params + 3 translation]
    SMPL expects: dictionary with transl, global_orient, body_pose, etc.
    """
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses).float()

    batch_size = poses.shape[0]
    device = poses.device

    # Extract translation (last 3 values)
    transl = poses[:, -3:]

    # Extract pose parameters (first 156 values)
    pose_params = poses[:, :-3]

    # Convert to SMPL format
    # Assuming pose_params are in axis-angle format
    # First 3: global orientation, rest: body pose (21 joints * 3 each = 63)
    if pose_params.shape[1] >= 66:  # 3 + 63
        global_orient = pose_params[:, :3]
        body_pose = pose_params[:, 3:66]
    else:
        # If we don't have enough parameters, pad with zeros
        global_orient = torch.zeros(batch_size, 3, device=device)
        body_pose = torch.zeros(batch_size, 63, device=device)
        available_params = min(pose_params.shape[1], 63)
        if available_params > 0:
            body_pose[:, :available_params] = pose_params[:, :available_params]

    return {
        "transl": transl,
        "global_orient": global_orient,
        "body_pose": body_pose,
    }


def get_root_positions(body_params):
    """Get root positions from body parameters."""
    if isinstance(body_params, dict):
        transl = body_params["transl"]
    else:
        transl = body_params[:, :3]

    base_root_position = torch.tensor(BASE_ROOT_POSITION_MALE, dtype=torch.float32, device=transl.device)
    root_positions = transl + base_root_position
    return root_positions


def get_facing_directions(body_params):
    """Get facing directions from body parameters."""
    if isinstance(body_params, dict):
        global_orient = body_params["global_orient"]
        # Convert axis-angle to rotation matrix
        global_orients = roma.rotvec_to_rotmat(global_orient)
    else:
        global_orients = roma.unitquat_to_rotmat(body_params[:, 3:7])

    facing_directions = global_orients[:, :, 2]  # extract local z-axis
    facing_directions[..., 2] = 0
    facing_directions /= torch.norm(facing_directions, dim=-1, keepdim=True)
    return facing_directions