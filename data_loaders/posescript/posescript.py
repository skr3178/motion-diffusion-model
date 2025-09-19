import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import json
from tqdm import tqdm


class PoseScript(data.Dataset):
    """
    PoseScript dataset for text-to-pose generation.
    Each item consists of a text description and corresponding SMPL pose parameters.
    """

    def __init__(self, split='train', num_frames=None, abs_path='.'):
        self.split = split
        self.abs_path = abs_path

        # PoseScript data paths
        self.posescript_dir = pjoin(abs_path, 'dataset', 'posescript_release')
        self.amass_dir = pjoin(abs_path, 'dataset', 'AMASS', 'SMPL-H-G')

        # Load mapping files
        ids_file = pjoin(self.posescript_dir, 'ids_2_dataset_sequence_and_frame_index_100k.json')
        text_file = pjoin(self.posescript_dir, 'posescript_human_6293.json')

        with open(ids_file, 'r') as f:
            self.ids_mapping = json.load(f)

        with open(text_file, 'r') as f:
            self.text_data = json.load(f)

        # Filter to only include IDs that have both mapping and text
        self.data_ids = []
        for id_str in self.ids_mapping.keys():
            if id_str in self.text_data:
                self.data_ids.append(id_str)

        print(f"PoseScript dataset loaded: {len(self.data_ids)} poses")

        # For now, use simple train/test split (80/20)
        split_idx = int(len(self.data_ids) * 0.8)
        if split == 'train':
            self.data_ids = self.data_ids[:split_idx]
        else:  # test or val
            self.data_ids = self.data_ids[split_idx:]

        print(f"Using {split} split: {len(self.data_ids)} poses")

    def _fix_amass_path(self, file_path):
        """Fix mismatched folder names between PoseScript and actual AMASS structure"""

        # Common path fixes
        path_mappings = {
            'Eyes_Japan_Dataset/': 'EyesJapanDataset/',
            'BioMotionLab_NTroje/': 'BMLrub/',
            'MPI_Limits/': 'PosePrior/',
            'MPI_HDM05/': 'HDM05/',
            'MPI_mosh/': 'MoSh/',
            'SSM_synced/': 'SSM/',
            'Transitions_mocap/': 'Transitions/',
            'DFaust_67/': 'DFaust/',
        }

        fixed_path = file_path
        for old_path, new_path in path_mappings.items():
            if old_path in fixed_path:
                fixed_path = fixed_path.replace(old_path, new_path)
                break

        return fixed_path

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        id_str = self.data_ids[idx]

        # Get AMASS file path and frame index
        dataset_name, file_path, frame_idx = self.ids_mapping[id_str]

        # Get text description
        text_description = self.text_data[id_str][0]  # First description

        # Fix path mapping for mismatched folder names
        fixed_file_path = self._fix_amass_path(file_path)

        # Load AMASS pose data
        amass_file = pjoin(self.amass_dir, fixed_file_path)

        try:
            amass_data = np.load(amass_file)

            # Extract pose parameters for the specific frame
            poses = amass_data['poses']  # Shape: (n_frames, 156)
            trans = amass_data['trans']  # Shape: (n_frames, 3)

            if frame_idx >= len(poses):
                # If frame index is out of bounds, use the last frame
                frame_idx = len(poses) - 1

            pose = poses[frame_idx]  # Shape: (156,)
            translation = trans[frame_idx]  # Shape: (3,)

            # Combine pose and translation
            # pose_data = np.concatenate([pose, translation])  # Shape: (159,)
            pose_data = np.concatenate([translation, pose])  # Shape: (159,)

            return {
                'pose': torch.from_numpy(pose_data).float(),
                'text': text_description,
                'length': 1,  # Single pose
                'id': id_str
            }

        except Exception as e:
            print(f"Error loading {amass_file}, frame {frame_idx}: {e}")
            # Return a zero pose as fallback
            pose_data = np.zeros(159)  # 156 pose + 3 translation
            return {
                'pose': torch.from_numpy(pose_data).float(),
                'text': "unknown pose",
                'length': 1,
                'id': id_str
            }


def collate_fn(batch):
    """Collate function for PoseScript dataset - compatible with TrainLoop"""
    poses = torch.stack([item['pose'] for item in batch])  # [bs, 159]
    texts = [item['text'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])  # [bs] - all 1s for poses
    ids = [item['id'] for item in batch]

    # Convert poses to motion format: [bs, 159] -> [bs, njoints=1, nfeats=159, nframes=1]
    bs = poses.shape[0]
    motion = poses.unsqueeze(1).unsqueeze(-1)  # [bs, 1, 159, 1]

    # Create mask for single poses (all True since we have single poses)
    mask = torch.ones(bs, 1, 1, 1, dtype=torch.bool)  # [bs, 1, 1, 1]

    # Create conditioning dict in expected format
    cond = {
        'y': {
            'text': texts,
            'mask': mask,
            'lengths': lengths
        }
    }

    return motion, cond