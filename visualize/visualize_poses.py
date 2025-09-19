"""
3D visualization of generated SMPL poses using interactive viewer.
"""

import argparse
import os
import numpy as np
import torch

try:
    import viser
    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False

# Import our SMPL utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.smpl_utils import load_body_model, convert_pose_format, SMPL_AVAILABLE


class PoseVisualizer:
    def __init__(self, args):
        if not VISER_AVAILABLE:
            raise ImportError("viser not available. Install with: pip install viser")
        if not SMPL_AVAILABLE:
            raise ImportError("SMPL dependencies not available. Install with: pip install smplx roma")

        self.device = args.device
        self.server = viser.ViserServer()

        # Load SMPL model
        try:
            self.body_model = load_body_model(gender=args.gender, device=self.device)
            print(f"Loaded SMPL {args.gender} model")
        except Exception as e:
            print(f"Error loading SMPL model: {e}")
            print("Make sure SMPL models are available and SMPL_MODEL_PATH is set")
            raise

        # Load generated poses
        self.poses = self.load_poses(args.pose_file)
        self.num_poses = len(self.poses)
        print(f"Loaded {self.num_poses} poses from {args.pose_file}")

        # Initialize viewer
        self.setup_scene()
        self.setup_gui()

    def load_poses(self, pose_file):
        """Load generated poses and convert to SMPL format."""
        if pose_file.endswith('.npy'):
            poses = np.load(pose_file)
            print(f"Loaded poses with shape: {poses.shape}")
        else:
            raise ValueError("Pose file must be .npy format")

        # Convert to tensor
        poses = torch.from_numpy(poses).float().to(self.device)
        return poses

    def setup_scene(self):
        """Setup 3D scene."""
        # Add coordinate axes
        self.server.scene.add_frame("world", wxyz=(1, 0, 0, 0), position=(0, 0, 0))

        # Add ground plane
        ground_vertices = np.array([
            [-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]
        ], dtype=np.float32)
        ground_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

        self.server.scene.add_mesh_simple(
            "ground",
            vertices=ground_vertices,
            faces=ground_faces,
            color=(0.9, 0.9, 0.9)
        )

    def setup_gui(self):
        """Setup interactive GUI controls."""
        # Add title
        self.server.gui.add_markdown("# SMPL Pose Visualization")
        self.server.gui.add_markdown(f"**Total poses:** {self.num_poses}")

        # Add pose selector
        if self.num_poses > 1:
            self.pose_idx_handle = self.server.gui.add_slider(
                "Pose Index",
                min=0,
                max=self.num_poses - 1,
                step=1,
                initial_value=0
            )

            # Update callback
            @self.pose_idx_handle.on_update
            def _update_pose():
                self.update_pose_visualization(self.pose_idx_handle.value)
        else:
            self.server.gui.add_markdown("Single pose loaded")

        # Add controls
        self.wireframe_handle = self.server.gui.add_checkbox("Wireframe", initial_value=False)
        @self.wireframe_handle.on_update
        def _update_wireframe():
            self.update_pose_visualization(getattr(self, 'current_pose_idx', 0))

        # Initialize with first pose
        self.current_pose_idx = 0
        self.update_pose_visualization(0)

    def update_pose_visualization(self, idx):
        """Update the displayed pose."""
        if idx < 0 or idx >= self.num_poses:
            return

        self.current_pose_idx = idx

        try:
            # Get single pose
            pose = self.poses[idx:idx+1]  # Keep batch dimension

            # Convert pose format
            smpl_params = convert_pose_format(pose)

            # Generate SMPL mesh
            output = self.body_model(smpl_params)
            vertices = output.vertices[0].cpu().numpy()  # First (and only) pose
            faces = self.body_model.faces.cpu().numpy()

            # Update visualization
            wireframe = self.wireframe_handle.value
            if wireframe:
                # Show wireframe
                self.server.scene.add_mesh_simple(
                    "pose_mesh",
                    vertices=vertices,
                    faces=faces,
                    color=(0.8, 0.4, 0.4),
                    wireframe=True
                )
            else:
                # Show solid mesh
                self.server.scene.add_mesh_simple(
                    "pose_mesh",
                    vertices=vertices,
                    faces=faces,
                    color=(0.8, 0.4, 0.4)
                )

            # Add pose info
            translation = smpl_params["transl"][0].cpu().numpy()
            self.server.gui.add_markdown(
                f"**Pose {idx}**  \n"
                f"Position: ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})"
            )

            print(f"Updated visualization to pose {idx}")

        except Exception as e:
            print(f"Error updating pose visualization: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Start the visualization server."""
        print("Starting SMPL pose visualization server...")
        print("Open your browser and go to the URL shown above")
        print("Press Ctrl+C to stop")

        # Keep server running
        try:
            while True:
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Shutting down visualization server...")


def main():
    parser = argparse.ArgumentParser(description="Visualize generated SMPL poses")
    parser.add_argument("--pose_file", type=str, required=True,
                       help="Path to .npy file containing generated poses")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--gender", type=str, default="neutral", choices=["male", "female", "neutral"],
                       help="SMPL model gender")

    args = parser.parse_args()

    # Check if pose file exists
    if not os.path.exists(args.pose_file):
        print(f"Error: Pose file not found: {args.pose_file}")
        return

    # Check dependencies
    if not VISER_AVAILABLE:
        print("Error: viser not available. Install with: pip install viser")
        return

    if not SMPL_AVAILABLE:
        print("Error: SMPL dependencies not available.")
        print("Install with: pip install smplx roma python-dotenv")
        return

    # Create and run visualizer
    try:
        visualizer = PoseVisualizer(args)
        visualizer.run()
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()