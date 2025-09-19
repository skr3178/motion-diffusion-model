"""
Generate pose samples from a trained pose diffusion model.
"""

import os
import numpy as np
import torch
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import data_loaders.humanml.utils.paramUtil as paramUtil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to model checkpoint")
    parser.add_argument("--text_prompt", default="", type=str, help="Text prompt for pose generation")
    parser.add_argument("--num_samples", default=10, type=int, help="Number of poses to generate")
    parser.add_argument("--output_dir", default="", type=str, help="Output directory")
    parser.add_argument("--seed", default=10, type=int, help="Random seed")
    parser.add_argument("--device", default=0, type=int, help="Device id to use")
    parser.add_argument("--guidance_param", default=2.5, type=float, help="Classifier-free guidance scale")
    parser.add_argument("--visualize", action="store_true", help="Generate 3D pose visualizations")

    args = parser.parse_args()

    fixseed(args.seed)

    # Setup output directory
    if args.output_dir == "":
        model_name = os.path.basename(os.path.dirname(args.model_path))
        args.output_dir = f"save/{model_name}/pose_samples_seed{args.seed}"
        if args.text_prompt:
            args.output_dir += "_" + args.text_prompt.replace(" ", "_")

    os.makedirs(args.output_dir, exist_ok=True)

    dist_util.setup_dist(args.device)

    print("Loading pose model...")

    # Create args for model creation (matching the checkpoint's architecture)
    class ModelArgs:
        def __init__(self):
            self.dataset = 'posescript'
            self.latent_dim = 512  # Match checkpoint dimension
            self.layers = 8
            self.arch = 'trans_enc'
            self.emb_trans_dec = False
            self.text_encoder_type = 'clip'
            self.pos_embed_max_len = 5000
            self.mask_frames = True
            self.cond_mask_prob = 0.1
            self.diffusion_steps = 1000
            self.noise_schedule = 'cosine'
            # Additional required attributes
            self.sigma_small = True
            self.lambda_vel = 0.0
            self.lambda_rcxyz = 0.0
            self.lambda_fc = 0.0

    model_args = ModelArgs()

    # Create dummy data loader for model creation
    from data_loaders.get_data import get_dataset_loader
    data = get_dataset_loader('posescript', batch_size=1, num_frames=1)

    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(model_args, data)

    # Load checkpoint
    load_saved_model(model, args.model_path)
    model.to(dist_util.dev())
    model.eval()

    print(f"Model type: {model.modeltype}")
    assert model.modeltype == 'pose', f"Expected pose model, got {model.modeltype}"

    print(f"Generating {args.num_samples} poses in batch...")

    # Use provided text or default
    if args.text_prompt:
        text = args.text_prompt
    else:
        text = "a person standing in a neutral pose"

    # Generate poses in batch
    poses = sample_poses_batch(model, diffusion, text, args.num_samples, args.guidance_param)

    # Convert to numpy and prepare texts
    all_poses = poses.squeeze(-1).squeeze(1).cpu().numpy()  # [batch_size, 159] for saving
    all_texts = [text] * args.num_samples

    # Save individual poses
    for i in range(args.num_samples):
        pose_file = os.path.join(args.output_dir, f"pose_{i:03d}.npy")
        np.save(pose_file, all_poses[i])

        # Save text description
        text_file = os.path.join(args.output_dir, f"pose_{i:03d}.txt")
        with open(text_file, 'w') as f:
            f.write(text)
    np.save(os.path.join(args.output_dir, "all_poses.npy"), all_poses)

    # Save all text descriptions
    with open(os.path.join(args.output_dir, "all_texts.txt"), 'w') as f:
        for i, text in enumerate(all_texts):
            f.write(f"{i:03d}: {text}\n")

    print(f"Generated poses saved to: {args.output_dir}")
    print(f"Pose shape: {all_poses.shape}")  # Should be [num_samples, 159]

    # Generate visualizations if requested
    if args.visualize:
        print("Starting 3D pose visualization...")
        import subprocess
        import sys

        # Path to visualization script
        viz_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualize", "visualize_poses.py")
        pose_file = os.path.join(args.output_dir, "all_poses.npy")
        device_str = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"

        try:
            # Launch visualization in subprocess
            cmd = [
                sys.executable, viz_script,
                "--pose_file", pose_file,
                "--device", device_str,
                "--gender", "neutral"
            ]

            print(f"Launching visualization: {' '.join(cmd)}")
            subprocess.run(cmd)

        except Exception as e:
            print(f"Error launching visualization: {e}")
            print("You can manually run visualization with:")
            print(f"python visualize/visualize_poses.py --pose_file {pose_file}")

    print("Done!")


def sample_poses_batch(model, diffusion, text, batch_size, guidance_param=2.5):
    """Sample multiple poses from the model in batch"""

    device = next(model.parameters()).device

    # Sample noise - use motion format [bs, njoints=1, nfeats=159, nframes=1]
    pose_shape = (batch_size, 1, 159, 1)  # [batch_size, njoints, nfeats, nframes]
    noise = torch.randn(pose_shape, device=device)

    # Pre-encode text to avoid issues in p_sample_loop
    text_embed = model.encode_text([text] * batch_size)

    # Prepare conditioning in expected format
    model_kwargs = {
        'y': {
            'text': [text] * batch_size,  # Keep text for compatibility
            'text_embed': text_embed,
            'mask': torch.ones(batch_size, 1, 1, 1, dtype=torch.bool, device=device),
            'lengths': torch.tensor([1] * batch_size, device=device)
        }
    }

    # Classifier-free guidance
    if guidance_param > 1.0:
        # Unconditional sample
        uncond_text_embed = model.encode_text([''] * batch_size)
        uncond_kwargs = {
            'y': {
                'text': [''] * batch_size,  # Keep text for compatibility
                'text_embed': uncond_text_embed,
                'mask': torch.ones(batch_size, 1, 1, 1, dtype=torch.bool, device=device),
                'lengths': torch.tensor([1] * batch_size, device=device),
                'uncond': True
            }
        }

        def model_fn(x_t, ts, **kwargs):
            # Conditional prediction
            cond_pred = model(x_t, ts, **model_kwargs)

            # Unconditional prediction
            uncond_pred = model(x_t, ts, **uncond_kwargs)

            # Classifier-free guidance
            pred = uncond_pred + guidance_param * (cond_pred - uncond_pred)
            return pred

        sample_fn = model_fn
    else:
        sample_fn = model

    # Sample
    poses = diffusion.p_sample_loop(
        sample_fn,
        pose_shape,
        noise=noise,
        model_kwargs=model_kwargs,
        device=device,
        progress=True
    )

    # Return poses in motion format [batch_size, 1, 159, 1] for rot2xyz compatibility
    return poses


def sample_pose(model, diffusion, text, guidance_param=2.5):
    """Sample a single pose from the model"""

    device = next(model.parameters()).device

    # Sample noise - use motion format [bs, njoints=1, nfeats=159, nframes=1]
    pose_shape = (1, 1, 159, 1)  # [batch_size, njoints, nfeats, nframes]
    noise = torch.randn(pose_shape, device=device)

    # Pre-encode text to avoid issues in p_sample_loop
    text_embed = model.encode_text([text])

    # Prepare conditioning in expected format
    model_kwargs = {
        'y': {
            'text': [text],  # Keep text for compatibility
            'text_embed': text_embed,
            'mask': torch.ones(1, 1, 1, 1, dtype=torch.bool, device=device),
            'lengths': torch.tensor([1], device=device)
        }
    }

    # Classifier-free guidance
    if guidance_param > 1.0:
        # Unconditional sample
        uncond_text_embed = model.encode_text([''])
        uncond_kwargs = {
            'y': {
                'text': [''],  # Keep text for compatibility
                'text_embed': uncond_text_embed,
                'mask': torch.ones(1, 1, 1, 1, dtype=torch.bool, device=device),
                'lengths': torch.tensor([1], device=device),
                'uncond': True
            }
        }

        def model_fn(x_t, ts, **kwargs):
            # Conditional prediction
            cond_pred = model(x_t, ts, **model_kwargs)

            # Unconditional prediction
            uncond_pred = model(x_t, ts, **uncond_kwargs)

            # Classifier-free guidance
            pred = uncond_pred + guidance_param * (cond_pred - uncond_pred)
            return pred

        sample_fn = model_fn
    else:
        sample_fn = model

    # Sample
    pose = diffusion.p_sample_loop(
        sample_fn,
        pose_shape,
        noise=noise,
        model_kwargs=model_kwargs,
        device=device,
        progress=True
    )

    # Extract pose from motion format [1, 1, 159, 1] -> [159]
    return pose.squeeze()  # Remove all singleton dimensions


if __name__ == "__main__":
    main()