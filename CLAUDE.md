# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

The codebase uses conda for environment management. Set up the environment:

```bash
conda env create -f environment.yml
conda activate mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

**Additional dependencies by task type:**
- Text-to-Motion: `bash prepare/download_smpl_files.sh && bash prepare/download_glove.sh && bash prepare/download_t2m_evaluators.sh`
- Action-to-Motion: `bash prepare/download_smpl_files.sh && bash prepare/download_recognition_models.sh`
- Unconstrained: `bash prepare/download_smpl_files.sh && bash prepare/download_recognition_models.sh && bash prepare/download_recognition_unconstrained_models.sh`

## Core Architecture

### Project Structure
- `model/` - Core model implementations including MDM and DiP architectures
- `diffusion/` - Diffusion process implementations (gaussian diffusion, sampling, etc.)
- `data_loaders/` - Dataset handling for HumanML3D, KIT-ML, HumanAct12, UESTC
- `train/` - Training scripts and loops
- `sample/` - Generation and sampling utilities
- `eval/` - Evaluation scripts and metrics
- `visualize/` - Visualization tools including SMPL mesh rendering
- `utils/` - Utilities for parsing, model creation, and data processing

### Key Models
- **MDM**: Transformer encoder-based motion diffusion model (`trans_enc`) - Original model with 1000 diffusion steps
- **DiP**: Autoregressive transformer decoder for ultra-fast generation (`trans_dec`) - Uses 10 steps, predicts 2-second chunks
- **Fast MDM**: 50-step encoder model for 20x faster generation with comparable results
- Text encoders: CLIP (default) or DistilBERT (`--text_encoder_type bert`)

## Development Commands

### Training Models

**Text-to-Motion (HumanML3D)**:
```bash
# Original MDM
python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512 --dataset humanml

# Fast 50-step model
python -m train.train_mdm --save_dir save/my_humanml_50steps --dataset humanml --diffusion_steps 50 --mask_frames --use_ema

# DiP (ultra-fast autoregressive)
python -m train.train_mdm --save_dir save/my_humanml_DiP --dataset humanml --arch trans_dec --text_encoder_type bert --diffusion_steps 10 --context_len 20 --pred_len 40 --mask_frames --use_ema --autoregressive --gen_guidance_param 7.5
```

**Action-to-Motion**:
```bash
python -m train.train_mdm --save_dir save/my_name --dataset {humanact12,uestc} --cond_mask_prob 0 --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1
```

### Generation

**Text-to-Motion**:
```bash
# From test prompts
python -m sample.generate --model_path ./save/model.pt --num_samples 10 --num_repetitions 3

# Custom text prompt
python -m sample.generate --model_path ./save/model.pt --text_prompt "a person walks forward"

# DiP generation (ultra-fast)
python -m sample.generate --model_path save/target_10steps_context20_predict40/model000200000.pt --autoregressive --guidance_param 7.5

# DiP with custom prompt
python -m sample.generate --model_path save/target_10steps_context20_predict40/model000200000.pt --autoregressive --guidance_param 7.5 --text_prompt "A person throws a ball"

# DiP with dynamic prompts (each line = 2 seconds of motion)
python -m sample.generate --model_path save/target_10steps_context20_predict40/model000200000.pt --autoregressive --guidance_param 7.5 --dynamic_text_path assets/example_dynamic_text_prompts.txt
```

**Action-to-Motion**:
```bash
python -m sample.generate --model_path ./save/humanact12/model.pt --action_name "drink"
```

### Evaluation

**Text-to-Motion**:
```bash
# Standard evaluation
python -m eval.eval_humanml --model_path ./save/model.pt

# DiP evaluation
python -m eval.eval_humanml --model_path save/DiP_no-target_10steps_context20_predict40/model000600343.pt --autoregressive --guidance_param 7.5

# With WandB logging
python -m eval.eval_humanml --model_path ./save/model.pt --train_platform_type WandBPlatform

# Multimodality metric
python -m eval.eval_humanml --model_path ./save/model.pt --eval_mode mm_short
```

**Action-to-Motion**:
```bash
python -m eval.eval_humanact12_uestc --model ./save/model.pt --eval_mode full
```

### Visualization
```bash
# Render SMPL mesh from stick figure animation
python -m visualize.render_mesh --input_path /path/to/sample.mp4
```

## Code Patterns

### Model Creation
Models are created via `utils.model_util.create_model_and_diffusion()` which handles:
- Architecture selection (`trans_enc`, `trans_dec`, `gru`)
- Text encoder type (`clip`, `bert`)
- Diffusion configuration and sampling parameters

### Training Platforms
Supports logging to:
- WandB: `--train_platform_type WandBPlatform`
- Tensorboard: `--train_platform_type TensorboardPlatform`
- No logging: `--train_platform_type NoPlatform`

### Common Arguments
- `--arch {trans_enc, trans_dec, gru}` - Model architecture
- `--text_encoder_type {clip, bert}` - Text encoder
- `--diffusion_steps N` - Number of diffusion steps (1000 default, 50 for fast models, 10 for DiP)
- `--mask_frames` - Fix masking bug (recommended for all new training)
- `--use_ema` - Exponential moving average (recommended for better performance)
- `--autoregressive` - Enable autoregressive generation (required for DiP)
- `--guidance_param N` - Classifier-free guidance scale (7.5 recommended for DiP)
- `--context_len N` - Context length for autoregressive models (20 for DiP)
- `--pred_len N` - Prediction length for autoregressive models (40 for DiP)
- `--eval_during_training` - Run evaluation during training (slower but better monitoring)
- `--gen_during_training` - Generate samples during training

## Dataset Requirements

Models require specific datasets in `./dataset/`:
- **HumanML3D**: Text-to-motion dataset (main dataset for text conditioning)
- **KIT-ML**: Text-to-motion dataset (alternative text dataset)
- **HumanAct12**: Action-to-motion dataset (12 action classes)
- **UESTC**: Action-to-motion dataset (40 action classes)

**Dataset Downloads:**
- HumanML3D: Download from [Google Drive](https://drive.google.com/drive/folders/1OZrTlAGRvLjXhXwnRiOC-oxYry1vf-Uu?usp=drive_link)
- Action datasets: `bash prepare/download_a2m_datasets.sh`
- Unconstrained datasets: `bash prepare/download_unconstrained_datasets.sh`

Pre-trained models should be placed in `./save/` directory.

## Model Checkpoints

**Text-to-Motion (HumanML3D):**
- [DiP (ultra-fast)](https://huggingface.co/guytevet/CLoSD/tree/main/checkpoints/dip/DiP_no-target_10steps_context20_predict40) - 40x faster generation
- [Fast MDM (50 steps)](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view?usp=sharing) - 20x faster than original
- [Original MDM](https://drive.google.com/file/d/1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821/view?usp=sharing) - Best quality from paper

## Development Workflow

1. **Setup**: Use conda environment and download dependencies for your task type
2. **Data**: Download required datasets to `./dataset/`
3. **Models**: Download pre-trained models to `./save/` or train your own
4. **Generation**: Use `sample.generate` with appropriate model and parameters
5. **Evaluation**: Use task-specific evaluation scripts
6. **Visualization**: Convert outputs to SMPL meshes with `visualize.render_mesh`