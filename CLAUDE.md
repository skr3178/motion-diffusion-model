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
- **MDM**: Transformer encoder-based motion diffusion model (trans_enc)
- **DiP**: Autoregressive transformer decoder for ultra-fast generation (trans_dec) 
- Both support CLIP and DistilBERT text encoders

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

# DiP generation
python -m sample.generate --model_path save/dip_model.pt --autoregressive --guidance_param 7.5
```

**Action-to-Motion**:
```bash
python -m sample.generate --model_path ./save/humanact12/model.pt --action_name "drink"
```

### Evaluation

**Text-to-Motion**:
```bash
python -m eval.eval_humanml --model_path ./save/model.pt
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
Models are created via `utils.model_util.create_model_and_diffusion()` which handles architecture selection (`trans_enc`, `trans_dec`, `gru`) and text encoder type (`clip`, `bert`).

### Training Platforms
Supports logging to WandB (`--train_platform_type WandBPlatform`), Tensorboard (`--train_platform_type TensorboardPlatform`), or no logging (`--train_platform_type NoPlatform`).

### Common Arguments
- `--arch {trans_enc, trans_dec, gru}` - Model architecture
- `--text_encoder_type {clip, bert}` - Text encoder
- `--diffusion_steps N` - Number of diffusion steps (1000 default, 50 for fast models, 10 for DiP)
- `--mask_frames` - Fix masking bug (recommended)
- `--use_ema` - Exponential moving average (recommended)
- `--autoregressive` - Enable autoregressive generation (DiP)
- `--guidance_param N` - Classifier-free guidance scale

## Dataset Requirements

Models require specific datasets in `./dataset/`:
- **HumanML3D**: Text-to-motion dataset
- **KIT-ML**: Text-to-motion dataset  
- **HumanAct12**: Action-to-motion dataset
- **UESTC**: Action-to-motion dataset

Pre-trained models should be placed in `./save/` directory.