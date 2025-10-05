# MDM Model Usage Commands Reference

This file contains all the commands to use your downloaded MDM models with the correct paths.

## 📁 Model Locations

All models are located in: `/home/skr/motion-diffusion-model/save/`

### Available Models:

## 🏗️ Model Architecture Overview

The Motion Diffusion Model (MDM) supports three main neural network architectures:

1. **Transformer Encoder (`trans_enc`)** - Uses self-attention layers to process motion sequences
2. **Transformer Decoder (`trans_dec`)** - Uses cross-attention with text conditioning
3. **GRU (`gru`)** - Uses Gated Recurrent Units for sequential processing

## 📊 Dataset Differences

- **HumanML3D**: Large-scale dataset with diverse human motions and detailed text descriptions
- **KIT**: Smaller dataset focused on specific motion patterns
- **HumanAct12**: Action-based dataset with 12 predefined action categories
- **UESTC**: Action-based dataset with different motion characteristics

## 🔤 Text Encoder Types

- **CLIP**: Vision-language model for text encoding (default)
- **BERT**: Bidirectional Encoder Representations from Transformers (DistilBERT)

---

#### Text-to-Motion Models (Recommended: Use the first one)

##### 🚀 **Performance Optimized Models**
- **`humanml_enc_512_50steps/`** - ⚡ **FASTEST** 
  - **Architecture**: Transformer Encoder (`trans_enc`)
  - **Text Encoder**: CLIP
  - **Diffusion Steps**: 50 (vs 1000 in original)
  - **Speed**: 20X faster than original
  - **Performance**: Comparable quality to original
  - **Dataset**: HumanML3D
  - **Best for**: Quick prototyping and real-time applications

- **`models_to_upload/humanml_trans_dec_512_bert/`** - 🎯 **HIGHEST PRECISION**
  - **Architecture**: Transformer Decoder (`trans_dec`)
  - **Text Encoder**: BERT (DistilBERT)
  - **Diffusion Steps**: 50
  - **Speed**: 20X faster than original
  - **Performance**: Improved precision over CLIP models
  - **Dataset**: HumanML3D
  - **Best for**: High-quality motion generation when precision matters

##### 📚 **Research/Original Models**
- **`humanml_trans_enc_512/`** - 📖 **ORIGINAL PAPER MODEL**
  - **Architecture**: Transformer Encoder (`trans_enc`)
  - **Text Encoder**: CLIP
  - **Diffusion Steps**: 1000
  - **Speed**: Baseline (slowest)
  - **Performance**: Reference implementation from paper
  - **Dataset**: HumanML3D
  - **Best for**: Reproducing paper results, research comparisons

- **`humanml_trans_dec_512/`** - 🔄 **DECODER ARCHITECTURE**
  - **Architecture**: Transformer Decoder (`trans_dec`)
  - **Text Encoder**: CLIP
  - **Diffusion Steps**: 1000
  - **Speed**: Baseline
  - **Performance**: Different attention mechanism than encoder
  - **Dataset**: HumanML3D
  - **Best for**: Comparing encoder vs decoder architectures

- **`humanml_trans_dec_emb_512/`** - 🎨 **DECODER WITH EMBEDDINGS**
  - **Architecture**: Transformer Decoder (`trans_dec`) with embedding modifications
  - **Text Encoder**: CLIP
  - **Diffusion Steps**: 1000
  - **Speed**: Baseline
  - **Performance**: Enhanced with additional embedding layers
  - **Dataset**: HumanML3D
  - **Best for**: Research on embedding strategies

##### 🌍 **Alternative Dataset Models**
- **`kit_trans_enc_512/`** - 🎭 **KIT DATASET MODEL**
  - **Architecture**: Transformer Encoder (`trans_enc`)
  - **Text Encoder**: CLIP
  - **Diffusion Steps**: 1000
  - **Speed**: Baseline
  - **Performance**: Trained on KIT dataset (different motion characteristics)
  - **Dataset**: KIT-ML
  - **Best for**: Motions specific to KIT dataset characteristics

---

#### Action-to-Motion Models

##### 🎯 **HumanAct12 Dataset Models**
- **`humanact12/`** - 📋 **STANDARD HUMANACT12**
  - **Architecture**: Transformer Encoder (`trans_enc`)
  - **Conditioning**: Action labels (12 predefined actions)
  - **Diffusion Steps**: 1000
  - **Dataset**: HumanAct12
  - **Actions**: drink, eat, jump, pick, pour, push, run, sit, stand, throw, walk, wave
  - **Best for**: Controlled action-based motion generation

- **`humanact12_no_fc/`** - 🔧 **HUMANACT12 WITHOUT FC LAYERS**
  - **Architecture**: Transformer Encoder (`trans_enc`) without fully connected layers
  - **Conditioning**: Action labels
  - **Diffusion Steps**: 1000
  - **Dataset**: HumanAct12
  - **Performance**: Different network structure, potentially faster inference
  - **Best for**: Comparing network architectures, faster inference

##### 🏃 **UESTC Dataset Models**
- **`uestc/`** - 🎪 **STANDARD UESTC**
  - **Architecture**: Transformer Encoder (`trans_enc`)
  - **Conditioning**: Action labels
  - **Diffusion Steps**: 1000
  - **Dataset**: UESTC
  - **Performance**: Different motion characteristics than HumanAct12
  - **Best for**: UESTC-specific motion patterns

- **`uestc_no_fc/`** - ⚡ **UESTC WITHOUT FC LAYERS**
  - **Architecture**: Transformer Encoder (`trans_enc`) without fully connected layers
  - **Conditioning**: Action labels
  - **Diffusion Steps**: 1000
  - **Dataset**: UESTC
  - **Performance**: Optimized for faster inference
  - **Best for**: Fast UESTC motion generation

---

#### Unconstrained Models
- **`unconstrained/`** - 🆓 **UNCONSTRAINED MOTION GENERATION**
  - **Architecture**: Transformer Encoder (`trans_enc`)
  - **Conditioning**: None (unconditional generation)
  - **Diffusion Steps**: 1000
  - **Dataset**: HumanAct12 (unconstrained)
  - **Performance**: Generates motions without text or action conditioning
  - **Best for**: Exploring natural motion patterns, creative applications

---

## 🔬 Technical Model Comparison

### Architecture Differences

| Architecture | Attention Type | Text Integration | Use Case |
|-------------|---------------|------------------|----------|
| **Transformer Encoder** | Self-attention | Text embedding added to sequence | Standard text-to-motion |
| **Transformer Decoder** | Cross-attention | Text as memory/key-value | Better text-motion alignment |
| **GRU** | Sequential processing | Text embedding concatenated | Simpler, faster training |

### Performance vs Speed Trade-offs

| Model Type | Speed | Quality | Use When |
|------------|-------|--------|----------|
| **50-step models** | ⚡⚡⚡ Very Fast | ⭐⭐⭐ High | Real-time applications, prototyping |
| **1000-step models** | ⚡ Slow | ⭐⭐⭐⭐ Very High | Research, final quality results |
| **BERT models** | ⚡⚡ Fast | ⭐⭐⭐⭐⭐ Highest | When precision is critical |
| **CLIP models** | ⚡⚡ Fast | ⭐⭐⭐⭐ High | General purpose, good balance |

### Dataset Characteristics

| Dataset | Size | Motion Types | Text Quality | Best For |
|---------|------|-------------|--------------|----------|
| **HumanML3D** | Large | Diverse, natural | High-quality descriptions | General text-to-motion |
| **KIT** | Medium | Specific patterns | Moderate descriptions | Specialized motions |
| **HumanAct12** | Small | 12 action categories | Action labels only | Controlled action generation |
| **UESTC** | Medium | Different style | Action labels only | Alternative action patterns |

---

## 🎯 Model Selection Guide

### Quick Decision Tree

**For Text-to-Motion:**
1. **Need speed?** → Use `humanml_enc_512_50steps/` (20X faster)
2. **Need highest quality?** → Use `models_to_upload/humanml_trans_dec_512_bert/` (BERT + 50 steps)
3. **Reproducing paper results?** → Use `humanml_trans_enc_512/` (original 1000 steps)
4. **Research on architectures?** → Compare `trans_enc` vs `trans_dec` models

**For Action-to-Motion:**
1. **HumanAct12 actions?** → Use `humanact12/` or `humanact12_no_fc/`
2. **UESTC actions?** → Use `uestc/` or `uestc_no_fc/`
3. **Need speed?** → Use `_no_fc` variants

**For Unconstrained Generation:**
1. **Creative exploration?** → Use `unconstrained/`

### Recommended Usage Patterns

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| **First-time user** | `humanml_enc_512_50steps/` | Fast, good quality, easy to use |
| **Production app** | `humanml_enc_512_50steps/` | Speed is crucial for user experience |
| **Research paper** | `humanml_trans_enc_512/` | Reproducible, reference implementation |
| **High-quality demo** | `models_to_upload/humanml_trans_dec_512_bert/` | Best precision available |
| **Architecture comparison** | Multiple models | Compare trans_enc vs trans_dec vs gru |
| **Action-specific** | `humanact12/` or `uestc/` | Designed for action conditioning |

---

## 🚀 Text-to-Motion Generation Commands

### Using the FASTEST Model (Recommended)

#### Generate from test set prompts (10 samples, 3 repetitions each):
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt --num_samples 10 --num_repetitions 3
```

#### Generate from your own text file:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt --input_text ./assets/example_text_prompts.txt
```

#### Generate from a single text prompt:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt --text_prompt "the person walked forward and is picking up his toolbox."
```

### Using Other Text-to-Motion Models

#### Original Paper Model (1000 diffusion steps):
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanml_trans_enc_512/model000475000.pt --text_prompt "a person is dancing"
```

#### Decoder Architecture:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanml_trans_dec_512/model000375000.pt --text_prompt "a person is walking"
```

#### Decoder with Embeddings:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanml_trans_dec_emb_512/model000425000.pt --text_prompt "a person is running"
```

#### BERT Text Encoder:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/models_to_upload/humanml_trans_dec_512_bert/model000600000.pt --text_prompt "a person is jumping"
```

#### KIT Dataset Model:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/kit_trans_enc_512/model000400000.pt --text_prompt "a person is sitting down"
```

---

## 🎯 Action-to-Motion Generation Commands

### HumanAct12 Actions

#### Generate from test set actions:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanact12/model000350000.pt --num_samples 10 --num_repetitions 3
```

#### Generate from action file:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanact12/model000350000.pt --action_file ./assets/example_action_names_humanact12.txt
```

#### Generate single action:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanact12/model000350000.pt --action_name "drink"
```

#### HumanAct12 without fully connected layers:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanact12_no_fc/model000750000.pt --action_name "walk"
```

### UESTC Actions

#### Generate from test set actions:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/uestc/model000950000.pt --num_samples 10 --num_repetitions 3
```

#### Generate single action:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/uestc/model000950000.pt --action_name "bend"
```

#### UESTC without fully connected layers:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/uestc_no_fc/model001550000.pt --action_name "jump"
```

---

## 🎨 Unconstrained Motion Generation

```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/unconstrained/model000450000.pt --num_samples 10 --num_repetitions 3
```

---

## 🎬 Motion Editing Commands

### Unconditioned Editing

#### In-between editing:
```bash
python -m sample.edit --model_path /home/skr/motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt --edit_mode in_between
```

#### Upper body editing:
```bash
python -m sample.edit --model_path /home/skr/motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt --edit_mode upper_body
```

### Text Conditioned Editing

```bash
python -m sample.edit --model_path /home/skr/motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt --edit_mode upper_body --text_condition "A person throws a ball"
```

---

## 🎭 SMPL Mesh Rendering

To create 3D mesh files from your generated motions:

```bash
python -m visualize.render_mesh --input_path /path/to/your/generated/mp4/file
```

**Example:**
```bash
python -m visualize.render_mesh --input_path /home/skr/motion-diffusion-model/save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_example_text_prompts/samples_00_to_02.mp4
```

---

## ⚙️ Customization Options

### Additional Flags You Can Use:

- **`--device`** - Specify GPU ID (e.g., `--device 0`)
- **`--seed`** - Set random seed for reproducible results
- **`--motion_length`** - Motion length in seconds (max 9.8 seconds for text-to-motion)
- **`--output_dir`** - Custom output directory

### Example with Custom Options:

```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt --text_prompt "a person is dancing" --motion_length 5.0 --seed 42 --device 0 --output_dir ./my_custom_output
```

---

## 🏃‍♂️ Quick Start Examples

### Test the Fastest Model:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt --text_prompt "a person is walking forward"
```

### Test Action-to-Motion:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/humanact12/model000350000.pt --action_name "drink"
```

### Test Unconstrained:
```bash
python -m sample.generate --model_path /home/skr/motion-diffusion-model/save/unconstrained/model000450000.pt --num_samples 5
```

---

## 📁 Output Files

When you run generation, you'll get:
- **`results.npy`** - Text prompts and xyz positions of generated animations
- **`sample##_rep##.mp4`** - Stick figure animations for each generated motion
- **`results.txt`** - Text descriptions of generated motions

For SMPL mesh rendering:
- **`sample##_rep##_smpl_params.npy`** - SMPL parameters
- **`sample##_rep##_obj/`** - Mesh files per frame (.obj format)

---

## 💡 Tips

1. **Start with the 50-steps model** (`humanml_enc_512_50steps`) for fastest results
2. **Use BERT model** (`humanml_trans_dec_512_bert`) for highest precision when quality matters most
3. **Use original model** (`humanml_trans_enc_512`) for research reproducibility
4. **Compare architectures** by testing both `trans_enc` and `trans_dec` models
5. **Use absolute paths** to avoid path issues
6. **Check GPU availability** with `--device 0` if you have multiple GPUs
7. **Experiment with different text prompts** for varied results
8. **Use `--seed` for reproducible results** when testing
9. **Choose dataset-specific models** (`kit_trans_enc_512`) for specialized motion patterns
10. **Use action models** (`humanact12`, `uestc`) for controlled action generation

---

## 🔧 Troubleshooting

If you encounter issues:
1. Make sure you're in the project root directory: `/home/skr/motion-diffusion-model/`
2. Activate your virtual environment: `source .venv/bin/activate`
3. Check that the model files exist in the specified paths
4. Ensure you have sufficient GPU memory for generation

---

*Generated on: $(date)*
*Project: Motion Diffusion Model (MDM)*
*Location: /home/skr/motion-diffusion-model/*
