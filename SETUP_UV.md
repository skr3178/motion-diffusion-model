# UV Setup Guide for Motion Diffusion Model

This guide provides step-by-step instructions for setting up the Motion Diffusion Model using UV instead of conda.

## Prerequisites

- Python 3.7 installed on your system (required by the project)
- CUDA-capable GPU (for PyTorch)
- ffmpeg installed

## Quick Setup (Automated)

If you want to use the automated setup script:

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Run the automated setup script
bash setup_uv.sh

# Activate the environment
source .venv/bin/activate
```

## Manual Setup (Step-by-Step)

If you prefer to set up manually or need to troubleshoot:

### Step 1: Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### Step 2: Install System Dependencies

```bash
sudo apt update
sudo apt install -y ffmpeg
```

### Step 3: Create Virtual Environment

```bash
# Verify Python 3.7 is installed
python3.7 --version

# Create virtual environment with Python 3.7
uv venv --python 3.7
```

### Step 4: Activate Environment and Install Dependencies

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install PyTorch with CUDA support first
uv pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
uv pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 5: Download Additional Dependencies

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

### Step 6: Verify Installation

```bash
# Test if everything is working
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

To activate the environment in the future:
```bash
source .venv/bin/activate
```

To deactivate:
```bash
deactivate
```

## Troubleshooting

### Python 3.7 not found
If Python 3.7 is not available, you need to install it first. Here are some options:

**Using pyenv (recommended):**
```bash
# Install pyenv if not already installed
curl https://pyenv.run | bash

# Install Python 3.7
pyenv install 3.7.13
pyenv global 3.7.13
```

**Using conda:**
```bash
conda install python=3.7
```

**Using apt (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.7 python3.7-venv python3.7-dev
```

### CUDA issues
Make sure you have the correct CUDA version installed. The setup uses CUDA 11.0. If you have a different version, you may need to install the appropriate PyTorch version.

### Permission issues
If you encounter permission issues with system packages, you may need to install them manually or use a different approach for system dependencies.

## Benefits of UV over Conda

- **Speed**: 10-100x faster dependency resolution
- **Simplicity**: Single tool for environment and package management
- **Compatibility**: Works with existing pip workflows
- **Modern**: Built with Rust for performance and reliability
