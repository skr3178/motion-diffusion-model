#!/bin/bash

# Setup script for Motion Diffusion Model using UV instead of conda

echo "Setting up Motion Diffusion Model with UV..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add UV to PATH
    export PATH="$HOME/.local/bin:$PATH"
    # Try to source cargo env if it exists
    if [ -f "$HOME/.cargo/env" ]; then
        source $HOME/.cargo/env
    fi
fi

# Install system dependencies (similar to conda environment)
echo "Installing system dependencies..."
echo "Note: You may need to install ffmpeg manually if not already installed:"
echo "sudo apt update && sudo apt install -y ffmpeg"

# Check if Python 3.7 is available (required by the project)
echo "Checking for Python 3.7 (required by the project)..."
if command -v python3.7 &> /dev/null; then
    PYTHON_VERSION="3.7"
    echo "Found Python 3.7"
else
    echo "Error: Python 3.7 is required by this project but not found."
    echo "Please install Python 3.7 using one of these methods:"
    echo "1. Using pyenv: pyenv install 3.7.13"
    echo "2. Using conda: conda install python=3.7"
    echo "3. Using apt: sudo apt install python3.7"
    echo "4. Download from python.org"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment with Python $PYTHON_VERSION..."
uv venv --python $PYTHON_VERSION

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install PyTorch with CUDA support first (this is critical for the environment)
echo "Installing PyTorch with CUDA 11.0 support..."
uv pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
echo "Installing other dependencies with UV..."
uv pip install -e .

# Download spaCy model
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Download additional dependencies
echo "Downloading additional dependencies..."
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh

echo ""
echo "Setup complete! 🎉"
echo ""
echo "To activate the environment, run:"
echo "source .venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "deactivate"
