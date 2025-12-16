#!/bin/bash

# Exit on error
set -e

echo "Starting setup for Jetson Orin (JetPack 6.0 / CUDA 12.6)..."

# 1. Install System Dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev libomp-dev git build-essential libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev ffmpeg python3-venv

# 2. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# 3. Install PyTorch for Jetson (JetPack 6.0 / 6.1 / 6.2)
# Using PyTorch v2.5.0 for JetPack 6.1 (Compatible with 6.2)
# Note: JetPack 6.2 is very new. If this fails, fallback to the v2.4.0 wheel below.
# TORCH_WHEEL_NAME="torch-2.5.0a0+872d972e41.nv24.08.17596369-cp310-cp310-linux_aarch64.whl"
# TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/${TORCH_WHEEL_NAME}"

# Using PyTorch v2.4.0 (Stable for JetPack 6.x)
# This version is widely tested on JetPack 6.0 and should work on 6.2
TORCH_WHEEL_NAME="torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl"
TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/${TORCH_WHEEL_NAME}"

if ! python -c "import torch" &> /dev/null; then
    echo "Installing PyTorch for Jetson..."
    echo "Downloading ${TORCH_WHEEL_NAME}..."
    wget -q --show-progress "${TORCH_URL}" -O "${TORCH_WHEEL_NAME}"
    
    echo "Installing PyTorch wheel..."
    pip install "${TORCH_WHEEL_NAME}"
    
    # Cleanup
    rm "${TORCH_WHEEL_NAME}"
else
    echo "PyTorch already installed."
fi

# 4. Install Torchvision (Compile from source)
# Torchvision version must match PyTorch version compatibility.
# For PyTorch 2.4, we generally use Torchvision 0.19
if ! python -c "import torchvision" &> /dev/null; then
    echo "Installing Torchvision from source (this may take a while)..."
    if [ -d "vision" ]; then
        rm -rf vision
    fi
    git clone --branch v0.19.0 https://github.com/pytorch/vision
    cd vision
    export BUILD_VERSION=0.19.0
    python setup.py install
    cd ..
    rm -rf vision
else
    echo "Torchvision already installed."
fi

# 5. Install Basic Requirements
echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt
pip install Cython packaging

# 6. Install NeMo (AI4Bharat Fork)
if [ ! -d "NeMo" ]; then
    echo "Cloning AI4Bharat NeMo repository..."
    git clone https://github.com/AI4Bharat/NeMo.git
fi

echo "Installing NeMo..."
cd NeMo
# The reinstall.sh script handles the installation of NeMo and its dependencies
bash reinstall.sh
cd ..

# 7. Download IndicWhisper Model
echo "Downloading IndicWhisper Hindi Model..."
python download_indic_whisper_model.py

echo "========================================================"
echo "Jetson Setup Complete!"
echo "To start, run: source venv/bin/activate"
echo "Then run inference: python test_indic_whisper.py <audio_file>"
echo "========================================================"
