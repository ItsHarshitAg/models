#!/bin/bash

# Exit on error
set -e

echo "Starting setup for Indic ASR models on Ubuntu..."

# 1. Install System Dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 git python3-venv build-essential

# 2. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# 3. Install PyTorch
# CHECK FOR JETSON
if [ -f /etc/nv_tegra_release ]; then
    echo "----------------------------------------------------------------"
    echo "WARNING: Jetson device detected!"
    echo "Please run the dedicated Jetson setup script instead:"
    echo "./setup_jetson.sh"
    echo "----------------------------------------------------------------"
    exit 1
else
    # Standard x86_64 installation
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# 4. Install Basic Requirements
echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt
pip install Cython packaging

# 5. Install NeMo (AI4Bharat Fork)
if [ ! -d "NeMo" ]; then
    echo "Cloning AI4Bharat NeMo repository..."
    git clone https://github.com/AI4Bharat/NeMo.git
fi

echo "Installing NeMo..."
cd NeMo
# The reinstall.sh script handles the installation of NeMo and its dependencies
bash reinstall.sh
cd ..

# 6. Download IndicWhisper Model
echo "Downloading IndicWhisper Hindi Model..."
python download_indic_whisper_model.py

echo "========================================================"
echo "Setup Complete!"
echo "To start, run: source venv/bin/activate"
echo "Then run inference: python test_indic_whisper.py <audio_file>"
echo "========================================================"
