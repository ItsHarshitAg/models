# Indic ASR Setup and Execution Guide (Ubuntu with GPU)

This guide details the steps to set up the environment and run inference using AI4Bharat's IndicWhisper and IndicConformer models on an Ubuntu system with an NVIDIA GPU.

## Prerequisites

- Ubuntu OS
- NVIDIA GPU with drivers installed (`nvidia-smi` should work)
- Python 3.8 or higher
- `git`, `ffmpeg`, `venv`

## Jetson Orin Setup (Automated)

We have provided a dedicated script for Jetson devices that handles the complex installation of PyTorch and Torchvision.

**Supported JetPack Versions:** JetPack 6.0, 6.1, 6.2

1.  **Make the script executable:**
    ```bash
    chmod +x setup_jetson.sh
    ```

2.  **Run the Jetson setup script:**
    ```bash
    ./setup_jetson.sh
    ```
    *Note: This script uses a PyTorch 2.4 wheel compatible with JetPack 6.x. If you need a specific version, edit the `TORCH_URL` in `setup_jetson.sh`.*

## Standard Ubuntu Setup (x86_64 only)

Use this section ONLY if you are on a standard desktop/server Ubuntu (NOT Jetson).

1. **Make the setup script executable:**

    ```bash
    chmod +x setup_ubuntu.
    ```

2. **Run the setup script:**
    This script will create a virtual environment, install dependencies, install NeMo, and download the IndicWhisper model.

    ```bash
    ./setup_ubuntu.sh
    ```

3. **Activate the environment:**

    ```bash
    source venv/bin/activate
    ```

## Manual Setup Steps

If you prefer to run steps manually:

1. **Install System Dependencies:**

    ```bash
    sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1 git python3-venv
    ```

2. **Create and Activate Virtual Environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install PyTorch (with CUDA support):**

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Note: Adjust 'cu118' based on your CUDA version (check with nvidia-smi)
    ```

4. **Install Python Dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install Cython packaging
    ```

5. **Install NeMo (Required for IndicConformer):**

    ```bash
    git clone https://github.com/AI4Bharat/NeMo.git
    cd NeMo
    bash reinstall.sh
    cd ..
    ```

6. **Download IndicWhisper Model:**

    ```bash
    python download_indic_whisper_model.py
    ```

## Running Inference

Ensure your virtual environment is active (`source venv/bin/activate`) and you have an audio file (e.g., `audio.wav`) in the directory.

### 1. Test IndicWhisper

```bash
python test_indic_whisper.py <path_to_audio_file>
```

*Example:* `python test_indic_whisper.py "hindi podcast.wav"`

### 2. Test IndicConformer

```bash
python test_indic_conformer.py <path_to_audio_file>
```

*Example:* `python test_indic_conformer.py "hindi podcast.wav"`

## Troubleshooting

- **CUDA/GPU Issues:** Run `python -c "import torch; print(torch.cuda.is_available())"` to verify GPU access. If `False`, reinstall PyTorch with the correct CUDA version.
- **NeMo Import Errors:** Ensure you ran `bash reinstall.sh` inside the `NeMo` directory and that the installation completed successfully.
- **Audio Format:** `ffmpeg` is required to handle various audio formats. Ensure it is installed.
