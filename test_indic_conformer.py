import torch
import os
import sys

# Note: This script requires the NeMo toolkit to be installed.
# Installation:
# pip install torch torchvision torchaudio
# pip install packaging
# pip install huggingface_hub
# git clone https://github.com/AI4Bharat/NeMo.git
# cd NeMo && bash reinstall.sh

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    print("Error: NeMo toolkit is not installed.")
    print("Please install it using the instructions in the file comments or the AI4Bharat website.")
    sys.exit(1)

# Get audio file from command line arguments or use default
if len(sys.argv) > 1:
    audio_file = sys.argv[1]
else:
    audio_file = "hindi podcast.wav"

# Check if file exists
if not os.path.exists(audio_file):
    print(f"Error: Audio file '{audio_file}' not found.")
    print("Usage: python test_indic_conformer.py <path_to_audio_file>")
    exit(1)

print(f"Processing audio file: {audio_file}")

# Model details
# Using the AI4Bharat IndicConformer model for Hindi
model_name = "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print(f"Loading model: {model_name}...")
# Load from Hugging Face
# NeMo models on HF can often be loaded directly if supported, or downloaded as .nemo files.
# Since we are using the NeMo library, we can try to restore from the HF hub if supported,
# or we might need to download the .nemo file first.
# The AI4Bharat documentation suggests downloading the checkpoint.
# However, newer NeMo versions support `from_pretrained`.
try:
    model = nemo_asr.models.EncDecRNNTModel.from_pretrained(model_name=model_name)
except Exception as e:
    print(f"Failed to load using from_pretrained: {e}")
    print("Attempting to load using EncDecCTCModel (hybrid models might need specific class)...")
    try:
        model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)
    except Exception as e2:
        print(f"Failed to load using EncDecCTCModel: {e2}")
        print("Please manually download the .nemo file from Hugging Face and update the path.")
        exit(1)

model.freeze()
model = model.to(device)

# Inference
print("Starting inference...")

# CTC Decoding
print("Running CTC Decoding...")
model.cur_decoder = 'ctc'
# Note: transcribe method signature might vary slightly based on NeMo version
try:
    ctc_text = model.transcribe([audio_file], batch_size=1, logprobs=False, language_id='hi')[0]
    print(f"CTC Transcription: {ctc_text}")
except Exception as e:
    print(f"CTC Inference failed: {e}")

# RNN-T Decoding
print("Running RNN-T Decoding...")
try:
    model.cur_decoder = 'rnnt'
    rnnt_text = model.transcribe([audio_file], batch_size=1, language_id='hi')[0]
    print(f"RNN-T Transcription: {rnnt_text}")
except Exception as e:
    print(f"RNN-T Inference failed: {e}")
