import torch
from transformers import pipeline
import os
import sys

# Get audio file from command line arguments or use default
if len(sys.argv) > 1:
    audio_file = sys.argv[1]
else:
    audio_file = "hindi podcast.wav"

# Check if file exists
if not os.path.exists(audio_file):
    print(f"Error: Audio file '{audio_file}' not found.")
    print("Usage: python test_indic_whisper.py <path_to_audio_file>")
    exit(1)

print(f"Processing audio file: {audio_file}")

# Model details
# The IndicWhisper model must be downloaded locally.
# Run `python download_indic_whisper_model.py` to download it.
model_path = "hindi_models/whisper-medium-hi_alldata_multigpu"

if not os.path.exists(model_path):
    print(f"Error: Model not found at '{model_path}'.")
    print("Please run 'python download_indic_whisper_model.py' to download the model first.")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    print(f"Loading model from: {model_path}...")
    whisper_asr = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
    )
except Exception as e:
    print(f"Could not load model from {model_path}: {e}")
    exit(1)

# Configure for Hindi
# Note: For openai/whisper, we can pass generate_kwargs={"language": "hindi"}
# For IndicWhisper, it might be baked in or similar.
lang_code = "hi"

# Force decoder IDs for Hindi if needed (mostly for multilingual models)
try:
    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language=lang_code, task="transcribe"
        )
    )
except Exception as e:
    print(f"Note: Could not set forced_decoder_ids (might not be needed for this model): {e}")

print("Starting inference...")
result = whisper_asr(audio_file)
print("\nTranscription Result:")
print("-" * 30)
print(result["text"])
print("-" * 30)
