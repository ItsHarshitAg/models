import os
import requests
import zipfile
from tqdm import tqdm

# URL for the Hindi IndicWhisper model
MODEL_URL = "https://indicwhisper.objectstore.e2enetworks.net/hindi_models.zip"
ZIP_FILE = "hindi_models.zip"
EXTRACT_DIR = "."

def download_file(url, filename):
    """Downloads a file with a progress bar."""
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return

    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def extract_zip(zip_path, extract_to):
    """Extracts a zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    # Download
    download_file(MODEL_URL, ZIP_FILE)
    
    # Extract
    extract_zip(ZIP_FILE, EXTRACT_DIR)
    
    print("\nModel setup complete.")
    print(f"Model directory should be at: {os.path.join(EXTRACT_DIR, 'hindi_models', 'whisper-medium-hi_alldata_multigpu')}")

if __name__ == "__main__":
    main()
