import os
import shutil
import urllib.request
import tarfile
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PADDLE_DIR = os.path.join(MODELS_DIR, "paddleocr")
STAMP_DIR = os.path.join(MODELS_DIR, "stamp_model")
LLAMA_DIR = os.path.join(MODELS_DIR, "llama-1b")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PADDLE_DIR, exist_ok=True)
os.makedirs(STAMP_DIR, exist_ok=True)
os.makedirs(LLAMA_DIR, exist_ok=True)

def download_file(url, dest_path):
    print(f"Downloading {os.path.basename(dest_path)}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Done.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def setup_paddle():
    print("\n--- 1. Setting up PaddleOCR ---")
    urls = {
        "det": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar",
        "cls": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
        "rec_en": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar"
    }
    
    for name, url in urls.items():
        filename = url.split("/")[-1]
        save_path = os.path.join(PADDLE_DIR, filename)
        
        if not os.path.exists(save_path.replace(".tar", "")): # Skip if already extracted
            download_file(url, save_path)
            print(f"📦 Extracting {filename}...")
            with tarfile.open(save_path) as tar:
                tar.extractall(path=PADDLE_DIR)
            os.remove(save_path)
        else:
            print(f"{name} already exists. Skipping.")

def setup_huggingface():
    print("\n--- 2. Setting up Hugging Face Models ---")
    
    # Stamp Model (Public)
    print("Downloading Stamp Detector (Ooredoo)...")
    try:
        snapshot_download(
            repo_id="Ooredoo-Group/ooredoo-stamp-detection",
            local_dir=STAMP_DIR,
            allow_patterns=["config.json", "*.safetensors", "preprocessor_config.json"]
        )
        print("Stamp model ready.")
    except Exception as e:
        print(f"Error: {e}")

    # Llama 3.2 1B (Gated - Requires Token)
    print("\n⏳ Downloading Llama 3.2 1B...")
    print("NOTE: You need a Hugging Face token with access to meta-llama/Llama-3.2-1B-Instruct.")
    token = os.environ.get("HF_TOKEN") or input("Paste your Hugging Face Token (or press Enter if logged in via CLI): ")
    
    try:
        snapshot_download(
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            local_dir=LLAMA_DIR,
            ignore_patterns=["*.pth", "original/*"],
            token=token if token else None
        )
        print("Llama model ready.")
    except Exception as e:
        print(f"Llama Download Failed: {e}")
        print("Tip: Run 'huggingface-cli login' or set HF_TOKEN environment variable.")

def check_sign_model():
    print("\n--- 3. Checking Custom Signature Model ---")
    sign_path = os.path.join(MODELS_DIR, "sign_model.pt")
    if os.path.exists(sign_path):
        print("'sign_model.pt' found.")
    else:
        print("WARNING: 'sign_model.pt' is missing!")
        print("This is a custom trained model. You must manually place it in:")
        print(f"   {sign_path}")
        print("Or download it from your Project Releases page if you uploaded it there.")

if __name__ == "__main__":
    setup_paddle()
    setup_huggingface()
    check_sign_model()
    print("\nSetup Complete! You can now run executable.py")