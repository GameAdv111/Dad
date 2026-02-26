from huggingface_hub import snapshot_download
import os

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIR = os.path.join("models", "all-MiniLM-L6-v2")

os.makedirs("models", exist_ok=True)

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
)

print("MODEL DOWNLOADED TO:", LOCAL_DIR)