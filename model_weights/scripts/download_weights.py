#!/usr/bin/env python

import os
import sys
from huggingface_hub import snapshot_download

weights_root_storage = os.path.expanduser("~")+"/Downloads/llm-weights-rs"
weights_external_storage="/Volumes/OVERFLOW/0.UW-Capstone/LLM-weights"
weights_dir = weights_external_storage if os.path.exists(weights_external_storage) else weights_root_storage

if not os.path.exists(weights_external_storage):
    print("External storage not mounted, downloading to root storage. \nRoot storage path exists: "+str(os.path.exists(weights_root_storage)))

def download_model(model_id):
    # Extract model name from model_id
    model_name = model_id.split("/")[-1]
    local_dir = os.path.join(weights_dir, model_name)

    # Download the model snapshot
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision="main",
    )
    print(f"Model '{model_id}' downloaded to '{local_dir}'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download_weights.py <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]
    download_model(model_id)
