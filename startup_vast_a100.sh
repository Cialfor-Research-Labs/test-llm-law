#!/bin/bash

# --- VAST.AI A100 STARTUP SETUP ---
# Run this on your new A100 instance to prepare the environment.

echo "--- 1. UPDATING SYSTEM ---"
apt-get update && apt-get install -y htop nvtop

echo "--- 2. INSTALLING PYTHON DEPENDENCIES ---"
pip install -U transformers accelerate bitsandbytes torch sentencepiece

echo "--- 3. CREATING CACHE DIRECTORY ---"
mkdir -p /workspace/hf_cache
export HF_HOME=/workspace/hf_cache

echo "--- 4. GPU VERIFICATION ---"
nvidia-smi

echo "--- 5. READY TO LOAD ---"
echo "You can now run: python test_load_optimized.py"
echo "Note: On A100, you can even comment out load_in_4bit to use 8-bit or BF16 for better quality!"
