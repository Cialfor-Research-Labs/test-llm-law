#!/bin/bash

# ==========================================================
# MASTER SETUP & RUN SCRIPT FOR LEGAL AI (SARVAM-30B)
# ==========================================================
# Instructions:
# 1. Start a fresh A100-80GB instance on Vast.ai with 300GB+ disk.
# 2. Run: bash setup_and_run.sh
# ==========================================================

# --- 1. USER CONFIGURATION (Update these if your Vast.ai ports change) ---
BACKEND_EXTERNAL_PORT="46146"
FRONTEND_EXTERNAL_PORT="47131"
VAST_IP="185.65.93.212"

# --- 2. DISK SAFETY CHECK ---
FREE_SPACE=$(df -k /workspace | tail -1 | awk '{print $4}') # in KB
REQUIRED_SPACE=$((250 * 1024 * 1024)) # 250GB in KB

if [ "$FREE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "[ERROR] You only have $(($FREE_SPACE/1024/1024))GB free on /workspace."
    echo "This model requires ~130GB for weights. Please expand your disk to 300GB+."
    exit 1
fi

# --- 3. SYSTEM & PYTHON DEPENDENCIES ---
echo "--- Step 1: Installing System Dependencies ---"
apt-get update && apt-get install -y python3-venv nodejs npm

# Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate

echo "--- Step 2: Installing PyTorch (CUDA 12.1 optimized) ---"
pip install --upgrade pip
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "--- Step 3: Installing Requirements ---"
pip install -r requirements.txt
pip install uvicorn gunicorn bitsandbytes accelerate # Ensure these are present

# --- 4. HUGGINGFACE CACHE SETUP ---
echo "--- Step 4: Configuring Large Cache Directory ---"
mkdir -p /workspace/hf_cache
export HF_HOME=/workspace/hf_cache

# --- 5. FRONTEND SETUP ---
echo "--- Step 5: Setting up Frontend ---"
cd ui
npm install

# Create .env with the current public IP/Port
echo "VITE_API_BASE_URL=http://${VAST_IP}:${BACKEND_EXTERNAL_PORT}" > .env
cd ..

# --- 6. BACKGROUND RUNNER ---
echo "--- Step 6: Starting Backend & Frontend in Background ---"

# Kill any existing processes
pkill -f uvicorn
pkill -f vite

export LLM_LOAD_8BIT=true
export LLM_LOAD_4BIT=false

# Start Backend
echo "Starting Backend... Logs: backend.log"
nohup python -m uvicorn retrieval_api:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

# Start Frontend
echo "Starting Frontend... Logs: frontend.log"
cd ui
nohup npm run dev -- --host > ../frontend.log 2>&1 &
cd ..

echo "=========================================================="
echo "SUCCESS: Everything is running in the background!"
echo "----------------------------------------------------------"
echo "Public UI: http://${VAST_IP}:${FRONTEND_EXTERNAL_PORT}"
echo "Public API: http://${VAST_IP}:${BACKEND_EXTERNAL_PORT}"
echo "----------------------------------------------------------"
echo "Note: The model (129GB) is currently downloading in backend.log."
echo "Wait ~10 minutes before the UI becomes responsive."
echo "Use 'tail -f backend.log' to monitor progress."
echo "=========================================================="
