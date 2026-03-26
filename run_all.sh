#!/bin/bash

# Configuration
export HF_HOME=/workspace/hf_cache
export LLM_LOAD_8BIT=true
export LLM_LOAD_4BIT=false

echo "--- STARTING LEGAL AI ASSISTANT (BACKGROUND MODE) ---"

# 1. Start Backend
echo "Starting Backend on port 8000..."
nohup python -m uvicorn retrieval_api:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID. Logs: backend.log"

# 2. Start Frontend
echo "Starting Frontend on port 5173..."
cd ui
nohup npm run dev -- --host > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend started with PID: $FRONTEND_PID. Logs: frontend.log"

echo "-------------------------------------------------------"
echo "Both processes are running in the background."
echo "To stop them, use: kill $BACKEND_PID $FRONTEND_PID"
echo "To view logs: tail -f backend.log or tail -f frontend.log"
echo "-------------------------------------------------------"
