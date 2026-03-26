# Installation and Setup Guide

This file explains how to install, run, and verify the Legal RAG Assistant.

## 1. Prerequisites

- Python 3.10+ (recommended: 3.12)
- Node.js 18+ and npm
- NVIDIA GPU with CUDA (for Cloud/Local Transformers) OR Ollama
- macOS/Linux shell (commands below use `zsh/bash`)

## 2. Backend Setup (Python)

From project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Cloud GPU Setup (Linux)

To run the **Sarvam-30b** model on a cloud GPU (e.g., A100, H100, RTX 3090/4090):

### GPU Requirements
- **Full Precision (bf16)**: 70-80GB VRAM (Single A100 80GB recommended).
- **Quantized (4-bit)**: 18-24GB VRAM (Single RTX 3090/4090, A10G, or L4 recommended).

### Running with Quantization
To save VRAM on smaller cards, enable 4-bit quantization via environment variables:

```bash
# For 4-bit (recommended for 24GB GPUs)
export LLM_LOAD_4BIT=true
python -m uvicorn retrieval_api:app --host 0.0.0.0 --port 8000
```

```bash
# For 8-bit
export LLM_LOAD_8BIT=true
python -m uvicorn retrieval_api:app --host 0.0.0.0 --port 8000
```

## 4. Frontend Setup (React/Vite)

```bash
cd ui
npm install
```

## 5. Run Backend API

```bash
source venv/bin/activate
python -m uvicorn retrieval_api:app --host 0.0.0.0 --port 8000 --reload
```

## 6. Run Frontend UI

```bash
cd ui
npm run dev
```

Open `http://localhost:3000`

## 7. Common Issues

- **Out of Memory (OOM)**: If the GPU runs out of memory, ensure `LLM_LOAD_4BIT=true` is set.
- **CUDA Errors**: Ensure `torch` is installed with the correct CUDA version for your Linux instance.
