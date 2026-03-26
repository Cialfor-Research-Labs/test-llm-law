import os
import shutil
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- CONFIGURATION ---
MODEL_ID = "sarvamai/sarvam-30b"
CACHE_DIR = "/workspace/hf_cache"

# Redirect HF Cache to /workspace (where there's usually more space)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

def check_disk_space(path="/workspace"):
    total, used, free = shutil.disk_usage(path)
    free_gb = free // (2**30)
    print(f"--- DISK CHECK for {path} ---")
    print(f"Total: {total // (2**30)} GB")
    print(f"Used: {used // (2**30)} GB")
    print(f"Free: {free_gb} GB")
    
    if free_gb < 120:
        print("\n[WARNING] Less than 120GB available. Downloading an 80GB model might fail during extraction!")
        print("Please ensure you have enough space on Vast.ai (usually via instance configuration).")
    else:
        print("\n[INFO] Disk space looks sufficient for download.")

def load_model_optimized():
    print(f"\n--- STARTING OPTIMIZED LOAD FOR {MODEL_ID} ---")
    
    # 1. Disk Check
    check_disk_space()
    
    # 2. Tokenizer
    print("\n1. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # 3. Quantization Config
    # SET THIS TO '8-bit' for A100 users, '4-bit' for smaller GPUs
    load_mode = os.getenv("LOAD_MODE", "4bit") 
    
    if load_mode == "8bit":
        print("2. Configuring 8-bit quantization (A100/A6000 Recommended)...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        print("2. Configuring 4-bit quantization (General fallback)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # 4. Final Model Load
    print("3. Loading Model into VRAM (this expects ~20GB free VRAM)...")
    try:
        # We add 'llm_int8_enable_fp32_cpu_offload=True' and offload support in case of small VRAM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            # Fallback for small GPUs
            llm_int8_enable_fp32_cpu_offload=True, 
            max_memory={0: "20GB", "cpu": "30GB"} # Adjust based on your GPU
        )
        print("\n[SUCCESS] Model loaded successfully!")
        
        # Quick sanity check
        prompt = "Explain the importance of the Indian Constitution."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print("\nChecking generation...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        if "Out of memory" in str(e):
            print("TIP: Your GPU might not have enough VRAM even for 4-bit (~20GB required).")

if __name__ == "__main__":
    load_model_optimized()
