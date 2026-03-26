import argparse
import json
import os
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from context_builder import build_context_pack, load_acts_chunk_lookup, run_retrieval

OUT_DIR = "llama_outputs"

# Global placeholders for the model and tokenizer
_model = None
_tokenizer = None


def get_model_and_tokenizer(model_name: str):
    global _model, _tokenizer
    if _model is None:
        print(f"Loading model: {model_name} (this may take a while)...")
        
        # Cloud/Linux Optimization: Support quantization via environment variables
        load_in_4bit = os.getenv("LLM_LOAD_4BIT", "false").lower() == "true"
        load_in_8bit = os.getenv("LLM_LOAD_8BIT", "false").lower() == "true"
        
        kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        
        if load_in_4bit:
            print("Enabling 4-bit quantization...")
            kwargs["load_in_4bit"] = True
        elif load_in_8bit:
            print("Enabling 8-bit quantization...")
            kwargs["load_in_8bit"] = True
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return _model, _tokenizer


def safe_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return (cleaned[:80] or "query").strip("_")


def build_llm_prompt(query: str, context: str) -> str:
    return f"""
You are an AI Legal Assistant trained to help users understand Indian law.

STRICT RULES:
- Use ONLY the provided context
- Do NOT hallucinate laws or sections
- If insufficient data, say so clearly

FORMAT:
Issue:
Relevant Law / Sections:
Explanation:
Conclusion:

-----------------------------------

USER QUERY:
{query}

LEGAL CONTEXT:
{context}

-----------------------------------

Generate a complete legal answer.
"""


def call_llm(model_name: str, prompt: str, timeout_sec: int) -> str:
    """
    Calls the LLM using Transformers (Sarvam-30b backend).
    """
    model, tokenizer = get_model_and_tokenizer(model_name)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Simple generation parameters; can be tuned
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Extract only the generated part (after the prompt)
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True)
    parser.add_argument("--llm-model", default="sarvamai/sarvam-30b")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    results = run_retrieval(args)
    acts_lookup = load_acts_chunk_lookup("JSON_acts")

    pack = build_context_pack(
        query=args.q,
        results=results,
        acts_lookup=acts_lookup,
        max_chars=45000,
    )

    context_text = "\n\n".join([
        f"{c.get('title')} - Section {c.get('section_number')}\n{(c.get('texts', {}) or {}).get('chunk_text', '')}"
        for c in pack.get("context_blocks", [])
    ])

    prompt = build_llm_prompt(args.q, context_text)
    answer = call_llm(args.llm_model, prompt, args.timeout)

    print("\n===== ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    main()
