import argparse
import json
import os
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Tuple

from context_builder import build_context_pack, load_acts_chunk_lookup, run_retrieval

OUT_DIR = "llama_outputs"


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


def call_ollama(model: str, prompt: str, timeout_sec: int) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    url = f"{host.rstrip('/')}/api/generate"

    req_data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 16000
        }
    }).encode("utf-8")

    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True)
    parser.add_argument("--llm-model", default="llama3.1")
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
    answer = call_ollama(args.llm_model, prompt, args.timeout)

    print("\n===== ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    main()