import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from typing import Dict, Tuple

from context_builder import build_context_pack, load_acts_chunk_lookup, run_retrieval
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = "llama_outputs"


def safe_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return (cleaned[:80] or "query").strip("_")


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


def build_llm_prompt(
    query: str,
    pack: Dict,
    priors: str = "",
    mode: str = "context_strong",
    priors_weight: str = "low",
    answer_mode: str = "answer_only",
    secondary_domains: str = "",
    statute_regime: str = "modern",
    conversation_history: str = "",
    known_facts: str = "",
) -> str:
    instruction = (
        "You are an Indian Litigation Strategist. "
        "PRIMARY RULE: Prioritize the provided context blocks for section/case specificity. "
        "REASONING RULE: If context is thin for foundational law principles, you may apply general Indian legal knowledge. "
        "Clearly distinguish: (a) retrieved-context grounded points and (b) general legal framework points. "
        "Do not invent fake sections, fake cases, or fabricated authorities. "
        "Never say you cannot answer when domain is clear; provide a practical General Legal Framework response instead. "
        "If relying on general legal principles, use cautious language like 'generally', 'typically', or 'subject to facts'. "
        "Prioritize practical remedies over academic citation dumps. "
        "Use concise, professional legal language focused on practical action. "
        "Do not output citation markers like [C1] or a citations section. "
        "When citing law, write human-readable references like 'Consumer Protection Act, 2019 - Section 2(47)'. "
        "If unsure about exact section number, refer to the Act generally instead of guessing section numbers. "
        "Always follow this rule: if some facts are missing, give the best possible legal answer first, then ask specific clarification questions. "
        "Never respond with only 'insufficient information'. "
        "STATUTE REGIME RULE: Always apply the modern Indian criminal framework: BNS, BNSS, and BSA. "
        "Do not refer to IPC, CrPC, or Indian Evidence Act unless explicitly required by user or context constraints. "
        "If a concept corresponds to older IPC framing, use the equivalent BNS/BNSS/BSA concept and focus on nature of offence over exact section number. "
        "Follow IRAC style but with these exact plain headings on separate lines: "
        "FACTS, LEGAL ISSUE, GROUNDS, ANALYSIS, PRAYER, LIMITS/UNCERTAINTY. "
        "Under GROUNDS, use only bullet lines in this exact format: "
        "- <Act Name> | Section <number or Unknown> | <one-line relevance>. "
        "Under PRAYER, provide numbered action steps (1., 2., 3.). "
        "For consumer matters, prefer remedy hierarchy: refund/replacement/repair first, compensation next, product liability only for injury/harm scenarios. "
        "If uncertain, output exactly: "
        "- No specific statutory provision identified with high confidence from retrieved context."
    )

    mode_instruction = (
        "MODE RULES:\n"
        f"- Current MODE: {mode}\n"
        f"- Legal Priors Importance: {priors_weight}\n"
        "- If MODE=context_strong: prioritize retrieved context and use priors only as support.\n"
        "- If MODE=context_weak: combine retrieved context with legal priors.\n"
        "- If MODE=priors_only: rely on legal priors/general framework and do not say 'insufficient context'.\n"
        "- If MODE=scenario: generate 2-3 plausible legal scenarios and remedies for each.\n"
        "- If query includes multiple legal issues, address the primary issue first and then briefly mention alternatives.\n"
    )

    return (
        f"System Instruction:\n{instruction}\n\n"
        f"{mode_instruction}\n"
        f"Domain Priors:\n{(priors or 'None')}\n\n"
        f"Statute Regime:\n{statute_regime}\n\n"
        f"Answer Mode:\n{answer_mode}\n\n"
        f"Secondary Domains (if any):\n{(secondary_domains or 'None')}\n\n"
        f"Conversation History:\n{(conversation_history or 'None')}\n\n"
        f"Known Facts:\n{(known_facts or 'None')}\n\n"
        "If multiple legal remedies exist, include a clear 'Alternative Legal Actions' segment after primary remedy.\n\n"
        f"User Query:\n{query}\n\n"
        f"Context:\n{pack['prompt_context']}\n\n"
        "Now produce the legal answer in the required structure."
    )


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


def save_outputs(query: str, model: str, pack: Dict, answer: str) -> Tuple[str, str, str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = safe_filename(query)

    context_path = os.path.join(OUT_DIR, f"{ts}_{slug}_context.json")
    answer_path = os.path.join(OUT_DIR, f"{ts}_{slug}_answer.md")
    meta_path = os.path.join(OUT_DIR, f"{ts}_{slug}_meta.json")

    with open(context_path, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    with open(answer_path, "w", encoding="utf-8") as f:
        f.write(answer.strip() + "\n")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "query": query,
                "model": model,
                "generated_at": ts,
                "context_path": context_path,
                "answer_path": answer_path,
                "citations_available": [c["citation_id"] for c in pack.get("citations", [])],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return context_path, answer_path, meta_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate legal answers using context builder + LLaMA (Ollama)")
    p.add_argument("--q", required=True)
    p.add_argument("--corpus", choices=["acts", "judgements", "all"], default="all")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--dense-k", type=int, default=100)
    p.add_argument("--bm25-k", type=int, default=100)
    p.add_argument("--dense-weight", type=float, default=0.6)
    p.add_argument("--bm25-weight", type=float, default=0.4)
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    p.add_argument("--rerank-top-n", type=int, default=50)
    p.add_argument("--rerank-batch-size", type=int, default=16)
    p.add_argument("--max-context-chars", type=int, default=45000)

    p.add_argument("--llm-model", default="sarvamai/sarvam-30b")
    p.add_argument("--llm-timeout-sec", type=int, default=300)
    p.add_argument("--no-llm", action="store_true", help="Build context only; skip model generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results = run_retrieval(args)
    acts_lookup = load_acts_chunk_lookup("JSON_acts")
    pack = build_context_pack(
        query=args.q,
        results=results,
        acts_lookup=acts_lookup,
        max_chars=args.max_context_chars,
    )

    prompt = build_llm_prompt(
        args.q,
        pack,
        priors="",
        mode="context_strong",
        priors_weight="low",
        answer_mode="answer_only",
        secondary_domains="",
        statute_regime="modern",
        conversation_history="",
        known_facts="",
    )

    if args.no_llm:
        answer = "[DRY RUN] Context prepared. LLM generation skipped (--no-llm)."
    else:
        answer = call_llm(args.llm_model, prompt, timeout_sec=args.llm_timeout_sec)

    context_path, answer_path, meta_path = save_outputs(args.q, args.llm_model, pack, answer)

    print(f"Retrieved results: {len(results)}")
    print(f"Context blocks used: {len(pack.get('context_blocks', []))}")
    print(f"Saved context: {context_path}")
    print(f"Saved answer: {answer_path}")
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
