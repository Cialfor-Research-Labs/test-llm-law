import os
import traceback
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from context_builder import build_context_pack, load_acts_chunk_lookup, run_retrieval
from llama_legal_answer import call_llm

# 🔥 IMPORTANT: Updated import
from dynamic_intake_engine import handle_query as handle_dynamic_intake


app = FastAPI(title="Legal RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST / RESPONSE MODELS
# =========================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    llm_model: str = "sarvamai/sarvam-30b"
    llm_timeout_sec: int = 300


class QueryResponse(BaseModel):
    ok: bool
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    context_blocks: List[Dict[str, Any]]
    meta: Dict[str, Any]


# =========================
# CLEAN LEGAL PROMPT
# =========================

def build_clean_prompt(query: str, context_blocks: List[Dict]) -> str:
    context_text = "\n\n".join([
        f"{c.get('title')} - Section {c.get('section_number')}\n"
        f"{(c.get('texts', {}) or {}).get('chunk_text', '')}"
        for c in context_blocks
    ])

    return f"""
You are an AI Legal Assistant trained to help users understand Indian law.

STRICT RULES:
- Use ONLY the provided legal context
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
{context_text}

-----------------------------------

Generate a complete legal answer.
"""


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():
    return {
        "ok": True,
        "time": datetime.utcnow().isoformat() + "Z"
    }


# =========================
# MAIN QUERY ENDPOINT
# =========================

@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    try:
        user_query = payload.query

        # 🔍 Retrieval
        args = SimpleNamespace(
            q=user_query,
            corpus="all",
            top_k=5,
            dense_k=100,
            bm25_k=100,
            dense_weight=0.6,
            bm25_weight=0.4,
            rerank=False,
            max_context_chars=14000
        )

        results = run_retrieval(args)

        acts_lookup = load_acts_chunk_lookup("JSON_acts")

        pack = build_context_pack(
            query=user_query,
            results=results,
            acts_lookup=acts_lookup,
            max_chars=14000,
        )

        # 🧠 Build Prompt
        prompt = build_clean_prompt(
            user_query,
            pack.get("context_blocks", [])
        )

        # 🤖 LLM Call
        answer = call_llm(
    model_name=payload.llm_model,
    prompt=prompt,
    timeout_sec=payload.llm_timeout_sec,
)

        return QueryResponse(
            ok=True,
            query=user_query,
            answer=answer,
            citations=pack.get("citations", []),
            context_blocks=pack.get("context_blocks", []),
            meta={
                "context_used": len(pack.get("context_blocks", [])),
                "time": datetime.utcnow().isoformat() + "Z"
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


# =========================
# DYNAMIC INTAKE (OPTIONAL)
# =========================

@app.post("/dynamic_intake")
def dynamic_intake(user_input: str):
    try:
        result = handle_dynamic_intake(user_input)
        return {
            "ok": True,
            "answer": result["text"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# RUN SERVER
# =========================

