import os
import re
import traceback
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from context_builder import build_context_pack, load_acts_chunk_lookup, run_retrieval
from answer_validator import (
    detect_domain_contamination,
    detect_era_mismatch,
    validate_applicable_law_section,
    validate_output_structure,
)
from fact_extractor import FactExtraction, extract_facts
from legal_router import DOMAIN_KEYWORDS, build_intent_route, classify_legal_issue, entity_override, get_statute_regime
from llama_legal_answer import build_llm_prompt, call_llm
from llm_judge import llm_relevance_judge
from safe_fallback import domain_safe_fallback
from sub_question_engine import run_subquestion_retrieval
from schema_intake_engine import FactStore, handle_input, load_schema_registry
from dynamic_intake_engine import (
    FactStore as DynamicFactStore,
    handle_dynamic_intake,
    load_dynamic_config,
)


APP_NAME = "Legal RAG API"
APP_VERSION = "0.1.0"
SESSION_STORE: Dict[str, Dict[str, Any]] = {}
SCHEMA_SESSIONS: Dict[str, Dict[str, Any]] = {}
SCHEMAS = load_schema_registry()
DYNAMIC_INTAKE_SESSIONS: Dict[str, Dict[str, Any]] = {}
DYNAMIC_INTAKE_CONFIG = load_dynamic_config()

app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    session_id: Optional[str] = None
    enable_intake_mode: bool = Field(default=True)
    reset_session: bool = Field(default=False)
    corpus: str = Field(default="acts", pattern="^(acts)$")
    legal_domain: str = Field(
        default="auto",
        pattern="^(auto|general|property|consumer|criminal|labour|contract)$",
    )
    top_k: int = Field(default=12, ge=1, le=100)
    dense_k: int = Field(default=100, ge=1, le=500)
    bm25_k: int = Field(default=100, ge=1, le=500)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    rerank: bool = Field(default=True)
    rerank_model: str = Field(default="BAAI/bge-reranker-base")
    rerank_top_n: int = Field(default=50, ge=1, le=500)
    rerank_batch_size: int = Field(default=16, ge=1, le=128)
    max_context_chars: int = Field(default=45000, ge=2000, le=150000)
    generate_answer: bool = Field(default=True)
    llm_model: str = Field(default="sarvamai/sarvam-30b")
    llm_timeout_sec: int = Field(default=300, ge=30, le=1800)
    strict_legal_validation: bool = Field(default=True)
    require_procedural_facts: bool = Field(default=True)
    max_regen_attempts: int = Field(default=1, ge=0, le=2)
    use_llm_judge: bool = Field(default=True)
    llm_judge_top_n: int = Field(default=5, ge=3, le=10)
    use_subquestions: bool = Field(default=True)
    subquestion_top_k: int = Field(default=8, ge=3, le=30)


class Citation(BaseModel):
    citation_id: str
    title: Optional[str] = None
    section_number: Optional[str] = None
    context_path: Optional[str] = None
    source_file: Optional[str] = None
    chunk_id: Optional[str] = None


class QueryResponse(BaseModel):
    ok: bool
    query: str
    answer: Optional[str] = None
    prompt_context: str
    citations: List[Citation]
    context_blocks: List[Dict[str, Any]]
    meta: Dict[str, Any]


class ExtractFactsRequest(BaseModel):
    query: str = Field(..., min_length=3)
    llm_model: str = Field(default="sarvamai/sarvam-30b")
    llm_timeout_sec: int = Field(default=120, ge=30, le=600)


class ExtractFactsResponse(BaseModel):
    ok: bool
    query: str
    facts: FactExtraction
    meta: Dict[str, Any]


class SchemaIntakeRequest(BaseModel):
    user_input: str = Field(..., min_length=2)
    session_id: Optional[str] = None
    reset_session: bool = Field(default=False)
    llm_model: str = Field(default="sarvamai/sarvam-30b")
    llm_timeout_sec: int = Field(default=180, ge=30, le=900)


class SchemaIntakeResponse(BaseModel):
    ok: bool
    session_id: str
    mode: str
    case_type: str
    text: str
    missing_fields: List[str]
    questions: List[str]
    facts: Dict[str, Any]


class DynamicIntakeRequest(BaseModel):
    user_input: str = Field(..., min_length=2)
    session_id: Optional[str] = None
    reset_session: bool = Field(default=False)
    llm_model: str = Field(default="sarvamai/sarvam-30b")
    llm_timeout_sec: int = Field(default=180, ge=30, le=900)


class DynamicIntakeResponse(BaseModel):
    ok: bool
    session_id: str
    mode: str
    task: str
    domain: str
    signals: List[str]
    signal_confidence: Dict[str, float]
    strategy_tracks: List[str]
    routed_handler: Optional[str] = None
    rag_used: bool
    retrieval_queries: List[str]
    retrieved_context_count: int
    retrieved_citations: List[Dict[str, Any]]
    required_field_count: int
    missing_fields: List[str]
    questions: List[str]
    text: str
    facts: Dict[str, Any]


class ReasoningRequest(BaseModel):
    user_input: str = Field(..., min_length=2)
    session_id: Optional[str] = None
    reset_session: bool = Field(default=False)
    llm_model: str = Field(default="sarvamai/sarvam-30b")


class ReasoningResponse(BaseModel):
    ok: bool
    session_id: str
    text: str


INTAKE_QUESTION_MAP: Dict[str, str] = {
    "incident_date": "When did the incident or cause of action arise? Please share the date if you know it.",
    "claim_amount": "What is the approximate claim amount in INR?",
    "requested_relief": "What outcome do you want to pursue: refund, replacement, repair, compensation, or another remedy?",
    "injury": "Did the incident cause any physical injury? If yes, what type?",
    "invoice": "Do you have proof of purchase (invoice or receipt)?",
    "warranty": "Was the product under warranty?",
    "seller_response": "Did the seller or manufacturer respond, refuse, or ignore your complaint?",
}


def _get_or_create_session(session_id: Optional[str], reset: bool = False) -> tuple[str, Dict[str, Any]]:
    sid = session_id or str(uuid.uuid4())
    if reset or sid not in SESSION_STORE:
        SESSION_STORE[sid] = {
            "history": [],
            "facts": {
                "incident_date": {"value": None, "confidence": 0.0},
                "claim_amount": {"value": None, "confidence": 0.0},
                "seller_response": {"value": None, "confidence": 0.0},
                "requested_relief": {"value": None, "confidence": 0.0},
                "injury": {"value": None, "confidence": 0.0},
                "invoice": {"value": None, "confidence": 0.0},
                "warranty": {"value": None, "confidence": 0.0},
            },
            "mode": "question",
            "asked_keys": [],
            "pending_fields": [],
        }
    return sid, SESSION_STORE[sid]


def _bool_from_text(text: str, pos: List[str], neg: List[str]) -> Optional[bool]:
    t = (text or "").lower()
    if any(n in t for n in neg):
        return False
    if any(p in t for p in pos):
        return True
    return None


def _merge_session_facts(session: Dict[str, Any], user_input: str, extracted: Optional[FactExtraction]) -> None:
    facts = session["facts"]
    t = (user_input or "").lower()

    def update_field(key: str, val: Any, conf: float):
        current = facts.get(key, {"value": None, "confidence": 0.0})
        if val is not None and conf >= current["confidence"]:
            facts[key] = {"value": val, "confidence": conf}

    if extracted:
        update_field("incident_date", extracted.incident_date, extracted.date_confidence)
        update_field("claim_amount", extracted.claim_amount_inr, extracted.amount_confidence)
        update_field("seller_response", extracted.seller_response, extracted.seller_confidence)
        update_field("requested_relief", extracted.requested_relief, extracted.relief_confidence)
        if extracted.injury_signal:
            update_field("injury", True, extracted.injury_confidence or 0.8)

    # Heuristic overrides (always 1.0 confidence or high enough to stick)
    injury = _bool_from_text(
        t,
        pos=["injury", "injured", "burn", "burns", "exploded", "explosion", "wound", "fracture"],
        neg=["no injury", "not injured", "unhurt"],
    )
    if injury is not None:
        update_field("injury", injury, 0.9)

    invoice = _bool_from_text(
        t,
        pos=["invoice", "receipt", "bill", "proof of purchase", "i have invoice", "have receipt"],
        neg=["no invoice", "lost invoice", "no receipt", "don't have invoice", "dont have invoice"],
    )
    if invoice is not None:
        update_field("invoice", invoice, 0.9)

    if any(k in t for k in ["warranty", "under warranty"]):
        warranty = _bool_from_text(
            t,
            pos=["under warranty", "in warranty", "warranty valid", "yes warranty"],
            neg=["no warranty", "not under warranty", "warranty expired", "not sure about warranty", "unknown warranty"],
        )
        if warranty is not None:
            update_field("warranty", warranty, 0.9)

    if any(k in t for k in ["seller", "manufacturer", "response", "refused", "ignored", "no response"]):
        if any(k in t for k in ["refused", "deny", "denied", "ignored", "no response", "not help"]):
            update_field("seller_response", "refused_or_no_response", 0.85)
        elif any(k in t for k in ["replaced", "refunded", "accepted", "responded"]):
            update_field("seller_response", "cooperative", 0.85)

    if "ignore" in t:
        update_field("seller_response", "ignored", 0.9)

    # Map very short yes/no replies to pending asked fields
    tokens = [x for x in re.split(r"\s+", t.strip()) if x]
    pending = list(session.get("pending_fields", []))
    if len(tokens) <= 4 and pending:
        yn = _bool_from_text(
            t,
            pos=["yes", "yeah", "yep", "have", "available", "done"],
            neg=["no", "not", "don't", "dont", "none"],
        )
        if yn is not None:
            key = str(pending[0])
            if key in {"invoice", "warranty", "injury"}:
                update_field(key, bool(yn), 1.0)
            elif key == "seller_response":
                update_field(key, "cooperative" if yn else "refused_or_no_response", 1.0)
            elif key == "requested_relief":
                update_field(key, "refund" if yn else "other", 1.0)


def detect_task(user_input: str) -> str:
    t = (user_input or "").lower()
    if any(k in t for k in ["draft notice", "legal notice", "send notice", "issue notice"]):
        return "draft_notice"
    if any(k in t for k in ["draft complaint", "consumer complaint", "file complaint", "prepare complaint"]):
        return "complaint"
    if any(k in t for k in ["estimate claim", "claim estimate", "how much can i claim", "value of my claim"]):
        return "estimate"
    return "advice"


def get_required_facts(domain: str, facts: Dict[str, Any]) -> List[str]:
    if domain != "consumer":
        return []

    required = [
        "incident_date",
        "claim_amount",
        "seller_response",
        "requested_relief",
    ]
    if facts.get("injury", {}).get("value") is True:
        required.append("injury")
    return required


def get_task_required_facts(task: str, domain: str, facts: Dict[str, Any]) -> List[str]:
    if domain != "consumer":
        return []
    if task == "draft_notice":
        return ["incident_date", "seller_response", "requested_relief"]
    if task == "complaint":
        return ["incident_date", "claim_amount", "seller_response", "requested_relief"]
    if task == "estimate":
        return ["claim_amount", "requested_relief"]
    return get_required_facts(domain, facts)


def generate_legal_notice(facts: Dict[str, Any]) -> str:
    incident_date = facts.get("incident_date") or "Unknown incident date"
    seller_response = facts.get("seller_response") or "No seller response recorded"
    requested_relief = facts.get("requested_relief") or "appropriate legal relief"
    claim_amount = facts.get("claim_amount")
    amount_line = f"Claim amount asserted: INR {claim_amount}." if claim_amount else "Claim amount is not yet specified."
    return (
        "LEGAL NOTICE\n\n"
        f"Incident Date\n{incident_date}\n\n"
        "Facts\n"
        "The claimant reports a consumer dispute and seeks formal redress.\n\n"
        "Seller Response\n"
        f"{seller_response}\n\n"
        "Demand\n"
        f"The claimant seeks {requested_relief}. {amount_line}\n\n"
        "Compliance\n"
        "You are called upon to resolve the grievance within 15 days, failing which appropriate proceedings may be initiated."
    )


def generate_complaint(facts: Dict[str, Any]) -> str:
    incident_date = facts.get("incident_date") or "Unknown incident date"
    seller_response = facts.get("seller_response") or "No seller response recorded"
    requested_relief = facts.get("requested_relief") or "appropriate relief"
    claim_amount = facts.get("claim_amount")
    amount_line = f"Approximate claim amount: INR {claim_amount}." if claim_amount else "Claim amount not yet specified."
    return (
        "CONSUMER COMPLAINT DRAFT\n\n"
        "FACTS\n"
        f"Cause of action date: {incident_date}.\n"
        f"Seller/manufacturer response: {seller_response}.\n"
        f"{amount_line}\n\n"
        "CAUSE OF ACTION\n"
        "A consumer dispute is asserted on the available facts.\n\n"
        "RELIEFS SOUGHT\n"
        f"The complainant seeks {requested_relief}.\n\n"
        "PRAYER\n"
        "The appropriate Consumer Commission may grant relief according to the pleaded facts and supporting documents."
    )


def generate_claim_estimate(facts: Dict[str, Any]) -> str:
    requested_relief = facts.get("requested_relief") or "consumer relief"
    claim_amount = facts.get("claim_amount")
    amount_line = f"Current known monetary figure: INR {claim_amount}." if claim_amount else "No claim amount has been captured yet."
    return (
        "CLAIM ESTIMATE\n\n"
        "ESTIMATE BASIS\n"
        f"The current estimate is being considered for {requested_relief}.\n\n"
        "AVAILABLE FIGURES\n"
        f"{amount_line}\n\n"
        "NEXT STEP\n"
        "Use invoices, repair costs, replacement value, and compensation heads to refine the claim range."
    )


def route_task(task: str, facts: Dict[str, Any]) -> Optional[str]:
    if task == "draft_notice":
        return generate_legal_notice(facts)
    if task == "complaint":
        return generate_complaint(facts)
    if task == "estimate":
        return generate_claim_estimate(facts)
    return None


def _format_question_response(session_facts: Dict[str, Any], keys: List[str]) -> str:
    lines = ["To assess your legal options, I need a few details:"]
    for i, key in enumerate(keys, start=1):
        field = session_facts.get(key, {"value": None, "confidence": 0.0})
        # If confidence is between 0.4 and 0.7, ask to confirm
        if 0.4 <= field["confidence"] < 0.7:
            val = field["value"]
            if key == "seller_response":
                val_text = "ignored" if val == "ignored" else "not responded"
                lines.append(f"{i}. Just to confirm, has the seller {val_text}?")
            elif key == "requested_relief":
                lines.append(f"{i}. Are you specifically looking for a {val}?")
            else:
                lines.append(f"{i}. {INTAKE_QUESTION_MAP.get(key, key)}")
        else:
            lines.append(f"{i}. {INTAKE_QUESTION_MAP.get(key, key)}")
    return "\n".join(lines)


def _history_to_text(history: List[Dict[str, str]], max_items: int = 6) -> str:
    if not history:
        return ""
    tail = history[-max_items:]
    return "\n".join([f"{m.get('role', 'user')}: {m.get('text', '')}" for m in tail])


def _facts_to_text(facts: Dict[str, Any]) -> str:
    parts = []
    for k in ["incident_date", "claim_amount", "requested_relief", "injury", "invoice", "warranty", "seller_response"]:
        parts.append(f"- {k}: {facts.get(k)}")
    return "\n".join(parts)


def _get_or_create_schema_session(session_id: Optional[str], reset: bool = False) -> tuple[str, FactStore]:
    sid = session_id or str(uuid.uuid4())
    if reset or sid not in SCHEMA_SESSIONS:
        fs = FactStore()
        SCHEMA_SESSIONS[sid] = {
            "fact_data": {},
            "asked_keys": [],
        }
        return sid, fs

    stored = SCHEMA_SESSIONS[sid]
    fs = FactStore()
    fs.data = dict(stored.get("fact_data", {}))
    for k in stored.get("asked_keys", []):
        fs.mark_asked(k)
    return sid, fs


def _save_schema_session(session_id: str, fact_store: FactStore) -> None:
    SCHEMA_SESSIONS[session_id] = {
        "fact_data": dict(fact_store.data),
        "asked_keys": sorted(list(getattr(fact_store, "_asked", set()))),
    }


def _get_or_create_dynamic_session(session_id: Optional[str], reset: bool = False) -> tuple[str, DynamicFactStore]:
    sid = session_id or str(uuid.uuid4())
    if reset or sid not in DYNAMIC_INTAKE_SESSIONS:
        fs = DynamicFactStore()
        DYNAMIC_INTAKE_SESSIONS[sid] = {"fact_data": {}, "asked_keys": [], "task": None}
        return sid, fs

    stored = DYNAMIC_INTAKE_SESSIONS[sid]
    fs = DynamicFactStore()
    fs.data = dict(stored.get("fact_data", {}))
    if stored.get("task"):
        fs.update("task", stored.get("task"))
    for k in stored.get("asked_keys", []):
        fs.mark_asked(k)
    return sid, fs


def _save_dynamic_session(session_id: str, fact_store: DynamicFactStore) -> None:
    DYNAMIC_INTAKE_SESSIONS[session_id] = {
        "fact_data": dict(fact_store.data),
        "asked_keys": sorted(list(getattr(fact_store, "_asked", set()))),
        "task": fact_store.get("task"),
    }


def determine_reasoning_mode(pack: Dict[str, Any]) -> str:
    blocks = pack.get("context_blocks", [])
    count = len(blocks)
    if count == 0:
        return "priors_only"

    raw_scores: List[float] = []
    for b in blocks:
        scores = b.get("scores") or {}
        val = scores.get("final_score", scores.get("hybrid_score"))
        if isinstance(val, (int, float)):
            raw_scores.append(float(val))

    if not raw_scores:
        avg_score = 0.0
    else:
        max_score = max(raw_scores)
        if max_score > 1.0:
            normalized = [s / max_score for s in raw_scores]
            avg_score = sum(normalized) / len(normalized)
        else:
            avg_score = sum(raw_scores) / len(raw_scores)

    if count >= 5 and avg_score > 0.65:
        return "context_strong"
    if count >= 2 and avg_score > 0.4:
        return "context_weak"
    return "priors_only"


def priors_weight_for_mode(mode: str) -> str:
    if mode == "priors_only":
        return "high"
    if mode == "scenario":
        return "high"
    if mode == "context_weak":
        return "medium"
    return "low"


def sanity_check(answer: str, domain: str) -> Dict[str, Any]:
    if not answer:
        return {"ok": True, "reason": "empty_answer"}

    lower = answer.lower()
    if domain == "consumer":
        if "product liability" in lower and "injury" not in lower:
            return {"ok": False, "reason": "product_liability_without_injury"}
        if "criminal" in lower or "crpc" in lower or "bnss" in lower:
            return {"ok": False, "reason": "consumer_cross_domain_criminal"}
    if domain == "criminal":
        if "refund" in lower:
            return {"ok": False, "reason": "criminal_with_consumer_refund"}
    return {"ok": True, "reason": "passed"}


def score_answer(answer: str) -> float:
    if not answer:
        return 0.0
    lower = answer.lower()
    headings = ["facts", "legal issue", "grounds", "analysis", "prayer", "limits/uncertainty"]
    heading_hits = sum(1 for h in headings if h in lower)
    section_hits = lower.count("section ")
    has_raw_markers = 1 if ("[c1]" in lower or "[c2]" in lower or "[c3]" in lower) else 0
    return float(len(answer)) + (40.0 * heading_hits) + (12.0 * section_hits) - (120.0 * has_raw_markers)


def contradiction_check(answer: str, domain: str) -> Dict[str, Any]:
    if not answer:
        return {"has_contradiction": False, "reason": "empty_answer"}
    text = answer.lower()

    if "criminal" in text and "refund" in text:
        return {"has_contradiction": True, "reason": "criminal_and_refund_mixed"}
    if "civil suit" in text and "consumer complaint" in text:
        return {"has_contradiction": True, "reason": "civil_and_consumer_primary_mixed"}

    # Domain-specific contradiction checks.
    if domain == "consumer" and ("bail" in text or "fir" in text):
        return {"has_contradiction": True, "reason": "consumer_answer_with_criminal_primary"}
    if domain == "criminal" and ("district commission" in text or "consumer commission" in text):
        return {"has_contradiction": True, "reason": "criminal_answer_with_consumer_forum"}

    return {"has_contradiction": False, "reason": "none"}


LAW_MAPPING = {
    "Indian Penal Code": "Bharatiya Nyaya Sanhita (BNS)",
    "IPC": "BNS",
    "Code of Criminal Procedure": "Bharatiya Nagarik Suraksha Sanhita (BNSS)",
    "CrPC": "BNSS",
    "Indian Evidence Act": "Bharatiya Sakshya Adhiniyam (BSA)",
    "Evidence Act": "BSA",
}


def normalize_laws(answer: str) -> str:
    if not answer:
        return answer
    normalized = answer
    for old, new in LAW_MAPPING.items():
        normalized = re.sub(r"\b" + re.escape(old) + r"\b", new, normalized, flags=re.IGNORECASE)
    return normalized


def statute_sanity(answer: str) -> Dict[str, Any]:
    if not answer:
        return {"ok": True, "reason": "empty_answer"}
    text = answer.lower()
    legacy_hits = []
    for term in ["ipc", "crpc", "indian penal code", "code of criminal procedure", "indian evidence act"]:
        if term in text:
            legacy_hits.append(term)
    if legacy_hits:
        return {"ok": False, "reason": "legacy_statute_leakage", "terms": legacy_hits}
    return {"ok": True, "reason": "modern_only"}


def detect_garbage_context(results: List[Dict[str, Any]], domain: str) -> bool:
    keywords = {
        "consumer": ["consumer", "defect", "refund", "injury", "product"],
        "property": ["tenant", "rent", "lease", "possession"],
        "labour": ["employee", "salary", "termination", "wages"],
        "criminal": ["fir", "police", "offence", "arrest"],
    }
    domain_terms = keywords.get(domain, [])
    if not results or not domain_terms:
        return False

    hits = 0
    for r in results:
        text = str(r.get("chunk_text", "")).lower()
        if any(t in text for t in domain_terms):
            hits += 1
    ratio = hits / max(len(results), 1)
    return ratio < 0.3


def is_user_uncertain(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["don't know", "dont know", "not sure", "unknown"])


def reconcile_domain(query: str, results: List[Dict[str, Any]], domain: str) -> str:
    q = (query or "").lower()
    combined = " ".join(str(r.get("chunk_text", "")) for r in results).lower()
    if "tenant" in q or "landlord" in q:
        return "property"
    if "consumer protection act" in combined:
        return "consumer"
    return domain


def should_ask_first(facts: Optional[FactExtraction]) -> bool:
    if not facts:
        return False
    return (not facts.incident_date) and (facts.claim_amount_inr is None)


def detect_secondary_domains(query: str, primary_domain: str) -> List[str]:
    q = (query or "").lower()
    hits: List[tuple[str, int]] = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if domain == primary_domain:
            continue
        count = sum(1 for kw in keywords if kw in q)
        if count > 0:
            hits.append((domain, count))
    hits.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in hits[:2]]


def compute_confidence(
    mode: str,
    validation_meta: Dict[str, Any],
    failure_reason: List[str],
    context_blocks: List[Dict[str, Any]],
) -> float:
    score = 1.0
    if mode == "priors_only":
        score -= 0.35
    elif mode == "context_weak":
        score -= 0.15

    if failure_reason:
        score -= 0.1 * len(set(failure_reason))

    source_conf = (validation_meta or {}).get("source_confidence", {})
    if source_conf.get("unverified"):
        score -= 0.1

    if len(context_blocks or []) < 3:
        score -= 0.1

    return max(0.0, min(score, 1.0))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "time": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    try:
        user_query = payload.query
        normalized_query = normalize_laws(user_query)
        session_id = None
        session = None
        if payload.enable_intake_mode:
            session_id, session = _get_or_create_session(payload.session_id, reset=payload.reset_session)
            session["history"].append({"role": "user", "text": user_query})

        if abs((payload.dense_weight + payload.bm25_weight) - 1.0) > 1e-6:
            raise HTTPException(
                status_code=422,
                detail="dense_weight + bm25_weight must equal 1.0",
            )

        facts_intake = None
        facts_error = None
        try:
            facts_intake = extract_facts(
                query=normalized_query,
                llm_model=payload.llm_model,
                llm_timeout_sec=min(payload.llm_timeout_sec, 120),
            )
        except Exception as exc:
            facts_error = str(exc)

        args = SimpleNamespace(
            q=normalized_query,
            corpus=payload.corpus,
            legal_domain=payload.legal_domain,
            top_k=payload.top_k,
            dense_k=payload.dense_k,
            bm25_k=payload.bm25_k,
            dense_weight=payload.dense_weight,
            bm25_weight=payload.bm25_weight,
            rerank=payload.rerank,
            rerank_model=payload.rerank_model,
            rerank_top_n=payload.rerank_top_n,
            rerank_batch_size=payload.rerank_batch_size,
            max_context_chars=payload.max_context_chars,
        )

        route = classify_legal_issue(normalized_query)
        if payload.legal_domain == "auto":
            resolved_domain = (
                facts_intake.legal_domain
                if facts_intake and facts_intake.legal_domain not in ("", "general")
                else route.domain
            )
        else:
            resolved_domain = payload.legal_domain
        resolved_domain = entity_override(normalized_query, resolved_domain)
        intent_route = build_intent_route(normalized_query, forced_domain=resolved_domain)
        statute_regime = intent_route.get("statute_regime") or get_statute_regime()
        args.intent_route = intent_route
        task = detect_task(user_query)
        missing: List[str] = []
        if session is not None:
            _merge_session_facts(session=session, user_input=user_query, extracted=facts_intake)
            uncertain = is_user_uncertain(user_query)
            required = (
                get_task_required_facts(task, resolved_domain, session["facts"])
                if task != "advice"
                else get_required_facts(resolved_domain, session["facts"])
            )
            # Question logic: Only ask if field is missing OR confidence is low (< 0.7)
            missing = [k for k in required if session["facts"].get(k, {}).get("confidence", 0.0) < 0.7]

            if missing and not uncertain:
                asked_set = set(session.get("asked_keys", []))
                # Only ask keys that haven't been asked OR are low confidence
                ask_keys = [k for k in missing if k not in asked_set or session["facts"].get(k, {}).get("confidence", 0.0) < 0.7][:3]
                if ask_keys:
                    session["mode"] = "question"
                    session["asked_keys"] = sorted(list(asked_set.union(set(ask_keys))))
                    session["pending_fields"] = ask_keys
                    q_text = _format_question_response(session["facts"], ask_keys)
                    session["history"].append({"role": "assistant", "text": q_text})
                    return QueryResponse(
                        ok=True,
                        query=payload.query,
                        answer=q_text,
                        prompt_context="",
                        citations=[],
                        context_blocks=[],
                        meta={
                            "session_id": session_id,
                            "intake_mode": "question",
                            "task": task,
                            "required_facts": required,
                            "missing_facts": missing,
                            "known_facts": dict(session["facts"]),
                            "issue_domain": resolved_domain,
                            "time": datetime.utcnow().isoformat() + "Z",
                        },
                    )
                session["mode"] = "question"
                session["pending_fields"] = missing[:3]
                q_text = "I still need these details before I can proceed: " + ", ".join(session["pending_fields"])
                session["history"].append({"role": "assistant", "text": q_text})
                return QueryResponse(
                    ok=True,
                    query=payload.query,
                    answer=q_text,
                    prompt_context="",
                    citations=[],
                    context_blocks=[],
                    meta={
                        "session_id": session_id,
                        "intake_mode": "question",
                        "task": task,
                        "required_facts": required,
                        "missing_facts": missing,
                        "known_facts": dict(session["facts"]),
                        "issue_domain": resolved_domain,
                        "time": datetime.utcnow().isoformat() + "Z",
                    },
                )

            if task != "advice":
                routed_output = route_task(task, session["facts"])
                if routed_output:
                    session["mode"] = "answer"
                    session["pending_fields"] = []
                    session["history"].append({"role": "assistant", "text": routed_output})
                    return QueryResponse(
                        ok=True,
                        query=payload.query,
                        answer=routed_output,
                        prompt_context="",
                        citations=[],
                        context_blocks=[],
                        meta={
                            "session_id": session_id,
                            "intake_mode": "task_reuse",
                            "task": task,
                            "required_facts": required,
                            "missing_facts": missing,
                            "known_facts": dict(session["facts"]),
                            "issue_domain": resolved_domain,
                            "time": datetime.utcnow().isoformat() + "Z",
                        },
                    )

        subq_meta: Dict[str, Any] = {"used": False}
        if payload.use_subquestions:
            results, subq_meta = run_subquestion_retrieval(
                base_args=args,
                resolved_domain=resolved_domain,
                per_query_top_k=payload.subquestion_top_k,
            )
        else:
            results = run_retrieval(args)

        judge_meta: Dict[str, Any] = {"applied": False}
        if payload.use_llm_judge and results:
            results, judge_meta = llm_relevance_judge(
                query=payload.query,
                domain=resolved_domain,
                results=results,
                llm_model=payload.llm_model,
                timeout_sec=min(payload.llm_timeout_sec, 90),
                top_n=payload.llm_judge_top_n,
            )

        pre_failure_reasons: List[str] = []
        if detect_garbage_context(results, resolved_domain):
            pre_failure_reasons.append("garbage_retrieval")

        reconciled_domain = reconcile_domain(normalized_query, results, resolved_domain)
        if reconciled_domain != resolved_domain:
            resolved_domain = entity_override(normalized_query, reconciled_domain)
            intent_route = build_intent_route(normalized_query, forced_domain=resolved_domain)
            statute_regime = intent_route.get("statute_regime") or get_statute_regime()
            args.intent_route = intent_route
            pre_failure_reasons.append("domain_reconciled")

        secondary_domains = detect_secondary_domains(normalized_query, resolved_domain)

        acts_lookup = load_acts_chunk_lookup("JSON_acts")
        pack = build_context_pack(
            query=payload.query,
            results=results,
            acts_lookup=acts_lookup,
            max_chars=payload.max_context_chars,
        )
        reasoning_mode = determine_reasoning_mode(pack)
        if "garbage_retrieval" in pre_failure_reasons:
            reasoning_mode = "priors_only"
        if is_user_uncertain(payload.query):
            reasoning_mode = "scenario"
        priors_weight = priors_weight_for_mode(reasoning_mode)
        if float(intent_route.get("confidence", 0.0)) < 0.6 and priors_weight == "high":
            priors_weight = "medium"
        answer_mode = "answer_only"
        if reasoning_mode in {"context_weak", "scenario"} or should_ask_first(facts_intake):
            answer_mode = "answer_and_ask"
        pack["reasoning_mode"] = reasoning_mode
        pack["priors_weight"] = priors_weight
        pack["answer_mode"] = answer_mode

        answer = None
        llm_error = None
        contamination_meta: Dict[str, Any] = {"applied": False, "contaminated": False, "terms": []}
        structure_meta: Dict[str, Any] = {"applied": False, "valid": True, "missing_headings": []}
        era_meta: Dict[str, Any] = {"applied": False, "mismatch": False}
        sanity_meta: Dict[str, Any] = {"applied": False, "ok": True, "reason": "not_run"}
        contradiction_meta: Dict[str, Any] = {"applied": False, "has_contradiction": False, "reason": "not_run"}
        statute_meta: Dict[str, Any] = {"applied": False, "ok": True, "reason": "not_run"}
        failure_reasons: List[str] = list(pre_failure_reasons)
        if reasoning_mode == "priors_only":
            failure_reasons.append("low_context")
        regen_attempts = 0
        if payload.generate_answer:
            prompt = build_llm_prompt(
                normalized_query,
                pack,
                priors=intent_route.get("priors", ""),
                mode=reasoning_mode,
                priors_weight=priors_weight,
                answer_mode=answer_mode,
                secondary_domains=", ".join(secondary_domains),
                statute_regime=statute_regime,
                conversation_history=_history_to_text(session["history"]) if session else "",
                known_facts=_facts_to_text(session["facts"]) if session else "",
            )
            try:
                answer = call_llm(
                    model_name=payload.llm_model,
                    prompt=prompt,
                    timeout_sec=payload.llm_timeout_sec,
                )
                answer = normalize_laws(answer)
            except Exception as exc:
                # Keep retrieval usable even when local LLM daemon is unavailable.
                llm_error = str(exc)
                failure_reasons.append("llm_unavailable")

        validation_meta: Dict[str, Any] = {"applied": False}
        relax_validation = False
        if answer and payload.strict_legal_validation:
            answer, validation_meta = validate_applicable_law_section(
                answer=answer,
                pack=pack,
                domain=resolved_domain,
            )
            if validation_meta.get("fallback_used"):
                failure_reasons.append("validator_conflict")
                relax_validation = True

        if answer:
            while True:
                if "low_context" in failure_reasons:
                    reasoning_mode = "priors_only"
                    priors_weight = "high"
                    pack["reasoning_mode"] = reasoning_mode
                    pack["priors_weight"] = priors_weight
                if "validator_conflict" in failure_reasons:
                    relax_validation = True

                contamination_meta = detect_domain_contamination(answer=answer, domain=resolved_domain)
                structure_meta = validate_output_structure(answer=answer)
                era_meta = detect_era_mismatch(answer=answer, pack=pack, domain=resolved_domain)
                sanity_meta = sanity_check(answer=answer, domain=resolved_domain)
                sanity_meta["applied"] = True
                contradiction_meta = contradiction_check(answer=answer, domain=resolved_domain)
                contradiction_meta["applied"] = True
                statute_meta = statute_sanity(answer=answer)
                statute_meta["applied"] = True

                has_issue = (
                    contamination_meta.get("contaminated")
                    or not structure_meta.get("valid", True)
                    or era_meta.get("mismatch")
                    or not sanity_meta.get("ok", True)
                    or contradiction_meta.get("has_contradiction", False)
                    or not statute_meta.get("ok", True)
                )
                if not has_issue:
                    break

                if regen_attempts >= payload.max_regen_attempts:
                    answer = domain_safe_fallback(resolved_domain)
                    failure_reasons.append("max_regen_reached")
                    break

                regen_attempts += 1
                constraints: List[str] = []
                if contamination_meta.get("contaminated"):
                    failure_reasons.append("domain_confusion")
                    constraints.append(
                        "Do not discuss these off-domain terms: "
                        + ", ".join(contamination_meta.get("terms", []))
                    )
                if era_meta.get("mismatch"):
                    constraints.append(
                        "Do not reference Consumer Protection Act, 1986. Use Consumer Protection Act, 2019 only when relevant."
                    )
                if not structure_meta.get("valid", True):
                    missing = ", ".join(structure_meta.get("missing_headings", []))
                    constraints.append(
                        f"Output must contain all required headings. Missing currently: {missing}."
                    )
                if structure_meta.get("has_raw_citation_markers"):
                    constraints.append("Do not output raw citation markers like [C1], [C2].")
                if not sanity_meta.get("ok", True):
                    failure_reasons.append("sanity_check_failed")
                    if sanity_meta.get("reason") == "product_liability_without_injury":
                        constraints.append(
                            "Do not prioritize product liability unless injury, harm, or property damage is explicitly stated in user facts."
                        )
                    if sanity_meta.get("reason") == "consumer_cross_domain_criminal":
                        constraints.append("Do not mix criminal-track remedy as primary in consumer-domain answers.")
                    if sanity_meta.get("reason") == "criminal_with_consumer_refund":
                        constraints.append("Do not mix consumer refund remedies as primary in criminal-domain answers.")
                if contradiction_meta.get("has_contradiction", False):
                    failure_reasons.append("contradiction_detected")
                    constraints.append("Focus on the primary legal remedy. Avoid mixing unrelated legal tracks.")
                if not statute_meta.get("ok", True):
                    failure_reasons.append("legacy_statute_leakage")
                    constraints.append(
                        "Do not use IPC, CrPC, or Indian Evidence Act terminology. Use BNS, BNSS, and BSA framing."
                    )

                regen_prompt = (
                    build_llm_prompt(
                        normalized_query,
                        pack,
                        priors=intent_route.get("priors", ""),
                        mode=reasoning_mode,
                        priors_weight=priors_weight,
                        answer_mode=answer_mode,
                        secondary_domains=", ".join(secondary_domains),
                        statute_regime=statute_regime,
                        conversation_history=_history_to_text(session["history"]) if session else "",
                        known_facts=_facts_to_text(session["facts"]) if session else "",
                    )
                    + "\n\nHard constraints:\n- "
                    + "\n- ".join(constraints)
                )
                try:
                    old_answer = answer
                    old_validation = validation_meta

                    candidate_answer = call_llm(
                        model_name=payload.llm_model,
                        prompt=regen_prompt,
                        timeout_sec=payload.llm_timeout_sec,
                    )
                    candidate_answer = normalize_laws(candidate_answer)

                    candidate_validation = old_validation
                    if payload.strict_legal_validation and not relax_validation:
                        candidate_answer, candidate_validation = validate_applicable_law_section(
                            answer=candidate_answer,
                            pack=pack,
                            domain=resolved_domain,
                        )
                        if candidate_validation.get("fallback_used"):
                            failure_reasons.append("validator_conflict")
                            relax_validation = True

                    if score_answer(candidate_answer) >= score_answer(old_answer):
                        answer = candidate_answer
                        validation_meta = candidate_validation
                    else:
                        answer = old_answer
                        validation_meta = old_validation
                        failure_reasons.append("regen_not_better")
                except Exception:
                    answer = domain_safe_fallback(resolved_domain)
                    failure_reasons.append("llm_regen_failed")
                    break

        if answer and facts_intake:
            procedural_lines: List[str] = []
            if facts_intake.limitation_deadline:
                procedural_lines.append(
                    f"Limitation check: estimated deadline is {facts_intake.limitation_deadline}."
                )
                if facts_intake.is_time_barred is True:
                    procedural_lines.append("Status: claim appears time-barred on current inputs.")
                elif facts_intake.is_time_barred is False:
                    procedural_lines.append("Status: claim appears within limitation on current inputs.")
            elif resolved_domain == "consumer":
                procedural_lines.append("Limitation check: incident date is missing; limitation cannot be computed yet.")
            if facts_intake.recommended_forum and facts_intake.recommended_forum != "Unknown":
                procedural_lines.append(
                    f"Pecuniary jurisdiction: recommended forum is {facts_intake.recommended_forum}."
                )
            elif resolved_domain == "consumer":
                procedural_lines.append(
                    "Pecuniary jurisdiction: claim amount is missing; share amount (INR) to determine District/State/National Commission."
                )
            if resolved_domain == "consumer":
                procedural_lines.append("Filing method: complaint can be filed online via the e-Daakhil portal.")
            if procedural_lines:
                answer += "\n\nProcedural Checks\n" + "\n".join(f"- {line}" for line in procedural_lines)

            clarification_questions: List[str] = []
            if answer_mode == "answer_and_ask":
                if facts_intake.claim_amount_inr is None:
                    clarification_questions.append("What is the claim amount (in INR)?")
                if resolved_domain == "consumer" and not facts_intake.requested_relief:
                    clarification_questions.append("Do you want refund, replacement, repair, or compensation?")
                if clarification_questions:
                    answer += "\n\nClarifications Needed\n" + "\n".join(f"- {q}" for q in clarification_questions)

        if payload.generate_answer and not answer:
            answer = domain_safe_fallback(resolved_domain)
            failure_reasons.append("empty_answer_fallback")

        confidence_score = compute_confidence(
            mode=reasoning_mode,
            validation_meta=validation_meta,
            failure_reason=failure_reasons,
            context_blocks=pack.get("context_blocks", []),
        )
        if session is not None:
            session["mode"] = "answer"
            session["pending_fields"] = []
            session["history"].append({"role": "assistant", "text": answer or ""})

        return QueryResponse(
            ok=True,
            query=payload.query,
            answer=answer,
            prompt_context=pack["prompt_context"],
            citations=[Citation(**c) for c in pack.get("citations", [])],
            context_blocks=pack.get("context_blocks", []),
            meta={
                "session_id": session_id if session_id else payload.session_id,
                "intake_mode": (session.get("mode") if session else "disabled"),
                "known_facts": (dict(session["facts"]) if session else {}),
                "retrieved_results": len(results),
                "context_blocks_used": len(pack.get("context_blocks", [])),
                "issue_domain": resolved_domain,
                "issue_domain_confidence": route.confidence,
                "issue_terms": route.matched_terms,
                "intent_route": intent_route,
                "query_normalized_for_modern_law": normalized_query,
                "legacy_law_auto_switched": normalized_query != user_query,
                "secondary_domains": secondary_domains,
                "reasoning_mode": reasoning_mode,
                "priors_weight": priors_weight,
                "answer_mode": answer_mode,
                "facts": facts_intake.model_dump() if facts_intake else None,
                "facts_error": facts_error,
                "llm_generated": payload.generate_answer,
                "llm_model": payload.llm_model if payload.generate_answer else None,
                "llm_error": llm_error,
                "strict_legal_validation": payload.strict_legal_validation,
                "validation": validation_meta,
                "contamination": contamination_meta,
                "structure": structure_meta,
                "era_mismatch": era_meta,
                "sanity": sanity_meta,
                "contradiction": contradiction_meta,
                "statute_sanity": statute_meta,
                "regen_attempts": regen_attempts,
                "llm_judge": judge_meta,
                "subquestions": subq_meta,
                "failure_reason": sorted(set(failure_reasons)),
                "confidence_score": confidence_score,
                "time": datetime.utcnow().isoformat() + "Z",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/extract_facts", response_model=ExtractFactsResponse)
def extract_facts_endpoint(payload: ExtractFactsRequest) -> ExtractFactsResponse:
    try:
        facts = extract_facts(
            query=payload.query,
            llm_model=payload.llm_model,
            llm_timeout_sec=payload.llm_timeout_sec,
        )
        return ExtractFactsResponse(
            ok=True,
            query=payload.query,
            facts=facts,
            meta={
                "llm_model": payload.llm_model,
                "time": datetime.utcnow().isoformat() + "Z",
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


@app.post("/schema_intake", response_model=SchemaIntakeResponse)
def schema_intake(payload: SchemaIntakeRequest) -> SchemaIntakeResponse:
    try:
        session_id, fact_store = _get_or_create_schema_session(
            session_id=payload.session_id,
            reset=payload.reset_session,
        )

        result = handle_input(
            user_input=payload.user_input,
            fact_store=fact_store,
            schemas=SCHEMAS,
            llm_model=payload.llm_model,
            llm_timeout_sec=payload.llm_timeout_sec,
        )

        _save_schema_session(session_id=session_id, fact_store=fact_store)

        return SchemaIntakeResponse(
            ok=True,
            session_id=session_id,
            mode=result.get("mode", "question"),
            case_type=result.get("case_type", fact_store.get("case_type") or "unknown"),
            text=result.get("text", ""),
            missing_fields=result.get("missing_fields", []),
            questions=result.get("questions", []),
            facts=dict(fact_store.data),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/dynamic_intake", response_model=DynamicIntakeResponse)
def dynamic_intake(payload: DynamicIntakeRequest) -> DynamicIntakeResponse:
    try:
        session_id, fact_store = _get_or_create_dynamic_session(
            session_id=payload.session_id,
            reset=payload.reset_session,
        )

        result = handle_dynamic_intake(
            user_input=payload.user_input,
            fact_store=fact_store,
            config=DYNAMIC_INTAKE_CONFIG,
            llm_model=payload.llm_model,
            llm_timeout_sec=payload.llm_timeout_sec,
        )

        _save_dynamic_session(session_id=session_id, fact_store=fact_store)

        return DynamicIntakeResponse(
            ok=True,
            session_id=session_id,
            mode=result.get("mode", "question"),
            task=result.get("task", str(fact_store.get("task") or "advice")),
            domain=result.get("domain", "general"),
            signals=result.get("signals", []),
            signal_confidence=result.get("signal_confidence", {}),
            strategy_tracks=result.get("strategy_tracks", []),
            routed_handler=result.get("routed_handler"),
            rag_used=bool(result.get("rag_used", False)),
            retrieval_queries=result.get("retrieval_queries", []),
            retrieved_context_count=int(result.get("retrieved_context_count", 0)),
            retrieved_citations=result.get("retrieved_citations", []),
            required_field_count=int(result.get("required_field_count", 0)),
            missing_fields=result.get("missing_fields", []),
            questions=result.get("questions", []),
            text=result.get("text", ""),
            facts=dict(fact_store.data),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


from legal_engine.orchestrator import LegalEngineOrchestrator
from legal_engine.dialog_manager import DialogManager

ORCHESTRATOR = LegalEngineOrchestrator()
DIALOG_MANAGER = DialogManager(ORCHESTRATOR)


@app.post("/legal_reasoning", response_model=ReasoningResponse)
def legal_reasoning_endpoint(payload: ReasoningRequest) -> ReasoningResponse:
    try:
        session_id = payload.session_id or str(uuid.uuid4())
        ORCHESTRATOR.model = payload.llm_model
        
        response_text = DIALOG_MANAGER.handle_user_input(
            conversation_id=session_id,
            text=payload.user_input
        )
        
        return ReasoningResponse(
            ok=True,
            session_id=session_id,
            text=response_text
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("retrieval_api:app", host=host, port=port, reload=False)
