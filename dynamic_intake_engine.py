import json
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from context_builder import build_context_pack, load_acts_chunk_lookup, run_retrieval
from llama_legal_answer import call_llm


DEFAULT_DYNAMIC_CONFIG_FILE = "dynamic_intake_config.json"
_ACTS_LOOKUP_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def load_dynamic_config(path: str = DEFAULT_DYNAMIC_CONFIG_FILE) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dynamic intake config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "core_fields" not in cfg or "domains" not in cfg or "signals" not in cfg:
        raise ValueError("Invalid dynamic intake config: requires core_fields/domains/signals")
    return cfg


class FactStore:
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {} # {"key": {"value": ..., "confidence": ...}}
        self._asked: set[str] = set()

    def update(self, key: str, value: Any, confidence: float = 1.0) -> None:
        if value is None:
            return
        if isinstance(value, str) and not value.strip():
            return
        current = self.data.get(key, {"value": None, "confidence": 0.0})
        if confidence >= current.get("confidence", 0.0):
            self.data[key] = {"value": value, "confidence": confidence}

    def update_many(self, values: Dict[str, Any], confidence: float = 1.0) -> None:
        for k, v in (values or {}).items():
            if isinstance(v, dict) and "value" in v and "confidence" in v:
                self.update(k, v["value"], v["confidence"])
            else:
                self.update(k, v, confidence)

    def get(self, key: str) -> Any:
        field = self.data.get(key)
        return field.get("value") if field else None

    def get_confidence(self, key: str) -> float:
        field = self.data.get(key)
        return field.get("confidence", 0.0) if field else 0.0

    def missing_fields(self, merged_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # A field is missing if it's not in data OR its confidence is low (< 0.7)
        return [
            field for field in merged_fields 
            if field["key"] not in self.data or self.data[field["key"]].get("confidence", 0.0) < 0.7
        ]

    def mark_asked(self, key: str) -> None:
        self._asked.add(key)

    def asked(self, key: str) -> bool:
        return key in self._asked

    def to_flat_dict(self) -> Dict[str, Any]:
        return {k: v.get("value") for k, v in self.data.items() if v.get("value") is not None}

    def to_json(self) -> str:
        return json.dumps(self.data, ensure_ascii=False, indent=2)


def _extract_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
    return {}


def _safe_confidence(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(val, 1.0))


def _get_acts_lookup() -> Dict[str, Dict[str, Any]]:
    global _ACTS_LOOKUP_CACHE
    if _ACTS_LOOKUP_CACHE is None:
        _ACTS_LOOKUP_CACHE = load_acts_chunk_lookup("JSON_acts")
    return _ACTS_LOOKUP_CACHE


def _fallback_signal_confidence(user_input: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    q = (user_input or "").lower()
    found: List[Dict[str, Any]] = []
    for signal, info in config.get("signals", {}).items():
        hints = [str(h).lower() for h in info.get("detector_hints", [])]
        hits = sum(1 for h in hints if h in q)
        if hits <= 0:
            continue
        confidence = min(0.95, 0.45 + (0.15 * hits))
        found.append({"name": signal, "confidence": confidence})
    found.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return found


def detect_signals_with_confidence(
    user_input: str, config: Dict[str, Any], llm_model: str, llm_timeout_sec: int
) -> List[Dict[str, Any]]:
    allowed_signals = list(config.get("signals", {}).keys())
    prompt = (
        "You are a legal signal detection engine.\n\n"
        "Task: extract only relevant legal signals from the allowed list and assign confidence (0 to 1).\n"
        "Rules:\n"
        "1. Use only allowed signals\n"
        "2. Do not invent signals\n"
        "3. Include only clearly supported signals\n"
        "4. If unsure, exclude signal\n"
        "5. Return valid JSON only\n\n"
        f"Allowed signals: {json.dumps(allowed_signals, ensure_ascii=False)}\n\n"
        "Few-shot examples:\n"
        "Input: I was fired without notice\n"
        "Output: {\"signals\":[{\"name\":\"termination\",\"confidence\":0.95}]}\n\n"
        "Input: My phone exploded and injured me\n"
        "Output: {\"signals\":[{\"name\":\"defect\",\"confidence\":0.92},{\"name\":\"injury\",\"confidence\":0.97}]}\n\n"
        "User input:\n"
        f"{user_input}\n\n"
        "Return JSON in format: {\"signals\":[{\"name\":\"<allowed>\",\"confidence\":0.0}]}"
    )
    try:
        raw = call_llm(model=llm_model, prompt=prompt, timeout_sec=llm_timeout_sec)
        parsed = _extract_json(raw)
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in parsed.get("signals", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name in seen or name not in allowed_signals:
                continue
            out.append({"name": name, "confidence": _safe_confidence(item.get("confidence", 0.0))})
            seen.add(name)
        if out:
            return out
    except Exception:
        pass
    return _fallback_signal_confidence(user_input=user_input, config=config)


def validate_detected_signals(
    user_input: str,
    detected_signals: List[Dict[str, Any]],
    config: Dict[str, Any],
    llm_model: str,
    llm_timeout_sec: int,
) -> List[Dict[str, Any]]:
    if not detected_signals:
        return []
    allowed = set(config.get("signals", {}).keys())
    prompt = (
        "You are a strict validator.\n"
        "Keep only detected signals that are clearly supported by the user input.\n"
        "Do not add any new signal.\n"
        "Return JSON only: {\"validated_signals\":[\"signal1\",\"signal2\"]}\n\n"
        f"User input:\n{user_input}\n\n"
        f"Detected signals:\n{json.dumps(detected_signals, ensure_ascii=False)}"
    )
    try:
        raw = call_llm(model=llm_model, prompt=prompt, timeout_sec=llm_timeout_sec)
        parsed = _extract_json(raw)
        valid_names = {
            s for s in parsed.get("validated_signals", []) if isinstance(s, str) and s in allowed
        }
        filtered = [s for s in detected_signals if s.get("name") in valid_names]
        if filtered:
            return filtered
    except Exception:
        pass
    # Conservative fallback: keep only medium confidence+ signals.
    return [s for s in detected_signals if _safe_confidence(s.get("confidence", 0.0)) >= 0.55]


def detect_signals_bundle(
    user_input: str, config: Dict[str, Any], llm_model: str, llm_timeout_sec: int
) -> List[Dict[str, Any]]:
    raw_detected = detect_signals_with_confidence(
        user_input=user_input,
        config=config,
        llm_model=llm_model,
        llm_timeout_sec=llm_timeout_sec,
    )
    return validate_detected_signals(
        user_input=user_input,
        detected_signals=raw_detected,
        config=config,
        llm_model=llm_model,
        llm_timeout_sec=llm_timeout_sec,
    )


def classify_domain(user_input: str, config: Dict[str, Any], llm_model: str, llm_timeout_sec: int) -> str:
    domains = list(config.get("domains", {}).keys())
    prompt = (
        "Classify legal domain. Return JSON only: {\"domain\":\"<one_allowed_value>\"}.\n"
        f"Allowed domains: {json.dumps(domains, ensure_ascii=False)}\n"
        f"User input:\n{user_input}"
    )
    try:
        raw = call_llm(model=llm_model, prompt=prompt, timeout_sec=llm_timeout_sec)
        parsed = _extract_json(raw)
        domain = str(parsed.get("domain", "")).strip()
        if domain in config["domains"]:
            return domain
    except Exception:
        pass

    q = user_input.lower()
    best_domain = "general" if "general" in config["domains"] else domains[0]
    best_score = -1
    for domain, info in config["domains"].items():
        hints = [str(h).lower() for h in info.get("classifier_hints", [])]
        score = sum(1 for h in hints if h in q)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


def detect_signals(user_input: str, config: Dict[str, Any], llm_model: str, llm_timeout_sec: int) -> List[str]:
    bundle = detect_signals_bundle(
        user_input=user_input,
        config=config,
        llm_model=llm_model,
        llm_timeout_sec=llm_timeout_sec,
    )
    return sorted(list({str(item.get("name", "")).strip() for item in bundle if item.get("name")}))


def detect_task_intent(user_input: str, config: Dict[str, Any], llm_model: str, llm_timeout_sec: int) -> str:
    text = (user_input or "").lower()
    # Deterministic keyword routing first.
    if "legal notice" in text or "draft notice" in text:
        return "draft_notice"
    if "complaint" in text or "file complaint" in text:
        return "file_complaint"
    if "what can i do" in text or "legal advice" in text:
        return "advice"

    allowed_tasks = list((config.get("tasks") or {}).keys()) or [
        "advice",
        "draft_notice",
        "file_complaint",
        "estimate_claim",
    ]
    prompt = (
        "You are a legal task intent classifier.\n"
        "Return JSON only: {\"task\":\"<allowed>\"}\n"
        f"Allowed tasks: {json.dumps(allowed_tasks, ensure_ascii=False)}\n"
        "Choose one best task from user input.\n\n"
        f"User input:\n{user_input}"
    )
    try:
        raw = call_llm(model=llm_model, prompt=prompt, timeout_sec=llm_timeout_sec)
        parsed = _extract_json(raw)
        task = str(parsed.get("task", "")).strip()
        if task in allowed_tasks:
            return task
    except Exception:
        pass

    text = (user_input or "").lower()
    best_task = "advice" if "advice" in allowed_tasks else allowed_tasks[0]
    best_score = -1
    for task in allowed_tasks:
        hints = [str(h).lower() for h in (config.get("tasks", {}).get(task, {}).get("task_hints", []) or [])]
        score = sum(1 for h in hints if h in text)
        if score > best_score:
            best_score = score
            best_task = task
    return best_task


def _has_explicit_task_request(user_input: str, config: Dict[str, Any]) -> bool:
    text = (user_input or "").lower()
    for task_cfg in (config.get("tasks") or {}).values():
        for hint in task_cfg.get("task_hints", []) or []:
            if str(hint).lower() in text:
                return True
    return False


def resolve_task(user_input: str, fact_store: FactStore, config: Dict[str, Any], llm_model: str, llm_timeout_sec: int) -> str:
    current = str(fact_store.get("task") or "").strip()
    detected = detect_task_intent(
        user_input=user_input,
        config=config,
        llm_model=llm_model,
        llm_timeout_sec=llm_timeout_sec,
    )
    if not current:
        return detected
    if _has_explicit_task_request(user_input=user_input, config=config):
        return detected
    return current


def _build_field_index(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for field in config.get("core_fields", []) or []:
        key = str(field.get("key", "")).strip()
        if key:
            index[key] = field
    for domain_cfg in (config.get("domains") or {}).values():
        for field in domain_cfg.get("fields", []) or []:
            key = str(field.get("key", "")).strip()
            if key and key not in index:
                index[key] = field
    for sig_cfg in (config.get("signals") or {}).values():
        for field in sig_cfg.get("fields", []) or []:
            key = str(field.get("key", "")).strip()
            if key and key not in index:
                index[key] = field
    return index


def task_required_field_defs(task: str, config: Dict[str, Any], fallback_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    task_cfg = (config.get("tasks") or {}).get(task, {})
    required_keys = [str(k).strip() for k in task_cfg.get("required_fields", []) if str(k).strip()]
    if not required_keys:
        return fallback_fields

    index = _build_field_index(config)
    defs: List[Dict[str, Any]] = []
    for key in required_keys:
        field = index.get(key) or {"key": key, "question": f"Please provide: {key}"}
        defs.append(field)
    return defs


def task_critical_field_defs(task: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_cfg = (config.get("tasks") or {}).get(task, {})
    critical_keys = [str(k).strip() for k in task_cfg.get("critical_fields", []) if str(k).strip()]
    if not critical_keys:
        return []
    index = _build_field_index(config)
    defs: List[Dict[str, Any]] = []
    for key in critical_keys:
        defs.append(index.get(key) or {"key": key, "question": f"Please provide: {key}"})
    return defs


def generate_legal_notice(facts_dict: Dict[str, Any]) -> str:
    # facts_dict can be a flat dict from FactStore.to_flat_dict()
    parties = str(facts_dict.get("parties_involved") or "Sender and Recipient").strip()
    incident = str(facts_dict.get("incident_description") or "Incident details not provided.").strip()
    harm = str(facts_dict.get("harm_or_loss") or "Harm/loss details not provided.").strip()
    evidence = str(facts_dict.get("evidence_available") or "Supporting records available upon request.").strip()

    return (
        "LEGAL NOTICE\n\n"
        f"Parties: {parties}\n\n"
        "Subject: Demand for compliance and remedial action\n\n"
        "1. Facts\n"
        f"{incident}\n\n"
        "2. Harm/Loss\n"
        f"{harm}\n\n"
        "3. Supporting Documents\n"
        f"{evidence}\n\n"
        "4. Demands\n"
        "- Remedy the grievance in accordance with law.\n"
        "- Provide written response and corrective action within 15 days of receipt of this notice.\n"
        "- Compensate for losses caused, as applicable.\n\n"
        "5. Reservation of Rights\n"
        "Failing compliance within the above timeline, appropriate legal proceedings shall be initiated at your risk as to costs and consequences.\n"
    )


def apply_signal_domain_boost(base_domain: str, signals: List[str], config: Dict[str, Any]) -> str:
    boosts = config.get("signal_domain_boost", {}) or {}
    if not boosts or not signals:
        return base_domain

    score_by_domain: Dict[str, int] = {}
    score_by_domain[base_domain] = score_by_domain.get(base_domain, 0) + 1
    for signal in signals:
        for domain in boosts.get(signal, []) or []:
            d = str(domain).strip()
            if not d:
                continue
            score_by_domain[d] = score_by_domain.get(d, 0) + 1

    best_domain = base_domain
    best_score = score_by_domain.get(base_domain, 0)
    for domain, score in score_by_domain.items():
        if domain in config.get("domains", {}) and score > best_score:
            best_domain = domain
            best_score = score
    return best_domain


def _map_domain_for_retrieval(domain: str) -> str:
    if domain in {"consumer", "property", "criminal", "contract"}:
        return domain
    if domain in {"employment", "labour"}:
        return "labour"
    return "general"


def build_retrieval_queries(facts: Dict[str, Any], signals: List[str], domain: str) -> List[str]:
    # Support both flat and structured facts for robustness
    def get_val(f_dict, k):
        v = f_dict.get(k)
        if isinstance(v, dict) and "value" in v:
            return v["value"]
        return v

    fact_pairs = []
    for k in (facts or {}).keys():
        if k in {"domain", "signals", "strategy_tracks", "strategy_hints", "signal_confidence"}:
            continue
        v = get_val(facts, k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        fact_pairs.append(f"{k}: {v}")
    compact_facts = "; ".join(fact_pairs)[:1200]
    signals_str = ", ".join(signals) if signals else "none"

    base = (
        f"Domain: {domain}. Legal issue signals: {signals_str}. Facts: {compact_facts}"
        if compact_facts
        else f"Domain: {domain}. Legal issue signals: {signals_str}."
    )
    q2 = f"Indian law {domain} dispute {signals_str} relevant sections remedies"
    q3 = f"{domain} legal provisions and case law for {signals_str}"
    queries = [base.strip(), q2.strip(), q3.strip()]
    seen: set[str] = set()
    uniq: List[str] = []
    for q in queries:
        if q and q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq


def retrieve_legal_context(
    queries: List[str],
    domain: str,
    top_k: int = 5,
    max_context_chars: int = 14000,
) -> Dict[str, Any]:
    if not queries:
        return {"queries": [], "context_blocks": [], "prompt_context": "", "citations": []}

    legal_domain = _map_domain_for_retrieval(domain)
    merged_by_key: Dict[str, Dict[str, Any]] = {}
    for q in queries:
        args = SimpleNamespace(
            q=q,
            corpus="all",
            top_k=top_k,
            dense_k=100,
            bm25_k=100,
            dense_weight=0.6,
            bm25_weight=0.4,
            rerank=False,
            rerank_model="BAAI/bge-reranker-base",
            rerank_top_n=50,
            rerank_batch_size=16,
            max_context_chars=max_context_chars,
            legal_domain=legal_domain,
            intent_route={},
        )
        try:
            results = run_retrieval(args)
        except Exception:
            results = []
        for item in results:
            key = str(item.get("chunk_id") or item.get("document_id") or item.get("title") or "") + "|" + str(
                item.get("corpus") or ""
            )
            if not key:
                continue
            old = merged_by_key.get(key)
            if old is None:
                merged_by_key[key] = item
                continue
            old_score = float(old.get("final_score", old.get("hybrid_score", 0.0)) or 0.0)
            new_score = float(item.get("final_score", item.get("hybrid_score", 0.0)) or 0.0)
            if new_score > old_score:
                merged_by_key[key] = item

    merged = list(merged_by_key.values())
    merged.sort(key=lambda x: float(x.get("final_score", x.get("hybrid_score", 0.0)) or 0.0), reverse=True)
    merged = merged[:top_k]
    pack = build_context_pack(
        query=queries[0],
        results=merged,
        acts_lookup=_get_acts_lookup(),
        max_chars=max_context_chars,
    )
    return {
        "queries": queries,
        "context_blocks": pack.get("context_blocks", []),
        "prompt_context": pack.get("prompt_context", ""),
        "citations": pack.get("citations", []),
    }


def merge_required_fields(config: Dict[str, Any], domain: str, signals: List[str]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for field in config.get("core_fields", []):
        key = str(field.get("key", "")).strip()
        if key and key not in seen:
            merged.append(field)
            seen.add(key)

    for field in config.get("domains", {}).get(domain, {}).get("fields", []):
        key = str(field.get("key", "")).strip()
        if key and key not in seen:
            merged.append(field)
            seen.add(key)

    for signal in signals:
        for field in config.get("signals", {}).get(signal, {}).get("fields", []):
            key = str(field.get("key", "")).strip()
            if key and key not in seen:
                merged.append(field)
                seen.add(key)

    return merged


def resolve_strategy_tracks(config: Dict[str, Any], signals: List[str]) -> List[str]:
    tracks_map = config.get("signal_strategy_map", {}) or {}
    resolved: List[str] = []
    seen: set[str] = set()
    for signal in signals:
        for track in tracks_map.get(signal, []) or []:
            t = str(track).strip()
            if t and t not in seen:
                resolved.append(t)
                seen.add(t)
    return resolved


def resolve_strategy_hints(config: Dict[str, Any], signals: List[str]) -> List[str]:
    hints: List[str] = []
    seen: set[str] = set()
    for signal in signals:
        signal_hints = config.get("signals", {}).get(signal, {}).get("strategy_hints", []) or []
        for hint in signal_hints:
            h = str(hint).strip()
            if h and h not in seen:
                hints.append(h)
                seen.add(h)
    return hints


def extract_dynamic_facts(
    user_input: str,
    required_fields: List[Dict[str, Any]],
    llm_model: str,
    llm_timeout_sec: int,
) -> Dict[str, Any]:
    allowed_keys = [f["key"] for f in required_fields if f.get("key")]
    prompt = (
        "Extract structured legal facts from user input.\n"
        "Return JSON only.\n"
        f"Allowed keys: {json.dumps(allowed_keys, ensure_ascii=False)}\n"
        "Rules:\n"
        "1. Only extract if clearly mentioned\n"
        "2. Do not guess\n"
        "3. Use null for keys that are not present\n\n"
        f"User input:\n{user_input}"
    )
    try:
        raw = call_llm(model=llm_model, prompt=prompt, timeout_sec=llm_timeout_sec)
        parsed = _extract_json(raw)
        out: Dict[str, Dict[str, Any]] = {}
        for k in allowed_keys:
            if k in parsed and parsed[k] is not None and str(parsed[k]).strip() != "":
                # High confidence by default for explicit LLM extraction in dynamic intake
                out[k] = {"value": parsed[k], "confidence": 0.9}
        return out
    except Exception:
        return _fallback_dynamic_extract(user_input, required_fields)


def _fallback_dynamic_extract(user_input: str, required_fields: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    text = (user_input or "").strip()
    lower = text.lower()
    out: Dict[str, Dict[str, Any]] = {}
    for field in required_fields:
        key = str(field.get("key", "")).strip()
        question = str(field.get("question", "")).lower()
        if not key:
            continue
        tokens = [t for t in re.split(r"[_\W]+", key.lower()) if len(t) >= 4]
        tokens.extend([t for t in re.split(r"\W+", question) if len(t) >= 4])
        if any(tok in lower for tok in set(tokens)):
            out[key] = {"value": text, "confidence": 0.5}
    return out


def generate_questions(missing_fields: List[Dict[str, Any]], fact_store: FactStore) -> List[str]:
    questions: List[str] = []
    for field in missing_fields:
        key = str(field.get("key", "")).strip()
        question = str(field.get("question", "")).strip()
        if not key or not question:
            continue
        if fact_store.asked(key):
            continue
        questions.append(question)
        fact_store.mark_asked(key)
    return questions


def format_question_block(questions: List[str], pending_fields: Optional[List[str]] = None) -> str:
    if not questions:
        if pending_fields:
            pending = ", ".join(pending_fields)
            return (
                "I still need a few pending details to continue. "
                f"Please share values for: {pending}."
            )
        return "Please share additional details so I can proceed."
    lines = ["To proceed, please share these details:"]
    for i, q in enumerate(questions, start=1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines)


def generate_output(
    fact_store: FactStore,
    domain: str,
    signals: List[str],
    strategy_tracks: List[str],
    strategy_hints: List[str],
    legal_context: Dict[str, Any],
    llm_model: str,
    llm_timeout_sec: int,
) -> str:
    context_blocks = legal_context.get("context_blocks", []) if legal_context else []
    context_for_prompt = []
    for b in context_blocks:
        law_name = b.get("title") or "Unknown law"
        section = b.get("section_number") or "N/A"
        source = b.get("source_file") or b.get("context_path") or "Unknown source"
        chunk_text = (b.get("texts", {}) or {}).get("chunk_text", "")
        context_for_prompt.append(
            {
                "law_name": law_name,
                "section": section,
                "text": chunk_text,
                "source": source,
            }
        )

    prompt = (
        "You are a legal assistant.\n"
        "Use ONLY the provided legal context to answer.\n"
        "Do NOT hallucinate laws. Prefer retrieved content over general knowledge.\n"
        "If context is missing for a point, say that clearly.\n\n"
        "Generate a structured legal intake strategy response with these exact headings:\n"
        "FACTS\nLEGAL ISSUE\nGROUNDS\nANALYSIS\nSTRATEGY\nPRAYER\n\n"
        "Use domain and active signals to shape strategy. Prefer practical, actionable steps.\n\n"
        f"Domain: {domain}\n"
        f"Signals: {json.dumps(signals, ensure_ascii=False)}\n"
        f"Strategy tracks: {json.dumps(strategy_tracks, ensure_ascii=False)}\n"
        f"Strategy hints: {json.dumps(strategy_hints, ensure_ascii=False)}\n"
        f"Facts JSON:\n{fact_store.to_json()}\n\n"
        f"LEGAL CONTEXT:\n{json.dumps(context_for_prompt, ensure_ascii=False, indent=2)}"
    )
    try:
        return call_llm(model=llm_model, prompt=prompt, timeout_sec=llm_timeout_sec).strip()
    except Exception:
        return (
            "FACTS\n"
            f"{fact_store.to_json()}\n\n"
            "LEGAL ISSUE\nPrimary legal issues should be mapped from the collected facts and active signals.\n\n"
            "GROUNDS\nApplicable legal grounds should be selected based on domain and evidence.\n\n"
            "ANALYSIS\nThe matter appears actionable subject to documentary verification.\n\n"
            "STRATEGY\nProceed with evidence consolidation, formal notice/complaint preparation, and forum selection.\n\n"
            "PRAYER\nSeek statutory relief, compensation, and protective orders as supported by facts."
        )


def _generate_notice_output(
    fact_store: FactStore,
    domain: str,
    signals: List[str],
    legal_context: Dict[str, Any],
    llm_model: str,
    llm_timeout_sec: int,
) -> str:
    context_blocks = legal_context.get("context_blocks", []) if legal_context else []
    compact_context = []
    for b in context_blocks[:5]:
        compact_context.append(
            {
                "law_name": b.get("title"),
                "section": b.get("section_number"),
                "text": (b.get("texts", {}) or {}).get("chunk_text", ""),
                "source": b.get("source_file") or b.get("context_path"),
            }
        )
    prompt = (
        "Draft a legal notice in plain, professional language.\n"
        "Use ONLY provided facts and legal context. Do not fabricate sections.\n"
        "Include: Subject, Facts, Legal Grounds, Demands, Compliance Deadline, Reservation of Rights.\n\n"
        f"Domain: {domain}\n"
        f"Signals: {json.dumps(signals, ensure_ascii=False)}\n"
        f"Facts:\n{fact_store.to_json()}\n\n"
        f"Legal Context:\n{json.dumps(compact_context, ensure_ascii=False, indent=2)}"
    )
    try:
        return call_llm(model=llm_model, prompt=prompt, timeout_sec=llm_timeout_sec).strip()
    except Exception:
        return "Unable to draft notice right now. Please retry."


def _generate_complaint_output(
    fact_store: FactStore,
    domain: str,
    signals: List[str],
    legal_context: Dict[str, Any],
    llm_model: str,
    llm_timeout_sec: int,
) -> str:
    context_blocks = legal_context.get("context_blocks", []) if legal_context else []
    compact_context = []
    for b in context_blocks[:5]:
        compact_context.append(
            {
                "law_name": b.get("title"),
                "section": b.get("section_number"),
                "text": (b.get("texts", {}) or {}).get("chunk_text", ""),
                "source": b.get("source_file") or b.get("context_path"),
            }
        )
    prompt = (
        "Prepare a complaint draft in structured legal format.\n"
        "Use ONLY the supplied facts and legal context.\n"
        "Sections:\nFACTS\nCAUSE OF ACTION\nJURISDICTION/FORUM\nGROUNDS\nRELIEFS SOUGHT\nPRAYER\n\n"
        f"Domain: {domain}\n"
        f"Signals: {json.dumps(signals, ensure_ascii=False)}\n"
        f"Facts:\n{fact_store.to_json()}\n\n"
        f"Legal Context:\n{json.dumps(compact_context, ensure_ascii=False, indent=2)}"
    )
    try:
        return call_llm(model=llm_model, prompt=prompt, timeout_sec=llm_timeout_sec).strip()
    except Exception:
        return "Unable to draft complaint right now. Please retry."


def _generate_claim_estimate_output(
    fact_store: FactStore,
    domain: str,
    signals: List[str],
    legal_context: Dict[str, Any],
    llm_model: str,
    llm_timeout_sec: int,
) -> str:
    context_blocks = legal_context.get("context_blocks", []) if legal_context else []
    compact_context = []
    for b in context_blocks[:5]:
        compact_context.append(
            {
                "law_name": b.get("title"),
                "section": b.get("section_number"),
                "text": (b.get("texts", {}) or {}).get("chunk_text", ""),
                "source": b.get("source_file") or b.get("context_path"),
            }
        )
    prompt = (
        "Estimate claim value range conservatively.\n"
        "Use ONLY provided facts and legal context.\n"
        "Output sections:\nESTIMATE BASIS\nESTIMATED RANGE (LOW-HIGH)\nASSUMPTIONS\nMISSING DATA NEEDED\n\n"
        f"Domain: {domain}\n"
        f"Signals: {json.dumps(signals, ensure_ascii=False)}\n"
        f"Facts:\n{fact_store.to_json()}\n\n"
        f"Legal Context:\n{json.dumps(compact_context, ensure_ascii=False, indent=2)}"
    )
    try:
        return call_llm(model=llm_model, prompt=prompt, timeout_sec=llm_timeout_sec).strip()
    except Exception:
        return "Unable to estimate claim right now. Please retry."


def route_task_execution(
    task: str,
    fact_store: FactStore,
    domain: str,
    signals: List[str],
    strategy_tracks: List[str],
    strategy_hints: List[str],
    legal_context: Dict[str, Any],
    llm_model: str,
    llm_timeout_sec: int,
) -> str:
    if task == "draft_notice":
        return generate_legal_notice(fact_store.to_flat_dict())
    if task == "file_complaint":
        return _generate_complaint_output(
            fact_store=fact_store,
            domain=domain,
            signals=signals,
            legal_context=legal_context,
            llm_model=llm_model,
            llm_timeout_sec=llm_timeout_sec,
        )
    if task == "estimate_claim":
        return _generate_claim_estimate_output(
            fact_store=fact_store,
            domain=domain,
            signals=signals,
            legal_context=legal_context,
            llm_model=llm_model,
            llm_timeout_sec=llm_timeout_sec,
        )
    # Default advice flow uses existing RAG-grounded structured response.
    return generate_output(
        fact_store=fact_store,
        domain=domain,
        signals=signals,
        strategy_tracks=strategy_tracks,
        strategy_hints=strategy_hints,
        legal_context=legal_context,
        llm_model=llm_model,
        llm_timeout_sec=llm_timeout_sec,
    )


def handle_dynamic_intake(
    user_input: str,
    fact_store: FactStore,
    config: Dict[str, Any],
    llm_model: str,
    llm_timeout_sec: int,
) -> Dict[str, Any]:
    task = resolve_task(
        user_input=user_input,
        fact_store=fact_store,
        config=config,
        llm_model=llm_model,
        llm_timeout_sec=llm_timeout_sec,
    )
    fact_store.update("task", task)

    domain = fact_store.get("domain")
    if not domain:
        domain = classify_domain(
            user_input=user_input,
            config=config,
            llm_model=llm_model,
            llm_timeout_sec=llm_timeout_sec,
        )

    prior_signals = set(fact_store.get("signals") or [])
    prior_conf = dict(fact_store.get("signal_confidence") or {})
    detected_bundle = detect_signals_bundle(
        user_input=user_input,
        config=config,
        llm_model=llm_model,
        llm_timeout_sec=llm_timeout_sec,
    )
    new_signals = {str(item.get("name", "")).strip() for item in detected_bundle if item.get("name")}
    new_conf = {
        str(item.get("name", "")).strip(): _safe_confidence(item.get("confidence", 0.0))
        for item in detected_bundle
        if item.get("name")
    }
    merged_conf = prior_conf.copy()
    for k, v in new_conf.items():
        merged_conf[k] = max(_safe_confidence(merged_conf.get(k, 0.0)), v)
    signals = sorted(list(prior_signals.union(new_signals)))
    domain = apply_signal_domain_boost(base_domain=domain, signals=signals, config=config)
    fact_store.update("domain", domain)
    fact_store.update("signals", signals)
    fact_store.update("signal_confidence", merged_conf)
    strategy_tracks = resolve_strategy_tracks(config=config, signals=signals)
    strategy_hints = resolve_strategy_hints(config=config, signals=signals)
    fact_store.update("strategy_tracks", strategy_tracks)
    fact_store.update("strategy_hints", strategy_hints)

    merged_fields = merge_required_fields(config=config, domain=domain, signals=signals)
    extracted = extract_dynamic_facts(
        user_input=user_input,
        required_fields=merged_fields,
        llm_model=llm_model,
        llm_timeout_sec=llm_timeout_sec,
    )
    fact_store.update_many(extracted)

    task_fields = task_required_field_defs(task=task, config=config, fallback_fields=merged_fields)
    missing = fact_store.missing_fields(task_fields)
    # For non-intake tasks, only ask when task-critical fields are missing.
    if task != "intake":
        critical_defs = task_critical_field_defs(task=task, config=config)
        missing = fact_store.missing_fields(critical_defs) if critical_defs else []
    if missing:
        questions = generate_questions(missing, fact_store)
        return {
            "mode": "question",
            "task": task,
            "domain": domain,
            "signals": signals,
            "signal_confidence": merged_conf,
            "strategy_tracks": strategy_tracks,
            "required_field_count": len(task_fields),
            "missing_fields": [f["key"] for f in missing],
            "questions": questions,
            "text": format_question_block(
                questions,
                pending_fields=[f["key"] for f in missing] if not questions else None,
            ),
        }

    retrieval_queries: List[str] = []
    legal_context: Dict[str, Any] = {"context_blocks": [], "citations": []}
    if task != "draft_notice":
        retrieval_queries = build_retrieval_queries(
            facts=fact_store.data,
            signals=signals,
            domain=domain,
        )
        legal_context = retrieve_legal_context(
            queries=retrieval_queries,
            domain=domain,
            top_k=5,
            max_context_chars=14000,
        )

    output = route_task_execution(
        task=task,
        fact_store=fact_store,
        domain=domain,
        signals=signals,
        strategy_tracks=strategy_tracks,
        strategy_hints=strategy_hints,
        legal_context=legal_context,
        llm_model=llm_model,
        llm_timeout_sec=llm_timeout_sec,
    )
    return {
        "mode": "answer",
        "task": task,
        "domain": domain,
        "signals": signals,
        "signal_confidence": merged_conf,
        "strategy_tracks": strategy_tracks,
        "routed_handler": task,
        "rag_used": bool(task != "draft_notice"),
        "retrieval_queries": retrieval_queries,
        "retrieved_context_count": len(legal_context.get("context_blocks", [])),
        "retrieved_context": legal_context.get("context_blocks", []),
        "retrieved_citations": legal_context.get("citations", []),
        "required_field_count": len(task_fields),
        "missing_fields": [],
        "questions": [],
        "text": output,
    }
