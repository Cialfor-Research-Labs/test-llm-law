import json
import re
from datetime import date, datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from jurisdiction_validator import consumer_forum_by_amount
from llama_legal_answer import call_llm
from statutory_checks import consumer_limitation, money_recovery_limitation


class FactExtraction(BaseModel):
    incident_date: Optional[str] = None
    date_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    claim_amount_inr: Optional[float] = None
    amount_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    seller_response: Optional[str] = None # "cooperative" | "refused_or_no_response" | "ignored"
    seller_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requested_relief: Optional[str] = None
    relief_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    location: Optional[str] = None
    legal_domain: str = Field(default="general")
    domain_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    party_type: Optional[str] = None
    cause_summary: str = Field(default="")
    injury_signal: bool = False
    injury_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    statute_regime: str = Field(default="unknown")
    limitation_deadline: Optional[str] = None
    is_time_barred: Optional[bool] = None
    recommended_forum: str = Field(default="Unknown")
    needs_incident_date: bool = False
    follow_up_question: Optional[str] = None


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _to_iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None


def _extract_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _parse_amount_inr(text: str) -> Optional[float]:
    q = (text or "").lower().replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|crore|crores)\b", q)
    if m:
        n = float(m.group(1))
        unit = m.group(2)
        if "crore" in unit:
            return n * 10_000_000
        return n * 100_000
    m2 = re.search(r"\b(?:rs\.?|inr)?\s*(\d{4,12})\b", q)
    if m2:
        return float(m2.group(1))
    return None


def _parse_date_from_query(text: str) -> Optional[date]:
    q = (text or "").strip()
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except Exception:
            pass

    m2 = re.search(
        r"\b(0?[1-9]|[12]\d|3[01])\s+"
        r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+"
        r"(20\d{2})\b",
        q,
        flags=re.IGNORECASE,
    )
    if m2:
        try:
            return datetime.strptime(f"{m2.group(1)} {m2.group(2)} {m2.group(3)}", "%d %b %Y").date()
        except Exception:
            try:
                return datetime.strptime(f"{m2.group(1)} {m2.group(2)} {m2.group(3)}", "%d %B %Y").date()
            except Exception:
                pass

    m3 = re.search(r"\b(20\d{2})\b", q)
    if m3:
        return date(int(m3.group(1)), 1, 1)
    return None


def _heuristic_domain(text: str) -> str:
    q = (text or "").lower()
    if any(k in q for k in ["defect", "defective", "refund", "seller", "consumer", "hospital", "builder", "rera"]):
        return "consumer"
    if any(k in q for k in ["landlord", "tenant", "evict", "possession", "lease", "property"]):
        return "property"
    if any(k in q for k in ["fir", "arrest", "bail", "cheat", "theft", "forgery", "police", "criminal"]):
        return "criminal"
    if any(k in q for k in ["salary", "wages", "termination", "employee", "employer", "gratuity"]):
        return "labour"
    if any(k in q for k in ["contract", "agreement", "breach", "damages"]):
        return "contract"
    return "general"


def enforce_minimum_facts(query: str, facts: FactExtraction) -> FactExtraction:
    q = (query or "").lower()

    # Force consumer domain for clear product-purchase flows.
    if any(k in q for k in ["bought", "ordered", "product", "phone", "delivered"]):
        facts.legal_domain = "consumer"

    # Force injury signal where explicit harm keywords appear.
    if any(k in q for k in ["injury", "injured", "exploded", "burn", "damage"]):
        facts.injury_signal = True

    # Ensure we always keep a cause summary.
    if not (facts.cause_summary or "").strip():
        facts.cause_summary = (query or "").strip()

    return facts


def heuristic_extract_facts(query: str, today: date) -> FactExtraction:
    incident = _parse_date_from_query(query)
    claim_amount = _parse_amount_inr(query)
    domain = _heuristic_domain(query)
    q = query.lower()

    relief = None
    if "refund" in q:
        relief = "refund"
    elif "replace" in q or "replacement" in q:
        relief = "replacement"
    elif "compensation" in q:
        relief = "compensation"
    elif "injunction" in q:
        relief = "injunction"

    party_type = None
    if "government hospital" in q or "govt hospital" in q:
        party_type = "government"
    elif "private hospital" in q:
        party_type = "private"

    facts = FactExtraction(
        incident_date=incident.isoformat() if incident else None,
        date_confidence=0.55 if incident else 0.0,
        claim_amount_inr=claim_amount,
        amount_confidence=0.7 if claim_amount is not None else 0.0,
        legal_domain=domain,
        domain_confidence=0.6,
        party_type=party_type,
        cause_summary=query.strip(),
        requested_relief=relief,
        relief_confidence=0.7 if relief else 0.0,
    )
    if "ignore" in q or "no reply" in q or "no response" in q:
        facts.seller_response = "ignored"
        facts.seller_confidence = 0.75
    elif "refused" in q or "denied" in q:
        facts.seller_response = "refused_or_no_response"
        facts.seller_confidence = 0.75

    facts = enforce_minimum_facts(query=query, facts=facts)
    return _derive_and_normalize(facts, today=today)


def _build_prompt(user_query: str, today: date) -> str:
    schema_json = {
        "incident_date": "YYYY-MM-DD|null",
        "date_confidence": "0.0-1.0",
        "claim_amount_inr": "number|null",
        "amount_confidence": "0.0-1.0",
        "seller_response": "cooperative|refused_or_no_response|ignored|null",
        "seller_confidence": "0.0-1.0",
        "location": "string|null",
        "legal_domain": "consumer|property|criminal|contract|labour|general",
        "domain_confidence": "0.0-1.0",
        "party_type": "private|government|unknown|null",
        "cause_summary": "string",
        "requested_relief": "refund|replacement|compensation|injunction|criminal_action|other|null",
        "relief_confidence": "0.0-1.0",
        "injury_signal": "boolean",
        "injury_confidence": "0.0-1.0",
        "statute_regime": "BNS/BNSS/BSA|unknown",
        "limitation_deadline": "YYYY-MM-DD|null",
        "is_time_barred": "true|false|null",
        "recommended_forum": "District Commission|State Commission|National Commission|Unknown",
        "needs_incident_date": "true|false",
        "follow_up_question": "string|null",
    }
    return (
        "You are a legal fact extraction engine for Indian law intake.\n"
        f"Today is {today.isoformat()} (YYYY-MM-DD).\n"
        "Return ONLY valid JSON matching the schema.\n\n"
        "Goals:\n"
        "1) Extract incident dates, amounts, locations, parties, and remedy intent from user text.\n"
        "2) Extract implicit meanings for seller_response. For example:\n"
        "   - 'they ignored me' or 'no reply' -> 'ignored'\n"
        "   - 'refused to help' or 'denied my claim' -> 'refused_or_no_response'\n"
        "   - 'they asked for my receipt' -> 'cooperative'\n"
        "3) Provide a confidence score (0.0 to 1.0) for every field based on how explicitly it is stated.\n"
        "4) Criminal matters should default to current Indian criminal framework: \"BNS/BNSS/BSA\".\n"
        "5) For consumer disputes:\n"
        "   - limitation_deadline = incident_date + 2 years (if date known)\n"
        "   - forum by claim_amount:\n"
        "     * <= 5000000 => \"District Commission\"\n"
        "     * <= 20000000 => \"State Commission\"\n"
        "     * > 20000000 => \"National Commission\"\n"
        "6) Never invent facts. Use null when unknown.\n\n"
        f"Schema:\n{json.dumps(schema_json, ensure_ascii=False, indent=2)}\n\n"
        f"User text:\n{user_query}"
    )


def _derive_and_normalize(facts: FactExtraction, today: date) -> FactExtraction:
    incident = _parse_iso_date(facts.incident_date)

    if facts.legal_domain == "criminal":
        facts.statute_regime = "BNS/BNSS/BSA"
        facts.needs_incident_date = False
        facts.follow_up_question = None

    if facts.legal_domain == "consumer":
        lim = consumer_limitation(incident_date=incident, today=today)
        facts.limitation_deadline = _to_iso(lim.deadline)
        facts.is_time_barred = lim.is_time_barred

        forum = consumer_forum_by_amount(facts.claim_amount_inr)
        facts.recommended_forum = forum.forum

    if facts.legal_domain == "contract" and facts.requested_relief in {"refund", "compensation", "other"}:
        lim = money_recovery_limitation(incident_date=incident, today=today)
        # Only fill if consumer limitation is not already used.
        if not facts.limitation_deadline:
            facts.limitation_deadline = _to_iso(lim.deadline)
            facts.is_time_barred = lim.is_time_barred

    # Ensure incident_date format consistency.
    facts.incident_date = _to_iso(incident)
    return facts


def extract_facts(query: str, llm_model: str, llm_timeout_sec: int, today: Optional[date] = None) -> FactExtraction:
    today = today or date.today()
    prompt = _build_prompt(query, today=today)
    try:
        raw = call_llm(
model=llm_model, prompt=prompt, timeout_sec=llm_timeout_sec)
        parsed = _extract_json_object(raw)
        facts = FactExtraction(**parsed)
        facts = enforce_minimum_facts(query=query, facts=facts)
        return _derive_and_normalize(facts, today=today)
    except Exception:
        return heuristic_extract_facts(query=query, today=today)
