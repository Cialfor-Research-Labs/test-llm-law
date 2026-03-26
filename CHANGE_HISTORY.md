# Change History

Purpose: This file records every meaningful code/documentation change in plain English.
Rule: Update this file whenever any file is created, edited, renamed, or deleted.

## Entry Format

- Date: `YYYY-MM-DD`
- Files changed: list of file paths
- What changed: plain-English summary
- Why changed: reason/business or technical goal
- Impact: expected behavior impact and any risk

---

## 2026-03-20

### Files changed

- `README.md`
- `legal_router.py`
- `retrieval_api.py`
- `sub_question_engine.py`
- `context_builder.py`
- `llm_judge.py`
- `fact_extractor.py`
- `jurisdiction_validator.py`
- `statutory_checks.py`
- `answer_validator.py`
- `safe_fallback.py`
- `llama_legal_answer.py`
- `run_eval.py`
- `eval_cases.json`
- `eval_report.json`

### What changed

- Added legal issue/domain routing before retrieval.
- Added intent-route constraints (include/exclude style guidance).
- Added sub-question retrieval flow to improve legal context coverage.
- Added statute-aware context handling and relevance controls.
- Added optional LLM judge stage to filter weak top chunks.
- Added structured fact extraction and procedural question handling.
- Added limitation and jurisdiction procedural checks.
- Added output validation layers for:
  - structure quality
  - domain contamination
  - legal-era mismatch (old/new criminal codes)
- Added safer fallback behavior when validation fails.
- Added evaluation pipeline and report generation for regression testing.

### Why changed

- To move from basic RAG output to a safer legal-assistant workflow.
- To reduce wrong-domain citations and generic legal answers.
- To improve actionability and procedural correctness.

### Impact

- Better domain relevance and reduced context bleed.
- Safer outputs when context quality is weak.
- Improved readiness for production-style legal answer checks.
- Residual risk remains if source corpus metadata is incomplete.

---

## 2026-03-20 (Documentation Update)

### Files changed

- `README.md`
- `INSTALL.md` (new)
- `CODE_DESCRIPTOR.md` (new)
- `CHANGE_HISTORY.md` (new)

### What changed

- Expanded product README with detailed architecture and usage sections.
- Added installation/run instructions for backend, frontend, Ollama, and evaluation.
- Added file-by-file code descriptor for maintainability.
- Created this change-history file with update rules and baseline entries.

### Why changed

- To provide complete project documentation in the `New code` folder.
- To make onboarding, operations, and maintenance clear for future development.

### Impact

- Faster onboarding for new contributors.
- Clear install path and execution workflow.
- Formal audit trail process now established for all future code changes.

---

## 2026-03-20 (Reasoning-First Reliability Upgrade)

### Files changed

- `legal_router.py`
- `sub_question_engine.py`
- `llm_judge.py`
- `llama_legal_answer.py`
- `answer_validator.py`
- `retrieval_api.py`

### What changed

- Strengthened domain locking for consumer-healthcare and consumer-ecommerce queries.
- Added stricter allowed-source/exclude-term routing signals for high-risk bleed scenarios.
- Upgraded remedy-first scoring to prioritize refund/replacement/repair and Section 39 style relief.
- Increased de-prioritization of product-liability sections when no injury signal exists.
- Added geography guardrails in the LLM judge to block foreign/non-target law contamination.
- Updated generation prompt to IRAC-style structured output with headings:
  - FACTS
  - LEGAL ISSUE
  - GROUNDS
  - ANALYSIS
  - PRAYER
  - LIMITS/UNCERTAINTY
- Updated output validator to support and enforce the new heading schema (with legacy compatibility).
- Enhanced procedural injection in responses:
  - limitation-status messaging when date is missing
  - jurisdiction-status messaging when amount is missing
  - explicit e-Daakhil filing guidance for consumer domain

### Why changed

- To move the assistant from broad semantic retrieval toward strategic legal reasoning behavior.
- To reduce contextual bleed, improve remedy relevance, and increase procedural usefulness.

### Impact

- Better domain precision in medical/ecommerce consumer scenarios.
- Lower chance of over-legalized remedies for routine consumer complaints.
- Improved answer structure for workflow use (facts, grounds, prayer).
- Better last-mile user guidance for forum and filing process.

---

## 2026-03-20 (Reasoning Litigator Prompt + Bedrock Validation Update)

### Files changed

- `llama_legal_answer.py`
- `answer_validator.py`
- `legal_router.py`
- `retrieval_api.py`
- `safe_fallback.py`

### What changed

- Changed prompt philosophy from strict context-only to guided open-book reasoning:
  - context remains primary
  - model can use clearly labeled general legal framework when context is thin
- Added domain priors in routing and passed priors into LLM prompt generation.
- Added bedrock statute allow-list in applicable-law validation so foundational acts are not auto-rejected solely due to retrieval miss.
- Updated fallback responses to provide strategic legal framework guidance instead of dead-end refusals.
- Wired the new prompt arguments into both initial generation and regeneration paths.

### Why changed

- To reduce contextual paralysis when retrieval coverage is incomplete.
- To preserve legal usefulness while retaining anti-hallucination controls.

### Impact

- Better continuity of legal guidance even under partial retrieval.
- Lower chance of empty or refusal-style answers for clear-domain questions.
- Validation remains strict for fabricated authorities while allowing core foundational statutes.

---

## 2026-03-20 (Reasoning Mode + Sanity Layer + Confidence Metadata)

### Files changed

- `retrieval_api.py`
- `llama_legal_answer.py`
- `answer_validator.py`

### What changed

- Added `determine_reasoning_mode(pack)` with three modes:
  - `context_strong` (>= 5 context blocks)
  - `context_weak` (2-4 context blocks)
  - `priors_only` (< 2 context blocks)
- Added priors-weighting strategy:
  - high for `priors_only`
  - medium for `context_weak`
  - low for `context_strong`
- Injected `MODE` and `Legal Priors Importance` into LLM prompts for both first-pass and regeneration calls.
- Added answer sanity layer in API loop:
  - for consumer domain, flags product-liability emphasis without injury signal.
  - triggers constrained regeneration when sanity fails.
- Added failure-memory metadata in API response:
  - tracks reasons such as `low_context`, `validator_conflict`, `domain_confusion`, `sanity_check_failed`, and LLM availability failures.
- Extended validator metadata with source-confidence buckets:
  - `context_based`
  - `priors_based`
  - `unverified`

### Why changed

- To stabilize decision-making under uncertain retrieval quality.
- To reduce wrong legal path selection even when sections look legally valid.
- To make outputs auditable for trust and debugging.

### Impact

- Model now gets explicit guidance on when to rely on context vs priors.
- Validator now exposes confidence provenance for legal-ground lines.
- Regeneration is now triggered for practical-sense failures, not just structural/citation issues.

---

## 2026-03-20 (Last-Mile Reliability Hardening)

### Files changed

- `legal_router.py`
- `retrieval_api.py`
- `llama_legal_answer.py`

### What changed

- Upgraded reasoning mode from quantity-only to quality-aware:
  - uses context block count and average retrieval score quality (normalized when scores exceed 1.0)
  - prevents low-relevance high-volume retrieval from being misclassified as strong context
- Added sub-domain priors:
  - `consumer_ecommerce_dispute`
  - `consumer_healthcare_dispute`
  - `consumer_real_estate_dispute`
  - fallback to domain priors when sub-domain specific prior is unavailable
- Expanded sanity checks:
  - consumer answer rejected if criminal-track language appears without context fit
  - criminal answer rejected if it drifts into consumer-refund remedy logic
- Added answer ranking in regeneration loop:
  - compares old vs regenerated candidate using structural + legal-reference signal scoring
  - keeps stronger answer and rejects weaker regeneration attempts
- Activated failure-memory feedback loop:
  - `low_context` forces `priors_only` mode with high priors weight
  - `validator_conflict` triggers relaxed validation for next regeneration attempts
- Added uncertainty-control instruction in prompt:
  - asks model to use cautious language ("generally", "typically", "subject to facts") when relying on general legal framework.

### Why changed

- To close real-world failure modes where retrieval quantity masked poor relevance.
- To improve practical answer stability under imperfect context conditions.

### Impact

- Better mode selection under noisy retrieval.
- Better legal relevance through sub-domain-specific strategic priors.
- Reduced chance of regeneration producing worse or cross-domain answers.

---

## 2026-03-20 (Second-Order Reliability Controls)

### Files changed

- `retrieval_api.py`
- `sub_question_engine.py`
- `llama_legal_answer.py`

### What changed

- Added explicit confidence scoring in API metadata:
  - aggregates reasoning mode, failure reasons, unverified validation signals, and context depth.
  - exposed as `meta.confidence_score`.
- Added contradiction detection layer:
  - detects mixed-track remedy conflicts (for example criminal + refund, civil suit + consumer complaint).
  - triggers constrained regeneration with primary-track focus instruction.
- Added secondary-domain detection:
  - identifies likely additional legal tracks from query keywords.
  - exposed as `meta.secondary_domains` for multi-issue visibility.
- Added priors-confidence dampening:
  - if intent confidence is low, high-priors mode is downgraded to medium.
- Added specificity bonus in sub-question retrieval scoring:
  - boosts chunks with legally actionable language (`section`, `shall`, `liable`, `compensation`, `refund`).
  - penalizes over-general chunks.
- Added precision downgrade guardrail in prompt:
  - model told to avoid guessing exact section numbers when uncertain.
- Added explicit multi-issue instruction in prompt:
  - address primary issue first, then briefly mention alternatives.

### Why changed

- To reduce second-order failure modes that appear under noisy retrieval or adversarial mixed queries.
- To make trust and debugging easier through explicit confidence and contradiction signals.

### Impact

- Better stability under imperfect retrieval and mixed-issue queries.
- Better traceability for QA and production monitoring.
- Lower chance of confident but internally inconsistent legal guidance.

---

## 2026-03-20 (User-Aware Self-Correction Fix Set)

### Files changed

- `fact_extractor.py`
- `legal_router.py`
- `retrieval_api.py`
- `sub_question_engine.py`
- `llama_legal_answer.py`

### What changed

- Added minimum-fact enforcement after LLM extraction:
  - hard overrides for obvious consumer purchase signals
  - explicit injury signal forcing on injury keywords
  - guaranteed non-empty cause summary
- Added entity-based domain override in routing:
  - landlord/tenant/rent/lease -> property
  - employer/salary/termination -> labour
- Added garbage retrieval detector:
  - low domain-term hit ratio marks retrieval as garbage and forces priors-first mode.
- Added answer-and-ask behavior:
  - in weak/scenario or missing-core-facts conditions, system gives usable answer plus clarification questions.
- Added scenario mode for uncertain users:
  - detects “don’t know/not sure/unknown” and switches generation behavior accordingly.
- Added post-retrieval domain reconciliation:
  - reconciles domain from query entities and retrieved context indicators.
- Added clarification trigger helper for missing incident date + amount.
- Added multi-track guidance:
  - prompt now requests primary remedy plus alternative legal actions when applicable.
- Added specificity bonus in sub-question scoring:
  - promotes actionable chunks and downranks over-general chunks.
- Added safety fallback when generation returns empty answer.

### Why changed

- To prevent early-stage decision errors from cascading into wrong legal tracks.
- To improve behavior when users provide incomplete or uncertain facts.

### Impact

- Reduced false “insufficient facts” behavior for obvious consumer/injury cases.
- Better domain precision for landlord/labour edge cases.
- Better UX through answer-first plus clarification follow-up.
- Better resilience against irrelevant retrieval contamination.

---

## 2026-03-20 (Modern Criminal Regime Enforcement)

### Files changed

- `fact_extractor.py`
- `legal_router.py`
- `llama_legal_answer.py`
- `retrieval_api.py`
- `eval_cases.json`

### What changed

- Removed criminal date-based regime questioning flow and forced criminal statute regime to modern framework (`BNS/BNSS/BSA`).
- Added routing-level statute regime function (`get_statute_regime`) and passed regime through intent route metadata.
- Updated criminal priors/source hints to prioritize modern statutes and downrank legacy IPC/CrPC/IEA retrieval hints.
- Added prompt-level criminal framework rules:
  - use BNS/BNSS/BSA by default
  - avoid legacy IPC/CrPC/IEA references unless explicitly required
  - use offence-concept equivalence over fake section precision
- Added law normalization layer in API:
  - case-insensitive replacement of legacy names with modern framework names.
- Added statute sanity guard:
  - detects IPC/CrPC/IEA leakage and triggers regeneration constraints.
- Removed criminal incident-date follow-up gate in API so responses do not stall on regime-date disambiguation.
- Updated evaluation suite:
  - switched criminal 2023 expected regime to modern framework
  - added `c13_modern_law_enforcement` test to enforce no IPC/CrPC leakage.

### Why changed

- To simplify user UX and consistently enforce current criminal-law framing across the full pipeline.

### Impact

- Cleaner criminal answers with modern statute framing.
- Reduced legacy-law leakage in generated outputs.
- Less friction from unnecessary criminal date clarifications.

---

## 2026-03-20 (Auto Legacy-to-Modern Switch Without Date Questions)

### Files changed

- `retrieval_api.py`

### What changed

- Added automatic query normalization from legacy criminal-law terms to modern framework terms before routing/retrieval/fact extraction.
- Enforced no date-question prompts in API flow:
  - removed incident-date clarification return path for limitation/date prompts.
  - removed date clarification questions from answer-and-ask follow-up block.
- Ensured reconciliation, routing, and secondary-domain detection use normalized query for consistency.
- Added metadata flags:
  - `query_normalized_for_modern_law`
  - `legacy_law_auto_switched`

### Why changed

- To automatically convert IPC/CrPC/IEA framing into BNS/BNSS/BSA workflow without asking users for incident dates.

### Impact

- Smoother user experience with modern-law consistency.
- Reduced friction from date-collection prompts.

---

## 2026-03-20 (Stateful Multi-Turn Intake Mode)

### Files changed

- `retrieval_api.py`
- `llama_legal_answer.py`

### What changed

- Added in-memory session state for multi-turn intake:
  - `session_id` support in `/query`
  - persisted conversation history, fact memory, mode, and question rounds
- Added intake controller flow:
  - question mode when required facts are missing
  - answer mode when minimum facts are satisfied
  - guardrail to avoid endless questioning (switches to answer/scenario behavior after repeated asks)
- Added fact tracker for conversational intake facts:
  - injury, invoice, warranty, seller response
- Added domain-aware missing-facts detector and mapped follow-up question generator.
- Injected conversation history and known facts into the LLM prompt.
- Added response metadata for intake operations:
  - `session_id`
  - `intake_mode`
  - `known_facts`
  - `required_facts` / `missing_facts` during question turns

### Why changed

- To move from one-shot answering into stateful, goal-directed legal intake and strategy flow.

### Impact

- System can now conduct consultation-style multi-turn questioning before final legal strategy output.
- Reduced inconsistent answers caused by missing context across turns.

---

## 2026-03-20 (Frontend Session Wiring for Multi-Turn Intake)

### Files changed

- `ui/src/components/LegalChat.tsx`

### What changed

- Removed separate `/extract_facts` pre-call and switched to direct `/query` flow.
- Added persistent `session_id` handling in frontend:
  - loads from localStorage on startup
  - sends `session_id` and `enable_intake_mode=true` on each message
  - stores updated `session_id` returned by backend.
- Added lightweight session-aware metadata display in assistant messages:
  - domain + intake mode
  - missing facts being collected
  - confidence score (if present)
- Kept existing backend validation error notes in UI output.

### Why changed

- To enable real end-user testing of stateful, multi-turn intake directly from frontend chat.

### Impact

- Users can now test consultation-style multi-turn behavior without using Swagger/Postman.
- Session continuity is maintained automatically across messages and reloads.

---

## 2026-03-20 (Frontend New Session Control)

### Files changed

- `ui/src/components/LegalChat.tsx`

### What changed

- Added `New Session` button in chat header.
- Button behavior:
  - resets local chat view to initial assistant greeting
  - sets `reset_session=true` for next `/query` request
  - keeps current `session_id` and instructs backend to reset server-side state cleanly.
- Added frontend state tracking for `resetNextSession`.
- Integrated `reset_session` payload field into `/query` request body.

### Why changed

- To let users start a fresh intake flow without manual storage clearing or API tools.

### Impact

- Cleaner UX for repeated testing and consultations.
- Reliable server/client session reset alignment.

---

## 2026-03-20 (Schema-Driven Intake and Reasoning Engine)

### Files changed

- `legal_case_schemas.json` (new)
- `schema_intake_engine.py` (new)
- `retrieval_api.py`

### What changed

- Added schema registry file with extensible case schemas:
  - `consumer_defect`
  - `employment_termination`
  - `criminal_complaint`
- Added reusable schema-driven intake engine:
  - `FactStore` with `update`, `get`, `missing_fields`
  - dynamic case classification from schema set
  - dynamic fact extraction against schema fields
  - batched missing-field question generation
  - structured final output generation with fixed headings:
    - FACTS
    - LEGAL ISSUE
    - GROUNDS
    - ANALYSIS
    - STRATEGY
    - PRAYER
- Added fallback schema-driven extractor (non-LLM path) so intake still progresses when local model is unavailable.
- Added dedicated API endpoint:
  - `POST /schema_intake`
  - supports `session_id`, `reset_session`, and multi-turn fact persistence.

### Why changed

- To replace hardcoded case branching with schema-defined intake/reasoning flow.
- To make system extensible by schema addition only.

### Impact

- New case types can be introduced by adding schemas without changing core intake loop.
- No repeated questions for already-asked fields in a session.
- Multi-turn schema-guided legal intake is now available as a first-class API flow.

---

## 2026-03-20 (Frontend Schema Intake Mode Toggle)

### Files changed

- `ui/src/components/LegalChat.tsx`

### What changed

- Added frontend toggle to switch between:
  - standard `/query` pipeline
  - schema-driven `/schema_intake` pipeline
- Persisted mode preference in localStorage.
- Updated request payloads dynamically based on selected mode.
- Unified session handling for both endpoints:
  - reuses/stores returned `session_id`
  - supports `reset_session` via existing New Session button.
- Added schema-intake response rendering:
  - displays `case_type`, `mode`, and missing fields in assistant message context.

### Why changed

- To allow immediate UI testing of the new schema-driven intake system without API tools.

### Impact

- Product now supports direct side-by-side testing of classic pipeline vs schema intake in frontend.

---

## 2026-03-20 (Intake Loop Repetition Fix)

### Files changed

- `retrieval_api.py`

### What changed

- Fixed repeated question loop in stateful intake mode:
  - asks only unasked missing fields (max 3) per round
  - avoids re-asking previously asked keys.
- Added pending-question mapping for short user replies:
  - maps terse yes/no style responses to last pending fact key.
- Improved seller-response detection:
  - accepts "ignore" phrasing as refusal/no-response signal.
- Clears pending fields after final answer generation.

### Why changed

- To stop repeated intake questions and improve understanding of short follow-up replies.

### Impact

- Better conversational continuity and reduced user frustration in multi-turn intake.

---

## 2026-03-20 (Dynamic 3-Layer Intake System)

### Files changed

- `dynamic_intake_config.json` (new)
- `dynamic_intake_engine.py` (new)
- `retrieval_api.py`

### What changed

- Added dynamic intake configuration with 3 layers:
  - global core fields
  - 8 domain field sets (consumer, employment, criminal, property, family, contract, cyber, financial, plus general fallback)
  - signal-based field sets (injury, fraud, termination, property_dispute)
- Added dynamic intake engine:
  - domain classification
  - signal detection
  - dynamic merge of core + domain + signal fields
  - schema-driven fact extraction
  - batched missing-field questioning
  - no-repeat question behavior via asked-field memory
- Added new API endpoint:
  - `POST /dynamic_intake`
  - supports session state (`session_id`, `reset_session`)
  - returns mode/domain/signals/missing fields/questions/facts/text.

### Why changed

- To implement a non-hardcoded, extensible intake system driven by configuration and runtime signals.

### Impact

- New domains/signals can be added by configuration changes without core flow rewrites.
- Intake now adapts dynamically to user input and activated legal signals.

---

## Update Checklist (Use For Every Future Change)

- Add a new dated section at top or bottom (keep order consistent).
- Mention all files touched.
- Describe changes in plain English (no vague "misc fixes").
- Note why the change was made.
- Note expected impact and any risk/trade-off.

---

## 2026-03-20 (Dynamic Intake Expansion: Domains, Signals, Strategy Tracks)

### Files changed

- `dynamic_intake_config.json`
- `dynamic_intake_engine.py`
- `retrieval_api.py`

### What changed

- Expanded dynamic intake domain coverage from basic set to production-oriented set:
  - consumer, employment, criminal, property, family, contract, cyber, financial,
  - insurance, tax_regulatory, personal_injury_tort, corporate_commercial, and general fallback.
- Expanded signal catalog with Tier-like practical triggers:
  - injury, death, fraud, harassment, termination, defect, non_payment, delay,
  - document_exists, no_response, partial_payment, verbal_agreement, online_transaction,
  - police_involved, legal_notice_sent, court_case_exists, threat, repeat_offense, property_dispute.
- Added signal-driven strategy metadata in config:
  - `strategy_hints` per signal
  - global `signal_strategy_map` for signal-to-strategy-track mapping.
- Updated dynamic intake engine to compute and persist strategy influence:
  - `resolve_strategy_tracks(...)`
  - `resolve_strategy_hints(...)`
  - injects tracks/hints into final structured output prompt.
- Updated dynamic intake API response to expose strategy layer:
  - added `strategy_tracks` in `DynamicIntakeResponse` and endpoint payload.

### Why changed

- To move from simple intake classification to signal-aware legal strategy shaping, without hardcoding case-specific logic.

### Impact

- Questioning remains schema-driven and extensible while now reflecting stronger cross-domain signals.
- Output generation now receives explicit strategy tracks/hints, improving consistency between intake findings and final legal reasoning.
- New domains/signals can still be added by config only, preserving extensibility.

### Follow-up refinement (same date)

- Added schema-driven signal-to-domain boosting to improve fallback domain classification when LLM is unavailable.
- Reduced generic corporate routing bias by tightening `corporate_commercial` classifier hints.
- Improved question mode behavior when all unique questions were already asked:
  - system now shows pending field keys instead of silently repeating or stalling.

### Follow-up refinement (controlled signal extraction)

- Upgraded dynamic signal detection to a constrained, JSON-only pipeline:
  - detection with allowed signals + confidence
  - strict validation pass to remove unsupported signals
  - conservative fallback filtering by confidence.
- Added few-shot style examples inside the detection prompt to reduce under-detection drift.
- Strengthened fact extraction prompt rules:
  - no guessing
  - null for missing fields
  - only extract clearly mentioned values.
- Added `signal_confidence` to dynamic intake outputs and API response model for observability.

### Follow-up refinement (post-intake RAG integration)

- Added schema-driven post-intake RAG flow in `dynamic_intake_engine.py`:
  - RAG executes only in answer mode (after required facts are collected).
  - Added retrieval query builder from `domain + signals + facts`.
  - Added multi-query retrieval (3 query variants), merge + dedupe, top-k context selection.
  - Added context formatting with law name, section, text, and source.
- Updated final generation prompt to be context-grounded:
  - "Use ONLY provided legal context"
  - "Do not hallucinate laws"
  - "If context is missing, say so"
- Added dynamic intake response metadata in `retrieval_api.py`:
  - `rag_used`, `retrieval_queries`, `retrieved_context_count`, `retrieved_citations`.

### Validation notes

- Confirmed question mode remains RAG-free.
- Confirmed answer mode performs retrieval and returns citations/context count.

### Follow-up refinement (task detection + routing layer)

- Added task-intent layer on top of dynamic intake with supported tasks:
  - `advice`, `draft_notice`, `file_complaint`, `estimate_claim`.
- Added `tasks` registry to `dynamic_intake_config.json` with task hints and task-required fields.
- Implemented task detection and session persistence:
  - detected task is stored in fact store and persisted in dynamic session state (`session["task"]` equivalent via `DYNAMIC_INTAKE_SESSIONS[session_id]["task"]`).
- Implemented task routing in `dynamic_intake_engine.py`:
  - `advice` -> existing RAG-grounded structured advice flow
  - `draft_notice` -> notice generator
  - `file_complaint` -> complaint draft generator
  - `estimate_claim` -> claim-estimate generator
- Changed missing-info behavior to be task-specific:
  - asks only missing fields required by current task
  - reuses existing collected facts (no full re-intake loop)
  - no repeated question keys due existing asked-key memory.
- Updated `/dynamic_intake` response model to include task-routing metadata:
  - `task`, `routed_handler`
- Updated dynamic session load/save helpers in `retrieval_api.py` to persist and restore task.

### Follow-up refinement (strict task mode behavior)

- Added deterministic task detection shortcuts:
  - "legal notice"/"draft notice" -> `draft_notice`
  - "complaint" -> `file_complaint`
  - "what can I do"/"legal advice" -> `advice`
- Added `critical_fields` per task in dynamic task config and enforced critical-only follow-ups for non-intake tasks.
- Implemented dedicated notice function `generate_legal_notice(facts)`:
  - template-based formatted notice output only
  - no extra questions
  - no repeated legal analysis blocks.
- Updated task router so `draft_notice` uses `generate_legal_notice(facts)` directly.
- Updated flow to skip RAG retrieval for `draft_notice` path (`rag_used=false`) while keeping existing advice/other task routing.
