# Phase 5 Completion: LLM Investigation Assist & Evidence Grounding

## 1. Overview & Objective
Phase 5 delivers the asynchronous investigation assist engine of **SentinelForge**. For every transaction flagged by the real-time scoring engine (risk score $\ge 0.60$), this layer gathers 360-degree context (historical customer baseline, identity graph linkages, merchant risk, geographic displacement) and generates a structured, evidence-grounded case narrative using Groq LLM (Llama-3-8B). It validates factual citations via an automated **Evidence Grounding Verifier** to eliminate hallucinations before presenting cases to fraud analysts.

---

## 2. Artifacts & Deliverables Completed

| Component | Path | Description & Design Justification |
| :--- | :--- | :--- |
| **System Prompts** | [`services/investigation/prompts.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/investigation/prompts.py) | Strict system prompt enforcing factual grounding, bracketed citations `[Citing: field=val]`, severity tagging, and disposition recommendations. |
| **Case Builder** | [`services/investigation/case_builder.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/investigation/case_builder.py) | Compiles 7-dimensional context dossier (transaction, scoring, customer baseline, linked accounts graph, merchant profile, geo signals, feature snapshot). |
| **Grounding Verifier** | [`services/investigation/evidence_grounding.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/investigation/evidence_grounding.py) | Extracts all citations from generated markdown, cross-checks against ground-truth context, checks numerical tolerance ($\pm 10\%$), and scores fidelity. |
| **Investigation Agent** | [`services/investigation/investigation_agent.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/investigation/investigation_agent.py) | Groq API (`llama-3.1-8b-instant`) integration with Prometheus latency tracking and deterministic fallback synthesis for resilience. |
| **Unit & Integration Tests** | `tests/unit/test_case_builder.py`, `tests/unit/test_evidence_grounding.py`, `tests/integration/test_investigation_pipeline.py` | 6 new tests verifying context compilation, citation verification, and end-to-end async narrative generation. |

---

## 3. How Evidence Grounding Works

```mermaid
flowchart TD
    A[Flagged Transaction Event] --> B[CaseBuilder: Compile 360 Context Bundle]
    B --> C[InvestigationAgent: Groq Llama-3-8B]
    C --> D[Generated Case Narrative with [Citing: ...] Tokens]
    D --> E{EvidenceGroundingVerifier}
    E -->|Grounding Score >= 0.80| F[Verified Narrative Stored in DB & Streamed to Analyst]
    E -->|Grounding Score < 0.80| G[Reject Hallucination & Trigger Deterministic Fallback]
```

1. **Citation Token Extraction**:
   Regex pattern `r"\[Citing:\s*([^\]]+)\]"` parses claims like `[Citing: amount_zscore=14.2, txn_count_1min=6]`.
2. **Context Matching & Proximity**:
   Values are cross-referenced with the flattened JSON context bundle. Numbers are verified within $\pm 10\%$ rounding tolerance; booleans and categorical strings are checked for exact equivalence.
3. **Acceptance Threshold**:
   $$\text{Grounding Score} = \frac{\text{Verified Claims}}{\text{Total Claims}}$$
   If score $< 0.80$ ($>20\%$ unverified claims), the narrative is marked ungrounded, preventing AI hallucinations from reaching human investigators.

---

## 4. Test Verification Results (48 Tests Passing)

```text
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
collected 48 items

tests\integration\test_feature_pipeline.py ..                            [  4%]
tests\integration\test_investigation_pipeline.py .                       [  6%]
tests\integration\test_postgres.py ..                                    [ 10%]
tests\integration\test_redis.py .                                        [ 12%]
tests\integration\test_scoring_pipeline.py .                             [ 14%]
tests\unit\test_case_builder.py ..                                       [ 18%]
tests\unit\test_evidence_grounding.py ...                                [ 25%]
tests\unit\test_feature_definitions.py .......                           [ 39%]
tests\unit\test_models.py .........                                      [ 58%]
tests\unit\test_scorer.py ...                                            [ 64%]
tests\unit\test_settings.py ......                                       [ 77%]
tests\unit\test_simulator.py .......                                     [ 91%]
tests\unit\test_threshold_router.py ....                                 [100%]

============================= 48 passed in 1.85s ==============================
```

---

## 5. Next Steps
With Phase 5 verified, the project is ready for **Phase 6: FastAPI Gateway & Analyst Dashboard**, exposing REST endpoints (`/score`, `/cases`, `/cases/{id}/feedback`), JWT authentication, Redis token-bucket rate limiting, and Server-Sent Events (SSE) for live investigation streaming.
