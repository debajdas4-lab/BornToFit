# BornToFit Healthcare Agent

An agentic medical-task assistant. One natural-language request becomes a **typed plan** (a dependency graph of tool calls) that is
executed in **parallel**, **role-checked**, **audited**, **traced** and **checkpointed**. It books and cancels appointments, reads and updates
patient records, exports **FHIR R4**, and answers disease questions from trusted sources with **corrective RAG**. The same guarded tools are
available to any agent through an **MCP server**.

It follows the InsightForge layout (launcher, UI, config, data, tools, memory, agent, evaluation, tests) and runs **fully offline** with no AI
subscription (Live Mode), or with an LLM (Smart Mode).

> **Demo only.** All patients, doctors and schedules are synthetic. Do not enter real patient information. General information, not medical
> advice. The identity switch is a demo of access control, **not authentication**. No encryption at rest or consent management.

## Run it

```powershell
python -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Tests, evaluation gate, MCP server:

```powershell
python -m pytest -q
python scripts/run_evaluation.py --min-score 0.95 --benchmark
python scripts/mcp_server.py --role doctor
```

Optional: copy `.env.example` to `.env` (or use Streamlit **Secrets**). You must set `OPENAI_MODEL` (or `ANTHROPIC_MODEL`) yourself when you add a key:
no model name is assumed. `pip install faiss-cpu` switches the vector index from NumPy to FAISS automatically.

## The sample scenario

> "My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?"

```
s1 find_patient (70, M, CKD)       s3 find_slots (Nephrology)      s5 search_medical_info (CKD treatment options)
        |                               |                               (independent: runs in parallel)
s2 get_patient_history (needs s1)       |
        \________ s4 book_appointment (needs s1, s2, s3) ________/
```
Five tool calls, three sequential hops, ten spans, two checkpoints.

## What is new compared with a prompt-chain assistant

| Capability | How it works here | Where |
|---|---|---|
| Typed planner + validation | LLM must return a JSON plan (JSON-schema structured output, with fallback to JSON mode); unknown tools, cycles, missing references and identifiers in search queries are rejected and the rules planner takes over | `agent/planner.py` |
| Parallel dependency graph | Independent steps run concurrently; results flow through `$step.path` references | `agent/executor.py` |
| Role-based access + patient scope | Deny by default; a family member reaches only their own patient, and cannot even *search* others; discovery and calls are filtered by role | `security/rbac.py` |
| Audit | Every authorisation decision (allow/deny, principal, tool, patient) is stored | `data/db.py` `audit_log` |
| Human approval + resume | Writes can be held; the held run is persisted as a checkpoint and finished later, even by a new process; the approver is re-authorised; a plan can never approve itself | `agent/agent.py` `resume()` |
| Corrective RAG | Retrieve (live/offline) -> grade relevance -> rewrite and retry -> fall back -> screen -> cite | `tools/medical_search.py` |
| Prompt-injection defence (OWASP LLM01) | Web results and EHR free text are treated as DATA: scanned, quarantined or neutralised before any LLM sees them | `security/injection.py` |
| Verified answers | LLM wording is rejected if it names a doctor/date/time not in the tool output or cites claims the source does not support | `agent/composer.py` |
| Tiered memory + patient brief | Working (session pronouns), episodic (trace), long-term (hybrid BM25 + vectors, FAISS optional) plus a consolidated per-patient brief that refreshes on every write | `memory/` |
| Observability | OpenTelemetry-style spans (guardrails, plan, each tool, each LLM call with token usage) stored and drawn as a waterfall | `observability/tracing.py` |
| FHIR R4 export | Patient, Condition, MedicationStatement, AllergyIntolerance, Observation, Flag, Encounter, Appointment, Practitioner as a collection Bundle with a structural validator | `tools/interop.py` |
| MCP server | stdio, newline-delimited JSON-RPC; works without a handshake (2026-07-28 style) and with `initialize` (older clients) | `mcp_server/server.py` |
| Red-team evaluation | 26 cases incl. RBAC, injection, PHI leakage, FHIR and approval; mutation tests prove each defence matters | `evaluation/`, `tests/test_security.py` |

## File-wise workflow (read in this order)

| # | File | What it does |
|---|---|---|
| 1 | `app.py` | Root launcher |
| 2 | `app/streamlit_app.py` | UI: 7 tabs (Assistant, Patient and doctor, Appointments, Medical info, Evaluation, Memory and logs, Interop and MCP) |
| 3 | `src/config.py` | Settings from `.env` / environment / Streamlit Secrets |
| 4 | `src/pipeline.py` | `build_app()` wires every layer |
| 5 | `src/data/db.py`, `generate_sample_data.py` | SQLite EHR, Doctor Schedule API, audit, spans, checkpoints; synthetic data |
| 6 | `src/security/rbac.py`, `guardrails.py`, `injection.py` | Access control, safety/PHI, injection defence |
| 7 | `src/tools/` | `patients`, `scheduling`, `medical_search` (corrective RAG), `interop` (FHIR), `registry`, `medical_terms` |
| 8 | `src/memory/` | Hybrid index, tiered memory, patient brief |
| 9 | `src/agent/planner.py` -> `executor.py` -> `composer.py` -> `agent.py` | Plan, execute, answer, checkpoint/resume |
| 10 | `src/mcp_server/server.py`, `scripts/mcp_server.py` | MCP tool server |
| 11 | `src/evaluation/evaluator.py`, `eval/eval_dataset.json` | Per-module evaluation, LLM judge or keyword grader |
| 12 | `tests/` | 76 tests |

## Trusted sources

| Source | Needs | Status |
|---|---|---|
| MedlinePlus Web Service | nothing (free; 85 requests/min/IP; cached 12 h) | live path implemented, **not tested against the live service** |
| WHO + MedlinePlus via Tavily (`include_domains`) | `TAVILY_API_KEY` | implemented, **not tested live** |
| Offline reference corpus | nothing | always available; paraphrased, static, **not clinician-reviewed** |

Bing Search APIs were retired on 11 Aug 2025, so they are not used. Results always show whether they are live or offline.

## Known limits

- Synthetic data; demo identity only; no encryption at rest, consent management or treatment-relationship scoping for doctors.
- The offline embedder is lexical (hashed n-grams), not semantic. Use `BORNTOFIT_EMBEDDINGS=remote` with an OpenAI-compatible key for dense embeddings.
- The FHIR export is shape-checked by this project's own validator, **not** the official HL7 validator, and carries no coded terminologies. FHIR instants use `BORNTOFIT_TZ_OFFSET` as an assumption.
- MCP: the `server/discover` result shape and the `ttlMs` cache hint are best-effort, and conformance is untested against official suites. For production, put the official SDK in front of the tool layer.
- LLM clients (OpenAI-compatible, Anthropic), remote embeddings, FAISS and live search were written to the public API shapes but **could not be exercised from the build sandbox** (no internet, no keys).
- In-memory SQLite per browser session: data resets on reload. Use `BORNTOFIT_DB_PATH` for a shared file.
- Orchestration is framework-free so it runs offline on free hosting; the plan graph maps onto LangGraph nodes if you want it.
