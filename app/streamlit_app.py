"""
BornToFit Healthcare Agent: agentic medical task automation (Streamlit UI).
Run from the project root:   streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent.executor import invoke_tool
from src.agent.models import Session
from src.config import get_settings
from src.evaluation.evaluator import benchmark_parallelism, run_evaluation
from src.mcp_server.server import MCPServer
from src.pipeline import build_app
from src.security.rbac import DEMO_PRINCIPALS, PermissionDenied
from src.tools.registry import ToolContext

st.set_page_config(page_title="BornToFit | Healthcare Agent", page_icon="🌿", layout="wide")

GREEN, DARK, AMBER = "#0E7C66", "#064E3B", "#B45309"
STATUS_ICON = {"ok": "✅ ok", "failed": "❌ failed", "skipped": "⏭️ skipped", "needs_confirmation": "⏸️ awaiting approval",
               "pending": "… pending", "running": "⏳ running"}
STATUS_COLOR = {"ok": "#2E7D32", "failed": "#B3261E", "skipped": "#9CA3AF", "needs_confirmation": AMBER}
KIND_COLOR = {"agent": "#064E3B", "guardrail": "#B3261E", "plan": "#6D28D9", "execute": "#0E7C66", "tool": "#2563EB", "llm": "#B45309",
              "compose": "#0891B2", "internal": "#6B7280"}

# label -> (query, demo identity to switch to or None)
SCENARIOS = {
    "Capstone: book nephrologist + treatments": ("My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. "
                                                 "Also, can you summarize latest treatment methods?", "attendant"),
    "History and alerts (by name)": ("Summarize Ramesh Kumar's medical history and alerts", "attendant"),
    "Book cardiologist, next week morning": ("I need an appointment with a cardiologist for Lakshmi Devi next week morning", "attendant"),
    "Add a record (attendant)": ("Add a note for Suresh Iyer: HbA1c 7.4% on follow-up visit", "attendant"),
    "Diet and exercise guidance (CKD)": ("What diet and exercise guidance is there for chronic kidney disease?", "attendant"),
    "Export FHIR bundle": ("Export Ramesh Kumar's records as FHIR", "attendant"),
    "Family member books for father": ("Book an appointment for my 70-year-old father", "guardian_p1001"),
    "RBAC: family asks about another patient": ("Summarize Suresh Iyer's medical history and alerts", "guardian_p1001"),
    "RBAC: family tries to add a record": ("Add a note for Ramesh Kumar: BP 130/80 today", "guardian_p1001"),
    "Ambiguous patient (asks back)": ("Book an appointment for my 70-year-old father", "attendant"),
    "Safety: emergency language": ("My father has chest pain and can't breathe", "attendant"),
    "Safety: diagnosis / dosage request": ("Do I have diabetes? Prescribe me metformin dosage", "attendant"),
}

st.markdown(
    f"""
    <style>
    div[data-testid="stDecoration"], div[data-testid="stAppDeployButton"], .stAppDeployButton, #MainMenu {{display:none !important;}}
    header[data-testid="stHeader"] {{background: transparent !important; height: 2.6rem !important; min-height: 2.6rem !important;}}
    .main .block-container, div[data-testid="stMainBlockContainer"] {{padding-top: .6rem !important; padding-bottom: 1rem !important; max-width: 100% !important;}}
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{gap: .35rem !important;}}
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{background: transparent; border: none; color: {DARK}; font-weight: 600;
        font-style: italic; text-align: left !important; justify-content: flex-start !important; padding: .1rem 0; box-shadow: none; width: 100%;}}
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {{color: {GREEN}; text-decoration: underline; background: transparent;}}
    div[role="dialog"] {{max-width: 660px !important; width: 92vw !important; max-height: 80vh !important; overflow-y: auto !important;}}
    .btf-banner {{background: linear-gradient(135deg, #052E22 0%, {GREEN} 55%, #34D399 100%); padding: .7rem 1.2rem; border-radius: 10px;
        margin: .45rem 0 .7rem 0; box-shadow: 0 3px 10px rgba(14,124,102,.25);}}
    .btf-banner h1 {{color:#fff; font-size:1.25rem; font-weight:800; margin:0 0 .2rem 0;}}
    .btf-banner p {{color:#ECFDF5; font-size:.78rem; margin:0 0 .4rem 0; line-height:1.4;}}
    .btf-badges span {{background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.4); color:#fff; font-size:.62rem; font-weight:600;
        padding:.15rem .55rem; border-radius:999px; margin-right:.3rem; white-space:nowrap; display:inline-block; margin-bottom:.15rem;}}
    </style>
    <div class="btf-banner">
      <h1>🌿 BornToFit Healthcare Agent</h1>
      <p>One request, several coordinated actions. A typed planner turns it into a dependency graph; guarded tools book appointments, read and
      update records, export FHIR and search trusted sources (MedlinePlus, WHO) with corrective RAG. Every call is role-checked, audited,
      traced and checkpointed, and every write can wait for human approval.</p>
      <div class="btf-badges"><span>🧭 Typed planner</span><span>⚡ Parallel DAG</span><span>🛂 RBAC + audit</span><span>🧪 Corrective RAG</span>
      <span>🛡️ Injection defence</span><span>⏸️ Human approval + resume</span><span>🧠 Tiered memory + patient brief</span><span>🔌 MCP server</span>
      <span>🏥 FHIR R4 export</span><span>🔭 Traces</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ state
if "pending_principal" in st.session_state:       # set by scenario buttons; must be applied BEFORE the selectbox is created
    st.session_state["principal_key"] = st.session_state.pop("pending_principal")


def init_state() -> None:
    """Each browser session gets its own in-memory demo database (no cross-user data sharing)."""
    if "app" not in st.session_state:
        with st.spinner("Preparing synthetic hospital data and the memory index..."):
            st.session_state.app = build_app(get_settings())
        st.session_state.session = Session()
        st.session_state.chat = []
        st.session_state.last_search = None
        st.session_state.eval_report = None


init_state()
app = st.session_state.app
session: Session = st.session_state.session
settings = app.settings


def fmt_ts(ts: str) -> str:
    return ts.replace("T", " ")[:16]


# ------------------------------------------------------------------ dialogs
@st.dialog("Why BornToFit?")
def dlg_why():
    st.markdown("**The problem:** appointments, patient records and disease information sit in separate tools, so one patient request needs "
                "several manual hops.\n\n**The idea:** an *agentic* assistant plans the whole request, calls the right tools in the right order "
                "(in parallel when it can), remembers patient context, shows its work, and stops to ask before risky actions.\n\n"
                "**Beyond a chatbot:** it acts (book, update, export), cites trusted sources, enforces who may see what, and keeps an audit trail.")


@st.dialog("Benefits")
def dlg_benefits():
    st.markdown("- **Fewer hops:** one sentence replaces identify, check history, find slot, book, look up information.\n"
                "- **Speed:** independent steps run in parallel (benchmark in the Evaluation tab).\n"
                "- **Safer care admin:** alerts and allergies surface at booking time.\n"
                "- **Trust:** every source is labelled live or offline; answers are grounded and citation-checked.\n"
                "- **Governance:** role checks, audit log, traces and an evaluation gate with red-team cases.\n"
                "- **Interoperability:** FHIR R4 export and an MCP server expose the same guarded tools to other systems.")


@st.dialog("Privacy and safety")
def dlg_privacy():
    st.markdown("- **Synthetic data only.** Do not enter real patient information.\n"
                "- **No diagnosis, prescribing or dosing.** General information only.\n"
                "- **Emergency language** stops the workflow and shows an emergency message.\n"
                "- **Roles and scope:** a family member can only reach their own patient; clinical staff can write; every decision is audited.\n"
                "- **PHI minimisation:** names become placeholders before an external LLM; public search queries are scrubbed.\n"
                "- **Prompt-injection defence:** web results are screened and quarantined; instruction-like text in records is stripped before it reaches an LLM.\n"
                "- **Approval:** writes can wait for a human; the plan can never approve itself.\n"
                "- **Logs** keep tool names, status and timing, never free-text clinical content.\n"
                "- **Not production-ready:** the identity switch is a demo, not authentication; no encryption at rest, consent management or clinician-reviewed content.")


@st.dialog("Standards and interoperability")
def dlg_standards():
    st.markdown("- **FHIR R4 (4.0.1)** collection Bundle export: Patient, Condition, MedicationStatement, AllergyIntolerance, Observation, Flag, "
                "Encounter, Appointment. Structural checks only; not run through the official HL7 validator; no coded terminologies.\n"
                "- **Model Context Protocol (MCP)** server over stdio. It works with the stateless 2026-07-28 flow and the older handshake flow. "
                "Conformance is untested against official suites; use an official SDK for production.\n"
                "- **MedlinePlus Web Service** and **WHO (via Tavily domain filter)** for live information; results are cached, rate-limited and labelled.")


@st.dialog("Roadmap")
def dlg_roadmap():
    st.markdown("1. **Real identity:** SSO/OIDC claims instead of the demo identity; treatment-relationship scoping for doctors.\n"
                "2. **Real integrations:** a FHIR server and hospital scheduling API behind the same tool interfaces.\n"
                "3. **Retrieval:** clinical embeddings, guideline ingestion, reranking, per-source freshness checks.\n"
                "4. **Orchestration:** map the plan graph onto LangGraph or an MCP gateway when workflows need loops.\n"
                "5. **Operations:** persistent database, OTLP trace export, alerting, clinician-reviewed content, encryption at rest.")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div style="font-size:2.0rem; font-weight:800; line-height:1.1; margin:0;">🌿 BornToFit</div>'
                '<div style="font-size:.85rem; color:#6B7280; font-weight:500; margin:.3rem 0 .6rem 0;">Healthcare Agent</div>', unsafe_allow_html=True)
    if app.llm:
        st.success(f"✅ Smart Mode ({app.llm.name})")
    else:
        st.markdown('<div style="background:#ECFDF5; padding:.6rem .8rem; border-radius:.5rem; border-left:4px solid #0E7C66; font-size:.72rem;">'
                    '⚡ <strong>Live Mode</strong>: planning, tools and answers run offline with no AI subscription. Add an LLM key and model to switch to Smart Mode.</div>',
                    unsafe_allow_html=True)
    stt = app.search.status()
    st.caption("Sources: " + ", ".join(x for x, on in (("MedlinePlus live", stt["medlineplus_live"]), ("WHO/Medline via Tavily", stt["tavily_who_medline"]),
                                                       ("offline corpus", True)) if on))
    st.markdown('<div style="font-weight:700; margin:.4rem 0 .1rem 0;">Learn more:</div>', unsafe_allow_html=True)
    for key, label, dlg in (("l1", "🔍 Why BornToFit?", dlg_why), ("l2", "💼 Know its Benefits", dlg_benefits), ("l3", "🔐 Privacy and safety", dlg_privacy),
                            ("l4", "🏥 Standards and interoperability", dlg_standards), ("l5", "🗺️ Roadmap", dlg_roadmap)):
        if st.button(label, key=key):
            dlg()
    st.divider()
    st.subheader("Session")
    keys = list(DEMO_PRINCIPALS)
    pkey = st.selectbox("Signed in as (demo identity)", keys, key="principal_key", format_func=lambda k: f"{DEMO_PRINCIPALS[k].label} · {DEMO_PRINCIPALS[k].role}",
                        help="Demo role switch to show access control. It is not authentication.")
    principal = DEMO_PRINCIPALS[pkey]
    allowed = [p for p in app.db.list_patients() if principal.sees(p["patient_id"])]
    labels = ["Auto-detect from message"] + [f"{p['patient_id']} · {p['name']}" for p in allowed]
    cur = next((i + 1 for i, p in enumerate(allowed) if p["patient_id"] == session.active_patient_id), 0)
    pick = st.selectbox("Active patient", labels, index=cur, help="Used when a message names no patient.")
    session.active_patient_id = None if pick == labels[0] else pick.split(" · ")[0]
    settings.auto_confirm_writes = st.toggle("Auto-book without approval", value=settings.auto_confirm_writes,
                                             help="Off: the agent proposes a slot and waits for your approval.")
    settings.parallel = st.toggle("Run independent steps in parallel", value=settings.parallel)
    if st.button("↺ Reset demo data", key="reset"):
        for k in ("app", "session", "chat", "last_search", "eval_report"):
            st.session_state.pop(k, None)
        st.rerun()
    st.caption(f"Memory index: {app.memory.stats()['backend']}")
    st.caption("Synthetic demo data. Not for real patient information.")


def tctx() -> ToolContext:
    return ToolContext(app.db, app.memory, app.search, settings, session, "ui", principal)


# ------------------------------------------------------------------ helpers for run rendering
def timeline_fig(plan) -> go.Figure:
    fig = go.Figure()
    for s in plan.steps:
        fig.add_bar(y=[f"{s.id} {s.tool}"], x=[max(s.end_ms - s.start_ms, 0.6)], base=[s.start_ms], orientation="h",
                    marker_color=STATUS_COLOR.get(s.status, "#6B7280"), hovertemplate=f"{s.goal}<br>%{{x:.1f}} ms<extra></extra>", showlegend=False)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=70 + 30 * max(len(plan.steps), 1), margin=dict(l=0, r=10, t=10, b=30), bargap=0.35,
                      xaxis_title="ms since execution start (overlapping bars = steps that ran in parallel)")
    return fig


def span_fig(spans: list[dict]) -> go.Figure:
    by_id = {s["span_id"]: s for s in spans}
    depth = lambda s: 0 if not s.get("parent_id") or s["parent_id"] not in by_id else 1 + depth(by_id[s["parent_id"]])  # noqa: E731
    fig = go.Figure()
    rows = sorted(spans, key=lambda s: (s["start_ms"], depth(s)))
    for s in rows:
        end = s["end_ms"] if s["end_ms"] is not None else s["start_ms"]
        fig.add_bar(y=["  " * depth(s) + s["name"]], x=[max(end - s["start_ms"], 0.3)], base=[s["start_ms"]], orientation="h",
                    marker_color=KIND_COLOR.get(s["kind"], "#6B7280") if s["status"] == "ok" else "#B3261E", showlegend=False,
                    hovertemplate=f"{s['name']} ({s['kind']})<br>%{{x:.1f}} ms<br>{json.dumps(s['attrs'], default=str)[:200]}<extra></extra>")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=60 + 24 * max(len(rows), 1), margin=dict(l=0, r=10, t=10, b=30), bargap=0.3, xaxis_title="ms since run start")
    return fig


def steps_df(plan) -> pd.DataFrame:
    return pd.DataFrame([{"step": s.id, "goal": s.goal, "tool": s.tool, "waits for": ", ".join(s.depends_on) or "-",
                          "status": STATUS_ICON.get(s.status, s.status), "ms": s.latency_ms, "note": s.error or ""} for s in plan.steps])


def render_run(entry: dict, idx: int) -> None:
    res = entry["result"]
    with st.expander("🧭 Plan and execution", expanded=idx == len(st.session_state.chat) - 1):
        p = res.plan
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Planner", p.planner.split(" ")[0])
        c2.metric("Steps / critical path", f"{len(p.steps)} / {p.critical_path()}")
        c3.metric("Wall time", f"{res.wall_ms:.0f} ms")
        c4.metric("If run one-by-one", f"{res.sequential_estimate_ms:.0f} ms")
        st.caption(f"Intents: {', '.join(p.intents) or 'none'}  |  {res.mode}  |  as {res.principal}  |  run {res.run_id}  |  checkpoint v{res.checkpoint_version}")
        for n in p.notes:
            st.caption(f"Note: {n}")
        if p.steps:
            st.dataframe(steps_df(p), hide_index=True, width="stretch")
            st.plotly_chart(timeline_fig(p), width="stretch", key=f"tl_{res.run_id}_{idx}_{res.checkpoint_version}")
        for sid in res.pending_approvals:
            s = next(x for x in p.steps if x.id == sid)
            prop = (s.result or {}).get("proposal", {})
            st.warning(f"Held for approval: {prop.get('doctor_name')} ({prop.get('specialty')}) at {fmt_ts(prop.get('start_ts', ''))}. {(s.result or {}).get('message', '')}")
            b1, b2 = st.columns(2)
            if b1.button("✅ Approve this booking", key=f"ap_{res.run_id}_{sid}"):
                entry["result"] = app.agent.resume(res.run_id, sid, True, session, principal)
                st.rerun()
            if b2.button("✖ Decline", key=f"dc_{res.run_id}_{sid}"):
                entry["result"] = app.agent.resume(res.run_id, sid, False, session, principal)
                st.rerun()
    with st.expander(f"🔭 Trace ({len(res.spans)} spans, {res.tokens} LLM tokens)"):
        if res.spans:
            st.plotly_chart(span_fig(res.spans), width="stretch", key=f"sp_{res.run_id}_{idx}_{res.checkpoint_version}")
            st.caption("OpenTelemetry-style spans: guardrails, plan, each tool call, compose. LLM spans show token usage when a model is configured.")
    if res.sources:
        with st.expander(f"📚 Sources ({len(res.sources)})"):
            for s in res.sources:
                badge = "🟢 live" if s["freshness"] == "live" else "🟠 offline reference"
                link = f" · [{s['url']}]({s['url']})" if s.get("url") else ""
                st.markdown(f"**[{s['id']}] {s['title']}**  \n{badge} · {s['source']}{link} · relevance {s.get('relevance')}")
    if res.quarantined:
        st.error(f"🛡️ {res.quarantined} retrieved item(s) were quarantined as possible prompt injection and never reached the answer.")


def handle(query: str) -> None:
    with st.spinner("Planning and executing..."):
        res = app.agent.run(query, session, principal)
    st.session_state.chat.append({"query": query, "result": res})
    for s in res.plan.steps:
        if s.tool == "search_medical_info" and isinstance(s.result, dict) and s.result.get("results") is not None:
            st.session_state.last_search = {"query": s.result["query_sent"], "result": s.result, "run": res.run_id}


# ------------------------------------------------------------------ tabs
tab_chat, tab_people, tab_appt, tab_info, tab_eval, tab_mem, tab_interop = st.tabs(
    ["💬 Assistant", "🧑‍⚕️ Patient and doctor", "📅 Appointments", "📚 Medical info", "🧪 Evaluation", "🧠 Memory and logs", "🏥 Interop and MCP"])

# ---------------- Assistant
with tab_chat:
    st.subheader("Ask for several things at once")
    st.caption(f"You are signed in as **{principal.label}** ({principal.role}). Scenarios marked with another identity switch it for you.")
    cols = st.columns(3)
    for i, (label, (q, who)) in enumerate(SCENARIOS.items()):
        if cols[i % 3].button(label, key=f"sc{i}", width="stretch"):
            st.session_state.pending_query = q
            if who and who != pkey:
                st.session_state.pending_principal = who
            st.rerun()
    for turn_i, turn in enumerate(st.session_state.chat):
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.markdown(turn["result"].answer)
            render_run(turn, turn_i)
    typed = st.chat_input("e.g. Book a nephrologist for my 70-year-old father with chronic kidney disease and summarise latest treatments")
    clicked = st.session_state.pop("pending_query", None)
    if typed or clicked:
        handle(typed or clicked)
        st.rerun()
    if st.session_state.chat and st.button("Clear conversation", key="clr"):
        st.session_state.chat = []
        session.turns.clear()
        session.last_patient_id = session.last_appointment_id = None
        st.rerun()

# ---------------- Patient and doctor
with tab_people:
    left, right = st.columns(2)
    with left:
        st.subheader("🧑 Patient view")
        if not allowed:
            st.info("This identity has no patient access.")
        else:
            pid = st.selectbox("Patient", [p["patient_id"] for p in allowed], format_func=lambda x: f"{x} · {app.db.get_patient(x)['name']}", key="pv_patient")
            p = app.db.get_patient(pid)
            m1, m2, m3 = st.columns(3)
            m1.metric("Age / sex", f"{p['age']} / {p['sex']}")
            m2.metric("Blood group", p["blood_group"])
            m3.metric("Primary condition", p["primary_condition"])
            hist = invoke_tool(tctx(), app.registry, "get_patient_history", {"patient_id": pid})
            st.info(hist["brief"])
            for a in hist["summary"]["alerts"]:
                st.warning(a)
            st.markdown("**Record timeline**")
            st.dataframe(pd.DataFrame(app.db.get_records(pid))[["rec_date", "rec_type", "title", "details", "author"]], hide_index=True, width="stretch", height=240)
            st.markdown("**Upcoming appointments**")
            ups = pd.DataFrame(hist["upcoming_appointments"])
            if len(ups):
                ups["when"] = ups["start_ts"].map(fmt_ts)
                st.dataframe(ups[["when", "doctor_name", "specialty", "reason"]], hide_index=True, width="stretch")
            else:
                st.caption("None booked.")
            with st.expander("✍️ Add a record (updates long-term memory and the patient brief)"):
                if principal.role == "patient":
                    st.caption("Your role cannot add records. Switch the demo identity to an attendant or doctor.")
                with st.form("add_rec", clear_on_submit=True):
                    f1, f2 = st.columns([1, 2])
                    rtype = f1.selectbox("Type", ["note", "diagnosis", "medication", "lab", "allergy", "alert", "visit"])
                    title = f2.text_input("Title")
                    details = st.text_area("Details (free text)")
                    if st.form_submit_button("Save record"):
                        try:
                            r = invoke_tool(tctx(), app.registry, "add_patient_record", {"patient_id": pid, "rec_type": rtype, "title": title or rtype.title(), "details": details})
                            if r["ok"]:
                                st.success(f"Saved record #{r['record']['record_id']}; memory re-indexed.")
                                st.rerun()
                            else:
                                st.error(r["message"])
                        except PermissionDenied as e:
                            st.error(f"Not permitted: {e}")
    with right:
        st.subheader("🩺 Doctor view")
        docs = app.db.list_doctors()
        did = st.selectbox("Doctor", [d["doctor_id"] for d in docs], format_func=lambda x: f"{app.db.get_doctor(x)['name']} · {app.db.get_doctor(x)['specialty']}", key="dv_doc")
        d = app.db.get_doctor(did)
        n1, n2 = st.columns(2)
        n1.metric("Specialty", d["specialty"])
        n2.metric("Location", d["location"])
        now_s = settings.now().strftime("%Y-%m-%dT%H:%M")
        sl = app.db.df("SELECT substr(start_ts,1,10) AS day, status, count(*) AS n FROM slots WHERE doctor_id=? AND start_ts>=? GROUP BY day, status ORDER BY day LIMIT 40", (did, now_s))
        if len(sl):
            fig = go.Figure()
            for status, color in (("booked", AMBER), ("open", GREEN)):
                x = sl[sl["status"] == status]
                fig.add_bar(x=x["day"], y=x["n"], name=status, marker_color=color)
            fig.update_layout(barmode="stack", height=250, margin=dict(l=0, r=0, t=25, b=0), title="Slots per day: booked vs open")
            st.plotly_chart(fig, width="stretch", key="doc_slots")
        if principal.role == "patient":
            st.caption("Booked appointments of other patients are hidden for this identity.")
        else:
            sched = pd.DataFrame(app.db.list_appointments(doctor_id=did, status="booked"))
            st.markdown("**Booked appointments**")
            if len(sched):
                sched["when"] = sched["start_ts"].map(fmt_ts)
                st.dataframe(sched[["when", "patient_name", "reason", "created_by"]], hide_index=True, width="stretch", height=280)
            else:
                st.caption("No bookings yet.")

# ---------------- Appointments (live)
with tab_appt:
    @st.fragment(run_every="5s")
    def tracker() -> None:
        st.subheader("Real-time appointment tracking")
        st.caption("Refreshes every 5 seconds. Bookings made by the assistant appear as soon as they are saved.")
        allappt = pd.DataFrame(app.db.list_appointments())
        now_s = settings.now().strftime("%Y-%m-%dT%H:%M")
        if allappt.empty:
            st.info("No appointments yet.")
            return
        if principal.role == "patient":
            allappt = allappt[allappt["patient_id"].isin(principal.patient_ids)]
            st.caption("Showing only the appointments you may access.")
            if allappt.empty:
                st.info("No appointments for your patients yet.")
                return
        up = allappt[(allappt["status"] == "booked") & (allappt["start_ts"] >= now_s)]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Upcoming booked", len(up))
        k2.metric("Booked by the assistant", int(((allappt["created_by"] == "agent") & (allappt["status"] == "booked")).sum()))
        k3.metric("Cancelled", int((allappt["status"] == "cancelled").sum()))
        week = (settings.now() + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M")
        k4.metric("Open slots, next 7 days", app.db.q("SELECT count(*) c FROM slots WHERE status='open' AND start_ts>=? AND start_ts<?", (now_s, week))[0]["c"])
        f1, f2 = st.columns(2)
        spec = f1.multiselect("Specialty", sorted(allappt["specialty"].unique()))
        stat = f2.multiselect("Status", sorted(allappt["status"].unique()), default=["booked"])
        view = allappt.copy()
        if spec:
            view = view[view["specialty"].isin(spec)]
        if stat:
            view = view[view["status"].isin(stat)]
        view["when"] = view["start_ts"].map(fmt_ts)
        st.dataframe(view.sort_values("start_ts")[["appt_id", "when", "patient_name", "doctor_name", "specialty", "reason", "status", "created_by"]], hide_index=True, width="stretch", height=280)
        cnt = up.groupby("specialty").size().reset_index(name="booked")
        opn = pd.DataFrame(app.db.q("SELECT d.specialty, count(*) AS open_slots FROM slots s JOIN doctors d ON d.doctor_id=s.doctor_id WHERE s.status='open' AND s.start_ts>=? AND s.start_ts<? GROUP BY d.specialty", (now_s, week)))
        merged = cnt.merge(opn, on="specialty", how="outer").fillna(0)
        fig = go.Figure()
        fig.add_bar(x=merged["specialty"], y=merged["booked"], name="booked", marker_color=AMBER)
        fig.add_bar(x=merged["specialty"], y=merged["open_slots"], name="open (next 7d)", marker_color=GREEN)
        fig.update_layout(barmode="group", height=270, margin=dict(l=0, r=0, t=25, b=0), title="Demand vs capacity by specialty")
        st.plotly_chart(fig, width="stretch", key="appt_chart")
        with st.expander("Cancel an appointment"):
            cand = up["appt_id"].tolist()
            if cand:
                pick = st.selectbox("Appointment", cand, key="cancel_pick", format_func=lambda i: f"#{i} · {up[up.appt_id == i].iloc[0]['patient_name']} · {up[up.appt_id == i].iloc[0]['specialty']} · {fmt_ts(up[up.appt_id == i].iloc[0]['start_ts'])}")
                if st.button("Cancel selected", key="do_cancel"):
                    try:
                        invoke_tool(tctx(), app.registry, "cancel_appointment", {"patient_id": up[up.appt_id == pick].iloc[0]["patient_id"], "appt_id": int(pick)})
                    except PermissionDenied as e:
                        st.error(f"Not permitted: {e}")
                    st.rerun()
    tracker()

# ---------------- Medical info
with tab_info:
    st.subheader("Latest retrieved medical information")
    st.caption("From the assistant's most recent search, or run your own. Queries are scrubbed of names, phone numbers and IDs before leaving the app.")
    q = st.text_input("Search trusted sources", placeholder="e.g. chronic kidney disease treatment options", key="info_q")
    if st.button("Search", key="info_go") and q.strip():
        r = invoke_tool(tctx(), app.registry, "search_medical_info", {"query": q.strip()})
        st.session_state.last_search = {"query": r.get("query_sent", q), "result": r, "run": "manual"}
    ls = st.session_state.last_search
    if not ls:
        st.info("Nothing retrieved yet. Ask the assistant for treatment information, or search above.")
    else:
        r = ls["result"]
        st.markdown(f"**Query sent:** `{ls['query']}`  ·  **Mode:** {'🟢 live' if r['mode'] == 'live' else '🟠 offline reference corpus'}")
        if r["mode"] == "offline":
            st.warning("Showing the built-in offline corpus (static, paraphrased). Enable live search for up-to-date information (MedlinePlus needs no key; Tavily adds WHO).")
        for e in r.get("errors", []):
            st.caption(f"Live source unavailable: {e}")
        with st.expander("🧪 Corrective RAG trace", expanded=False):
            st.dataframe(pd.DataFrame(r["crag"]["attempts"]), hide_index=True, width="stretch")
            st.caption(f"Query rewritten: {r['crag']['rewritten']}  ·  Fell back after quarantine: {r['crag']['quarantined_fallback']}  ·  Relevance threshold 0.5")
        for qd in r.get("quarantined", []):
            st.error(f"🛡️ Quarantined: **{qd['title']}** ({qd['source']}): {', '.join(qd['reasons'])}")
        for d in r["results"]:
            with st.container(border=True):
                st.markdown(f"**{d['title']}**  ·  relevance {d.get('relevance')}")
                st.write(d["snippet"])
                st.caption(d["source"] + (f" · retrieved {d['retrieved_at']}" if d.get("retrieved_at") else "") + (f" · {d['url']}" if d.get("url") else ""))
        st.caption(r["disclaimer"])

# ---------------- Evaluation
with tab_eval:
    st.subheader("Model, tool and red-team evaluation")
    st.caption("Runs 26 scenarios on a fresh, isolated database each: planning, patient resolution, booking, retrieval, safety, RBAC, prompt injection, privacy, FHIR and the approval flow. "
               "Uses the LLM judge when a model is configured, otherwise a keyword grader. Live search is disabled so results are reproducible.")
    use_llm = bool(app.llm) and st.checkbox("Evaluate Smart Mode (LLM judge and composer)", value=False)
    if st.button("▶ Run evaluation suite", type="primary", key="run_eval"):
        bar = st.progress(0.0, text="Starting...")
        st.session_state.eval_report = run_evaluation(settings, llm=app.llm, use_llm=use_llm, progress=lambda f, cid: bar.progress(f, text=f"Case {cid}"))
        bar.empty()
    rep = st.session_state.eval_report
    if rep:
        a, b, c = st.columns(3)
        a.metric("Overall score", f"{rep['overall'] * 100:.0f}%")
        b.metric("Cases passed", f"{int(round(rep['pass_rate'] * rep['n']))} / {rep['n']}")
        c.metric("Judge", "LLM" if use_llm else "keyword grader")
        l, r2 = st.columns(2)
        fig = go.Figure(go.Bar(x=rep["by_module"]["avg_score"], y=rep["by_module"]["module"], orientation="h", marker_color=GREEN))
        fig.update_layout(height=320, title="Average score by module", margin=dict(l=0, r=0, t=30, b=0), xaxis_range=[0, 1])
        l.plotly_chart(fig, width="stretch", key="ev_mod")
        fig2 = go.Figure(go.Bar(x=rep["by_check"]["pass_rate"], y=rep["by_check"]["check"], orientation="h", marker_color=DARK))
        fig2.update_layout(height=320, title="Pass rate by check", margin=dict(l=0, r=0, t=30, b=0), xaxis_range=[0, 1])
        r2.plotly_chart(fig2, width="stretch", key="ev_chk")
        st.dataframe(rep["cases"], hide_index=True, width="stretch")
    st.divider()
    st.markdown("**Tool success across this session** (from the tool log)")
    tl = app.db.df("SELECT tool, status, count(*) AS n FROM tool_logs GROUP BY tool, status")
    if len(tl):
        piv = tl.pivot_table(index="tool", columns="status", values="n", fill_value=0).reset_index()
        piv["success_rate"] = ((piv["ok"] if "ok" in piv else 0) / piv.drop(columns="tool").sum(axis=1)).round(2)
        st.dataframe(piv, hide_index=True, width="stretch")
    else:
        st.caption("No tool calls yet. Run a scenario in the Assistant tab.")
    st.divider()
    st.markdown("**Parallel vs sequential execution** (same request, simulated per-tool latency)")
    lat = st.slider("Simulated latency per tool call (ms)", 50, 400, 150, 50)
    if st.button("Run benchmark", key="bench"):
        with st.spinner("Benchmarking..."):
            bm = benchmark_parallelism(settings, latency_ms=lat, repeats=2)
        x1, x2, x3 = st.columns(3)
        x1.metric("Sequential", f"{bm['sequential']:.0f} ms")
        x2.metric("Parallel (dependency graph)", f"{bm['parallel']:.0f} ms")
        x3.metric("Speed-up", f"{bm['speedup']}x")
        st.caption("Five tool calls, three sequential hops: patient lookup, slot search and the medical search are independent.")

# ---------------- Memory and logs
with tab_mem:
    st.subheader("Memory traces, planning breakdown, audit and tool logs")
    ms = app.memory.stats()
    a, b, c = st.columns(3)
    a.metric("Long-term memory chunks", ms["chunks"])
    b.metric("Trace events", ms["trace_events"])
    c.metric("Retrieval backend", ms["backend"].split(" (")[0])
    st.markdown("**Working memory (this session)**")
    st.json({"identity": principal.short(), "active_patient": session.active_patient_id, "last_patient": session.last_patient_id,
             "last_condition": session.last_condition, "last_appointment": session.last_appointment_id, "recent_turns": len(session.turns)}, expanded=False)
    if st.session_state.chat:
        last = st.session_state.chat[-1]["result"]
        with st.expander("Planning breakdown of the latest run", expanded=True):
            st.json({"query": last.query, "planner": last.plan.planner, "intents": last.plan.intents,
                     "steps": [{"id": s.id, "tool": s.tool, "args": s.args, "depends_on": s.depends_on, "status": s.status} for s in last.plan.steps]}, expanded=False)
    st.markdown("**Episodic memory trace** (reads, writes, recalls, plans)")
    tr = pd.DataFrame(app.memory.traces[-120:][::-1])
    st.dataframe(tr, hide_index=True, width="stretch", height=230) if len(tr) else st.caption("No events yet.")
    st.markdown("**Tool call log**")
    logs = app.db.df("SELECT ts, run_id, step_id, tool, status, latency_ms, error, side_effect FROM tool_logs ORDER BY log_id DESC LIMIT 200")
    if len(logs):
        st.dataframe(logs, hide_index=True, width="stretch", height=230)
        agg = logs.groupby(["tool", "status"]).size().reset_index(name="n")
        fig = go.Figure()
        for stt_, color in (("ok", "#2E7D32"), ("failed", "#B3261E"), ("skipped", "#9CA3AF"), ("needs_confirmation", AMBER)):
            x = agg[agg["status"] == stt_]
            if len(x):
                fig.add_bar(x=x["tool"], y=x["n"], name=stt_, marker_color=color)
        fig.update_layout(barmode="stack", height=250, margin=dict(l=0, r=0, t=25, b=0), title="Tool outcomes")
        st.plotly_chart(fig, width="stretch", key="log_chart")
    else:
        st.caption("No tool calls logged yet.")
    st.markdown("**Access audit** (every authorisation decision)")
    aud = app.db.df("SELECT ts, run_id, principal, role, tool, patient_id, decision, reason FROM audit_log ORDER BY audit_id DESC LIMIT 200")
    st.dataframe(aud, hide_index=True, width="stretch", height=230) if len(aud) else st.caption("No decisions yet.")
    st.markdown("**Agent runs, spans and checkpoints**")
    runs = app.db.df("SELECT ts, run_id, principal, query, intents, n_steps, ok, latency_ms, planner, mode, tokens FROM runs ORDER BY ts DESC LIMIT 50")
    if len(runs):
        st.dataframe(runs, hide_index=True, width="stretch", height=190)
        rid = st.selectbox("Inspect run", runs["run_id"].tolist(), key="inspect_run")
        spans = app.db.q("SELECT * FROM spans WHERE trace_id=? ORDER BY start_ms", (rid,))
        if spans:
            st.plotly_chart(span_fig([{**s, "attrs": json.loads(s["attrs"])} for s in spans]), width="stretch", key="insp_spans")
        st.dataframe(pd.DataFrame(app.db.checkpoint_history(rid)), hide_index=True, width="stretch")
        st.caption("Checkpoints are saved after planning, after execution (or when awaiting approval) and after approval, so a held write survives a restart.")
    else:
        st.caption("No runs yet.")
    with st.expander("🔎 Test long-term memory retrieval (hybrid BM25 + vectors)"):
        rp = st.selectbox("Patient", [p["patient_id"] for p in allowed] or ["P1001"], key="mem_p", format_func=lambda x: f"{x} · {app.db.get_patient(x)['name']}")
        rq = st.text_input("Query", value="medicines to avoid", key="mem_q")
        if rq and principal.sees(rp):
            hits = app.memory.index.search(rq, k=5, where=lambda m: m.get("patient_id") == rp)
            st.dataframe(pd.DataFrame([{"score": h.score, "matched_by": h.matched_by, "kind": h.meta.get("kind"), "chunk": h.text} for h in hits]), hide_index=True, width="stretch")

# ---------------- Interop and MCP
with tab_interop:
    st.subheader("FHIR R4 export")
    st.caption("Collection Bundle with Patient, Condition, MedicationStatement, AllergyIntolerance, Observation, Flag, Encounter and Appointment resources. "
               "Structural checks only; not validated with the official HL7 validator; clinical text is not coded.")
    if allowed:
        fpid = st.selectbox("Patient", [p["patient_id"] for p in allowed], format_func=lambda x: f"{x} · {app.db.get_patient(x)['name']}", key="fhir_p")
        try:
            r = invoke_tool(tctx(), app.registry, "export_fhir_bundle", {"patient_id": fpid})
            ca, cb = st.columns(2)
            ca.metric("Resources", len(r["bundle"]["entry"]))
            cb.metric("Structure checks", "passed" if r["ok"] else f"{len(r['validation_errors'])} issue(s)")
            st.caption(", ".join(f"{k} x{v}" for k, v in r["counts"].items()))
            st.download_button("⬇ Download FHIR bundle (JSON)", data=json.dumps(r["bundle"], indent=2), file_name=f"{fpid}_fhir_r4_bundle.json", mime="application/fhir+json")
            with st.expander("Preview first resources"):
                st.json({"resourceType": "Bundle", "type": "collection", "entry": r["bundle"]["entry"][:3]}, expanded=False)
        except PermissionDenied as e:
            st.error(f"Not permitted: {e}")
    else:
        st.info("This identity has no patient access.")
    st.divider()
    st.subheader("MCP server")
    st.markdown("The same guarded tools are available to any MCP client over stdio. Role and patient scope are fixed at launch; discovery lists only the tools "
                "the role may call; every call is audited. Run it from the project root:")
    st.code("python scripts/mcp_server.py --role doctor\npython scripts/mcp_server.py --role patient --patients P1001\n"
            "python scripts/mcp_server.py --role attendant --db borntofit.db   # shared SQLite file", language="bash")
    st.code(json.dumps({"mcpServers": {"borntofit": {"command": "python", "args": ["scripts/mcp_server.py", "--role", "doctor"]}}}, indent=2), language="json")
    st.caption("Spec note: 2026-07-28 removes the initialize handshake. This server answers tools/list and tools/call without a handshake and also supports `initialize` for older clients.")
    listing = MCPServer(app, principal).handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    st.markdown(f"**Tools a client would see as {principal.role}:**")
    st.dataframe(pd.DataFrame([{"tool": t["name"], "read-only": t["annotations"]["readOnlyHint"], "destructive": t["annotations"]["destructiveHint"],
                                "description": t["description"][:110]} for t in listing]), hide_index=True, width="stretch")
