from conftest import CAPSTONE, FakeLLM
from src.agent.models import Session
from src.observability.tracing import RunTrace, traced_chat
from src.pipeline import build_app


def test_run_has_a_complete_hierarchical_trace(app):
    r = app.agent.run(CAPSTONE, Session(), simulated_latency_ms=60)      # deterministic overlap for independent tools
    names = [s["name"] for s in r.spans]
    root = next(s for s in r.spans if s["name"] == "agent.run")
    assert {"guardrails", "plan", "execute", "compose", "tool:find_patient", "tool:book_appointment", "tool:search_medical_info"} <= set(names)
    ids = {s["span_id"] for s in r.spans}
    assert all(s["parent_id"] in ids for s in r.spans if s["parent_id"]) and root["parent_id"] is None
    assert all(s["end_ms"] >= s["start_ms"] for s in r.spans) and root["end_ms"] >= max(s["end_ms"] for s in r.spans if s is not root) - 1e-6
    tools = [s for s in r.spans if s["kind"] == "tool"]
    assert any(a["start_ms"] < b["end_ms"] and b["start_ms"] < a["end_ms"] for a in tools for b in tools if a is not b)   # parallel overlap
    assert len(app.db.q("SELECT * FROM spans WHERE trace_id=?", (r.run_id,))) == len(r.spans)


def test_failed_tool_marks_span_error(app):
    r = app.agent.run("Summarize Zed Unknown's medical history", Session())
    assert any(s["kind"] == "tool" and s["status"] == "error" for s in r.spans)


def test_llm_spans_carry_token_usage(settings):
    import json
    plan = json.dumps({"intents": ["medical_info"], "steps": [{"id": "s1", "goal": "g", "tool": "search_medical_info", "args": {"query": "type 2 diabetes treatment options"}, "depends_on": []}]})
    from src.security.guardrails import MEDICAL_DISCLAIMER
    app = build_app(settings, llm=FakeLLM(plan, "Metformin is first-line [S1]. " + MEDICAL_DISCLAIMER))
    r = app.agent.run("What are the latest treatment options for type 2 diabetes?", Session())
    llm_spans = [s for s in r.spans if s["kind"] == "llm"]
    assert len(llm_spans) == 2 and r.tokens == 36 and all(s["attrs"]["tokens_in"] == 11 for s in llm_spans)
    assert r.mode == "Smart Mode (LLM)" and app.db.q("SELECT tokens FROM runs")[0]["tokens"] == 36


def test_traced_chat_without_trace_and_error_status():
    llm = FakeLLM("hi")
    assert traced_chat(None, None, llm, "x", "s", "u") == "hi"
    t = RunTrace("t")
    try:
        with t.span("boom", "internal"):
            raise RuntimeError("bad")
    except RuntimeError:
        pass
    assert t.spans[0]["status"] == "error" and "RuntimeError" in t.spans[0]["attrs"]["error"]
