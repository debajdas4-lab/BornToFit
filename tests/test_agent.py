import json
import threading

from conftest import CAPSTONE, FakeLLM
from src.agent.models import Session
from src.data.db import SlotUnavailable
from src.pipeline import build_app
from src.tools.registry import ToolContext


def call(app, tool, principal=None, **kw):
    ctx = ToolContext(app.db, app.memory, app.search, app.settings, Session(), "t", principal)
    return app.registry.get(tool).fn(ctx, **kw)


def agent_appts(app):
    return [a for a in app.db.list_appointments(status="booked") if a["created_by"] == "agent"]


def test_find_patient_disambiguates(app):
    r = call(app, "find_patient", age=70, sex="M", condition="chronic kidney disease")
    assert r["ok"] and r["patient"]["patient_id"] == "P1001"
    amb = call(app, "find_patient", age=70, sex="M")
    assert amb["status"] == "ambiguous" and len(amb["candidates"]) == 2
    assert call(app, "find_patient", name="Zed Unknown")["status"] == "not_found"


def test_history_has_alerts_brief_and_memory(app):
    h = call(app, "get_patient_history", patient_id="P1001")
    assert any("NSAID" in a for a in h["summary"]["alerts"]) and h["memory_hits"] and h["brief"].startswith("PATIENT BRIEF")


def test_double_booking_impossible_under_concurrency(app):
    slot = app.db.open_slots("Nephrology", "2026-10-06T00:00", "2026-10-30T00:00", limit=1)[0]["slot_id"]
    results, lock = [], threading.Lock()

    def worker(pid):
        try:
            app.db.book(pid, slot, "race"); out = "ok"
        except SlotUnavailable:
            out = "taken"
        with lock:
            results.append(out)
    ts = [threading.Thread(target=worker, args=(p,)) for p in ("P1001", "P1002", "P1003", "P1004", "P1005")]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert results.count("ok") == 1 and results.count("taken") == 4


def test_booking_self_heals_when_slot_taken(app):
    slots = app.db.open_slots("Nephrology", "2026-10-06T00:00", "2026-10-30T00:00", limit=2)
    app.db.book("P1005", slots[0]["slot_id"], "someone else")
    r = call(app, "book_appointment", patient_id="P1001", slot_id=slots[0]["slot_id"], specialty="Nephrology")
    assert r["status"] == "booked" and r["retries"] == 1 and r["appointment"]["slot_id"] != slots[0]["slot_id"]


def test_capstone_end_to_end(app, session):
    r = app.agent.run(CAPSTONE, session)
    assert r.ok and r.patient_id == "P1001" and session.last_patient_id == "P1001"
    assert "Nephrology" in r.answer and r.sources and r.sources[0]["freshness"] == "offline" and "not medical advice" in r.answer
    b = agent_appts(app)
    assert len(b) == 1 and b[0]["patient_id"] == "P1001" and b[0]["specialty"] == "Nephrology"


def test_ambiguous_patient_blocks_writes_but_info_still_runs(app):
    r = app.agent.run("Book an appointment for my 70-year-old father and tell me latest treatment options for hypertension", Session())
    assert r.needs_clarification and "Ramesh Kumar" in r.answer and not agent_appts(app) and r.sources


def test_emergency_short_circuits(app):
    r = app.agent.run("my father has chest pain and can't breathe, book a cardiologist", Session())
    assert r.emergency and not r.plan.steps and "emergency" in r.answer.lower() and not agent_appts(app)


def test_parallel_faster_than_sequential(app):
    from src.evaluation.evaluator import benchmark_parallelism
    b = benchmark_parallelism(app.settings, latency_ms=120, repeats=1)
    assert b["parallel"] < b["sequential"] and b["speedup"] > 1.3


def test_free_text_never_logged_and_record_updates_memory_and_brief(app, session):
    app.agent.run("Add a note for Suresh Iyer: HbA1c 7.4% on follow-up visit", session)
    logs = app.db.df("SELECT * FROM tool_logs WHERE tool='add_patient_record'")
    assert len(logs) == 1 and "HbA1c" not in logs.iloc[0]["args"] and "chars" in logs.iloc[0]["args"]
    assert "HbA1c 7.4%" in " ".join(h.text for h in app.memory.recall("P1002", "HbA1c follow-up"))


def test_cancel_by_just_booked_reference(app):
    s = Session()
    app.agent.run("Book a nephrologist for Meena Pillai next week", s)
    r = app.agent.run("Cancel the appointment you just booked for Meena Pillai", s)
    assert r.ok and "cancelled" in r.answer.lower()


def plan_json(confirm):
    return json.dumps({"intents": ["book_appointment"], "steps": [
        {"id": "s1", "goal": "find", "tool": "find_patient", "args": {"name": "[PATIENT_1]"}, "depends_on": []},
        {"id": "s2", "goal": "slots", "tool": "find_slots", "args": {"specialty": "Nephrology"}, "depends_on": []},
        {"id": "s3", "goal": "book", "tool": "book_appointment", "depends_on": ["s1", "s2"],
         "args": {"patient_id": "$s1.patient.patient_id", "slot_id": "$s2.best_slot.slot_id", "specialty": "Nephrology", "confirm": confirm}}]})


def test_llm_plan_redacted_executes_and_structured_output_requested(settings):
    llm = FakeLLM(plan_json(False), "x")
    app = build_app(settings, llm=llm)
    r = app.agent.run("Book a nephrologist for Ramesh Kumar next week", Session())
    assert "Ramesh" not in llm.calls[0]["user"] and "[PATIENT_1]" in llm.calls[0]["user"] and llm.calls[0]["json_schema"]
    assert r.plan.planner == "llm" and r.ok and agent_appts(app)[0]["patient_id"] == "P1001"


def test_plan_can_never_approve_itself(settings):
    settings.auto_confirm_writes = False
    app = build_app(settings, llm=FakeLLM(plan_json(True), "x"))
    r = app.agent.run("Book a nephrologist for Ramesh Kumar next week", Session())
    assert r.pending_approvals == ["s3"] and not agent_appts(app)            # confirm:true from the plan is stripped


def test_invalid_llm_plan_falls_back_to_rules(settings):
    bad = json.dumps({"intents": [], "steps": [{"id": "s1", "goal": "x", "tool": "delete_all_records", "args": {}, "depends_on": []}]})
    app = build_app(settings, llm=FakeLLM(bad, "x"))
    r = app.agent.run("Book a nephrologist for Ramesh Kumar next week", Session())
    assert "fallback" in r.plan.planner and r.ok


def test_composer_rejects_invented_doctor_and_bad_citations(settings):
    from src.agent.composer import compose, grounding_ok, verify_citations
    from src.security.guardrails import MEDICAL_DISCLAIMER
    ev = json.dumps({"booking": {"doctor": "Dr. Anjali Rao", "when": "Mon 12 Oct 2026, 09:00-09:30"}})
    assert grounding_ok("Booked Dr. Anjali Rao at 9:00", ev) and not grounding_ok("Dr. Fake Person at 09:00", ev) and not grounding_ok("Dr. Anjali Rao at 17:45", ev)
    src = [{"id": "S1", "title": "CKD", "snippet": "Blood pressure control slows kidney disease progression."}]
    assert verify_citations("Blood pressure control slows progression [S1].", src)[0] == 1.0
    assert verify_citations("Doctors recommend surgery for everyone immediately [S1].", src)[0] == 0.0
    assert verify_citations("No citation at all.", src)[0] == 0.0
    app = build_app(settings, llm=None)
    r = app.agent.run("What are the latest treatment options for type 2 diabetes?", Session())
    llm = FakeLLM("Surgery cures diabetes instantly [S1]. " + MEDICAL_DISCLAIMER)
    out = compose(r.plan, None, False, llm)
    assert not out.used_llm and "Surgery" not in out.answer and any("citation check" in n for n in out.notes)
