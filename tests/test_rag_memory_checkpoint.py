import pytest

from conftest import FakeLLM
from src.agent.agent import HealthAgent
from src.agent.models import Session
from src.pipeline import build_app
from src.tools.medical_search import MedicalSearch, grade, parse_medlineplus, rewrite_query
from src.tools.registry import TransientToolError


def held(settings, who):
    settings.auto_confirm_writes = False
    app = build_app(settings, llm=None)
    r = app.agent.run("Book a nephrologist for Ramesh Kumar next week", Session(), who)
    return app, r


# ---------------------------------------------------------------- corrective RAG
def test_grader_and_rewrite():
    on = {"title": "Chronic kidney disease: treatment overview", "snippet": "Treatment of chronic kidney disease aims to slow progression."}
    off = {"title": "Type 2 diabetes: management overview", "snippet": "Metformin first line; screening for kidney disease."}
    assert grade("chronic kidney disease treatment options", on) >= 0.9
    assert grade("chronic kidney disease treatment options", off) < 0.5          # right words, wrong condition
    assert rewrite_query("what is new for ckd latest") == "chronic kidney disease treatment overview"


def test_crag_rewrites_then_gives_up_on_off_topic(settings):
    r = MedicalSearch(settings).search("quantum entanglement")
    assert r["results"] == [] and r["crag"]["rewritten"] and len(r["crag"]["attempts"]) == 4


def test_crag_discards_irrelevant_live_results_and_uses_offline(settings):
    settings.live_search = True
    ms = MedicalSearch(settings)
    ms._medlineplus = lambda q, n: [{"title": "Seasonal allergies", "snippet": "Pollen causes sneezing.", "url": "u", "source": "MedlinePlus.gov", "freshness": "live", "retrieved_at": "now"}]
    r = ms.search("chronic kidney disease treatment options")
    assert r["mode"] == "offline" and r["crag"]["attempts"][0]["kept"] == 0 and all("kidney" in d["title"].lower() for d in r["results"])


def test_live_failure_falls_back_with_error_recorded(settings, monkeypatch):
    settings.live_search = True
    ms = MedicalSearch(settings)
    monkeypatch.setattr(ms, "_medlineplus", lambda q, n: (_ for _ in ()).throw(TransientToolError("down")))
    r = ms.search("chronic kidney disease treatment")
    assert r["mode"] == "offline" and r["errors"] and r["results"][0]["freshness"] == "offline"


def test_medlineplus_parser():
    xml = ('<nlmSearchResult><list><document url="https://medlineplus.gov/x.html"><content name="title">&lt;span&gt;Kidney Disease&lt;/span&gt;</content>'
           '<content name="FullSummary">&lt;p&gt;Kidneys filter blood.&lt;/p&gt;</content></document></list></nlmSearchResult>')
    assert parse_medlineplus(xml) == [{"title": "Kidney Disease", "snippet": "Kidneys filter blood.", "url": "https://medlineplus.gov/x.html"}]


def test_lifestyle_query_prefers_the_diet_document(settings):
    r = MedicalSearch(settings).search("chronic kidney disease diet and exercise lifestyle")
    assert "diet" in r["results"][0]["title"].lower()
    r2 = MedicalSearch(settings).search("type 2 diabetes treatment options")
    assert "management" in r2["results"][0]["title"].lower()


# ---------------------------------------------------------------- memory
def test_hybrid_retrieval_is_patient_scoped(app):
    hits = app.memory.recall("P1002", "HbA1c metformin diabetes")
    assert hits and all(h.id.startswith("P1002") for h in hits)
    assert not app.memory.recall("P1002", "xyzzy nonsense token")


def test_patient_brief_is_consolidated_and_refreshed_on_write(app):
    b0 = app.memory.brief("P1002")
    assert "type 2 diabetes" in b0 and "Glimepiride" in b0
    app.registry.get("add_patient_record").fn(
        __import__("src.tools.registry", fromlist=["ToolContext"]).ToolContext(app.db, app.memory, app.search, app.settings, None, "t", None),
        patient_id="P1002", rec_type="alert", title="New alert", details="Recent hypoglycaemia episode")
    assert "hypoglycaemia" in app.memory.brief("P1002").lower() and app.memory.recall("P1002", "patient brief hypoglycaemia")[0].meta["kind"] in ("brief", "alert")


# ---------------------------------------------------------------- checkpoint / human approval
def test_held_write_survives_restart_and_resumes_from_database(settings, P):
    app, r = held(settings, P["attendant"])
    assert r.pending_approvals == ["s4"] and not [a for a in app.db.list_appointments() if a["created_by"] == "agent"]
    fresh = HealthAgent(app.db, app.memory, app.registry, app.search, app.settings, None)       # new object = "process restart"
    r2 = fresh.resume(r.run_id, "s4", True, Session(), P["attendant"])
    assert r2.ok and "Appointment booked" in r2.answer and not r2.pending_approvals
    assert len([a for a in app.db.list_appointments(status="booked") if a["created_by"] == "agent"]) == 1
    assert [c["label"] for c in app.db.checkpoint_history(r.run_id)] == ["planned", "awaiting_approval", "approved"]


def test_decline_changes_nothing_and_is_not_resumable_twice(settings, P):
    app, r = held(settings, P["attendant"])
    r2 = app.agent.resume(r.run_id, "s4", False, Session(), P["attendant"])
    assert "declined" in r2.answer and not [a for a in app.db.list_appointments() if a["created_by"] == "agent"]
    with pytest.raises(ValueError):
        app.agent.resume(r.run_id, "s4", True, Session(), P["attendant"])
    with pytest.raises(ValueError):
        app.agent.resume("nope", "s1", True, Session(), P["attendant"])


def test_alternative_slot_requires_approval_even_with_auto_confirm(app):
    r = app.agent.run("Book a cardiologist for Ramesh Kumar today", Session())      # nothing open "today" -> alternative
    assert r.pending_approvals and "waiting for your approval" in r.answer
