from datetime import datetime

from conftest import CAPSTONE
from src.agent.models import Plan, Session, Step
from src.agent.planner import parse_when, validate_plan
from src.tools.registry import default_registry


def plan_for(app, text, session=None):
    return app.agent.rules.plan(text, session or Session())


def test_capstone_plan_has_expected_dag(app):
    p = plan_for(app, CAPSTONE)
    assert set(p.intents) == {"book_appointment", "medical_info"}
    by_tool = {s.tool: s for s in p.steps}
    assert set(by_tool) == {"find_patient", "get_patient_history", "find_slots", "book_appointment", "search_medical_info"}
    assert by_tool["find_patient"].args == {"age": 70, "sex": "M", "condition": "chronic kidney disease"}
    assert by_tool["find_slots"].depends_on == [] and by_tool["search_medical_info"].depends_on == []
    assert set(by_tool["book_appointment"].depends_on) == {"s1", "s2", "s3"}
    assert p.critical_path() == 3          # 5 tool calls, only 3 sequential hops


def test_search_query_never_contains_identifiers(app):
    p = plan_for(app, "Summarize Ramesh Kumar's history and latest treatment options for his kidney disease")
    q = [s for s in p.steps if s.tool == "search_medical_info"][0]
    assert "Ramesh" not in str(q.args) and "P1001" not in str(q.args)


def test_past_treatments_is_history_not_search(app):
    assert plan_for(app, "Summarize past diagnoses and treatments for Ramesh Kumar").intents == ["medical_history"]


def test_no_patient_asks_for_clarification(app):
    p = plan_for(app, "Book a nephrologist")
    assert p.clarification and not p.steps


def test_active_patient_and_pronoun_memory(app):
    assert plan_for(app, "Book a cardiologist next week", Session(active_patient_id="P1004")).steps[0].args == {"patient_id": "P1004"}
    assert plan_for(app, "Book a nephrologist for her", Session(last_patient_id="P1003")).steps[0].args == {"patient_id": "P1003"}


def test_export_and_lifestyle_intents(app):
    p = plan_for(app, "Export Ramesh Kumar's records as FHIR")
    assert p.intents == ["fhir_export"] and [s.tool for s in p.steps] == ["find_patient", "export_fhir_bundle"]
    q = plan_for(app, "What diet and exercise guidance is there for chronic kidney disease?")
    assert q.intents == ["medical_info"] and "diet and exercise lifestyle" in q.steps[0].args["query"]


def test_condition_asked_about_does_not_identify_patient(app):
    p = plan_for(app, "Book an appointment for my 70-year-old father and tell me latest treatment options for hypertension")
    assert "condition" not in p.steps[0].args          # "options for hypertension" is a question, not the father's condition


def test_parse_when():
    now = datetime(2026, 10, 5, 10, 0)
    assert parse_when("Friday afternoon", now)["date_from"].startswith("2026-10-09")
    assert parse_when("next week morning", now)["date_from"].startswith("2026-10-12")
    assert "date_from" not in parse_when("sometime", now)


def test_validate_plan_catches_cycles_unknown_tools_and_phi_in_search():
    bad = Plan(steps=[Step(id="s1", goal="x", tool="nope"),
                      Step(id="s2", goal="y", tool="find_slots", args={"specialty": "Nephrology"}, depends_on=["s3"]),
                      Step(id="s3", goal="z", tool="find_slots", args={"specialty": "Nephrology"}, depends_on=["s2"]),
                      Step(id="s4", goal="w", tool="search_medical_info", args={"query": "[PATIENT_1] kidney"})])
    errs = " ".join(validate_plan(bad, default_registry()))
    assert "unknown tool" in errs and "cyclic" in errs and "identifier" in errs
