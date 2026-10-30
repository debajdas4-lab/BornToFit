"""Headless UI tests with Streamlit's AppTest (runs the real app script, every tab)."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(__file__))


def _arrow_safe_dataframe(data=None, *a, **k):
    """This sandbox's pyarrow/NumPy pair is mismatched, so st.dataframe is stubbed, but the most common real failure
    (object columns mixing Python types, which Arrow cannot serialise) is still asserted."""
    import pandas as pd
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            kinds = {type(v) for v in df[c].dropna()}
            assert len(kinds) <= 1, f"mixed types in column {c!r}: {kinds}"


@pytest.fixture
def at(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "dataframe", _arrow_safe_dataframe)
    monkeypatch.setenv("BORNTOFIT_FIXED_NOW", "2026-10-05T10:00")
    monkeypatch.setenv("BORNTOFIT_LIVE_SEARCH", "false")
    monkeypatch.setenv("BORNTOFIT_LLM_PROVIDER", "none")
    a = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=90)
    a.run()
    return a


def agent_appts(at):
    return [a for a in at.session_state["app"].db.list_appointments(status="booked") if a["created_by"] == "agent"]


def click(at, prefix):
    next(b for b in at.button if b.label.startswith(prefix)).click().run()


def test_app_loads_all_tabs_without_exceptions(at):
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 7 and any("BornToFit" in m.value for m in at.markdown)


def test_capstone_scenario_books_and_renders_trace(at):
    click(at, "Capstone")
    assert not at.exception, [e.value for e in at.exception]
    answer = " ".join(m.value for m in at.chat_message[1].markdown)
    assert "Ramesh Kumar" in answer and "Appointment booked" in answer and len(agent_appts(at)) == 1
    assert any(e.label.startswith("🔭 Trace") for e in at.expander)


def test_follow_up_memory_pronoun(at):
    at.chat_input[0].set_value("Summarize Ramesh Kumar's medical history and alerts").run()
    at.chat_input[0].set_value("Book a nephrologist for him next week").run()
    assert not at.exception, [e.value for e in at.exception]
    assert [a["patient_id"] for a in agent_appts(at)] == ["P1001"]


def test_approval_and_decline_buttons(at):
    next(t for t in at.sidebar.toggle if t.label.startswith("Auto-book")).set_value(False).run()
    at.chat_input[0].set_value("Book a nephrologist for Ramesh Kumar next week").run()
    assert not agent_appts(at)
    click(at, "✅ Approve")
    assert not at.exception, [e.value for e in at.exception]
    assert len(agent_appts(at)) == 1
    at.chat_input[0].set_value("Book a cardiologist for Lakshmi Devi next week").run()
    click(at, "✖ Decline")
    assert len(agent_appts(at)) == 1 and "declined" in " ".join(m.value for m in at.chat_message[-1].markdown)


def test_rbac_scenarios_switch_identity_and_deny(at):
    click(at, "RBAC: family tries to add a record")
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["principal_key"] == "guardian_p1001"
    assert "Not permitted" in " ".join(m.value for m in at.chat_message[1].markdown)
    assert not [r for r in at.session_state["app"].db.get_records("P1001") if "BP 130/80" in r["details"]]
    click(at, "RBAC: family asks about another patient")
    assert "couldn't identify" in " ".join(m.value for m in at.chat_message[-1].markdown)


def test_fhir_export_scenario_and_interop_tab(at):
    click(at, "Export FHIR")
    assert not at.exception, [e.value for e in at.exception]
    assert "FHIR R4 bundle ready" in " ".join(m.value for m in at.chat_message[1].markdown)
    assert any(e.label == "Preview first resources" for e in at.expander)          # the FHIR panel rendered for the selected patient
    assert any("Structure checks" in m.label for m in at.metric)


def test_evaluation_tab_runs_the_full_suite(at):
    click(at, "▶ Run evaluation suite")
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["eval_report"]["overall"] >= 0.95 and at.session_state["eval_report"]["n"] == 26
