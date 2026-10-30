"""Final answer. Smart Mode: LLM wording over SANITISED evidence, then verified (entities, times, citations);
anything that fails verification is replaced by the deterministic answer. Live Mode: deterministic template."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from ..memory.vector_store import tokenize
from ..observability.tracing import RunTrace, traced_chat
from ..security.guardrails import CLINICAL_ADVICE_NOTE, MEDICAL_DISCLAIMER
from ..security.injection import neutralize_obj
from .llm_factory import LLMClient, LLMError
from .models import Plan
from .prompts import COMPOSER_SYSTEM


@dataclass
class ComposeResult:
    answer: str
    sources: list[dict]
    used_llm: bool
    faithfulness: float = 1.0
    quarantined: int = 0
    notes: list[str] = field(default_factory=list)


def when_text(start: str, end: str | None = None) -> str:
    txt = datetime.fromisoformat(start).strftime("%a %d %b %Y, %H:%M")
    return txt + (f"-{datetime.fromisoformat(end).strftime('%H:%M')}" if end else "")


def build_evidence(plan: Plan) -> tuple[dict, list[dict], int]:
    ev: dict = {"failures": []}
    sources: list[dict] = []
    quarantined = 0
    for s in plan.steps:
        r = s.result if isinstance(s.result, dict) else {}
        if s.status in ("failed", "skipped"):
            ev["failures"].append({"step": s.id, "tool": s.tool, "reason": s.error or "skipped", "candidates": r.get("candidates") if r else None})
        if s.tool == "find_patient" and r.get("ok"):
            p = r["patient"]
            ev["patient"] = {"patient_id": p["patient_id"], "age": p["age"], "sex": p["sex"], "primary_condition": p["primary_condition"],
                             "matched_on": r.get("matched_on")}
        elif s.tool == "get_patient_history" and r.get("ok"):
            ev["history"] = r["summary"]
            ev["brief"] = r.get("brief", "")
            ev["upcoming_appointments"] = [{"doctor": a["doctor_name"], "specialty": a["specialty"], "when": when_text(a["start_ts"], a["end_ts"])}
                                           for a in r["upcoming_appointments"]]
            ev["recalled_memory"] = [h["text"][:160] for h in r.get("memory_hits", [])[:3]]
        elif s.tool == "find_slots" and r.get("ok"):
            ev["slot_search"] = {"specialty": r["specialty"], "window_met": r["window_met"], "note": r.get("note"),
                                 "options": [{"doctor": x["doctor_name"], "when": when_text(x["start_ts"], x["end_ts"])} for x in r["slots"][:3]]}
        elif s.tool == "book_appointment" and r:
            if r.get("status") == "booked":
                a = r["appointment"]
                ev["booking"] = {"status": "booked", "appointment_id": a["appt_id"], "doctor": a["doctor_name"], "specialty": a["specialty"],
                                 "when": when_text(a["start_ts"], a["end_ts"]), "reason": a["reason"], "retries": r.get("retries", 0)}
            elif r.get("status") == "pending_confirmation":
                x = r["proposal"]
                ev["booking"] = {"status": "pending_confirmation", "step": s.id, "doctor": x["doctor_name"], "specialty": x["specialty"],
                                 "when": when_text(x["start_ts"], x["end_ts"]), "location": x.get("location"), "why_pending": r.get("message")}
        elif s.tool == "cancel_appointment" and r.get("ok"):
            a = r["appointment"]
            ev["cancellation"] = {"doctor": a["doctor_name"], "specialty": a["specialty"], "when": when_text(a["start_ts"], a["end_ts"])}
        elif s.tool == "add_patient_record" and r.get("ok"):
            rec = r["record"]
            ev["record_added"] = {"record_id": rec["record_id"], "type": rec["rec_type"], "title": rec["title"], "date": rec["rec_date"]}
        elif s.tool == "export_fhir_bundle" and r.get("bundle"):
            ev["fhir"] = {"resources": len(r["bundle"]["entry"]), "counts": r["counts"], "valid": r["ok"], "issues": r["validation_errors"][:3]}
        elif s.tool == "search_medical_info" and r.get("ok"):
            for d in r["results"]:
                sources.append({"id": f"S{len(sources) + 1}", "title": d["title"], "source": d["source"], "url": d.get("url"),
                                "freshness": d["freshness"], "retrieved_at": d.get("retrieved_at"), "snippet": d["snippet"],
                                "relevance": d.get("relevance")})
            ev["medical_info_mode"] = r["mode"]
            ev["crag"] = {"attempts": len(r["crag"]["attempts"]), "rewritten": r["crag"]["rewritten"]}
            quarantined += len(r.get("quarantined", []))
        elif s.tool == "search_medical_info" and r.get("quarantined"):
            quarantined += len(r["quarantined"])
    if sources:
        ev["sources"] = sources
    if quarantined:
        ev["quarantined_items"] = quarantined
    return ev, sources, quarantined


def _bullets(items: list[str], n: int = 3) -> str:
    return "\n".join(f"- {i}" for i in items[:n]) if items else "- None on file"


def compose_offline(plan: Plan, ev: dict, name: str | None, clinical_advice: bool) -> str:
    out: list[str] = []
    if clinical_advice:
        out.append(f"> {CLINICAL_ADVICE_NOTE}")
    if "patient" in ev:
        p = ev["patient"]
        out.append(f"**Patient identified:** {name or p['patient_id']} ({p['patient_id']}), {p['age']} y, {p['sex']}; primary condition: "
                   f"{p['primary_condition']}. Matched on: {', '.join(p['matched_on'] or [])}.")
    if "history" in ev:
        h = ev["history"]
        out += ["**Medical history summary**", "*Diagnoses*\n" + _bullets(h["diagnoses"]), "*Current medicines*\n" + _bullets(h["medications"], 2),
                "*Latest labs*\n" + _bullets(h["labs"], 2), "*Alerts*\n" + _bullets(h["alerts"], 4)]
        if ev.get("upcoming_appointments"):
            out.append("*Upcoming appointments*\n" + _bullets([f"{a['specialty']} with {a['doctor']}, {a['when']}" for a in ev["upcoming_appointments"]]))
    b = ev.get("booking")
    if b and b["status"] == "booked":
        out.append(f"**Appointment booked:** {b['doctor']} ({b['specialty']}) on **{b['when']}**, ref #{b['appointment_id']}."
                   + (" The first slot was taken, so I moved to the next open one." if b.get("retries") else ""))
        if "history" in ev and ev["history"]["alerts"]:
            out.append("*Noted for the clinic:* " + "; ".join(a.split(". ", 1)[-1].rstrip(".") for a in ev["history"]["alerts"][:2]) + ".")
    elif b:
        out.append(f"**Appointment proposed, waiting for your approval:** {b['doctor']} ({b['specialty']}) on **{b['when']}**. {b['why_pending']}")
    ss = ev.get("slot_search")
    if ss and ss.get("note") and not (b and b["status"] == "booked" and ss["window_met"]):
        out.append(f"*Scheduling note:* {ss['note']}")
    if "cancellation" in ev:
        c = ev["cancellation"]
        out.append(f"**Appointment cancelled:** {c['specialty']} with {c['doctor']}, {c['when']}. The slot is open again.")
    if "record_added" in ev:
        r = ev["record_added"]
        out.append(f"**Record added:** {r['type']} \"{r['title']}\" (#{r['record_id']}, {r['date']}). Memory index and patient brief updated.")
    if "fhir" in ev:
        f = ev["fhir"]
        out.append(f"**FHIR R4 bundle ready:** {f['resources']} resources ({', '.join(f'{k} x{v}' for k, v in f['counts'].items())}). "
                   + ("Structure checks passed." if f["valid"] else f"Structure issues: {'; '.join(f['issues'])}.") + " Download it from the Interop tab.")
    if ev.get("sources"):
        mode = ev.get("medical_info_mode")
        out.append("**Trusted-source information (live)**" if mode == "live"
                   else "**Medical information (offline reference corpus, not live; verify on MedlinePlus/WHO)**")
        for i, s in enumerate(ev["sources"][:3]):
            lim = 1100 if i == 0 else 320
            out.append(f"- **[{s['id']}] {s['title']}**: {s['snippet'][:lim]}{'...' if len(s['snippet']) > lim else ''}")
    if ev.get("quarantined_items"):
        out.append(f"*Security:* {ev['quarantined_items']} retrieved item(s) were blocked because they contained instruction-like text (possible prompt injection).")
    for f in ev["failures"]:
        reason = f["reason"]
        if f["tool"] == "cancel_appointment" and f.get("candidates"):
            opts = "; ".join(f"#{a['appt_id']} {a['specialty']} with {a['doctor_name']}, {when_text(a['start_ts'], a['end_ts'])}" for a in f["candidates"])
            out.append(f"**Which appointment?** I found several upcoming ones: {opts}. Reply with the reference number, e.g. \"cancel #{f['candidates'][0]['appt_id']}\".")
        elif f["tool"] == "find_patient" and f.get("candidates"):
            names = "; ".join(f"{c['name']} ({c['patient_id']}, {c['age']} y, {c['primary_condition']})" for c in f["candidates"])
            out.append(f"**I need one more detail.** Several patients match: {names}. Tell me the name or ID and I'll continue.")
        elif reason.startswith("permission denied"):
            out.append(f"**Not permitted:** {reason.removeprefix('permission denied: ')}. Nothing was changed.")
        elif reason == "declined by the user":
            out.append("**Not booked:** you declined the proposed slot. Nothing was changed.")
        elif f["tool"] == "find_patient":
            out.append(f"**I couldn't identify the patient:** {reason} Please give a name, patient ID, or age and condition that you have access to.")
        elif reason != "an earlier step did not succeed":
            out.append(f"**Could not complete `{f['tool']}`:** {reason}")
    if len(out) == 0 or (not any(k in ev for k in ("patient", "history", "booking", "cancellation", "record_added", "sources", "fhir")) and not ev["failures"]):
        out.append("I wasn't able to complete that request.")
    out.append(f"_{MEDICAL_DISCLAIMER}_")
    return "\n\n".join(out)


def grounding_ok(answer: str, evidence_json: str) -> bool:
    """Doctors, dates and clock times in an LLM answer must exist in the evidence."""
    ev_low = evidence_json.lower()
    if any(dr.lower() not in ev_low for dr in re.findall(r"Dr\. [A-Z][a-z]+ [A-Z][a-z]+", answer)):
        return False
    norm = lambda t: t.lstrip("0")  # noqa: E731
    ev_times = {norm(t) for t in re.findall(r"\b\d{1,2}:\d{2}\b", evidence_json)}
    return all(norm(t) in ev_times for t in re.findall(r"\b\d{1,2}:\d{2}\b", answer)) and \
        all(d in evidence_json for d in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", answer))


def verify_citations(answer: str, sources: list[dict]) -> tuple[float, list[str]]:
    """Claim-level check: every sentence citing [S#] must be lexically supported by the cited source(s)."""
    src = {s["id"]: set(tokenize(s["title"] + " " + s["snippet"])) for s in sources}
    checked = supported = 0
    bad: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", answer):
        tags = re.findall(r"\[S(\d+)\]", sent)
        if not tags:
            continue
        checked += 1
        words = [t for t in tokenize(re.sub(r"\[S\d+\]", "", sent)) if not t.isdigit() and len(t) > 3]
        cited = set().union(*[src.get(f"S{t}", set()) for t in tags])
        if not words or sum(w in cited for w in words) / len(words) >= 0.5:
            supported += 1
        else:
            bad.append(sent.strip()[:160])
    if sources and checked == 0:
        return 0.0, ["no citations in a sourced answer"]
    return (supported / checked if checked else 1.0), bad


def compose(plan: Plan, name: str | None, clinical_advice: bool, llm: LLMClient | None, trace: RunTrace | None = None,
            parent: str | None = None) -> ComposeResult:
    ev, sources, quarantined = build_evidence(plan)
    fallback = compose_offline(plan, ev, name, clinical_advice)
    base = ComposeResult(fallback, sources, False, 1.0, quarantined)
    if llm is None or not plan.steps:
        return base
    try:
        safe_ev = neutralize_obj(ev)                         # EHR free text / web snippets are DATA: strip instruction-like text
        ev_json = json.dumps(safe_ev, default=str)
        raw = traced_chat(trace, parent, llm, "compose", COMPOSER_SYSTEM.format(disclaimer=MEDICAL_DISCLAIMER),
                          f"Evidence JSON:\n{ev_json}", temperature=0.1, max_tokens=700)
        text = raw.replace("[PATIENT]", name or "the patient").strip()
        if not grounding_ok(text, ev_json) or MEDICAL_DISCLAIMER not in text:
            base.notes.append("LLM answer failed grounding check; used deterministic answer.")
            return base
        faith, bad = verify_citations(text, sources)
        if faith < 0.6:
            base.notes.append(f"LLM answer failed citation check ({faith:.2f}); used deterministic answer.")
            base.faithfulness = faith
            return base
        return ComposeResult((f"> {CLINICAL_ADVICE_NOTE}\n\n" if clinical_advice else "") + text, sources, True, faith, quarantined)
    except LLMError as e:
        base.notes.append(f"LLM composer failed ({str(e)[:80]}); used deterministic answer.")
        return base
