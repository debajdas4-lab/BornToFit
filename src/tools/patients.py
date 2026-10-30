"""EHR tools: identify patient, retrieve/summarise history, add/update records."""
from __future__ import annotations

from ..data.db import REC_TYPES
from .medical_terms import canonical_condition, detect_conditions
from .registry import ToolContext, ToolSpec


def _public(p: dict) -> dict:
    return {k: p[k] for k in ("patient_id", "name", "age", "sex", "primary_condition", "allergies", "blood_group", "guardian")}


def _has_condition(ctx: ToolContext, p: dict, cond: str) -> bool:
    if canonical_condition(p["primary_condition"]) == cond:
        return True
    return any(r["rec_type"] == "diagnosis" and cond in detect_conditions(f"{r['title']} {r['details']}")
               for r in ctx.db.get_records(p["patient_id"]))


def find_patient(ctx: ToolContext, name: str | None = None, age: int | None = None, sex: str | None = None,
                 condition: str | None = None, patient_id: str | None = None) -> dict:
    """Candidates are filtered to what the caller may see, so a family member's search can never reveal
    (or be confused by) other patients."""
    if patient_id:
        p = ctx.db.get_patient(patient_id)
        if not p or not ctx.can_see(p["patient_id"]):
            return {"ok": False, "status": "not_found", "message": f"No patient with ID {patient_id} is available to you."}
        ctx.memory.trace("read", "patient resolved by ID", p["patient_id"], ctx.run_id)
        return {"ok": True, "status": "found", "patient": _public(p), "matched_on": ["patient_id"]}
    if not any([name, age, sex, condition]):
        return {"ok": False, "status": "no_criteria", "message": "No patient details were given."}
    cands = [p for p in ctx.db.search_patients(name=name, age=age, sex=sex) if ctx.can_see(p["patient_id"])]
    matched = [k for k, v in (("name", name), ("age", age), ("sex", sex)) if v]
    if condition:
        cond = canonical_condition(condition) or condition
        cands = [p for p in cands if _has_condition(ctx, p, cond)]
        matched.append("condition")
    if len(cands) == 1:
        p = cands[0]
        ctx.memory.trace("read", f"patient resolved by {', '.join(matched)}", p["patient_id"], ctx.run_id)
        return {"ok": True, "status": "found", "patient": _public(p), "matched_on": matched}
    if not cands:
        return {"ok": False, "status": "not_found", "message": "No matching patient found.", "criteria": matched}
    return {"ok": False, "status": "ambiguous", "needs_clarification": True, "message": f"{len(cands)} patients match.",
            "criteria": matched, "candidates": [{"patient_id": c["patient_id"], "name": c["name"], "age": c["age"], "sex": c["sex"],
                                                 "primary_condition": c["primary_condition"]} for c in cands[:6]]}


def _fmt(r: dict) -> str:
    return f"{r['rec_date']}: {r['title']}. {r['details']}"


def get_patient_history(ctx: ToolContext, patient_id: str, focus: str | None = None) -> dict:
    p = ctx.db.get_patient(patient_id)
    if not p:
        return {"ok": False, "message": f"Unknown patient {patient_id}."}
    recs = ctx.db.get_records(patient_id)
    by = lambda t: [r for r in recs if r["rec_type"] == t]  # noqa: E731
    alerts = [_fmt(r) for r in by("alert")]
    if p["allergies"] and p["allergies"] != "None known":
        alerts.append(f"Allergies: {p['allergies']}")
    now_s = ctx.now().strftime("%Y-%m-%dT%H:%M")
    appts = [a for a in ctx.db.list_appointments(patient_id=patient_id, status="booked") if a["start_ts"] >= now_s]
    summary = {"diagnoses": [_fmt(r) for r in by("diagnosis")][:6], "medications": [_fmt(r) for r in by("medication")][:3],
               "labs": [_fmt(r) for r in by("lab")][:3], "alerts": alerts[:6], "visits": [_fmt(r) for r in by("visit")][:3],
               "notes": [_fmt(r) for r in by("note")][:3]}
    hits = ctx.memory.recall(patient_id, focus or f"{p['primary_condition']} history medications alerts", k=4, run_id=ctx.run_id)
    ctx.memory.trace("read", f"history: {len(recs)} records, {len(appts)} upcoming appts", patient_id, ctx.run_id)
    return {"ok": True, "patient_id": patient_id, "patient": _public(p), "summary": summary, "brief": ctx.memory.brief(patient_id),
            "upcoming_appointments": appts, "n_records": len(recs),
            "memory_hits": [{"id": h.id, "text": h.text, "score": h.score, "matched_by": h.matched_by} for h in hits]}


def add_patient_record(ctx: ToolContext, patient_id: str, rec_type: str, title: str = "", details: str = "",
                       rec_date: str | None = None, author: str = "attendant") -> dict:
    rec_type = (rec_type or "note").lower()
    if rec_type not in REC_TYPES:
        return {"ok": False, "message": f"rec_type must be one of {', '.join(REC_TYPES)}."}
    if not ctx.db.get_patient(patient_id):
        return {"ok": False, "message": f"Unknown patient {patient_id}."}
    if not (details or "").strip():
        return {"ok": False, "message": "Record details are empty."}
    role = ctx.principal.role if ctx.principal else author
    rec = ctx.db.add_record(patient_id, rec_type, (title or rec_type.title())[:120], details.strip(), rec_date=rec_date, author=role)
    n = ctx.memory.index_patient(patient_id)
    ctx.memory.trace("write", f"added {rec_type} record #{rec['record_id']}; re-indexed {n} chunks + brief", patient_id, ctx.run_id)
    return {"ok": True, "record": rec, "reindexed_chunks": n}


def update_patient_record(ctx: ToolContext, record_id: int, title: str | None = None, details: str | None = None,
                          rec_type: str | None = None) -> dict:
    old = ctx.db.get_record(int(record_id))
    if not old:
        return {"ok": False, "message": f"No record #{record_id}."}
    rec = ctx.db.update_record(int(record_id), title=title, details=details, rec_type=rec_type)
    n = ctx.memory.index_patient(old["patient_id"])
    ctx.memory.trace("write", f"updated record #{record_id}; re-indexed {n} chunks + brief", old["patient_id"], ctx.run_id)
    return {"ok": True, "record": rec, "reindexed_chunks": n}


SPECS = [
    ToolSpec("find_patient", "Identify a patient from name, age, sex, condition or ID. Returns found / ambiguous / not_found. Only patients the caller may access are considered.",
             {"name": {"type": "string"}, "age": {"type": "integer"}, "sex": {"type": "string", "enum": ["M", "F"]},
              "condition": {"type": "string"}, "patient_id": {"type": "string"}}, find_patient),
    ToolSpec("get_patient_history", "Retrieve and structure a patient's diagnoses, medications, labs, alerts and upcoming appointments, plus the consolidated patient brief and relevant long-term memory.",
             {"patient_id": {"type": "string"}, "focus": {"type": "string"}}, get_patient_history, required=["patient_id"]),
    ToolSpec("add_patient_record", "Add a structured or free-text record (diagnosis, medication, lab, allergy, alert, visit, note) to the EHR. Clinical staff only.",
             {"patient_id": {"type": "string"}, "rec_type": {"type": "string", "enum": list(REC_TYPES)}, "title": {"type": "string"},
              "details": {"type": "string"}, "rec_date": {"type": "string"}}, add_patient_record, side_effect=True,
             required=["patient_id", "rec_type", "details"]),
    ToolSpec("update_patient_record", "Update an existing EHR record. Clinical staff only.",
             {"record_id": {"type": "integer"}, "title": {"type": "string"}, "details": {"type": "string"}, "rec_type": {"type": "string"}},
             update_patient_record, side_effect=True, required=["record_id"]),
]
