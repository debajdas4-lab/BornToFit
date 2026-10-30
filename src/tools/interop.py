"""FHIR R4 (4.0.1) export of a patient's record as a collection Bundle, plus a structural validator.

This is an interoperability *shape* export. It is NOT validated against the official HL7 validator and uses no
coded terminologies (SNOMED/LOINC/RxNorm): clinical text is carried in `text` fields."""
from __future__ import annotations

import base64
import uuid
from datetime import datetime

from .registry import ToolContext, ToolSpec

BASE = "https://borntofit.example/fhir"
_NS = uuid.UUID("6f1c2a54-0d8e-4b7a-9e61-3a5b7c1d2e90")
SYS_COND = "http://terminology.hl7.org/CodeSystem/condition-clinical"
SYS_ALLERGY = "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical"
SYS_ACT = "http://terminology.hl7.org/CodeSystem/v3-ActCode"

# Required (min cardinality 1) elements for the resource types we emit, from the R4 definitions.
REQUIRED = {
    "Patient": [], "Practitioner": [],
    "Condition": ["subject"], "MedicationStatement": ["status", "medication[x]", "subject"],
    "AllergyIntolerance": ["patient"], "Observation": ["status", "code"],
    "Appointment": ["status", "participant"], "Flag": ["status", "code", "subject"],
    "Encounter": ["status", "class"], "DocumentReference": ["status", "content"],
}
VALID_STATUS = {"Appointment": {"proposed", "pending", "booked", "arrived", "fulfilled", "cancelled", "noshow", "entered-in-error", "checked-in", "waitlist"},
                "Observation": {"registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error", "unknown"},
                "MedicationStatement": {"active", "completed", "entered-in-error", "intended", "stopped", "on-hold", "unknown", "not-taken"}}


def _iso(ts: str, tz: str) -> str:
    return datetime.fromisoformat(ts).strftime("%Y-%m-%dT%H:%M:%S") + tz


def _ref(rt: str, rid: str) -> dict:
    return {"reference": f"{rt}/{rid}"}


def build_bundle(db, patient_id: str, tz: str = "+05:30", now: datetime | None = None) -> dict:
    p = db.get_patient(patient_id)
    pid = p["patient_id"]
    res: list[dict] = [{"resourceType": "Patient", "id": pid,
                        "identifier": [{"system": "urn:borntofit:patient-id", "value": pid}],
                        "name": [{"text": p["name"]}], "gender": {"M": "male", "F": "female"}.get(p["sex"], "unknown"),
                        "extension": [{"url": "urn:borntofit:age-years", "valueInteger": int(p["age"])}]}]
    subj = _ref("Patient", pid)
    active = lambda sysu: {"coding": [{"system": sysu, "code": "active"}]}  # noqa: E731
    if p["allergies"] and p["allergies"] != "None known":
        res.append({"resourceType": "AllergyIntolerance", "id": f"{pid}-allergy-profile", "clinicalStatus": active(SYS_ALLERGY),
                    "code": {"text": p["allergies"]}, "patient": subj})
    for r in db.get_records(pid):
        rid, t, d = f"{pid}-r{r['record_id']}", r["rec_type"], r["rec_date"]
        if t == "diagnosis":
            res.append({"resourceType": "Condition", "id": rid, "clinicalStatus": active(SYS_COND), "code": {"text": r["title"]},
                        "subject": subj, "recordedDate": d, "note": [{"text": r["details"]}]})
        elif t == "medication":
            res.append({"resourceType": "MedicationStatement", "id": rid, "status": "active", "medicationCodeableConcept": {"text": r["details"]},
                        "subject": subj, "dateAsserted": d})
        elif t == "allergy":
            res.append({"resourceType": "AllergyIntolerance", "id": rid, "clinicalStatus": active(SYS_ALLERGY),
                        "code": {"text": r["details"]}, "patient": subj, "recordedDate": d})
        elif t == "lab":
            res.append({"resourceType": "Observation", "id": rid, "status": "final", "category": [{"text": "laboratory"}],
                        "code": {"text": r["title"]}, "subject": subj, "effectiveDateTime": d, "valueString": r["details"]})
        elif t == "alert":
            res.append({"resourceType": "Flag", "id": rid, "status": "active", "code": {"text": f"{r['title']}: {r['details']}"}, "subject": subj})
        elif t == "visit":
            res.append({"resourceType": "Encounter", "id": rid, "status": "finished", "class": {"system": SYS_ACT, "code": "AMB"},
                        "subject": subj, "period": {"start": d}, "reasonCode": [{"text": f"{r['title']}: {r['details']}"}]})
        else:
            res.append({"resourceType": "DocumentReference", "id": rid, "status": "current", "subject": subj, "date": None,
                        "description": r["title"], "content": [{"attachment": {"contentType": "text/plain",
                                                                              "data": base64.b64encode(r["details"].encode()).decode()}}]})
            res[-1].pop("date")
    docs: dict[str, dict] = {}
    for a in db.list_appointments(patient_id=pid):
        doc = db.get_doctor(a["doctor_id"])
        docs[doc["doctor_id"]] = {"resourceType": "Practitioner", "id": doc["doctor_id"], "name": [{"text": doc["name"]}]}
        res.append({"resourceType": "Appointment", "id": f"{pid}-a{a['appt_id']}", "status": "booked" if a["status"] == "booked" else "cancelled",
                    "serviceType": [{"text": a["specialty"]}], "reasonCode": [{"text": a["reason"] or ""}],
                    "start": _iso(a["start_ts"], tz), "end": _iso(a["end_ts"], tz),
                    "participant": [{"actor": subj, "status": "accepted"},
                                    {"actor": {**_ref("Practitioner", doc["doctor_id"]), "display": doc["name"]}, "status": "accepted"}]})
    res += list(docs.values())
    return {"resourceType": "Bundle", "type": "collection", "timestamp": (now or datetime.now()).strftime("%Y-%m-%dT%H:%M:%S") + tz,
            "entry": [{"fullUrl": f"{BASE}/{r['resourceType']}/{r['id']}", "resource": r} for r in res]}


def validate_bundle(bundle: dict) -> list[str]:
    """Structural checks only (not the official HL7 validator)."""
    errs: list[str] = []
    if bundle.get("resourceType") != "Bundle" or bundle.get("type") != "collection":
        errs.append("not a collection Bundle")
    ids = set()
    for i, e in enumerate(bundle.get("entry", [])):
        r = e.get("resource", {})
        rt, rid = r.get("resourceType"), r.get("id")
        if not rt or not rid:
            errs.append(f"entry {i}: missing resourceType/id")
            continue
        if (rt, rid) in ids:
            errs.append(f"duplicate {rt}/{rid}")
        ids.add((rt, rid))
        if e.get("fullUrl") != f"{BASE}/{rt}/{rid}":
            errs.append(f"{rt}/{rid}: fullUrl mismatch")
        for req in REQUIRED.get(rt, []):
            if req == "medication[x]":
                ok = "medicationCodeableConcept" in r or "medicationReference" in r
            else:
                ok = req in r and r[req] not in (None, "", [], {})
            if not ok:
                errs.append(f"{rt}/{rid}: missing required element '{req}'")
        if rt in VALID_STATUS and r.get("status") not in VALID_STATUS[rt]:
            errs.append(f"{rt}/{rid}: invalid status {r.get('status')!r}")
    for e in bundle.get("entry", []):
        stack = [e.get("resource", {})]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                if set(x) == {"reference"} or ("reference" in x and isinstance(x["reference"], str)):
                    rt, _, rid = x["reference"].partition("/")
                    if (rt, rid) not in ids:
                        errs.append(f"dangling reference {x['reference']}")
                stack += list(x.values())
            elif isinstance(x, list):
                stack += x
    return errs


def export_fhir_bundle(ctx: ToolContext, patient_id: str) -> dict:
    if not ctx.db.get_patient(patient_id):
        return {"ok": False, "message": f"Unknown patient {patient_id}."}
    b = build_bundle(ctx.db, patient_id, ctx.settings.tz_offset, ctx.now())
    errs = validate_bundle(b)
    counts: dict[str, int] = {}
    for e in b["entry"]:
        counts[e["resource"]["resourceType"]] = counts.get(e["resource"]["resourceType"], 0) + 1
    ctx.memory.trace("read", f"FHIR export: {len(b['entry'])} resources, {len(errs)} structural issue(s)", patient_id, ctx.run_id)
    return {"ok": not errs, "bundle": b, "counts": counts, "validation_errors": errs, "fhir_version": "4.0.1",
            "message": "FHIR R4 bundle ready." if not errs else "Bundle built with structural issues."}


SPECS = [ToolSpec("export_fhir_bundle", "Export a patient's record as a FHIR R4 collection Bundle (Patient, Condition, MedicationStatement, AllergyIntolerance, Observation, Flag, Encounter, Appointment).",
                  {"patient_id": {"type": "string"}}, export_fhir_bundle, required=["patient_id"])]
