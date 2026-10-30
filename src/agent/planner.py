"""Planners: deterministic RulePlanner (offline) and LLMPlanner (schema-validated, falls back to rules)."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from ..observability.tracing import RunTrace, traced_chat
from ..security.guardrails import Redactor
from ..tools.medical_terms import detect_conditions, detect_patient_conditions, detect_specialty, specialty_for_condition
from .llm_factory import LLMClient, LLMError, extract_json
from .models import Plan, Session, Step
from .prompts import PLAN_SCHEMA, planner_system

WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
RELATION_SEX = {"father": "M", "dad": "M", "papa": "M", "husband": "M", "son": "M", "brother": "M", "grandfather": "M",
                "mother": "F", "mom": "F", "mum": "F", "wife": "F", "daughter": "F", "sister": "F", "grandmother": "F"}

RE_UNKNOWN_NAME = re.compile(r"\b(?:for|patient|named)\s+((?:Dr\.\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")
RE_POSSESSIVE_NAME = re.compile(r"\b((?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+)'s\b")
RE_APPT_REF = re.compile(r"(?:#|\bref(?:erence)?\s*#?\s*|\bappointment\s+(?:no\.?\s*|number\s*)?)(\d{1,6})\b", re.I)
RE_JUST_BOOKED = re.compile(r"\b(just booked|that appointment|the one (?:i|you) (?:just )?booked|appointment (?:you|i) just)\b", re.I)
RE_AGE = re.compile(r"(\d{1,3})\s*[- ]?\s*(?:years?|yrs?)\s*[- ]?\s*old|\baged?\s+(\d{1,3})\b|\b(\d{2,3})\s*y/?o\b", re.I)
RE_PID = re.compile(r"\bP\d{4}\b", re.I)
RE_PRONOUN = re.compile(r"\b(him|her|he|she|his|hers|them|their)\b", re.I)
RE_BOOK_VERB = re.compile(r"\b(book|schedule|arrange|reserve|set up|fix)\b", re.I)
RE_BOOK_NEED = re.compile(r"\b(appointment|slot|consult\w*|visit|doctor|specialist)\b", re.I)
RE_WANT_SEE = re.compile(r"\b(need|want|like|get)\b.*\b(appointment|to see|to consult)\b|\bsee (a|an|the) \w*(ologist|doctor|specialist|physician|gp)\b", re.I)
RE_CANCEL = re.compile(r"\bcancel\b", re.I)
RE_HISTORY = re.compile(r"\b(history|histories|records?|medical summary|diagnoses|alerts?|medications?|allergies|past treatments?|previous visits?)\b", re.I)
RE_INFO_STRONG = re.compile(r"\b(latest|newest|options|methods|symptoms?|causes?|prevention|guidelines?|treatments?|therap(?:y|ies)|diet|exercise|lifestyle|nutrition)\b", re.I)
RE_INFO_WEAK = re.compile(r"\b(what is|tell me about|information|manage(?:ment)?|how is .{0,30} treated)\b", re.I)
RE_PAST_TREAT = re.compile(r"\b(?:past|previous|earlier|prior)\s+(?:diagnoses|treatments?)(?:\s*(?:,|and|&)\s*(?:diagnoses|treatments?))*", re.I)
RE_ADD = re.compile(r"^\s*(?:please\s+)?(add|record|log|enter|note down|update)\b", re.I)
RE_ADD_OBJ = re.compile(r"\b(note|record|history|diagnos\w*|medication|lab|result|allerg\w*|alert|visit)\b", re.I)
RE_EXPORT = re.compile(r"\bfhir\b|\b(export|download)\b.{0,30}\b(records?|history|bundle|data)\b", re.I)


def parse_when(text: str, now: datetime) -> dict:
    """Interpret 'tomorrow', 'next week', 'Friday afternoon' ... into an ISO window."""
    t, today = text.lower(), now.date()
    part = "morning" if "morning" in t else "afternoon" if "afternoon" in t else None
    lo = hi = None
    label = "earliest available"
    if "tomorrow" in t:
        lo, hi, label = today + timedelta(days=1), today + timedelta(days=2), "tomorrow"
    elif re.search(r"\btoday\b", t):
        lo, hi, label = today, today + timedelta(days=1), "today"
    elif "next week" in t:
        mon = today + timedelta(days=7 - today.weekday())
        lo, hi, label = mon, mon + timedelta(days=7), "next week"
    elif "this week" in t:
        lo, hi, label = today, today + timedelta(days=7 - today.weekday()), "this week"
    else:
        for name, wd in WEEKDAYS.items():
            if re.search(rf"\b{name}\b", t):
                lo = today + timedelta(days=(wd - today.weekday()) % 7 or 7)
                hi, label = lo + timedelta(days=1), name.title()
                break
    out = {"part_of_day": part, "label": label + (f" ({part})" if part else "")}
    if lo:
        out["date_from"], out["date_to"] = lo.isoformat() + "T00:00", hi.isoformat() + "T00:00"
    return out


def extract_entities(text: str, patients: list[dict]) -> dict:
    ent: dict = {}
    if m := RE_PID.search(text):
        ent["patient_id"] = m.group(0).upper()
    low = text.lower()
    for p in patients:
        if p["name"].lower() in low:
            ent["name"] = p["name"]
            break
    if "name" not in ent:
        firsts: dict[str, list[dict]] = {}
        for p in patients:
            firsts.setdefault(p["name"].split()[0].lower(), []).append(p)
        for f, plist in firsts.items():
            if len(plist) == 1 and re.search(rf"\b{f}\b", low):
                ent["name"] = plist[0]["name"]
                break
    m = RE_UNKNOWN_NAME.search(text)
    cand = m.group(1) if m else None
    if cand is None and (pm := RE_POSSESSIVE_NAME.search(text)):
        cand = " ".join(pm.group(1).split()[-2:])          # "Summarize Zed Unknown's history" -> "Zed Unknown"
    if "name" not in ent and "patient_id" not in ent and cand:
        if not cand.lower().startswith("dr") and not detect_conditions(cand) and not detect_specialty(cand) \
                and not any(w in cand.lower() for w in WEEKDAYS):
            ent["name"] = cand          # an unknown name: let find_patient answer "not found"
    if m := RE_AGE.search(text):
        ent["age"] = int(next(g for g in m.groups() if g))
    for rel, sex in RELATION_SEX.items():
        if re.search(rf"\b{rel}\b", low):
            ent["sex"], ent["relation"] = sex, rel
            break
    if conds := detect_conditions(text):
        ent["condition"] = conds[0]
    if pconds := detect_patient_conditions(text):
        ent["patient_condition"] = pconds[0]
    if sp := detect_specialty(text):
        ent["specialty"] = sp
    return ent


class RulePlanner:
    name = "rules"

    def __init__(self, db, now_fn):
        self.db, self.now = db, now_fn

    def plan(self, text: str, session: Session) -> Plan:
        ent = extract_entities(text, self.db.list_patients())
        low = text.lower()
        info_text = RE_PAST_TREAT.sub(" ", text)            # "past treatments" is history, not a literature search
        explicit = any(k in ent for k in ("patient_id", "name", "age", "sex"))
        patient_ref = explicit or bool(session.active_patient_id) or bool(session.last_patient_id and RE_PRONOUN.search(text))

        wants_cancel = bool(RE_CANCEL.search(text))
        wants_book = (not wants_cancel) and bool((RE_BOOK_VERB.search(text) and (RE_BOOK_NEED.search(text) or ent.get("specialty")))
                                                 or RE_WANT_SEE.search(text) or (ent.get("specialty") and re.search(r"\bappointment\b", low)))
        wants_add = bool(RE_ADD.search(text) and RE_ADD_OBJ.search(text))
        wants_export = bool(RE_EXPORT.search(text)) and not wants_add
        wants_hist = bool(RE_HISTORY.search(text)) and not wants_add and not wants_export
        wants_info = (not wants_add) and (not wants_export) and (
            (bool(RE_INFO_STRONG.search(info_text)) and bool(ent.get("condition") or patient_ref))
            or (bool(RE_INFO_WEAK.search(info_text)) and bool(ent.get("condition"))))
        intents = [n for f, n in ((wants_book, "book_appointment"), (wants_cancel, "cancel_appointment"), (wants_hist, "medical_history"),
                                  (wants_info, "medical_info"), (wants_add, "add_record"), (wants_export, "fhir_export")) if f]
        plan = Plan(intents=intents, planner=self.name)
        if not intents:
            plan.clarification = ("I can book or cancel appointments, summarise and update patient records, export a FHIR record bundle, and look up "
                                  "trusted disease information. For example: \"Book a nephrologist for Ramesh Kumar next week and summarise "
                                  "latest treatment options.\"")
            return plan
        needs_patient = bool({"book_appointment", "cancel_appointment", "medical_history", "add_record", "fhir_export"} & set(intents))

        steps: list[Step] = []
        nid = lambda: f"s{len(steps) + 1}"  # noqa: E731

        # 1. who is the patient?
        pid_step = None
        if needs_patient or (wants_info and not ent.get("condition")):
            find_args: dict = {}
            if explicit:
                find_args = {k: ent[k] for k in ("patient_id", "name", "age", "sex") if k in ent}
                if ent.get("patient_condition"):
                    find_args["condition"] = ent["patient_condition"]
            elif session.active_patient_id:
                find_args = {"patient_id": session.active_patient_id}
            elif session.last_patient_id and RE_PRONOUN.search(text):
                find_args = {"patient_id": session.last_patient_id}
            elif ent.get("patient_condition"):
                find_args = {"condition": ent["patient_condition"]}
            if find_args:
                pid_step = nid()
                steps.append(Step(id=pid_step, goal="Identify the patient and context", tool="find_patient", args=find_args))
            elif needs_patient:
                plan.clarification = ("Which patient is this for? Please give a name, patient ID, or age and condition "
                                      "(for example \"my 70-year-old father with chronic kidney disease\").")
                return plan
        pid_ref = f"${pid_step}.patient.patient_id" if pid_step else None

        # 2. history (context + alerts)
        hist_step = None
        if pid_step and ({"medical_history", "book_appointment"} & set(intents)):
            hist_step = nid()
            steps.append(Step(id=hist_step, goal="Retrieve the patient's medical history", tool="get_patient_history",
                              args={"patient_id": pid_ref}, depends_on=[pid_step]))

        # 3. scheduling (independent of the patient lookup -> parallel)
        if wants_book:
            spec = ent.get("specialty") or specialty_for_condition(ent.get("condition")) or "General Medicine"
            if not ent.get("specialty") and not specialty_for_condition(ent.get("condition")):
                plan.notes.append("No specialty named; defaulting to General Medicine.")
            when = parse_when(text, self.now())
            slot_args = {"specialty": spec, **{k: when[k] for k in ("date_from", "date_to", "part_of_day") if when.get(k)}}
            slot_step = nid()
            steps.append(Step(id=slot_step, goal=f"Find open {spec} slots ({when['label']})", tool="find_slots", args=slot_args))
            reason = f"Consultation: {ent['condition']}" if ent.get("condition") else f"{spec} consultation"
            steps.append(Step(id=nid(), goal="Book the best available slot", tool="book_appointment",
                              args={"patient_id": pid_ref, "slot_id": f"${slot_step}.best_slot.slot_id", "reason": reason, "specialty": spec,
                                    "window_met": f"${slot_step}.window_met"},
                              depends_on=[d for d in (pid_step, hist_step, slot_step) if d]))
        if wants_cancel and pid_step:
            args = {"patient_id": pid_ref}
            if ref := RE_APPT_REF.search(text):
                args["appt_id"] = int(ref.group(1))
            elif RE_JUST_BOOKED.search(text) and session.last_appointment_id:
                args["appt_id"] = session.last_appointment_id
            elif ent.get("specialty"):
                args["specialty"] = ent["specialty"]
            steps.append(Step(id=nid(), goal="Cancel the upcoming appointment", tool="cancel_appointment", args=args, depends_on=[pid_step]))

        # 4. disease information (independent -> parallel)
        if wants_info:
            facet = ("diet and exercise lifestyle" if re.search(r"\b(diet|exercise|lifestyle|nutrition)\b", low)
                     else "symptoms" if re.search(r"symptom", low) else "causes" if re.search(r"\bcause", low)
                     else "prevention" if re.search(r"prevent", low) else "treatment options")
            if ent.get("condition"):
                steps.append(Step(id=nid(), goal=f"Search trusted sources: {ent['condition']} ({facet})", tool="search_medical_info",
                                  args={"query": f"{ent['condition']} {facet}"}))
            elif pid_step:
                steps.append(Step(id=nid(), goal=f"Search trusted sources for the patient's condition ({facet})", tool="search_medical_info",
                                  args={"query": facet, "condition": f"${pid_step}.patient.primary_condition"}, depends_on=[pid_step]))

        # 5. EHR write / export
        if wants_add and pid_step:
            body = text.split(":", 1)[1].strip() if ":" in text else RE_ADD.sub("", text, count=1).strip()
            rtype = ("allergy" if re.search(r"allerg", low) else "medication" if re.search(r"medicat|prescri|tablet", low)
                     else "lab" if re.search(r"\b(lab|result|hba1c|egfr|tsh|bp reading)\b", low)
                     else "diagnosis" if re.search(r"diagnos", low) else "alert" if re.search(r"\balert\b", low)
                     else "visit" if re.search(r"\bvisit\b", low) else "note")
            steps.append(Step(id=nid(), goal=f"Add a {rtype} record to the EHR", tool="add_patient_record",
                              args={"patient_id": pid_ref, "rec_type": rtype, "title": body[:60] or rtype.title(), "details": body or text},
                              depends_on=[pid_step]))
        if wants_export and pid_step:
            steps.append(Step(id=nid(), goal="Export the patient's record as a FHIR R4 bundle", tool="export_fhir_bundle",
                              args={"patient_id": pid_ref}, depends_on=[pid_step]))
        plan.steps = steps
        return plan


def validate_plan(plan: Plan, registry) -> list[str]:
    errs, ids = [], {s.id for s in plan.steps}
    if len(ids) != len(plan.steps):
        errs.append("duplicate step ids")
    for s in plan.steps:
        if s.tool not in registry:
            errs.append(f"unknown tool {s.tool}")
        for d in s.depends_on:
            if d not in ids:
                errs.append(f"{s.id} depends on missing {d}")
        for v in s.args.values():
            if isinstance(v, str) and v.startswith("$"):
                ref = v[1:].split(".")[0]
                if ref not in ids:
                    errs.append(f"{s.id} references missing step {ref}")
                elif ref not in s.depends_on:
                    s.depends_on.append(ref)                  # repair: a reference implies a dependency
    done: set[str] = set()
    pending = list(plan.steps)
    while pending:
        ready = [s for s in pending if all(d in done for d in s.depends_on)]
        if not ready:
            errs.append("cyclic dependencies")
            break
        done |= {s.id for s in ready}
        pending = [s for s in pending if s.id not in done]
    for s in plan.steps:
        if s.tool == "search_medical_info" and re.search(r"\[PATIENT_\d+\]|\bP\d{4}\b", str(s.args.get("query", ""))):
            errs.append("patient identifier inside an external search query")
    return errs


class LLMPlanner:
    name = "llm"

    def __init__(self, llm: LLMClient, registry, fallback: RulePlanner):
        self.llm, self.registry, self.fallback = llm, registry, fallback

    def plan(self, text: str, session: Session, redactor: Redactor, trace: RunTrace | None = None, parent: str | None = None) -> Plan:
        ctx = {"active_patient_id": session.active_patient_id, "last_patient_id": session.last_patient_id,
               "now": self.fallback.now().isoformat(timespec="minutes")}
        try:
            raw = traced_chat(trace, parent, self.llm, "plan", planner_system(self.registry.catalog(), ctx), redactor.apply(text),
                              json_schema=PLAN_SCHEMA, max_tokens=900)
            data = extract_json(raw)
            plan = Plan(intents=data.get("intents", []), clarification=data.get("clarification"), planner="llm",
                        steps=[Step(**{k: v for k, v in s.items() if k in {"id", "goal", "tool", "args", "depends_on"}})
                               for s in data.get("steps", [])])
            if errs := validate_plan(plan, self.registry):
                raise LLMError("invalid plan: " + "; ".join(errs))
            return plan
        except Exception as e:
            fb = self.fallback.plan(text, session)
            fb.planner = "rules (LLM planner fallback)"
            fb.notes.append(f"LLM planner failed: {str(e)[:120]}")
            return fb
