"""Safety and privacy controls: emergency triage, diagnosis/dosage refusal, PHI redaction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MEDICAL_DISCLAIMER = ("General information only, not medical advice. It does not replace a clinician's "
                      "judgement; please confirm decisions with the treating doctor.")
EMERGENCY_MESSAGE = (
    "**This may be a medical emergency.** Please call your local emergency number or go to the nearest "
    "emergency department now. I have not booked or changed anything. Once the person is safe, I can help "
    "with follow-up appointments and records.")
CLINICAL_ADVICE_NOTE = ("I can't diagnose conditions or advise on prescriptions or doses. I can share general "
                        "educational information and help you book a clinician who can.")

_EMERGENCY = [r"\bchest pain\b", r"\b(can'?t|cannot|unable to) breathe\b", r"\bdifficulty breathing\b", r"\bsuicid",
              r"\boverdos", r"\bunconscious\b", r"\bsevere bleeding\b", r"\b(having|had) a (heart attack|stroke)\b",
              r"\bface (is )?drooping\b", r"\bnot breathing\b"]
_DIAGNOSE = [r"\bdiagnose (me|him|her|them)\b", r"\bdo (i|he|she|they) have\b.*\b(cancer|diabetes|disease)\b",
             r"\bwhat (disease|condition) (do|does) (i|he|she) have\b", r"\bprescribe\b",
             r"\bhow (much|many) (mg|tablets|pills)\b", r"\b(dose|dosage)\b",
             r"\bshould (i|he|she) (stop|skip|increase|reduce|double)\b"]


@dataclass
class SafetyFlags:
    emergency: bool = False
    clinical_advice: bool = False
    notes: list[str] = field(default_factory=list)


def classify(text: str) -> SafetyFlags:
    t = (text or "").lower()
    f = SafetyFlags()
    if any(re.search(p, t) for p in _EMERGENCY):
        f.emergency = True
        f.notes.append("emergency language detected")
    if any(re.search(p, t) for p in _DIAGNOSE):
        f.clinical_advice = True
        f.notes.append("request for diagnosis/prescription/dosage")
    return f


class Redactor:
    """Swaps patient names for stable placeholders before text goes to an external LLM."""

    def __init__(self, names: list[str]):
        self.names = sorted({n for n in names if n}, key=len, reverse=True)
        self.map: dict[str, str] = {}

    def apply(self, text: str) -> str:
        out = text or ""
        for n in self.names:
            rx = re.compile(re.escape(n), re.I)
            if rx.search(out):
                out = rx.sub(self.map.setdefault(n, f"[PATIENT_{len(self.map) + 1}]"), out)
        return out

    def restore(self, value):
        if isinstance(value, str):
            for name, token in self.map.items():
                value = value.replace(token, name)
            return value
        if isinstance(value, dict):
            return {k: self.restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.restore(v) for v in value]
        return value


_PHONE = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PID = re.compile(r"\bP\d{3,5}\b", re.I)


def redact_for_external(text: str, names: list[str] | None = None) -> str:
    """Strict scrub for anything leaving the system (public search APIs)."""
    out = Redactor(names or []).apply(text or "")
    out = _EMAIL.sub("", _PHONE.sub("", _PID.sub("", out)))
    out = re.sub(r"\[PATIENT_\d+\]", "", out)
    out = re.sub(r"\b(my|his|her|their)\s+(father|mother|dad|mom|mum|husband|wife|son|daughter)\b", "", out, flags=re.I)
    return re.sub(r"\s{2,}", " ", out).strip(" ,.-")
