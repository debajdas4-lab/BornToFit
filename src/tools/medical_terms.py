"""Shared vocabulary: conditions, aliases and the specialty that usually treats them."""
from __future__ import annotations

import re

CONDITIONS: dict[str, dict] = {
    "chronic kidney disease": {"aliases": ["ckd", "kidney disease", "kidney failure", "renal failure", "renal disease", "chronic renal"], "specialty": "Nephrology"},
    "type 2 diabetes": {"aliases": ["diabetes", "diabetic", "t2dm", "type 2 diabetes mellitus", "type ii diabetes", "blood sugar"], "specialty": "Endocrinology"},
    "hypertension": {"aliases": ["high blood pressure", "hypertensive"], "specialty": "Cardiology"},
    "heart failure": {"aliases": ["cardiac failure", "chf", "hfref"], "specialty": "Cardiology"},
    "COPD": {"aliases": ["copd", "chronic obstructive pulmonary disease", "emphysema"], "specialty": "Pulmonology"},
    "asthma": {"aliases": ["asthmatic", "bronchial asthma"], "specialty": "Pulmonology"},
    "osteoarthritis": {"aliases": ["arthritis", "knee pain", "joint pain"], "specialty": "Orthopedics"},
    "hypothyroidism": {"aliases": ["thyroid", "underactive thyroid"], "specialty": "Endocrinology"},
    "migraine": {"aliases": ["migraines", "headache"], "specialty": "Neurology"},
    "stroke": {"aliases": ["cerebrovascular accident"], "specialty": "Neurology"},
    "psoriasis": {"aliases": ["eczema", "skin rash"], "specialty": "Dermatology"},
    "gastroesophageal reflux": {"aliases": ["gerd", "acid reflux", "heartburn"], "specialty": "Gastroenterology"},
}

SPECIALTY_WORDS: dict[str, str] = {
    "nephrologist": "Nephrology", "nephrology": "Nephrology", "kidney specialist": "Nephrology", "kidney doctor": "Nephrology",
    "cardiologist": "Cardiology", "cardiology": "Cardiology", "heart specialist": "Cardiology", "heart doctor": "Cardiology",
    "endocrinologist": "Endocrinology", "endocrinology": "Endocrinology", "diabetologist": "Endocrinology",
    "pulmonologist": "Pulmonology", "pulmonology": "Pulmonology", "lung specialist": "Pulmonology",
    "neurologist": "Neurology", "neurology": "Neurology",
    "orthopedic": "Orthopedics", "orthopaedic": "Orthopedics", "orthopedist": "Orthopedics", "orthopedics": "Orthopedics", "bone doctor": "Orthopedics",
    "dermatologist": "Dermatology", "dermatology": "Dermatology", "skin doctor": "Dermatology",
    "gastroenterologist": "Gastroenterology", "gastroenterology": "Gastroenterology",
    "general physician": "General Medicine", "general practitioner": "General Medicine", "family doctor": "General Medicine",
    "general medicine": "General Medicine", "gp": "General Medicine", "physician": "General Medicine",
}


def _compile(terms: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b", re.I)


_COND_RE = {c: _compile([c, *m["aliases"]]) for c, m in CONDITIONS.items()}
_SPEC_RE = _compile(list(SPECIALTY_WORDS))
_CUE = r"\b(?:has|have|having|with|suffers? from|suffering from|diagnosed with|known case of|history of)\s+(?:(?:a|an|the|severe|mild|early|advanced)\s+)?"
_PATIENT_COND_RE = {c: re.compile(_CUE + "(?:" + rx.pattern + ")", re.I) for c, rx in _COND_RE.items()}


def detect_conditions(text: str) -> list[str]:
    hits = [(m.start(), c) for c, rx in _COND_RE.items() if (m := rx.search(text or ""))]
    return [c for _, c in sorted(hits)]


def detect_patient_conditions(text: str) -> list[str]:
    """Conditions described as the patient's own ("father has CKD"), not merely asked about."""
    return [c for c, rx in _PATIENT_COND_RE.items() if rx.search(text or "")]


def detect_specialty(text: str) -> str | None:
    m = _SPEC_RE.search(text or "")
    return SPECIALTY_WORDS[m.group(0).lower()] if m else None


def specialty_for_condition(cond: str | None) -> str | None:
    return CONDITIONS.get(cond or "", {}).get("specialty")


def canonical_condition(text: str | None) -> str | None:
    found = detect_conditions(text or "")
    return found[0] if found else None


def count_condition_mentions(text: str, cond: str) -> int:
    rx = _COND_RE.get(cond)
    return len(rx.findall(text or "")) if rx else 0
