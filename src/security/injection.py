"""Indirect prompt-injection defence (OWASP LLM01). Anything that is DATA (web snippets, EHR free text,
memory chunks) is scanned; instruction-like content is quarantined or neutralised before it can reach an LLM."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

THRESHOLD = 0.6
_PATTERNS: list[tuple[str, float, str]] = [
    (r"\b(ignore|disregard|forget|override)\b.{0,25}\b(previous|prior|above|earlier|all|any|your)\b.{0,25}\b(instructions?|prompts?|rules?|guidelines?|polic(?:y|ies))\b", 0.9, "override instructions"),
    (r"\b(reveal|print|show|output|leak|repeat)\b.{0,20}\b(system|hidden|developer|initial)\b.{0,12}\b(prompt|instructions?|message)\b", 0.9, "system prompt exfiltration"),
    (r"\byou are now\b.{0,20}\b(developer|dan|jailbroken|unrestricted|admin|root)\b", 0.8, "role hijack"),
    (r"\bnew instructions?\s*:", 0.7, "injected instructions"),
    (r"\b(cancel|delete|remove|drop|wipe)\b.{0,15}\b(all|every)\b.{0,15}\b(appointments?|records?|patients?)\b", 0.8, "destructive instruction"),
    (r"\b(send|post|forward|exfiltrate|upload|email)\b.{0,70}\b(to|at)\b\s*https?://", 0.8, "data exfiltration"),
    (r"!\[[^\]]*\]\(\s*https?://[^)\s]*\?[^)\s]*\)", 0.8, "markdown image exfiltration"),
    (r"<\s*(script|iframe|object|embed)\b", 0.8, "active HTML"),
    (r"(?im)^\s*(system|assistant|developer)\s*:", 0.6, "role marker"),
    (r"\b(do not|don't|never)\b.{0,20}\b(tell|inform|mention|warn)\b.{0,20}\b(the user|patient|anyone)\b", 0.7, "concealment instruction"),
    (r"[A-Za-z0-9+/]{120,}={0,2}", 0.5, "long encoded blob"),
]
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")


@dataclass
class Finding:
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def suspicious(self) -> bool:
        return self.score >= THRESHOLD


def scan(text: str) -> Finding:
    t = text or ""
    f = Finding()
    for pat, w, why in _PATTERNS:
        if re.search(pat, t, re.I | re.S):
            f.score = min(1.0, f.score + w)
            f.reasons.append(why)
    if len(_ZERO_WIDTH.findall(t)) > 2:
        f.score = min(1.0, f.score + 0.5)
        f.reasons.append("hidden zero-width characters")
    return f


def neutralize(text: str) -> str:
    """Replace instruction-like sentences with a visible marker (keeps the rest of the data usable)."""
    if not text or not scan(text).suspicious:
        return text
    parts = re.split(r"(?<=[.!?\n])\s+", _ZERO_WIDTH.sub("", text))
    kept = [("[removed: instruction-like text]" if scan(p).score >= 0.5 else p) for p in parts]
    out = " ".join(kept)
    return out if not scan(out).suspicious else "[removed: instruction-like text]"


def neutralize_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return neutralize(obj)
    if isinstance(obj, dict):
        return {k: neutralize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [neutralize_obj(v) for v in obj]
    return obj
