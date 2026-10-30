"""Three memory tiers: working (session), episodic (trace of reads/writes/plans), long-term (vector index).
Long-term memory also holds a consolidated PATIENT BRIEF per patient (the brief's 'patient summaries')."""
from __future__ import annotations

import threading
from datetime import datetime

from ..data.db import Database
from .vector_store import Hit, HashEmbedder, HybridIndex, RemoteEmbedder


def build_brief(db: Database, pid: str, now: datetime | None = None) -> str:
    """Deterministic consolidation of a patient's timeline into one retrievable summary."""
    p = db.get_patient(pid)
    if not p:
        return ""
    recs = db.get_records(pid)
    pick = lambda t, n: [r for r in recs if r["rec_type"] == t][:n]  # noqa: E731
    dx = "; ".join(r["title"] for r in pick("diagnosis", 4)) or "none recorded"
    meds = "; ".join(r["details"] for r in pick("medication", 1)) or "none recorded"
    lab = "; ".join(f"{r['title']}: {r['details']}" for r in pick("lab", 1)) or "none recorded"
    alerts = "; ".join(r["details"] for r in pick("alert", 3)) or "none"
    allergy = p["allergies"] if p["allergies"] and p["allergies"] != "None known" else "none known"
    n_up = len([a for a in db.list_appointments(patient_id=pid, status="booked") if a["start_ts"] >= (now or datetime.now()).strftime("%Y-%m-%dT%H:%M")])
    return (f"PATIENT BRIEF {p['name']} ({pid}): {p['age']}-year-old {p['sex']}, primary condition {p['primary_condition']}. "
            f"Diagnoses: {dx}. Current medicines: {meds}. Latest lab: {lab}. Alerts: {alerts}. Allergies: {allergy}. "
            f"Upcoming appointments: {n_up}.")


class MemoryManager:
    def __init__(self, db: Database, settings=None):
        emb = HashEmbedder()
        if settings is not None and getattr(settings, "embeddings", "hash") == "remote" and settings.openai_api_key:
            emb = RemoteEmbedder(settings.openai_base_url, settings.openai_api_key, "text-embedding-3-small")
        self.db, self.settings = db, settings
        self.index = HybridIndex(emb)
        self.traces: list[dict] = []
        self._lock = threading.Lock()
        self._briefs: dict[str, str] = {}

    def _now(self) -> datetime:
        return self.settings.now() if self.settings else datetime.now()

    def trace(self, kind: str, detail: str, patient_id: str | None = None, run_id: str | None = None) -> None:
        with self._lock:
            self.traces.append({"ts": datetime.now().strftime("%H:%M:%S.%f")[:-3], "kind": kind, "patient_id": patient_id or "",
                                "run_id": run_id or "", "detail": detail})
            del self.traces[:-500]

    def brief(self, patient_id: str) -> str:
        return self._briefs.get(patient_id) or build_brief(self.db, patient_id, self._now())

    def index_patient(self, patient_id: str) -> int:
        self.index.remove_where(lambda m: m.get("patient_id") == patient_id)
        p = self.db.get_patient(patient_id)
        if not p:
            return 0
        b = build_brief(self.db, patient_id, self._now())
        self._briefs[patient_id] = b
        self.index.add(f"{patient_id}:brief", b, {"patient_id": patient_id, "kind": "brief"})
        self.index.add(f"{patient_id}:profile",
                       f"Patient profile: {p['age']}-year-old {p['sex']}; primary condition {p['primary_condition']}; "
                       f"allergies {p['allergies']}; blood group {p['blood_group']}.", {"patient_id": patient_id, "kind": "profile"})
        n = 2
        for r in self.db.get_records(patient_id):
            self.index.add(f"{patient_id}:r{r['record_id']}", f"{r['rec_date']} {r['rec_type'].upper()}: {r['title']}. {r['details']}",
                           {"patient_id": patient_id, "kind": r["rec_type"], "record_id": r["record_id"], "rec_date": r["rec_date"]})
            n += 1
        return n

    def index_all(self) -> int:
        return sum(self.index_patient(p["patient_id"]) for p in self.db.list_patients())

    def recall(self, patient_id: str, query: str, k: int = 4, run_id: str | None = None) -> list[Hit]:
        hits = self.index.search(query, k=k, where=lambda m: m.get("patient_id") == patient_id)
        self.trace("recall", f"query='{query[:60]}' -> {len(hits)} chunk(s)", patient_id, run_id)
        return hits

    def stats(self) -> dict:
        return {"chunks": len(self.index), "backend": self.index.backend, "trace_events": len(self.traces)}
