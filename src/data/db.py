"""SQLite: EHR + Doctor Schedule API + audit, tool logs, spans and agent checkpoints (thread-safe)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from typing import Any, Iterable

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients(
  patient_id TEXT PRIMARY KEY, name TEXT NOT NULL, age INTEGER, sex TEXT,
  phone TEXT, guardian TEXT, primary_condition TEXT, blood_group TEXT, allergies TEXT);
CREATE TABLE IF NOT EXISTS doctors(
  doctor_id TEXT PRIMARY KEY, name TEXT NOT NULL, specialty TEXT NOT NULL,
  languages TEXT, location TEXT, years INTEGER);
CREATE TABLE IF NOT EXISTS slots(
  slot_id INTEGER PRIMARY KEY AUTOINCREMENT, doctor_id TEXT NOT NULL,
  start_ts TEXT NOT NULL, end_ts TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open');
CREATE INDEX IF NOT EXISTS ix_slots ON slots(status, start_ts);
CREATE TABLE IF NOT EXISTS appointments(
  appt_id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL, doctor_id TEXT NOT NULL,
  slot_id INTEGER NOT NULL, reason TEXT, status TEXT NOT NULL DEFAULT 'booked',
  created_at TEXT, created_by TEXT);
CREATE TABLE IF NOT EXISTS records(
  record_id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL, rec_date TEXT,
  rec_type TEXT NOT NULL, title TEXT, details TEXT, structured TEXT, author TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS ix_records ON records(patient_id, rec_date);
CREATE TABLE IF NOT EXISTS tool_logs(
  log_id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, run_id TEXT, step_id TEXT, tool TEXT,
  status TEXT, latency_ms REAL, args TEXT, error TEXT, side_effect INTEGER);
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, ts TEXT, principal TEXT, query TEXT, intents TEXT, n_steps INTEGER,
  ok INTEGER, latency_ms REAL, planner TEXT, mode TEXT, tokens INTEGER);
CREATE TABLE IF NOT EXISTS spans(
  trace_id TEXT, span_id TEXT, parent_id TEXT, name TEXT, kind TEXT, start_ms REAL, end_ms REAL,
  status TEXT, attrs TEXT);
CREATE INDEX IF NOT EXISTS ix_spans ON spans(trace_id);
CREATE TABLE IF NOT EXISTS audit_log(
  audit_id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, run_id TEXT, principal TEXT, role TEXT,
  tool TEXT, patient_id TEXT, decision TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS checkpoints(
  run_id TEXT, version INTEGER, ts TEXT, label TEXT, state TEXT, PRIMARY KEY(run_id, version));
"""

REC_TYPES = ("diagnosis", "medication", "lab", "allergy", "alert", "visit", "note")

_APPT_SQL = (
    "SELECT a.appt_id, a.patient_id, p.name AS patient_name, a.doctor_id, d.name AS doctor_name, "
    "d.specialty, s.slot_id, s.start_ts, s.end_ts, a.reason, a.status, a.created_by, a.created_at "
    "FROM appointments a JOIN slots s ON s.slot_id=a.slot_id "
    "JOIN doctors d ON d.doctor_id=a.doctor_id JOIN patients p ON p.patient_id=a.patient_id")


class SlotUnavailable(Exception):
    pass


class Database:
    def __init__(self, path: str = ":memory:"):
        self._lock = threading.RLock()
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ---- primitives -----------------------------------------------------------
    def q(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def x(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, tuple(params))
            self.conn.commit()
            return cur

    def df(self, sql: str, params: Iterable[Any] = ()) -> pd.DataFrame:
        return pd.DataFrame(self.q(sql, params))

    def is_seeded(self) -> bool:
        return bool(self.q("SELECT 1 FROM patients LIMIT 1"))

    # ---- patients and doctors ---------------------------------------------------
    def get_patient(self, pid: str) -> dict | None:
        r = self.q("SELECT * FROM patients WHERE patient_id=?", (pid,))
        return r[0] if r else None

    def list_patients(self) -> list[dict]:
        return self.q("SELECT * FROM patients ORDER BY patient_id")

    def search_patients(self, name: str | None = None, age: int | None = None, sex: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM patients WHERE 1=1", []
        if name:
            sql += " AND lower(name) LIKE ?"
            params.append(f"%{name.lower()}%")
        if age is not None:
            sql += " AND age=?"
            params.append(int(age))
        if sex:
            sql += " AND upper(sex)=?"
            params.append(sex.upper())
        return self.q(sql + " ORDER BY patient_id", params)

    def list_doctors(self) -> list[dict]:
        return self.q("SELECT * FROM doctors ORDER BY specialty, name")

    def get_doctor(self, did: str) -> dict | None:
        r = self.q("SELECT * FROM doctors WHERE doctor_id=?", (did,))
        return r[0] if r else None

    # ---- Doctor Schedule API ------------------------------------------------------
    def open_slots(self, specialty: str, date_from: str, date_to: str, part_of_day: str | None = None,
                   doctor_id: str | None = None, limit: int = 5) -> list[dict]:
        sql = ("SELECT s.slot_id, s.doctor_id, d.name AS doctor_name, d.specialty, d.location, "
               "s.start_ts, s.end_ts FROM slots s JOIN doctors d ON d.doctor_id=s.doctor_id "
               "WHERE s.status='open' AND lower(d.specialty)=lower(?) AND s.start_ts>=? AND s.start_ts<?")
        params: list[Any] = [specialty, date_from, date_to]
        if part_of_day == "morning":
            sql += " AND substr(s.start_ts,12,5) < '12:00'"
        elif part_of_day == "afternoon":
            sql += " AND substr(s.start_ts,12,5) >= '12:00'"
        if doctor_id:
            sql += " AND s.doctor_id=?"
            params.append(doctor_id)
        return self.q(sql + " ORDER BY s.start_ts, s.doctor_id LIMIT ?", [*params, int(limit)])

    def get_appointment(self, appt_id: int) -> dict | None:
        r = self.q(_APPT_SQL + " WHERE a.appt_id=?", (appt_id,))
        return r[0] if r else None

    def list_appointments(self, patient_id: str | None = None, doctor_id: str | None = None,
                          status: str | None = None) -> list[dict]:
        sql, params = _APPT_SQL + " WHERE 1=1", []
        for col, val in (("a.patient_id", patient_id), ("a.doctor_id", doctor_id), ("a.status", status)):
            if val:
                sql += f" AND {col}=?"
                params.append(val)
        return self.q(sql + " ORDER BY s.start_ts", params)

    def book(self, patient_id: str, slot_id: int, reason: str, created_by: str = "agent",
             now: str | None = None) -> dict:
        """Atomic: the UPDATE only succeeds while the slot is still open."""
        with self._lock:
            slot = self.q("SELECT * FROM slots WHERE slot_id=?", (slot_id,))
            if not slot:
                raise SlotUnavailable(f"Slot {slot_id} does not exist.")
            clash = self.q("SELECT 1 FROM appointments a JOIN slots s ON s.slot_id=a.slot_id "
                           "WHERE a.patient_id=? AND a.status='booked' AND s.start_ts=?",
                           (patient_id, slot[0]["start_ts"]))
            if clash:
                raise SlotUnavailable("The patient already has an appointment at that time.")
            cur = self.conn.execute("UPDATE slots SET status='booked' WHERE slot_id=? AND status='open'", (slot_id,))
            if cur.rowcount != 1:
                self.conn.rollback()
                raise SlotUnavailable(f"Slot {slot_id} was just taken.")
            cur = self.conn.execute(
                "INSERT INTO appointments(patient_id,doctor_id,slot_id,reason,status,created_at,created_by) VALUES (?,?,?,?,?,?,?)",
                (patient_id, slot[0]["doctor_id"], slot_id, reason, "booked",
                 now or datetime.now().isoformat(timespec="seconds"), created_by))
            self.conn.commit()
            return self.get_appointment(cur.lastrowid)  # type: ignore[return-value]

    def cancel(self, appt_id: int) -> dict | None:
        with self._lock:
            a = self.get_appointment(appt_id)
            if not a or a["status"] != "booked":
                return None
            self.conn.execute("UPDATE appointments SET status='cancelled' WHERE appt_id=?", (appt_id,))
            self.conn.execute("UPDATE slots SET status='open' WHERE slot_id=?", (a["slot_id"],))
            self.conn.commit()
            return self.get_appointment(appt_id)

    # ---- EHR ----------------------------------------------------------------------
    def add_record(self, patient_id: str, rec_type: str, title: str, details: str, rec_date: str | None = None,
                   structured: dict | None = None, author: str = "attendant") -> dict:
        cur = self.x(
            "INSERT INTO records(patient_id,rec_date,rec_type,title,details,structured,author,updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (patient_id, rec_date or datetime.now().date().isoformat(), rec_type, title, details,
             json.dumps(structured or {}), author, datetime.now().isoformat(timespec="seconds")))
        return self.get_record(cur.lastrowid)  # type: ignore[arg-type,return-value]

    def get_record(self, record_id: int) -> dict | None:
        r = self.q("SELECT * FROM records WHERE record_id=?", (record_id,))
        return r[0] if r else None

    def update_record(self, record_id: int, **fields: Any) -> dict | None:
        sets = {k: v for k, v in fields.items() if k in {"title", "details", "rec_type", "rec_date"} and v is not None}
        if not sets:
            return self.get_record(record_id)
        sets["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.x(f"UPDATE records SET {', '.join(k + '=?' for k in sets)} WHERE record_id=?", [*sets.values(), record_id])
        return self.get_record(record_id)

    def get_records(self, patient_id: str) -> list[dict]:
        return self.q("SELECT * FROM records WHERE patient_id=? ORDER BY rec_date DESC, record_id DESC", (patient_id,))

    # ---- observability, audit, checkpoints --------------------------------------
    def log_tool(self, run_id: str, step_id: str, tool: str, status: str, latency_ms: float, args: dict,
                 error: str | None, side_effect: bool) -> None:
        self.x("INSERT INTO tool_logs(ts,run_id,step_id,tool,status,latency_ms,args,error,side_effect) VALUES (?,?,?,?,?,?,?,?,?)",
               (datetime.now().isoformat(timespec="seconds"), run_id, step_id, tool, status, round(latency_ms, 2),
                json.dumps(args, default=str)[:600], error, int(side_effect)))

    def log_run(self, run_id: str, principal: str, query: str, intents: list[str], n_steps: int, ok: bool,
                latency_ms: float, planner: str, mode: str, tokens: int = 0) -> None:
        self.x("INSERT OR REPLACE INTO runs(run_id,ts,principal,query,intents,n_steps,ok,latency_ms,planner,mode,tokens) "
               "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
               (run_id, datetime.now().isoformat(timespec="seconds"), principal, query[:300], ",".join(intents),
                n_steps, int(ok), round(latency_ms, 1), planner, mode, tokens))

    def log_spans(self, spans: list[dict]) -> None:
        with self._lock:
            self.conn.executemany(
                "INSERT INTO spans(trace_id,span_id,parent_id,name,kind,start_ms,end_ms,status,attrs) VALUES (?,?,?,?,?,?,?,?,?)",
                [(s["trace_id"], s["span_id"], s["parent_id"], s["name"], s["kind"], round(s["start_ms"], 2),
                  round(s["end_ms"] or s["start_ms"], 2), s["status"], json.dumps(s["attrs"], default=str)[:800]) for s in spans])
            self.conn.commit()

    def log_audit(self, run_id: str, principal: str, role: str, tool: str, patient_id: str | None,
                  decision: str, reason: str) -> None:
        self.x("INSERT INTO audit_log(ts,run_id,principal,role,tool,patient_id,decision,reason) VALUES (?,?,?,?,?,?,?,?)",
               (datetime.now().isoformat(timespec="seconds"), run_id, principal, role, tool, patient_id, decision, reason))

    def save_checkpoint(self, run_id: str, state: dict, label: str) -> int:
        with self._lock:
            v = (self.q("SELECT max(version) m FROM checkpoints WHERE run_id=?", (run_id,))[0]["m"] or 0) + 1
            self.conn.execute("INSERT INTO checkpoints(run_id,version,ts,label,state) VALUES (?,?,?,?,?)",
                              (run_id, v, datetime.now().isoformat(timespec="seconds"), label, json.dumps(state, default=str)))
            self.conn.commit()
            return v

    def load_checkpoint(self, run_id: str, version: int | None = None) -> dict | None:
        sql = "SELECT * FROM checkpoints WHERE run_id=?" + (" AND version=?" if version else "") + " ORDER BY version DESC LIMIT 1"
        r = self.q(sql, (run_id, version) if version else (run_id,))
        return {**r[0], "state": json.loads(r[0]["state"])} if r else None

    def checkpoint_history(self, run_id: str) -> list[dict]:
        return self.q("SELECT version, ts, label FROM checkpoints WHERE run_id=? ORDER BY version", (run_id,))
