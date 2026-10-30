"""Doctor Schedule API tools: find slots, book (self-healing + approval gate), cancel."""
from __future__ import annotations

from datetime import timedelta

from ..data.db import SlotUnavailable
from .registry import ToolContext, ToolSpec


def _fmt_slot(s: dict) -> str:
    return f"{s['start_ts'].replace('T', ' ')} with {s['doctor_name']}"


def find_slots(ctx: ToolContext, specialty: str, date_from: str | None = None, date_to: str | None = None,
               part_of_day: str | None = None, doctor_id: str | None = None, limit: int = 5) -> dict:
    now = ctx.now()
    floor = now.strftime("%Y-%m-%dT%H:%M")
    d_from = max(date_from or floor, floor)
    d_to = date_to or (now + timedelta(days=14)).strftime("%Y-%m-%dT%H:%M")
    slots = ctx.db.open_slots(specialty, d_from, d_to, part_of_day, doctor_id, limit)
    window_met, note = bool(slots), None
    if not slots:   # widen: nearest alternatives within 21 days, any time of day
        slots = ctx.db.open_slots(specialty, floor, (now + timedelta(days=21)).strftime("%Y-%m-%dT%H:%M"), None, None, limit)
        note = "No open slot matched the requested window; showing the nearest alternatives." if slots else None
    if not slots:
        return {"ok": False, "message": f"No open {specialty} slots in the next 21 days.", "specialty": specialty}
    return {"ok": True, "specialty": specialty, "slots": slots, "best_slot": slots[0], "window_met": window_met, "note": note}


def book_appointment(ctx: ToolContext, patient_id: str, slot_id: int, reason: str = "Consultation",
                     specialty: str | None = None, window_met: bool = True, confirm: bool = False) -> dict:
    """Without `confirm`, a write is held for human approval whenever auto-booking is off or the slot is an
    alternative to what was asked. The agent never sets `confirm` itself: only the approval path does."""
    patient = ctx.db.get_patient(patient_id)
    if not patient:
        return {"ok": False, "message": f"Unknown patient {patient_id}."}
    slot = ctx.db.q("SELECT s.*, d.name AS doctor_name, d.specialty, d.location FROM slots s "
                    "JOIN doctors d ON d.doctor_id=s.doctor_id WHERE s.slot_id=?", (int(slot_id),))
    if not slot:
        return {"ok": False, "message": f"Slot {slot_id} not found."}
    if (not ctx.settings.auto_confirm_writes or not window_met) and not confirm:
        why = ("auto-booking is off" if not ctx.settings.auto_confirm_writes
               else "your preferred time was unavailable, so this is an alternative")
        ctx.memory.trace("plan", f"booking proposed for slot {slot_id}; awaiting approval ({why})", patient_id, ctx.run_id)
        return {"ok": True, "status": "pending_confirmation", "proposal": slot[0], "reason": reason,
                "message": f"Proposed {_fmt_slot(slot[0])}. Waiting for approval because {why}."}
    tried, current = [], int(slot_id)
    for _ in range(3):   # self-healing: if the slot was just taken, move to the next open one
        try:
            appt = ctx.db.book(patient_id, current, reason, created_by="agent", now=ctx.now().isoformat(timespec="seconds"))
            ctx.memory.trace("write", f"booked appt #{appt['appt_id']} {appt['specialty']} {appt['start_ts']}", patient_id, ctx.run_id)
            ctx.memory.index_patient(patient_id)
            return {"ok": True, "status": "booked", "appointment": appt, "retries": len(tried), "message": f"Booked {_fmt_slot(appt)}."}
        except SlotUnavailable as e:
            tried.append(str(e))
            if not specialty:
                break
            nxt = ctx.db.open_slots(specialty, ctx.now().strftime("%Y-%m-%dT%H:%M"),
                                    (ctx.now() + timedelta(days=21)).strftime("%Y-%m-%dT%H:%M"), None, None, 1)
            if not nxt:
                break
            current = nxt[0]["slot_id"]
    return {"ok": False, "message": "Could not book: " + "; ".join(tried)}


def cancel_appointment(ctx: ToolContext, patient_id: str, appt_id: int | None = None, specialty: str | None = None) -> dict:
    now_s = ctx.now().strftime("%Y-%m-%dT%H:%M")
    appts = [a for a in ctx.db.list_appointments(patient_id=patient_id, status="booked") if a["start_ts"] >= now_s]
    if specialty:
        appts = [a for a in appts if a["specialty"].lower() == specialty.lower()]
    if appt_id:
        appts = [a for a in appts if a["appt_id"] == int(appt_id)]
    if not appts:
        return {"ok": False, "message": "No upcoming appointment found to cancel."}
    if len(appts) > 1:
        return {"ok": False, "needs_clarification": True, "message": "More than one upcoming appointment; which one?", "candidates": appts}
    done = ctx.db.cancel(appts[0]["appt_id"])
    ctx.memory.trace("write", f"cancelled appt #{appts[0]['appt_id']}", patient_id, ctx.run_id)
    return {"ok": True, "status": "cancelled", "appointment": done,
            "message": f"Cancelled the {done['specialty']} appointment on {done['start_ts'].replace('T', ' ')}."}


SPECS = [
    ToolSpec("find_slots", "Query the Doctor Schedule API for open slots for a specialty, optionally in a date window / part of day.",
             {"specialty": {"type": "string"}, "date_from": {"type": "string", "description": "ISO datetime"}, "date_to": {"type": "string"},
              "part_of_day": {"type": "string", "enum": ["morning", "afternoon"]}, "doctor_id": {"type": "string"}, "limit": {"type": "integer"}},
             find_slots, required=["specialty"]),
    ToolSpec("book_appointment", "Book a slot for a patient. Self-heals if the slot was just taken. Held for approval unless `confirm` is true (set only by the human-approval path or an MCP client's own approval UI).",
             {"patient_id": {"type": "string"}, "slot_id": {"type": "integer"}, "reason": {"type": "string"}, "specialty": {"type": "string"},
              "window_met": {"type": "boolean"}, "confirm": {"type": "boolean"}},
             book_appointment, side_effect=True, required=["patient_id", "slot_id"]),
    ToolSpec("cancel_appointment", "Cancel a patient's upcoming appointment.",
             {"patient_id": {"type": "string"}, "appt_id": {"type": "integer"}, "specialty": {"type": "string"}},
             cancel_appointment, side_effect=True, destructive=True, required=["patient_id"]),
]
