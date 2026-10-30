"""Synthetic patients, doctors, schedule and EHR history. No real patient data."""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from .db import Database

# (id, name, specialty, languages, location, years, working weekdays Mon=0)
DOCTORS = [
    ("D01", "Dr. Anjali Rao", "Nephrology", "English, Kannada", "Block A, Room 12", 18, {0, 2, 4}),
    ("D02", "Dr. Vikram Shah", "Nephrology", "English, Hindi", "Block A, Room 14", 12, {1, 3, 5}),
    ("D03", "Dr. Priya Menon", "Cardiology", "English, Malayalam", "Block B, Room 3", 15, {0, 1, 3}),
    ("D04", "Dr. Arjun Nair", "Cardiology", "English, Malayalam", "Block B, Room 5", 9, {2, 4, 5}),
    ("D05", "Dr. Kavita Desai", "Endocrinology", "English, Hindi", "Block C, Room 2", 14, {0, 2, 4}),
    ("D06", "Dr. Farah Ali", "Endocrinology", "English, Urdu", "Block C, Room 4", 8, {1, 3, 5}),
    ("D07", "Dr. Rohit Verma", "General Medicine", "English, Hindi", "Block D, Room 1", 11, {0, 1, 2, 3, 4}),
    ("D08", "Dr. Sunita Iyer", "General Medicine", "English, Tamil", "Block D, Room 2", 20, {1, 2, 3, 4, 5}),
    ("D09", "Dr. Imran Khan", "Pulmonology", "English, Urdu", "Block E, Room 6", 13, {0, 3, 4}),
    ("D10", "Dr. Neha Kulkarni", "Neurology", "English, Marathi", "Block E, Room 8", 10, {1, 2, 5}),
    ("D11", "Dr. Sanjay Patil", "Orthopedics", "English, Marathi", "Block F, Room 1", 16, {0, 2, 3}),
    ("D12", "Dr. Meera Joshi", "Dermatology", "English, Hindi", "Block F, Room 5", 7, {1, 4, 5}),
    ("D13", "Dr. Karthik Reddy", "Gastroenterology", "English, Telugu", "Block G, Room 2", 12, {0, 1, 4}),
]
TIMES = [(9, 0), (9, 30), (10, 0), (10, 30), (11, 0), (11, 30), (14, 0), (14, 30), (15, 0), (15, 30)]

# (id, name, age, sex, primary condition, allergies, guardian)
PATIENTS = [
    ("P1001", "Ramesh Kumar", 70, "M", "chronic kidney disease", "Penicillin", "Anita Kumar (daughter)"),
    ("P1002", "Suresh Iyer", 70, "M", "type 2 diabetes", "Sulfa drugs", "Latha Iyer (wife)"),
    ("P1003", "Meena Pillai", 64, "F", "chronic kidney disease", "None known", "Ravi Pillai (son)"),
    ("P1004", "Lakshmi Devi", 58, "F", "hypertension", "None known", "Gopal Devi (husband)"),
    ("P1005", "Arun Mehta", 45, "M", "asthma", "Dust, pollen", "Self"),
    ("P1006", "Fatima Sheikh", 67, "F", "heart failure", "Aspirin sensitivity", "Zaid Sheikh (son)"),
    ("P1007", "George Thomas", 73, "M", "COPD", "None known", "Mary Thomas (wife)"),
    ("P1008", "Kavya Nair", 52, "F", "hypothyroidism", "None known", "Self"),
    ("P1009", "Harish Gowda", 61, "M", "osteoarthritis", "NSAIDs (stomach upset)", "Self"),
    ("P1010", "Sunita Rao", 39, "F", "migraine", "None known", "Self"),
    ("P1011", "Mohammed Rafi", 55, "M", "type 2 diabetes", "None known", "Self"),
    ("P1012", "Deepa Menon", 48, "F", "psoriasis", "None known", "Self"),
    ("P1013", "Vijay Singh", 66, "M", "hypertension", "None known", "Sunita Singh (wife)"),
    ("P1014", "Anjali Bhat", 34, "F", "asthma", "None known", "Self"),
    ("P1015", "Prakash Hegde", 78, "M", "heart failure", "None known", "Nikhil Hegde (son)"),
    ("P1016", "Rina Das", 29, "F", "gastroesophageal reflux", "None known", "Self"),
]


def _ago(now: datetime, days: int) -> str:
    return (now - timedelta(days=days)).date().isoformat()


def _records_for(cond: str, allergies: str, rng: random.Random) -> list[tuple]:
    """(days_ago, type, title, details)"""
    recs: list[tuple] = []
    if cond == "chronic kidney disease":
        recs = [
            (2400, "diagnosis", "Hypertension", "Diagnosed with essential hypertension; started on ARB therapy."),
            (1700, "diagnosis", "Type 2 diabetes mellitus", "Diagnosed with type 2 diabetes mellitus; metformin started."),
            (420, "diagnosis", "Chronic kidney disease stage G3a", "Persistent reduction in eGFR with albuminuria; referred to nephrology."),
            (120, "lab", "Kidney function panel", "eGFR 47 mL/min/1.73 m2; urine albumin-creatinine ratio 180 mg/g; potassium 4.9 mmol/L."),
            (118, "medication", "Current medications", "Telmisartan 40 mg once daily; Amlodipine 5 mg once daily; Atorvastatin 20 mg at night."),
            (60, "visit", "Nephrology review", "BP 138/82. Advised low-salt diet and hydration review. Repeat labs in 3 months."),
            (60, "alert", "Medication safety", "Avoid NSAID painkillers (kidney disease). Dose-adjust renally cleared drugs."),
        ]
    elif cond == "type 2 diabetes":
        recs = [
            (2900, "diagnosis", "Type 2 diabetes mellitus", "Diagnosed with type 2 diabetes mellitus."),
            (200, "lab", "HbA1c", "HbA1c 8.1%; fasting glucose 142 mg/dL; urine albumin normal."),
            (195, "medication", "Current medications", "Metformin 500 mg twice daily; Glimepiride 1 mg once daily."),
            (90, "visit", "Endocrinology review", "Counselled on diet and activity. Foot and eye screening advised."),
            (90, "alert", "Hypoglycaemia risk", "Monitor for low blood sugar on sulfonylurea; carry fast-acting glucose."),
        ]
    elif cond == "hypertension":
        recs = [
            (1800, "diagnosis", "Essential hypertension", "Diagnosed with essential hypertension."),
            (150, "medication", "Current medications", "Amlodipine 5 mg once daily."),
            (45, "visit", "Cardiology review", "Home BP log average 142/88; advised sodium reduction and repeat review."),
        ]
    elif cond == "asthma":
        recs = [
            (3000, "diagnosis", "Bronchial asthma", "Allergic asthma; triggers include dust and cold air."),
            (160, "medication", "Current medications", "Budesonide-formoterol inhaler; salbutamol inhaler as needed."),
            (75, "visit", "Pulmonology review", "Peak flow stable. Inhaler technique reviewed. Written action plan updated."),
            (75, "alert", "Trigger avoidance", "Avoid dust exposure; seek care if reliever needed more than twice a week."),
        ]
    elif cond == "heart failure":
        recs = [
            (900, "diagnosis", "Heart failure with reduced ejection fraction", "Echocardiogram: LVEF 38%."),
            (130, "medication", "Current medications", "Sacubitril-valsartan, Metoprolol succinate, Spironolactone, Dapagliflozin, Furosemide as directed."),
            (50, "visit", "Cardiology review", "Mild ankle swelling. Daily weight log reviewed. Fluid advice reinforced."),
            (50, "alert", "Self-monitoring", "Report weight gain over 2 kg in 3 days or worsening breathlessness."),
        ]
    elif cond == "COPD":
        recs = [
            (1500, "diagnosis", "Chronic obstructive pulmonary disease", "Former smoker; spirometry FEV1 58% predicted."),
            (140, "medication", "Current medications", "Tiotropium inhaler once daily; salbutamol as needed."),
            (70, "visit", "Pulmonology review", "Pulmonary rehabilitation recommended. Pneumococcal and flu vaccines up to date."),
        ]
    elif cond == "hypothyroidism":
        recs = [
            (2100, "diagnosis", "Primary hypothyroidism", "Diagnosed with primary hypothyroidism."),
            (100, "lab", "Thyroid panel", "TSH 3.2 mIU/L on current dose."),
            (100, "medication", "Current medications", "Levothyroxine 75 mcg each morning on an empty stomach."),
        ]
    elif cond == "osteoarthritis":
        recs = [
            (800, "diagnosis", "Osteoarthritis, both knees", "Knee pain on stairs; X-ray shows joint space narrowing."),
            (90, "medication", "Current medications", "Paracetamol as needed; topical diclofenac gel."),
            (90, "visit", "Orthopedics review", "Physiotherapy programme and weight management advised."),
            (90, "alert", "Medication safety", "Oral NSAIDs not tolerated (stomach upset)."),
        ]
    elif cond == "migraine":
        recs = [
            (1200, "diagnosis", "Migraine without aura", "Recurrent unilateral headaches with nausea."),
            (80, "visit", "Neurology review", "Headache diary started; trigger review (sleep, screen time)."),
        ]
    elif cond == "psoriasis":
        recs = [
            (1000, "diagnosis", "Plaque psoriasis", "Scaly plaques on elbows and knees."),
            (60, "medication", "Current medications", "Topical corticosteroid and emollient."),
        ]
    else:
        recs = [(500, "diagnosis", cond.title(), f"Diagnosed with {cond}.")]
    if allergies and allergies != "None known":
        recs.append((rng.randint(300, 900), "allergy", "Allergy", f"Allergy on file: {allergies}."))
    return recs


def seed(db: Database, now: datetime | None = None, rng_seed: int = 42, days: int = 12, prebook_ratio: float = 0.14) -> None:
    now = now or datetime.now()
    rng = random.Random(rng_seed)
    for did, name, spec, lang, loc, yrs, _ in DOCTORS:
        db.conn.execute("INSERT INTO doctors VALUES (?,?,?,?,?,?)", (did, name, spec, lang, loc, yrs))
    for pid, name, age, sex, cond, allergy, guardian in PATIENTS:
        db.conn.execute("INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?)",
                        (pid, name, age, sex, f"+91-90000-{rng.randint(10000, 99999)}", guardian, cond,
                         rng.choice(["A+", "B+", "O+", "AB+", "O-"]), allergy))
        for days_ago, rtype, title, details in _records_for(cond, allergy, rng):
            db.conn.execute(
                "INSERT INTO records(patient_id,rec_date,rec_type,title,details,structured,author,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (pid, _ago(now, days_ago), rtype, title, details, "{}", "seed", now.isoformat(timespec="seconds")))
    start_day = (now + timedelta(days=1)).date()
    slot_ids: list[int] = []
    for offset in range(days):
        day = start_day + timedelta(days=offset)
        if day.weekday() == 6:      # no Sunday clinics
            continue
        for did, *_rest, wd in DOCTORS:
            if day.weekday() not in wd:
                continue
            for h, m in TIMES:
                st = datetime(day.year, day.month, day.day, h, m)
                cur = db.conn.execute("INSERT INTO slots(doctor_id,start_ts,end_ts,status) VALUES (?,?,?,'open')",
                                      (did, st.strftime("%Y-%m-%dT%H:%M"), (st + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M")))
                slot_ids.append(cur.lastrowid)
    db.conn.commit()
    pids = [p[0] for p in PATIENTS]
    for slot_id in slot_ids:                       # realistic pre-booked load
        if rng.random() < prebook_ratio:
            try:
                db.book(rng.choice(pids), slot_id, "Follow-up consultation", created_by="seed",
                        now=now.isoformat(timespec="seconds"))
            except Exception:
                pass
