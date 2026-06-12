"""
Generates 10 synthetic patient JSON files and lab_reference.csv.
Run: python data/patients.py
"""
import json, csv, os

DIR = os.path.join(os.path.dirname(__file__), "patients")
CSV = os.path.join(os.path.dirname(__file__), "lab_reference.csv")

RECORDS = [
    {"patient_id": "PT001", "ground_truth": "P1",
     "symptoms": "Severe crushing chest pain radiating to left arm and jaw. Profuse sweating, nausea, 45 min onset.",
     "history": {"age": 67, "gender": "Male", "conditions": ["Hypertension", "Diabetes"]},
     "labs": {"troponin_ngL": 520, "bp_systolic": 195, "wbc": 12500, "oxygen_sat": 90, "glucose_mgdL": 280, "rr": 24}},

    {"patient_id": "PT002", "ground_truth": "P1",
     "symptoms": "Worst headache of life, sudden onset. Neck stiffness, photophobia, vomiting x2, mild confusion.",
     "history": {"age": 42, "gender": "Female", "conditions": ["Migraines"]},
     "labs": {"troponin_ngL": 15, "bp_systolic": 178, "wbc": 16200, "oxygen_sat": 97, "glucose_mgdL": 105, "rr": 20}},

    {"patient_id": "PT003", "ground_truth": "P1",
     "symptoms": "Worsening SOB over 3 days. Productive cough, fever, chills. COPD patient on inhalers.",
     "history": {"age": 72, "gender": "Male", "conditions": ["COPD", "Hypertension"]},
     "labs": {"troponin_ngL": 30, "bp_systolic": 145, "wbc": 18900, "oxygen_sat": 86, "glucose_mgdL": 115, "rr": 28}},

    {"patient_id": "PT004", "ground_truth": "P2",
     "symptoms": "Sharp right-sided chest pain worse on breathing. Mild SOB, temp 38.8°C. Recent long-haul flight.",
     "history": {"age": 35, "gender": "Female", "conditions": ["OCP use"]},
     "labs": {"troponin_ngL": 55, "bp_systolic": 128, "wbc": 11200, "oxygen_sat": 92, "glucose_mgdL": 98, "rr": 22}},

    {"patient_id": "PT005", "ground_truth": "P1",
     "symptoms": "Severe diffuse abdominal pain 9/10. Rigid abdomen, vomiting x5, 4 days constipation.",
     "history": {"age": 58, "gender": "Male", "conditions": ["Peptic ulcer disease"]},
     "labs": {"troponin_ngL": 18, "bp_systolic": 88, "wbc": 22000, "oxygen_sat": 95, "glucose_mgdL": 130, "rr": 26}},

    {"patient_id": "PT006", "ground_truth": "P3",
     "symptoms": "Right knee pain after bicycle fall. Swelling and bruising. Partial weight-bearing. GCS 15.",
     "history": {"age": 28, "gender": "Male", "conditions": []},
     "labs": {"troponin_ngL": 12, "bp_systolic": 120, "wbc": 9200, "oxygen_sat": 99, "glucose_mgdL": 95, "rr": 16}},

    {"patient_id": "PT007", "ground_truth": "P1",
     "symptoms": "Diabetic, no insulin for 2 days. Polyuria, polydipsia, confusion, fruity breath, Kussmaul breathing.",
     "history": {"age": 23, "gender": "Female", "conditions": ["Type 1 Diabetes"]},
     "labs": {"troponin_ngL": 20, "bp_systolic": 98, "wbc": 14500, "oxygen_sat": 94, "glucose_mgdL": 480, "rr": 32}},

    {"patient_id": "PT008", "ground_truth": "P4",
     "symptoms": "Mild sore throat and runny nose for 3 days. Low-grade fever 37.5°C. Wants antibiotic.",
     "history": {"age": 31, "gender": "Female", "conditions": []},
     "labs": {"troponin_ngL": 10, "bp_systolic": 118, "wbc": 9800, "oxygen_sat": 99, "glucose_mgdL": 92, "rr": 15}},

    {"patient_id": "PT009", "ground_truth": "P2",
     "symptoms": "Elderly male found on floor, confused, disoriented. Temp 38.9°C. Urinary incontinence.",
     "history": {"age": 82, "gender": "Male", "conditions": ["Dementia", "Hypertension", "CKD stage 3"]},
     "labs": {"troponin_ngL": 38, "bp_systolic": 155, "wbc": 17800, "oxygen_sat": 93, "glucose_mgdL": 118, "rr": 21}},

    {"patient_id": "PT010", "ground_truth": "P3",
     "symptoms": "Lower back pain radiating down right leg, 2 weeks after lifting. Pain 5/10, no bowel/bladder changes.",
     "history": {"age": 45, "gender": "Male", "conditions": ["Obesity"]},
     "labs": {"troponin_ngL": 11, "bp_systolic": 132, "wbc": 7600, "oxygen_sat": 98, "glucose_mgdL": 108, "rr": 16}},
]

LAB_RANGES = [
    {"parameter": "Troponin I",   "unit": "ng/L",        "normal": "< 40",      "critical": "≥ 400 → P1"},
    {"parameter": "WBC",          "unit": "cells/μL",    "normal": "4000–10000","critical": "> 15000 sepsis"},
    {"parameter": "BP Systolic",  "unit": "mmHg",        "normal": "90–139",    "critical": "< 90 or ≥ 180"},
    {"parameter": "SpO2",         "unit": "%",           "normal": "≥ 95",      "critical": "< 88 → P1"},
    {"parameter": "Glucose",      "unit": "mg/dL",       "normal": "70–140",    "critical": "> 300 DKA"},
    {"parameter": "Resp Rate",    "unit": "breaths/min", "normal": "12–20",     "critical": "> 30 distress"},
]

def generate():
    os.makedirs(DIR, exist_ok=True)
    for rec in RECORDS:
        path = os.path.join(DIR, f"{rec['patient_id']}.json")
        json.dump(rec, open(path, "w"), indent=2)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LAB_RANGES[0].keys())
        w.writeheader(); w.writerows(LAB_RANGES)
    print(f"[Data] {len(RECORDS)} patient files + lab_reference.csv created.")

if __name__ == "__main__":
    generate()
