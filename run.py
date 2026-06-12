"""
MediRoute – main runner.

Usage:
    python run.py                   # all 10 patients
    python run.py --patient PT001   # single patient
"""
import argparse, csv, json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.rag    import ingest
from pipeline.graph  import build_graph
from store.memory    import ltm_save, ltm_count
from data.patients   import generate, RECORDS
from core.settings   import PATIENTS_DIR, RESULTS_CSV, PRIORITIES


def ensure_setup():
    if not os.listdir(PATIENTS_DIR) if os.path.exists(PATIENTS_DIR) else True:
        generate()
    ingest()


def make_initial_state(rec: dict) -> dict:
    return {
        "patient_id":  rec["patient_id"],
        "symptoms":    rec["symptoms"],
        "history":     rec.get("history", {}),
        "labs":        rec.get("labs", {}),
        "rag_context": None,
        "findings":    None,
        "ltm_hit":     None,
        "retry_count": 0,
        "paused":      False,
        "error":       None,
        "priority":    None,
        "confidence":  None,
        "reasoning":   None,
        "alert_flags": [],
    }


def run_patient(graph, rec: dict) -> dict:
    pid = rec["patient_id"]
    print(f"\n{'='*55}\n  {pid} | {rec['symptoms'][:60]}…\n{'='*55}")
    t0    = time.time()
    state = graph.invoke(make_initial_state(rec))
    elapsed = time.time() - t0

    gt      = rec.get("ground_truth", "?")
    correct = "✓" if state.get("priority") == gt else "✗"
    flag    = "⏸ PAUSED" if state.get("paused") else ""

    print(f"  Result   : {state.get('priority')} – {PRIORITIES.get(state.get('priority',''),'?')} {flag}")
    print(f"  Expected : {gt}  {correct}")
    print(f"  Confidence: {state.get('confidence', 0):.2f}  |  {elapsed:.1f}s")
    print(f"  Reasoning: {(state.get('reasoning') or '')[:120]}")

    # Save to LTM
    if state.get("priority") and state.get("confidence", 0) > 0:
        ltm_save(pid, rec["symptoms"], state["priority"],
                 state["confidence"], state.get("reasoning", ""))

    return state


def save_csv(results: list, records: list):
    gt_map = {r["patient_id"]: r.get("ground_truth","?") for r in records}
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient_id","priority","confidence","paused","correct","flags","reasoning"])
        w.writeheader()
        for s in results:
            pid = s["patient_id"]
            w.writerow({
                "patient_id": pid,
                "priority":   s.get("priority",""),
                "confidence": round(s.get("confidence") or 0, 2),
                "paused":     s.get("paused", False),
                "correct":    s.get("priority") == gt_map.get(pid),
                "flags":      "|".join(s.get("alert_flags", [])),
                "reasoning":  (s.get("reasoning") or "")[:200],
            })
    print(f"\n[Output] Results → {RESULTS_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", help="Run a single patient by ID")
    args = parser.parse_args()

    ensure_setup()
    graph   = build_graph()
    records = [r for r in RECORDS if not args.patient or r["patient_id"] == args.patient]

    results = [run_patient(graph, r) for r in records]

    if len(results) > 1:
        save_csv(results, records)

    correct = sum(1 for s, r in zip(results, records) if s.get("priority") == r.get("ground_truth"))
    paused  = sum(1 for s in results if s.get("paused"))
    print(f"\n{'─'*40}")
    print(f"  {len(results)} patients | {correct}/{len(results)} correct | {paused} paused")
    print(f"  LTM cases stored: {ltm_count()}")
    print(f"{'─'*40}")


if __name__ == "__main__":
    main()
