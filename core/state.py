from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class TriageState(TypedDict):
    # ── Input ─────────────────────────────────────────────
    patient_id:     str
    symptoms:       str
    history:        Dict[str, Any]
    labs:           Dict[str, Any]

    # ── Intermediate ──────────────────────────────────────
    rag_context:    Optional[str]
    findings:       Optional[str]   # researcher output
    ltm_hit:        Optional[str]   # similar prior case from ChromaDB

    # ── Control ───────────────────────────────────────────
    retry_count:    int
    paused:         bool
    error:          Optional[str]

    # ── Output ────────────────────────────────────────────
    priority:       Optional[str]   # P1-P4
    confidence:     Optional[float]
    reasoning:      Optional[str]
    alert_flags:    List[str]
