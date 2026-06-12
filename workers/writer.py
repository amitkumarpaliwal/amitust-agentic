"""
Writer node.
Turns the researcher's findings into a structured triage JSON report.
"""
import json, re
from core.llm   import llm
from core.state import TriageState
from core.settings import PRIORITIES

SYSTEM = (
    "You are a senior triage nurse. "
    "Output ONLY a single valid JSON object — no preamble, no markdown fences.\n"
    'Schema: {"priority":"P1|P2|P3|P4","confidence":0.0-1.0,'
    '"reasoning":"2-3 sentences","alert_flags":["..."]}'
)

def writer_node(state: TriageState) -> dict:
    prompt = (
        f"FINDINGS:\n{state.get('findings','(none)')}\n\n"
        f"PROTOCOL CONTEXT:\n{state.get('rag_context','(none)')}\n\n"
        f"PRIOR CASE HINT:\n{state.get('ltm_hit','None')}"
    )

    try:
        raw = llm.invoke([("system", SYSTEM), ("human", prompt)]).content
        # Strip accidental markdown fences
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        d   = json.loads(raw)
    except Exception as e:
        return {**state, "error": f"Writer parse error: {e}", "retry_count": state["retry_count"] + 1}

    priority = d.get("priority", "P3").upper()
    if priority not in PRIORITIES:
        priority = "P3"

    return {
        **state,
        "priority":    priority,
        "confidence":  float(d.get("confidence", 0.5)),
        "reasoning":   d.get("reasoning", ""),
        "alert_flags": d.get("alert_flags", []),
        "error":       None,
    }
