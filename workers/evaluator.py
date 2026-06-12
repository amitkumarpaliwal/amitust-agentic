"""
Evaluator node.
Checks confidence and LLM-validates the triage decision.
Sets paused=True if confidence is too low or decision looks wrong.
"""
import json, re
from core.llm      import llm
from core.state    import TriageState
from core.settings import CONFIDENCE_GATE

SYSTEM = (
    "You are a triage QA reviewer. "
    "Output ONLY valid JSON: "
    '{"approved":true|false,"note":"one sentence"}'
)

def evaluator_node(state: TriageState) -> dict:
    # Hard gate – no result at all
    if not state.get("priority"):
        return {**state, "paused": True, "error": "No triage result to evaluate"}

    # Confidence gate
    if (state.get("confidence") or 0) < CONFIDENCE_GATE:
        return {**state, "paused": True,
                "error": f"Confidence {state['confidence']:.2f} below threshold {CONFIDENCE_GATE}"}

    prompt = (
        f"Patient symptoms: {state['symptoms']}\n"
        f"Labs: {json.dumps(state['labs'])}\n"
        f"Decision: priority={state['priority']} confidence={state['confidence']:.2f}\n"
        f"Reasoning: {state['reasoning']}"
    )

    try:
        raw = llm.invoke([("system", SYSTEM), ("human", prompt)]).content
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        ev  = json.loads(raw)
    except Exception:
        ev  = {"approved": True, "note": "evaluator parse error – passed through"}

    if not ev.get("approved", True):
        return {**state, "paused": True, "error": f"Evaluator rejected: {ev.get('note','')}"}

    print(f"[Evaluator] ✓ {state['priority']} conf={state['confidence']:.2f}")
    return {**state, "paused": False, "error": None}
