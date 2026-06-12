"""
MediRoute LangGraph workflow.
Nodes: researcher → writer → evaluator → supervisor
Conditional edges handle retry, pause, and graceful failure.
"""
from langgraph.graph import StateGraph, END
from core.state       import TriageState
from core.settings    import MAX_RETRIES
from workers.researcher import researcher_node
from workers.writer     import writer_node
from workers.evaluator  import evaluator_node


# ── Supervisor ────────────────────────────────────────────────────

def supervisor_node(state: TriageState) -> dict:
    """Decides: retry, graceful-fail, or pass through."""
    if state.get("paused"):
        return state
    if state.get("error") and not state.get("priority"):
        if state["retry_count"] < MAX_RETRIES:
            print(f"[Supervisor] retry {state['retry_count']}/{MAX_RETRIES}")
            return {**state, "retry_count": state["retry_count"] + 1}
        # Graceful failure – safe default
        print("[Supervisor] max retries – graceful failure")
        return {**state, "paused": True,
                "priority": "P2", "confidence": 0.0,
                "reasoning": (f"AI triage failed after {MAX_RETRIES} retries "
                              f"({state['error']}). Defaulting to P2 – human review required."),
                "alert_flags": ["AI_FAILURE", "HUMAN_REVIEW_REQUIRED"]}
    return state


# ── Routing functions (conditional edges) ────────────────────────

def after_researcher(s):
    return "supervisor" if s.get("error") else "writer"

def after_writer(s):
    return "supervisor" if (s.get("error") or not s.get("priority")) else "evaluator"

def after_evaluator(s):
    return "paused_end" if s.get("paused") else "ok_end"

def after_supervisor(s):
    if s.get("paused"):          return "paused_end"
    if s["retry_count"] >= MAX_RETRIES: return "paused_end"
    return "researcher"


# ── Build graph ───────────────────────────────────────────────────

def build_graph():
    g = StateGraph(TriageState)

    g.add_node("researcher", researcher_node)
    g.add_node("writer",     writer_node)
    g.add_node("evaluator",  evaluator_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("ok_end",     lambda s: s)
    g.add_node("paused_end", lambda s: s)

    g.set_entry_point("researcher")

    g.add_conditional_edges("researcher", after_researcher,
                             {"writer": "writer", "supervisor": "supervisor"})
    g.add_conditional_edges("writer",     after_writer,
                             {"evaluator": "evaluator", "supervisor": "supervisor"})
    g.add_conditional_edges("evaluator",  after_evaluator,
                             {"ok_end": "ok_end", "paused_end": "paused_end"})
    g.add_conditional_edges("supervisor", after_supervisor,
                             {"researcher": "researcher", "paused_end": "paused_end"})

    g.add_edge("ok_end",     END)
    g.add_edge("paused_end", END)

    return g.compile()
