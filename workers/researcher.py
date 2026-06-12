"""
Researcher node.
Looks up RAG context + memory, then asks the LLM to extract clinical findings.
"""
import json
from core.llm     import llm
from core.state   import TriageState
from pipeline.rag import retrieve
from store.memory import stm_save, stm_load, ltm_recall

SYSTEM = (
    "You are a medical data analyst in an ED. "
    "Extract key clinical findings as a numbered list. "
    "End with: CRITICAL FLAGS: <comma-separated abnormal values or 'None'>"
)

def researcher_node(state: TriageState) -> dict:
    pid = state["patient_id"]

    # STM: restore prior state if same patient re-submitted this session
    prior = stm_load(pid)
    if prior:
        print(f"[Researcher] STM hit for {pid}")

    rag  = retrieve(f"{state['symptoms']} {json.dumps(state['labs'])}")
    ltm  = ltm_recall(state["symptoms"])

    prompt = (
        f"SYMPTOMS:\n{state['symptoms']}\n\n"
        f"HISTORY:\n{json.dumps(state['history'], indent=2)}\n\n"
        f"LABS:\n{json.dumps(state['labs'], indent=2)}\n\n"
        f"PROTOCOL CONTEXT:\n{rag or '(none)'}"
    )

    try:
        findings = llm.invoke([
            ("system", SYSTEM),
            ("human",  prompt),
        ]).content
    except Exception as e:
        return {**state, "error": str(e), "retry_count": state["retry_count"] + 1}

    stm_save(pid, {"symptoms": state["symptoms"], "findings": findings})
    return {**state, "rag_context": rag, "findings": findings, "ltm_hit": ltm, "error": None}
