"""
Short-Term Memory  – plain Python dict, lives for one session.
Long-Term Memory   – ChromaDB persistent collection, survives restarts.
"""
import os, json, time
from typing import Optional
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from core.settings import CHROMA_DIR, LTM_COLLECTION, EMBED_MODEL

# ── shared embedding function ─────────────────────────────────────
_ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

def _ltm_collection():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR).get_or_create_collection(
        LTM_COLLECTION, embedding_function=_ef
    )

# ── Short-Term Memory ─────────────────────────────────────────────
_stm: dict = {}

def stm_save(pid: str, snapshot: dict):
    _stm[pid] = snapshot

def stm_load(pid: str) -> Optional[dict]:
    return _stm.get(pid)

# ── Long-Term Memory ──────────────────────────────────────────────
def ltm_save(pid: str, symptoms: str, priority: str, confidence: float, reasoning: str):
    col = _ltm_collection()
    col.upsert(
        ids       = [f"{pid}_{int(time.time())}"],
        documents = [symptoms],
        metadatas = [{"pid": pid, "priority": priority,
                      "confidence": str(confidence), "reasoning": reasoning[:300]}],
    )

def ltm_recall(symptoms: str, threshold: float = 0.70) -> Optional[str]:
    col = _ltm_collection()
    if col.count() == 0:
        return None
    r = col.query(query_texts=[symptoms], n_results=1)
    if not r["documents"][0]:
        return None
    sim = 1.0 - (r["distances"][0][0] / 2.0)
    if sim < threshold:
        return None
    m = r["metadatas"][0][0]
    return (f"[Prior case – similarity {sim:.2f}] "
            f"Priority={m['priority']} Confidence={m['confidence']}\n"
            f"Reasoning: {m['reasoning']}")

def ltm_count() -> int:
    return _ltm_collection().count()
