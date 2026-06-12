"""
RAG pipeline.
ingest() loads triage_protocol.txt into ChromaDB.
retrieve() fetches the most relevant chunks for a query.
"""
import os, re
from typing import List
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from core.settings import CHROMA_DIR, RAG_COLLECTION, EMBED_MODEL, PROTOCOL_FILE

_ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

def _collection():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR).get_or_create_collection(
        RAG_COLLECTION, embedding_function=_ef
    )

def _chunks(text: str, size=400, overlap=80) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) < size:
            buf = (buf + " " + p).strip()
        else:
            if buf: out.append(buf)
            buf = (buf[-overlap:] + " " + p).strip()
    if buf: out.append(buf)
    return out

def ingest(force=False):
    col = _collection()
    if not force and col.count() > 0:
        return col.count()
    if not os.path.exists(PROTOCOL_FILE):
        _write_default_protocol()
    text   = open(PROTOCOL_FILE).read()
    chunks = _chunks(text)
    col.upsert(
        ids       = [f"c{i}" for i in range(len(chunks))],
        documents = chunks,
        metadatas = [{"src": "protocol", "i": i} for i in range(len(chunks))],
    )
    return len(chunks)

def retrieve(query: str, n=3) -> str:
    col = _collection()
    if col.count() == 0:
        ingest()
    r = col.query(query_texts=[query], n_results=min(n, col.count()))
    chunks = r["documents"][0] if r["documents"] else []
    return "\n\n".join(f"[Excerpt {i+1}] {c}" for i, c in enumerate(chunks))

def _write_default_protocol():
    os.makedirs(os.path.dirname(PROTOCOL_FILE), exist_ok=True)
    open(PROTOCOL_FILE, "w").write("""
TRIAGE PROTOCOL – ED v4.2

P1 – IMMEDIATE (life-threatening)
- Cardiac arrest / pulselessness
- SpO2 < 88%, severe respiratory distress (RR > 30)
- Troponin ≥ 400 ng/L with chest pain
- Systolic BP < 70 or > 220 with neuro symptoms
- Active stroke (FAST positive within 4 h)
- GCS < 13, altered consciousness
- Anaphylaxis with airway involvement
- DKA: glucose > 400, Kussmaul breathing

P2 – URGENT
- Chest pain with troponin 100–399 ng/L
- SpO2 88–93%, moderate dyspnoea
- Systolic BP 70–90 or 180–220
- Fever > 39.5 °C with focal source
- WBC > 15 000 suggesting sepsis
- New confusion in elderly (age > 65)
- Severe pain > 7/10, unexplained

P3 – LESS URGENT
- Pain 4–6/10, stable vitals
- Fever < 39 °C, immunocompetent
- Minor trauma, non-neurovascular fractures
- Mild COPD/asthma exacerbation

P4 – NON-URGENT
- Mild viral illness, runny nose, sore throat
- Medication refills, routine follow-ups
- Minor superficial wounds

LAB CRITICAL VALUES
Troponin I: normal < 40 ng/L | high ≥ 400 ng/L → P1
WBC: normal 4 000–10 000 | sepsis risk > 15 000
SpO2: normal ≥ 95 % | critical < 88 %
BP systolic: shock < 90 | crisis ≥ 180
Glucose: hypo < 70 | DKA risk > 300 mg/dL
RR: normal 12–20 | distress > 30

CONFIDENCE RULE
Reduce confidence by 0.15 for each missing critical field.
Pause for human review when confidence < 0.60.
""")
