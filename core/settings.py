import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM (Groq cloud) ──────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_MODEL    = "llama-3.3-70b-versatile"

# ── Embeddings (offline sentence-transformers) ────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"

# ── ChromaDB paths ────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
CHROMA_DIR    = os.path.join(BASE_DIR, "..", "store", "chroma_db")
RAG_COLLECTION  = "triage_protocols"
LTM_COLLECTION  = "triage_ltm"

# ── Data paths ────────────────────────────────────────────────────
DATA_DIR      = os.path.join(BASE_DIR, "..", "data")
PATIENTS_DIR  = os.path.join(DATA_DIR, "patients")
PROTOCOL_FILE = os.path.join(DATA_DIR, "triage_protocol.txt")
RESULTS_CSV   = os.path.join(DATA_DIR, "results.csv")

# ── Triage control ────────────────────────────────────────────────
CONFIDENCE_GATE = 0.60
MAX_RETRIES     = 3
PRIORITIES      = {"P1": "Immediate", "P2": "Urgent", "P3": "Less Urgent", "P4": "Non-Urgent"}
