# Agentic-AI-Capstone

MediRoute is an Agentic AI-powered medical triage system that combines Retrieval-Augmented Generation (RAG) and AI agents to analyze symptoms, retrieve relevant medical knowledge, assess urgency, and recommend appropriate healthcare pathways. Built using LLMs, vector databases, and multi-agent workflows.

# MediRoute — AI Triage Assistant

Multi-agent LangGraph system using **Groq** (cloud LLM) + **ChromaDB** (RAG + memory).

## Project structure

```
mediroute2/
├── core/
│   ├── llm.py          ← ChatGroq instance (used by every worker)
│   ├── settings.py     ← all config / env vars
│   └── state.py        ← TriageState TypedDict
├── pipeline/
│   ├── graph.py        ← LangGraph workflow + routing
│   └── rag.py          ← ChromaDB RAG ingest + retrieve
├── workers/
│   ├── researcher.py   ← node: extract clinical findings
│   ├── writer.py       ← node: produce triage JSON
│   └── evaluator.py    ← node: confidence gate
├── store/
│   └── memory.py       ← STM (dict) + LTM (ChromaDB)
├── data/
│   └── patients.py     ← 10 synthetic patients + lab CSV
├── tests/
│   └── tests.py        ← full test suite (all 4 criteria)
├── run.py              ← main entry point
├── .env.example        ← copy to .env and add your key
└── requirements.txt
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free Groq API key at https://console.groq.com
# 3. Create your .env file
cp .env.example .env
# edit .env → add GROQ_API_KEY=your_key_here

# 4. Run all 10 patients
python run.py

# 5. Run a single patient
python run.py --patient PT001

# 6. Run tests
python tests/tests.py
```

## How the LLM is loaded

```python
# core/llm.py — this is the entire LLM setup
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # reads .env file

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),   # reads key from .env
    model_name="llama-3.3-70b-versatile", # model on Groq's servers
    temperature=0,                         # 0 = deterministic output
)
```

Every worker imports it as: `from core.llm import llm`
Then calls it as: `llm.invoke([("system","..."), ("human","...")])`.

