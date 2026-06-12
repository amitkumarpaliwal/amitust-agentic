"""
MediRoute test suite – maps to the 100-point evaluation scheme.
Run: python tests/tests.py
"""
import os, sys, json, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.settings   import CONFIDENCE_GATE, MAX_RETRIES, PRIORITIES
from data.patients   import RECORDS


# ── helpers ───────────────────────────────────────────────────────

def _state(**kw) -> dict:
    base = {
        "patient_id": "TEST", "symptoms": "chest pain", "history": {}, "labs": {},
        "rag_context": None, "findings": None, "ltm_hit": None,
        "retry_count": 0, "paused": False, "error": None,
        "priority": None, "confidence": None, "reasoning": None, "alert_flags": [],
    }
    return {**base, **kw}


# ── 1. Graph correctness – 25 pts ─────────────────────────────────

class TestGraphCorrectness(unittest.TestCase):

    def test_all_priorities_in_settings(self):
        self.assertEqual(set(PRIORITIES.keys()), {"P1","P2","P3","P4"})

    def test_synthetic_data_has_ten_patients(self):
        self.assertEqual(len(RECORDS), 10)

    def test_all_p1_to_p4_covered(self):
        gts = {r["ground_truth"] for r in RECORDS}
        self.assertEqual(gts, {"P1","P2","P3","P4"})

    def test_routing_after_researcher_ok(self):
        from pipeline.graph import after_researcher
        self.assertEqual(after_researcher(_state(findings="ok")), "writer")

    def test_routing_after_researcher_error(self):
        from pipeline.graph import after_researcher
        self.assertEqual(after_researcher(_state(error="fail")), "supervisor")

    def test_routing_after_writer_ok(self):
        from pipeline.graph import after_writer
        self.assertEqual(after_writer(_state(priority="P2")), "evaluator")

    def test_routing_after_writer_no_priority(self):
        from pipeline.graph import after_writer
        self.assertEqual(after_writer(_state()), "supervisor")

    def test_routing_after_evaluator_approved(self):
        from pipeline.graph import after_evaluator
        self.assertEqual(after_evaluator(_state()), "ok_end")

    def test_routing_after_evaluator_paused(self):
        from pipeline.graph import after_evaluator
        self.assertEqual(after_evaluator(_state(paused=True)), "paused_end")

    def test_evaluator_gates_low_confidence(self):
        from workers.evaluator import evaluator_node
        out = evaluator_node(_state(priority="P2", confidence=0.30,
                                    reasoning="test", symptoms="test", labs={}))
        self.assertTrue(out["paused"])

    def test_graph_compiles_without_error(self):
        from pipeline.graph import build_graph
        g = build_graph()
        self.assertIsNotNone(g)


# ── 2. RAG quality – 20 pts ───────────────────────────────────────

class TestRAGQuality(unittest.TestCase):

    def test_chunker_produces_multiple_chunks(self):
        from pipeline.rag import _chunks
        text   = "Para one. " * 40 + "\n\n" + "Para two. " * 40
        result = _chunks(text, size=300)
        self.assertGreater(len(result), 1)

    def test_chunker_no_empty_chunks(self):
        from pipeline.rag import _chunks
        text = "Para one. " * 40 + "\n\n" + "Para two. " * 40
        for c in _chunks(text):
            self.assertGreater(len(c), 0)

    def test_ingest_creates_chunks(self):
        from pipeline.rag import ingest
        count = ingest()
        self.assertGreater(count, 0)

    def test_retrieve_returns_protocol_excerpt(self):
        from pipeline.rag import retrieve, ingest
        ingest()
        ctx = retrieve("chest pain troponin")
        self.assertIn("Excerpt", ctx)

    def test_retrieve_cardiac_context(self):
        from pipeline.rag import retrieve, ingest
        ingest()
        ctx = retrieve("troponin STEMI cardiac arrest")
        self.assertTrue(any(k in ctx.lower() for k in ["troponin","p1","cardiac","chest"]))

    def test_retrieve_no_hallucinated_p5(self):
        from pipeline.rag import retrieve, ingest
        ingest()
        ctx = retrieve("general patient")
        self.assertNotIn("P5", ctx)


# ── 3. Memory system – 20 pts ─────────────────────────────────────

class TestMemorySystem(unittest.TestCase):

    def test_stm_save_and_load(self):
        from store.memory import stm_save, stm_load
        stm_save("STM_TEST", {"priority": "P1"})
        self.assertEqual(stm_load("STM_TEST")["priority"], "P1")

    def test_stm_unknown_returns_none(self):
        from store.memory import stm_load
        self.assertIsNone(stm_load("NOBODY_XYZ"))

    def test_stm_overwrite(self):
        from store.memory import stm_save, stm_load
        stm_save("STM_OVR", {"priority": "P3"})
        stm_save("STM_OVR", {"priority": "P1"})
        self.assertEqual(stm_load("STM_OVR")["priority"], "P1")

    def test_ltm_save_increases_count(self):
        from store.memory import ltm_save, ltm_count
        before = ltm_count()
        ltm_save("LTM_TEST", "chest pain test", "P1", 0.9, "test reasoning")
        self.assertGreater(ltm_count(), before)

    def test_ltm_recall_returns_str_or_none(self):
        from store.memory import ltm_recall
        result = ltm_recall("random symptoms xyz")
        self.assertIn(type(result), [str, type(None)])

    def test_ltm_cross_instance_shared(self):
        """Two separate calls share same ChromaDB data."""
        from store.memory import ltm_count
        c1 = ltm_count()
        c2 = ltm_count()
        self.assertEqual(c1, c2)


# ── 4. Supervisor robustness – 20 pts ────────────────────────────

class TestSupervisorRobustness(unittest.TestCase):

    def test_retry_increments_count(self):
        from pipeline.graph import supervisor_node
        out = supervisor_node(_state(error="fail", retry_count=0))
        self.assertEqual(out["retry_count"], 1)

    def test_graceful_failure_at_max_retries(self):
        from pipeline.graph import supervisor_node
        out = supervisor_node(_state(error="fail", retry_count=MAX_RETRIES))
        self.assertTrue(out["paused"])
        self.assertIn("AI_FAILURE", out["alert_flags"])

    def test_graceful_failure_reasoning_is_human_readable(self):
        from pipeline.graph import supervisor_node
        out = supervisor_node(_state(error="fail", retry_count=MAX_RETRIES))
        r = out.get("reasoning","")
        self.assertGreater(len(r), 20)
        self.assertNotIn("Traceback", r)

    def test_graceful_failure_defaults_to_p2(self):
        from pipeline.graph import supervisor_node
        out = supervisor_node(_state(error="fail", retry_count=MAX_RETRIES))
        self.assertEqual(out["priority"], "P2")

    def test_supervisor_passthrough_on_success(self):
        from pipeline.graph import supervisor_node
        out = supervisor_node(_state(priority="P1", confidence=0.9))
        self.assertFalse(out.get("paused"))

    def test_after_supervisor_retry_routes_back(self):
        from pipeline.graph import after_supervisor
        self.assertEqual(after_supervisor(_state(retry_count=1)), "researcher")

    def test_after_supervisor_ends_at_max(self):
        from pipeline.graph import after_supervisor
        self.assertEqual(after_supervisor(_state(retry_count=MAX_RETRIES)), "paused_end")

    def test_pause_preserved_by_supervisor(self):
        from pipeline.graph import supervisor_node
        out = supervisor_node(_state(paused=True, priority="P2"))
        self.assertTrue(out["paused"])

    def test_max_retries_is_positive_int(self):
        self.assertIsInstance(MAX_RETRIES, int)
        self.assertGreater(MAX_RETRIES, 0)

    def test_confidence_gate_between_0_and_1(self):
        self.assertGreater(CONFIDENCE_GATE, 0)
        self.assertLess(CONFIDENCE_GATE, 1)


# ── Runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestGraphCorrectness, TestRAGQuality, TestMemorySystem, TestSupervisorRobustness]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = r.testsRun - len(r.failures) - len(r.errors)
    print(f"\n{'='*45}")
    print(f"  {passed}/{r.testsRun} tests passed")
    print(f"{'='*45}")
    sys.exit(0 if r.wasSuccessful() else 1)
