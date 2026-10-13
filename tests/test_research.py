"""Offline contract tests. No API credits, embedding downloads or corpus writes."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from api.main import app
from generation.generator import configured_models, stream_answer


def evidence(query, limit):
    return [SimpleNamespace(payload={"source": "data/raw/example.pdf", "page": 8, "text": "A sample provision."}, score=0.81)]


class ResearchTests(unittest.TestCase):
    def test_fallback_events_and_answer_share_evidence(self):
        error = Exception("private provider details")
        error.code = 429
        client = SimpleNamespace(models=SimpleNamespace(generate_content=Mock(side_effect=[error, SimpleNamespace(text="Answer [1]")])))
        events = list(stream_answer("Question", search_fn=evidence, client=client, models=["first-model", "second-model"]))
        self.assertEqual([e["type"] for e in events], ["retrieval_started", "retrieval_complete", "model_started", "model_failed", "model_started", "result"])
        self.assertEqual(events[3]["next_model"], "second-model")
        self.assertEqual(events[3]["reason"], "Rate limit or quota reached")
        self.assertNotIn("private provider", json.dumps(events))
        self.assertEqual(events[-1]["model"], "second-model")
        self.assertEqual(events[-1]["sources"][0]["document"], "example.pdf")
        self.assertEqual(events[-1]["sources"][0]["excerpt"], "A sample provision.")

    def test_empty_evidence_never_calls_model(self):
        client = Mock()
        events = list(stream_answer("Question", search_fn=lambda q, n: [], client=client))
        self.assertIsNone(events[-1]["model"])
        client.models.generate_content.assert_not_called()

    def test_empty_answer_falls_back_and_exhaustion_raises(self):
        client = SimpleNamespace(models=SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(text=""))))
        iterator = stream_answer("Question", search_fn=evidence, client=client, models=["first"])
        with self.assertRaisesRegex(RuntimeError, "All configured models"):
            list(iterator)
        self.assertEqual(client.models.generate_content.call_count, 1)

    def test_configured_models_are_deduplicated_and_bounded(self):
        with patch.dict("os.environ", {"GEMINI_MODELS": " models/a, a,b,c,d,e,f "}):
            self.assertEqual(configured_models(), ["a", "b", "c", "d", "e", "f"])

    def test_validation(self):
        with TestClient(app) as api:
            for question in ["   ", "a", "x" * 2001]:
                self.assertEqual(api.post("/ask/stream", json={"question": question}).status_code, 422)

    def test_streaming_error_is_sanitized(self):
        with patch("api.main.stream_answer", side_effect=ValueError("secret provider response")):
            with TestClient(app) as api:
                response = api.post("/ask/stream", json={"question": "Valid question"})
        self.assertIn("application/x-ndjson", response.headers["content-type"])
        self.assertEqual(response.json()["type"], "error")
        self.assertNotIn("secret provider", response.text)

    def test_unexpected_runtime_errors_are_sanitized(self):
        with patch("api.main.stream_answer", side_effect=RuntimeError("private database path")):
            with TestClient(app) as api:
                response = api.post("/ask/stream", json={"question": "Valid question"})
        self.assertNotIn("private database", response.text)

    def test_missing_key_fails_before_loading_retrieval(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                list(stream_answer("Question"))

    def test_stream_contract_and_trim(self):
        events = [{"type": "model_started", "model": "test-model"}, {"type": "result", "answer": "Evidence [1]"}]
        with patch("api.main.stream_answer", return_value=iter(events)) as generate:
            with TestClient(app) as api:
                response = api.post("/ask/stream", json={"question": "  A question  "})
            generate.assert_called_once_with("A question")
        self.assertEqual([json.loads(line) for line in response.text.splitlines()], events)
        self.assertEqual(response.headers["x-accel-buffering"], "no")

    def test_legacy_endpoint_and_health(self):
        answer = {"question": "Question", "answer": "Answer", "sources": []}
        with patch("api.main.generate_answer", return_value=answer):
            with TestClient(app) as api:
                self.assertEqual(api.post("/ask", json={"question": "Question"}).json(), answer)
                self.assertEqual(api.get("/health").json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
