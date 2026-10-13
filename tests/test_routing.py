import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from generation import routing
from generation.generator import stream_answer


def catalog_model(name, actions=None):
    return SimpleNamespace(name="models/" + name, supported_actions=actions or ["generateContent"])


def fake_client(names):
    return SimpleNamespace(models=SimpleNamespace(
        list=Mock(return_value=SimpleNamespace(page=[catalog_model(name) for name in names])),
        generate_content=Mock(return_value=SimpleNamespace(text="Grounded answer [1]")),
    ))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        routing._catalog.clear()
        routing._cooldowns.clear()
        routing._winners.clear()

    def test_discovery_filters_specialized_models_and_caches(self):
        client = fake_client(["gemini-good-flash", "gemini-good-pro", "gemini-flash-image", "gemini-flash-tts", "gemini-embedding-001", "gemini-omni-flash", "gemini-robotics", "gemini-flash-live", "gemini-nano-banana"])
        first, cached, ok = routing.discover_models(client)
        self.assertEqual(first, ["gemini-good-flash", "gemini-good-pro"])
        self.assertFalse(cached)
        self.assertTrue(ok)
        self.assertTrue(routing.discover_models(client)[1])
        client.models.list.assert_called_once()

    def test_models_without_generate_content_are_excluded(self):
        client = fake_client([])
        client.models.list.return_value.page = [catalog_model("gemini-test", ["countTokens"])]
        self.assertEqual(routing.discover_models(client)[0], [])

    def test_catalog_expiry_refreshes(self):
        client = fake_client(["gemini-flash"])
        with patch("generation.routing.monotonic", return_value=0):
            routing.discover_models(client)
        with patch("generation.routing.monotonic", return_value=901):
            routing.discover_models(client)
        self.assertEqual(client.models.list.call_count, 2)

    def test_discovery_failure_keeps_extra_fallbacks(self):
        client = fake_client([])
        client.models.list.side_effect = TimeoutError()
        chain, _, _, ok = routing.model_plan(client, ["preferred"])
        self.assertFalse(ok)
        self.assertEqual(chain[0], "preferred")
        self.assertGreater(len(chain), 3)

    def test_preferred_order_deduplication_and_cap(self):
        client = fake_client([f"gemini-{i}-flash" for i in range(30)])
        chain, _, _, _ = routing.model_plan(client, ["gemini-3-flash", "gemini-3-flash"])
        self.assertEqual(len(chain), 16)
        self.assertEqual(chain[0], "gemini-3-flash")
        self.assertEqual(len(chain), len(set(chain)))

    def test_cooldown_skips_then_retries_failed_model(self):
        client = fake_client(["gemini-failed", "gemini-good"])
        error = SimpleNamespace(code=429)
        with patch("generation.routing.monotonic", return_value=100):
            routing.record_failure("gemini-failed", error, "Rate limit reached")
            chain, skipped, _, _ = routing.model_plan(client, [])
        self.assertEqual(chain, ["gemini-good"])
        self.assertEqual(skipped[0]["model"], "gemini-failed")
        with patch("generation.routing.monotonic", return_value=161):
            self.assertIn("gemini-failed", routing.model_plan(client, [])[0])

    def test_previous_winner_is_tried_first(self):
        client = fake_client(["gemini-good"])
        routing.record_success("gemini-good")
        chain, _, _, _ = routing.model_plan(client, ["preferred"])
        self.assertEqual(chain, ["gemini-good", "preferred"])

    def test_account_state_is_isolated(self):
        client = fake_client(["gemini-good"])
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-account-a"}):
            routing.discover_models(client)
            routing.record_failure("gemini-good", SimpleNamespace(code=429), "Quota")
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-account-b"}):
            chain, _, cached, _ = routing.model_plan(client, [])
        self.assertFalse(cached)
        self.assertEqual(chain, ["gemini-good"])

    def test_fourth_model_succeeds_after_first_three_fail(self):
        client = fake_client(["gemini-fourth-flash"])
        unavailable = Exception("private provider payload")
        unavailable.code = 404
        client.models.generate_content.side_effect = [unavailable, unavailable, unavailable, SimpleNamespace(text="Answer [1]")]
        evidence = lambda query, limit: [SimpleNamespace(payload={"source": "law.pdf", "page": 1, "text": "Evidence"}, score=.8)]
        with patch.dict("os.environ", {"GEMINI_MODELS": "first,second,third"}):
            events = list(stream_answer("A question", client=client, search_fn=evidence))
        self.assertEqual([event["model"] for event in events if event["type"] == "model_started"], ["first", "second", "third", "gemini-fourth-flash"])
        self.assertEqual(events[-1]["model"], "gemini-fourth-flash")
        self.assertEqual(events[-1]["type"], "result")


if __name__ == "__main__":
    unittest.main()
