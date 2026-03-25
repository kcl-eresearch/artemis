"""Extended API endpoint tests — health, root, streaming, auth, build_summary."""

import json
import time
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi.testclient import TestClient

from artemis.config import Settings
from artemis.errors import UpstreamServiceError
from artemis.main import (
    _build_summary,
    _result_items,
    _message_output,
    app,
    summary_circuit,
)
from artemis.models import (
    DeepResearchRun,
    SearchResult,
    TokenUsage,
)


def _default_settings(**overrides) -> Settings:
    defaults = dict(
        searxng_api_base="http://localhost:8888",
        searxng_timeout_seconds=30.0,
        litellm_base_url="http://localhost:11434/api",
        litellm_api_key=None,
        llm_timeout_seconds=120.0,
        summary_model="qwen3.5:9b",
        summary_max_tokens=2000,
        enable_summary=True,
        deep_research_stages=2,
        deep_research_passes=1,
        deep_research_subqueries=5,
        deep_research_results_per_query=10,
        deep_research_max_tokens=4000,
        deep_research_content_extraction=True,
        deep_research_pages_per_section=3,
        deep_research_content_max_chars=3000,
        shallow_research_stages=1,
        shallow_research_passes=1,
        shallow_research_subqueries=3,
        shallow_research_results_per_query=5,
        shallow_research_max_tokens=2500,
        shallow_research_content_extraction=False,
        shallow_research_pages_per_section=2,
        shallow_research_content_max_chars=1500,
        allowed_origins=tuple(),
        artemis_api_key=None,
        log_level="INFO",
        cache_enabled=True,
        search_cache_ttl_seconds=3600,
        content_cache_ttl_seconds=86400,
        cache_max_entries=1000,
        embedding_model=None,
        semantic_similarity_threshold=0.92,
        log_format="text",
        playwright_context_recycle_pages=50,
        playwright_max_html_bytes=5242880,
        synthesis_tool_rounds=0,
        supervised_research=False,
        researcher_max_tool_rounds=15,
        research_brief_enabled=True,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class RootAndHealthTestCase(unittest.TestCase):
    def setUp(self) -> None:
        summary_circuit.consecutive_failures = 0
        summary_circuit.opened_until = 0.0
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    def test_root_returns_message(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("Artemis", response.json()["message"])

    @patch("artemis.main.httpx.AsyncClient")
    def test_health_returns_status(self, mock_client_cls: MagicMock) -> None:
        mock_resp = MagicMock(status_code=200)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_ctx

        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["status"], ("healthy", "degraded"))
        self.assertIn("checks", body)
        self.assertIn("summary_enabled", body)
        self.assertIn("auth_enabled", body)
        self.assertIn("summary_circuit_open", body)

    def test_health_reports_circuit_open(self) -> None:
        summary_circuit.opened_until = time.time() + 300
        with patch("artemis.main.httpx.AsyncClient") as mock_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_cls.return_value = mock_ctx
            response = self.client.get("/health")
        self.assertTrue(response.json()["summary_circuit_open"])

    def test_health_reports_circuit_closed(self) -> None:
        summary_circuit.opened_until = 0.0
        with patch("artemis.main.httpx.AsyncClient") as mock_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_cls.return_value = mock_ctx
            response = self.client.get("/health")
        self.assertFalse(response.json()["summary_circuit_open"])


class AuthEdgeCasesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    @patch("artemis.main.search_searxng", new_callable=AsyncMock)
    def test_wrong_token_rejected(self, mock_search: AsyncMock) -> None:
        mock_search.return_value = []
        with patch("artemis.main.get_settings", return_value=_default_settings(artemis_api_key="correct")):
            response = self.client.post(
                "/search",
                json={"query": "test"},
                headers={"Authorization": "Bearer wrong-token"},
            )
            self.assertEqual(response.status_code, 401)

    @patch("artemis.main.search_searxng", new_callable=AsyncMock)
    def test_no_auth_header_rejected(self, mock_search: AsyncMock) -> None:
        mock_search.return_value = []
        with patch("artemis.main.get_settings", return_value=_default_settings(artemis_api_key="secret")):
            response = self.client.post("/search", json={"query": "test"})
            self.assertEqual(response.status_code, 401)

    @patch("artemis.main.search_searxng", new_callable=AsyncMock)
    def test_no_api_key_configured_allows_access(self, mock_search: AsyncMock) -> None:
        mock_search.return_value = []
        with patch("artemis.main.get_settings", return_value=_default_settings(artemis_api_key=None)):
            response = self.client.post("/search", json={"query": "test"})
            self.assertEqual(response.status_code, 200)


class ResultItemsHelperTestCase(unittest.TestCase):
    def test_converts_search_results(self) -> None:
        results = [
            SearchResult(title="T1", url="https://a.com", snippet="S1", date="2025-01-01"),
            SearchResult(title="T2", url="https://b.com", snippet="S2"),
        ]
        items = _result_items(results)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].id, 0)
        self.assertEqual(items[0].url, "https://a.com")
        self.assertEqual(items[0].date, "2025-01-01")
        self.assertEqual(items[1].id, 1)
        self.assertIsNone(items[1].date)

    def test_empty_results(self) -> None:
        self.assertEqual(_result_items([]), [])


class MessageOutputHelperTestCase(unittest.TestCase):
    def test_creates_message(self) -> None:
        msg = _message_output("Hello world")
        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.status, "completed")
        self.assertEqual(len(msg.content), 1)
        self.assertEqual(msg.content[0].text, "Hello world")
        self.assertIsNotNone(msg.id)


class BuildSummaryTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        summary_circuit.consecutive_failures = 0
        summary_circuit.opened_until = 0.0

    @patch("artemis.main.summarize_results", new_callable=AsyncMock)
    async def test_returns_summary_on_success(self, mock_summarize: AsyncMock) -> None:
        mock_summarize.return_value = {
            "summary": "Great summary",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        results = [SearchResult(title="T", url="https://a.com", snippet="S")]

        summary, usage, warnings = await _build_summary("test", results)

        self.assertEqual(summary, "Great summary")
        self.assertIsNotNone(usage)
        self.assertEqual(warnings, [])

    async def test_returns_none_when_summary_disabled(self) -> None:
        with patch("artemis.main.get_settings", return_value=_default_settings(enable_summary=False)):
            results = [SearchResult(title="T", url="https://a.com", snippet="S")]
            summary, usage, warnings = await _build_summary("test", results)
            self.assertIsNone(summary)
            self.assertEqual(warnings, [])

    async def test_returns_none_when_no_results(self) -> None:
        summary, usage, warnings = await _build_summary("test", [])
        self.assertIsNone(summary)

    async def test_returns_none_when_circuit_open(self) -> None:
        summary_circuit.opened_until = time.time() + 300
        results = [SearchResult(title="T", url="https://a.com", snippet="S")]
        summary, usage, warnings = await _build_summary("test", results)
        self.assertIsNone(summary)
        self.assertGreater(len(warnings), 0)
        self.assertIn("temporarily disabled", warnings[0])

    @patch("artemis.main.summarize_results", new_callable=AsyncMock)
    async def test_records_failure_on_error(self, mock_summarize: AsyncMock) -> None:
        mock_summarize.side_effect = UpstreamServiceError("LLM down")
        results = [SearchResult(title="T", url="https://a.com", snippet="S")]

        summary, usage, warnings = await _build_summary("test", results)

        self.assertIsNone(summary)
        self.assertGreater(len(warnings), 0)
        self.assertEqual(summary_circuit.consecutive_failures, 1)


class SearchEndpointUsageTestCase(unittest.TestCase):
    """Test that /search enriches usage with search-specific metrics."""

    def setUp(self) -> None:
        summary_circuit.consecutive_failures = 0
        summary_circuit.opened_until = 0.0
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    @patch("artemis.main.summarize_results", new_callable=AsyncMock)
    @patch("artemis.main.search_searxng", new_callable=AsyncMock)
    def test_search_includes_usage_metrics(
        self, mock_search: AsyncMock, mock_summarize: AsyncMock
    ) -> None:
        mock_search.return_value = [
            SearchResult(title="Test Result", url="https://example.com", snippet="A test snippet"),
        ]
        mock_summarize.return_value = {
            "summary": "Summary text",
            "usage": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
        }

        response = self.client.post("/search", json={"query": "test"})
        body = response.json()
        usage = body["usage"]

        self.assertEqual(usage["search_requests"], 1)
        self.assertGreater(usage["citation_tokens"], 0)
        self.assertEqual(usage["prompt_tokens"], 50)
        self.assertEqual(usage["completion_tokens"], 20)


class ResponsesEndpointTestCase(unittest.TestCase):
    def setUp(self) -> None:
        summary_circuit.consecutive_failures = 0
        summary_circuit.opened_until = 0.0
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    @patch("artemis.main.summarize_results", new_callable=AsyncMock)
    @patch("artemis.main.search_searxng", new_callable=AsyncMock)
    def test_fast_search_response_structure(
        self, mock_search: AsyncMock, mock_summarize: AsyncMock
    ) -> None:
        mock_search.return_value = [
            SearchResult(title="R1", url="https://a.com", snippet="S1"),
        ]
        mock_summarize.return_value = {"summary": "Summary", "usage": None}

        response = self.client.post("/v1/responses", json={"input": "test"})
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["model"], "artemis-search")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(len(body["output"]), 2)  # message + search_results
        self.assertEqual(body["output"][0]["type"], "message")
        self.assertEqual(body["output"][1]["type"], "search_results")

    @patch("artemis.main.deep_research", new_callable=AsyncMock)
    def test_deep_research_response_structure(self, mock_dr: AsyncMock) -> None:
        mock_dr.return_value = DeepResearchRun(
            essay="Research essay",
            results=[SearchResult(title="R", url="https://r.com", snippet="S")],
            sub_queries=["q1"],
            stages_completed=2,
            usage=TokenUsage(input_tokens=100, output_tokens=200, total_tokens=300),
        )

        response = self.client.post(
            "/v1/responses", json={"input": "topic", "preset": "deep-research"}
        )
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["model"], "artemis-deep-research")
        self.assertIn("Research essay", body["output"][0]["content"][0]["text"])
        self.assertEqual(body["usage"]["total_tokens"], 300)

    @patch("artemis.main.deep_research", new_callable=AsyncMock)
    def test_shallow_research_response_structure(self, mock_dr: AsyncMock) -> None:
        mock_dr.return_value = DeepResearchRun(
            essay="Shallow research essay",
            results=[SearchResult(title="R", url="https://r.com", snippet="S")],
            sub_queries=["q1"],
            stages_completed=1,
            usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        )

        with patch("artemis.main.get_settings", return_value=_default_settings()):
            response = self.client.post(
                "/v1/responses", json={"input": "topic", "preset": "shallow-research"}
            )
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["model"], "artemis-shallow-research")
        self.assertIn("Shallow research essay", body["output"][0]["content"][0]["text"])
        self.assertEqual(body["usage"]["total_tokens"], 30)


class StreamingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    @patch("artemis.main.summarize_results", new_callable=AsyncMock)
    @patch("artemis.main.search_searxng", new_callable=AsyncMock)
    def test_fast_search_streaming(self, mock_search: AsyncMock, mock_summarize: AsyncMock) -> None:
        mock_search.return_value = [
            SearchResult(title="R1", url="https://a.com", snippet="Snippet text"),
        ]
        mock_summarize.return_value = {"summary": "Streamed summary", "usage": None}

        response = self.client.post(
            "/v1/responses", json={"input": "test", "streaming": True}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")
        body = response.text
        self.assertIn("[Starting research on: test]", body)
        self.assertIn("[Searching...]", body)

    @patch("artemis.main.deep_research", new_callable=AsyncMock)
    def test_deep_research_streaming(self, mock_dr: AsyncMock) -> None:
        mock_dr.return_value = DeepResearchRun(
            essay="Streamed essay content",
            results=[SearchResult(title="R", url="https://r.com", snippet="S")],
            sub_queries=["q1"],
            stages_completed=1,
            usage=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
        )

        response = self.client.post(
            "/v1/responses",
            json={"input": "test topic", "preset": "deep-research", "streaming": True},
        )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("[Starting research on: test topic]", body)
        self.assertIn("Streamed essay content", body)
        self.assertIn("[USAGE]", body)

    @patch("artemis.main.deep_research", new_callable=AsyncMock)
    def test_shallow_research_streaming(self, mock_dr: AsyncMock) -> None:
        mock_dr.return_value = DeepResearchRun(
            essay="Shallow streamed essay",
            results=[SearchResult(title="R", url="https://r.com", snippet="S")],
            sub_queries=["q1"],
            stages_completed=1,
            usage=TokenUsage(input_tokens=5, output_tokens=15, total_tokens=20),
        )

        with patch("artemis.main.get_settings", return_value=_default_settings()):
            response = self.client.post(
                "/v1/responses",
                json={
                    "input": "test topic",
                    "preset": "shallow-research",
                    "streaming": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("[Starting research on: test topic]", body)
        self.assertIn("Shallow streamed essay", body)
        self.assertIn("[USAGE]", body)

    @patch("artemis.main.deep_research", new_callable=AsyncMock)
    def test_streaming_usage_is_valid_json(self, mock_dr: AsyncMock) -> None:
        usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30, search_requests=2)
        mock_dr.return_value = DeepResearchRun(
            essay="Essay", results=[], sub_queries=[], stages_completed=1, usage=usage,
        )

        response = self.client.post(
            "/v1/responses",
            json={"input": "t", "preset": "deep-research", "streaming": True},
        )

        # Extract USAGE line and parse it
        for line in response.text.split("\n"):
            if "[USAGE]" in line:
                json_str = line.split("[USAGE] ")[1]
                parsed = json.loads(json_str)
                self.assertEqual(parsed["input_tokens"], 10)
                self.assertEqual(parsed["search_requests"], 2)
                break
        else:
            self.fail("No [USAGE] line found in streaming response")
