"""Unit tests for the Artemis FastAPI app."""

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from artemis.config import Settings
from artemis.errors import UpstreamServiceError
from artemis.main import app, summary_circuit
from artemis.models import DeepResearchRun, SearchResult, TokenUsage


class APITestCase(unittest.TestCase):
    def setUp(self) -> None:
        summary_circuit.consecutive_failures = 0
        summary_circuit.opened_until = 0.0
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    def test_search_request_validation(self) -> None:
        response = self.client.post("/search", json={})
        self.assertEqual(response.status_code, 422)

    @patch("artemis.main.search_searxng", new_callable=AsyncMock)
    def test_search_requires_api_key_when_configured(
        self, mock_search_searxng: AsyncMock
    ) -> None:
        mock_search_searxng.return_value = []

        with patch(
            "artemis.main.get_settings",
            return_value=Settings(
                searxng_api_base="http://localhost:8888",
                searxng_timeout_seconds=30.0,
                litellm_base_url="https://api.openai.com/v1",
                litellm_api_key=None,
                llm_timeout_seconds=60.0,
                summary_model="arc:apex",
                summary_max_tokens=1024,
                enable_summary=True,
                deep_research_stages=2,
                deep_research_passes=1,
                deep_research_subqueries=5,
                deep_research_results_per_query=10,
                deep_research_max_tokens=4000,
                deep_research_content_extraction=True,
                deep_research_pages_per_section=3,
                deep_research_content_max_chars=3000,
                allowed_origins=tuple(),
                artemis_api_key="secret-token",
                log_level="INFO",
            ),
        ):
            unauthorized = self.client.post("/search", json={"query": "fastapi"})
            self.assertEqual(unauthorized.status_code, 401)

            authorized = self.client.post(
                "/search",
                json={"query": "fastapi"},
                headers={"Authorization": "Bearer secret-token"},
            )
            self.assertEqual(authorized.status_code, 200)

    @patch("artemis.main.deep_research", new_callable=AsyncMock)
    def test_deep_research_uses_requested_max_steps(
        self, mock_deep_research: AsyncMock
    ) -> None:
        mock_deep_research.return_value = DeepResearchRun(
            essay="Structured report",
            results=[
                SearchResult(
                    title="Example",
                    url="https://example.com/report",
                    snippet="A structured result",
                    date="2026-01-01",
                )
            ],
            sub_queries=["example"],
            stages_completed=5,
            usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        )

        response = self.client.post(
            "/v1/responses",
            json={"input": "example", "preset": "deep-research", "max_steps": 5},
        )

        self.assertEqual(response.status_code, 200)
        mock_deep_research.assert_awaited_once_with(
            "example", stages=5, passes=1, outline=None
        )
        self.assertEqual(response.json()["usage"]["total_tokens"], 30)

    @patch("artemis.main.summarize_results", new_callable=AsyncMock)
    @patch("artemis.main.search_searxng", new_callable=AsyncMock)
    def test_fast_search_returns_warning_when_summary_fails(
        self, mock_search_searxng: AsyncMock, mock_summarize_results: AsyncMock
    ) -> None:
        mock_search_searxng.return_value = [
            SearchResult(
                title="Example",
                url="https://example.com/article",
                snippet="Snippet text for fallback output.",
                date="2026-01-01",
            )
        ]
        mock_summarize_results.side_effect = UpstreamServiceError(
            "The LLM backend timed out."
        )

        response = self.client.post("/v1/responses", json={"input": "example"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["warnings"],
            ["LLM summarization is unavailable; returning search results only."],
        )
        self.assertIn(
            "Snippet text for fallback output.",
            body["output"][0]["content"][0]["text"],
        )
