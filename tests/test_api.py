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
                litellm_base_url="http://localhost:11434/api",
                litellm_api_key=None,
                llm_timeout_seconds=120.0,
                summary_model="qwen3.5:9b",
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
                shallow_research_stages=1,
                shallow_research_passes=1,
                shallow_research_subqueries=3,
                shallow_research_results_per_query=5,
                shallow_research_max_tokens=4000,
                shallow_research_content_extraction=False,
                shallow_research_pages_per_section=2,
                shallow_research_content_max_chars=2000,
                allowed_origins=tuple(),
                artemis_api_key="secret-token",
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
        settings = Settings(
            searxng_api_base="http://localhost:8888",
            searxng_timeout_seconds=30.0,
            litellm_base_url="http://localhost:11434/api",
            litellm_api_key=None,
            llm_timeout_seconds=120.0,
            summary_model="qwen3.5:9b",
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
        )

        with patch("artemis.main.get_settings", return_value=settings):
            response = self.client.post(
                "/v1/responses",
                json={"input": "example", "preset": "deep-research", "max_steps": 5},
            )

        self.assertEqual(response.status_code, 200)
        mock_deep_research.assert_awaited_once_with(
            "example",
            stages=5,
            passes=1,
            sub_queries_per_stage=5,
            results_per_query=10,
            max_tokens=4000,
            outline=None,
            content_extraction=True,
            pages_per_section=3,
            content_max_chars=3000,
        )
        self.assertEqual(response.json()["usage"]["total_tokens"], 30)

    @patch("artemis.main.deep_research", new_callable=AsyncMock)
    def test_shallow_research_uses_shallow_settings(
        self, mock_deep_research: AsyncMock
    ) -> None:
        mock_deep_research.return_value = DeepResearchRun(
            essay="Concise report",
            results=[],
            sub_queries=["example"],
            stages_completed=1,
            usage=TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15),
        )

        with patch(
            "artemis.main.get_settings",
            return_value=Settings(
                searxng_api_base="http://localhost:8888",
                searxng_timeout_seconds=30.0,
                litellm_base_url="http://localhost:11434/api",
                litellm_api_key=None,
                llm_timeout_seconds=120.0,
                summary_model="qwen3.5:9b",
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
            ),
        ):
            response = self.client.post(
                "/v1/responses",
                json={"input": "example", "preset": "shallow-research"},
            )

        self.assertEqual(response.status_code, 200)
        mock_deep_research.assert_awaited_once_with(
            "example",
            stages=1,
            passes=1,
            sub_queries_per_stage=3,
            results_per_query=5,
            max_tokens=2500,
            outline=None,
            content_extraction=False,
            pages_per_section=2,
            content_max_chars=1500,
        )
        self.assertEqual(response.json()["model"], "artemis-shallow-research")

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
