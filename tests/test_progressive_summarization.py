"""Unit tests for the progressive content summarization feature."""

import unittest
from unittest.mock import AsyncMock, patch

from artemis.errors import UpstreamServiceError
from artemis.models import SearchResult, TokenUsage
from artemis.researcher import (
    summarize_single_result,
    summarize_results_progressively,
)


class SummarizeSingleResultTestCase(unittest.IsolatedAsyncioTestCase):
    """Test individual result summarization."""

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_returns_summary_and_usage(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": "Concise summary of the page content.",
            "usage": {"input_tokens": 80, "output_tokens": 30, "total_tokens": 110},
        }

        summary, usage = await summarize_single_result(
            topic="AI safety",
            section="Overview",
            content="A" * 1000,
            url="https://example.com",
            title="Example Page",
        )

        self.assertEqual(summary, "Concise summary of the page content.")
        self.assertEqual(usage["total_tokens"], 110)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_short_content_skips_llm(self, mock_llm: AsyncMock) -> None:
        """Content shorter than max_chars is returned as-is without an LLM call."""
        summary, usage = await summarize_single_result(
            topic="test",
            section="S",
            content="Short content",
            max_chars=800,
        )

        self.assertEqual(summary, "Short content")
        self.assertIsNone(usage)
        mock_llm.assert_not_called()

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_llm_failure_returns_truncated_content(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = UpstreamServiceError("LLM down")
        long_content = "A" * 2000

        summary, usage = await summarize_single_result(
            topic="test",
            section="S",
            content=long_content,
            max_chars=800,
        )

        self.assertEqual(len(summary), 800)
        self.assertIsNone(usage)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_empty_llm_response_returns_truncated(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": "",
            "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
        }

        summary, usage = await summarize_single_result(
            topic="test",
            section="S",
            content="B" * 1000,
            max_chars=800,
        )

        self.assertEqual(len(summary), 800)
        self.assertIsNotNone(usage)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_prompt_includes_topic_and_section(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": "Summary.",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

        await summarize_single_result(
            topic="quantum computing",
            section="Applications: Industrial use cases",
            content="C" * 1000,
            url="https://qc.example.com",
            title="QC Applications",
        )

        messages = mock_llm.call_args[1]["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        self.assertIn("quantum computing", user_msg)
        self.assertIn("Applications: Industrial use cases", user_msg)
        self.assertIn("QC Applications", user_msg)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_summary_truncated_to_max_chars(self, mock_llm: AsyncMock) -> None:
        """Even if the LLM returns a long summary, it's capped at max_chars."""
        mock_llm.return_value = {
            "content": "X" * 2000,
            "usage": {"input_tokens": 10, "output_tokens": 500, "total_tokens": 510},
        }

        summary, _ = await summarize_single_result(
            topic="test",
            section="S",
            content="Y" * 1000,
            max_chars=800,
        )

        self.assertEqual(len(summary), 800)


class SummarizeResultsProgressivelyTestCase(unittest.IsolatedAsyncioTestCase):
    """Test batch progressive summarization across sections."""

    def _make_results(self, urls: list[str]) -> list[SearchResult]:
        return [
            SearchResult(title=f"Page {i}", url=url, snippet=f"Snippet for {url}")
            for i, url in enumerate(urls)
        ]

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_summarizes_all_results(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": "Summarized.",
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        }

        section_results = {
            "S1": self._make_results(["https://a.com", "https://b.com"]),
            "S2": self._make_results(["https://c.com"]),
        }
        outline = [
            {"section": "S1", "description": "D1"},
            {"section": "S2", "description": "D2"},
        ]
        # Content map has long content that will trigger summarization
        content_map = {
            "https://a.com": "A" * 1000,
            "https://b.com": "B" * 1000,
            "https://c.com": "C" * 1000,
        }

        summarized, usage = await summarize_results_progressively(
            section_results=section_results,
            topic="test",
            outline=outline,
            content_map=content_map,
        )

        # All 3 URLs should be in the summarized map
        self.assertEqual(len(summarized), 3)
        self.assertIn("https://a.com", summarized)
        self.assertIn("https://c.com", summarized)
        # Usage accumulated from all 3 calls
        self.assertEqual(usage.total_tokens, 90)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_partial_failure_keeps_successful(self, mock_llm: AsyncMock) -> None:
        """If one summarization fails, others still succeed."""
        mock_llm.side_effect = [
            {"content": "Good summary.", "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            UpstreamServiceError("LLM down"),
        ]

        section_results = {
            "S1": self._make_results(["https://a.com", "https://b.com"]),
        }
        outline = [{"section": "S1", "description": "D1"}]
        content_map = {
            "https://a.com": "A" * 1000,
            "https://b.com": "B" * 1000,
        }

        summarized, usage = await summarize_results_progressively(
            section_results=section_results,
            topic="test",
            outline=outline,
            content_map=content_map,
        )

        # First succeeded, second failed — only first in map
        # The second one falls back inside summarize_single_result (truncated),
        # so it should still appear in the map
        self.assertIn("https://a.com", summarized)
        self.assertIn("https://b.com", summarized)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_empty_sections_returns_empty(self, mock_llm: AsyncMock) -> None:
        summarized, usage = await summarize_results_progressively(
            section_results={},
            topic="test",
            outline=[],
        )

        self.assertEqual(summarized, {})
        self.assertEqual(usage.total_tokens, 0)
        mock_llm.assert_not_called()

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_progress_callback_invoked(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": "Summary.",
            "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
        }

        section_results = {"S1": self._make_results(["https://a.com"])}
        outline = [{"section": "S1", "description": "D1"}]
        content_map = {"https://a.com": "A" * 1000}

        progress_calls: list[tuple[str, str]] = []

        def cb(stage: str, msg: str):
            progress_calls.append((stage, msg))

        await summarize_results_progressively(
            section_results=section_results,
            topic="test",
            outline=outline,
            content_map=content_map,
            progress_callback=cb,
        )

        stages = [s for s, _ in progress_calls]
        self.assertIn("summarization", stages)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_short_content_not_summarized(self, mock_llm: AsyncMock) -> None:
        """Results with content shorter than max_chars skip LLM calls."""
        section_results = {"S1": self._make_results(["https://a.com"])}
        outline = [{"section": "S1", "description": "D1"}]
        content_map = {"https://a.com": "Short"}

        summarized, usage = await summarize_results_progressively(
            section_results=section_results,
            topic="test",
            outline=outline,
            content_map=content_map,
        )

        # Short content returned as-is, no LLM call
        self.assertEqual(summarized["https://a.com"], "Short")
        self.assertEqual(usage.total_tokens, 0)
        mock_llm.assert_not_called()


class DeepResearchProgressiveSummarizationTestCase(unittest.IsolatedAsyncioTestCase):
    """Test progressive summarization integration in the deep_research pipeline."""

    _BRIEF_RESPONSE = {
        "content": '{"research_question": "test topic", "scope": "", "search_guidance": ""}',
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
    }

    @patch("artemis.researcher.summarize_results_progressively", new_callable=AsyncMock)
    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_progressive_summarization_called_when_content_extracted(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
        mock_progressive: AsyncMock,
    ) -> None:
        from artemis.researcher import deep_research

        results = [
            SearchResult(title=f"Result {i} about test topic", url=f"https://r{i}.com", snippet=f"Relevant test topic content {i}")
            for i in range(3)
        ]

        mock_llm.side_effect = [
            self._BRIEF_RESPONSE,
            # generate_outline
            {"content": '[{"section": "S1", "description": "D1"}]',
             "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}},
            # generate_subqueries
            {"content": '["query 1"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            # synthesize_essay
            {"content": "Final essay.",
             "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}},
        ]
        mock_search.return_value = results
        mock_select.return_value = (results[:2], TokenUsage())
        mock_enrich.return_value = {
            "https://r0.com": "Long content " * 100,
            "https://r1.com": "More content " * 100,
        }
        mock_progressive.return_value = (
            {"https://r0.com": "Summarized r0", "https://r1.com": "Summarized r1"},
            TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        )

        result = await deep_research("test topic", stages=1, passes=1)

        self.assertEqual(result.essay, "Final essay.")
        mock_progressive.assert_awaited_once()

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    @patch("artemis.researcher.get_settings")
    async def test_progressive_summarization_skipped_when_disabled(
        self,
        mock_settings,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        """When progressive_summarization=False, no summarization LLM calls are made."""
        from artemis.config import Settings
        from artemis.researcher import deep_research

        settings = Settings(
            searxng_api_base="http://localhost:8888",
            searxng_timeout_seconds=30.0,
            litellm_base_url="http://localhost:11434/api",
            litellm_api_key=None,
            llm_timeout_seconds=120.0,
            summary_model="test-model",
            summary_max_tokens=1000,
            enable_summary=True,
            deep_research_stages=1,
            deep_research_passes=1,
            deep_research_subqueries=5,
            deep_research_results_per_query=5,
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
            progressive_summarization=False,
            progressive_summary_max_chars=800,
            progressive_summary_max_tokens=500,
        )
        mock_settings.return_value = settings

        results = [
            SearchResult(title="R1", url="https://r1.com", snippet="Content 1")
        ]

        mock_llm.side_effect = [
            self._BRIEF_RESPONSE,
            {"content": '[{"section": "S1", "description": "D1"}]',
             "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}},
            {"content": '["query 1"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            {"content": "Essay without summarization.",
             "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}},
        ]
        mock_search.return_value = results
        mock_select.return_value = (results, TokenUsage())
        mock_enrich.return_value = {"https://r1.com": "Extracted content " * 50}

        result = await deep_research("test topic", stages=1, passes=1)

        self.assertEqual(result.essay, "Essay without summarization.")
        # Only 4 LLM calls: brief + outline + subqueries + synthesis (no summarization)
        self.assertEqual(mock_llm.call_count, 4)
