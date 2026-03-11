"""Tests for researcher helper functions and deep_research orchestration."""

import unittest
from unittest.mock import AsyncMock, patch, MagicMock

from artemis.errors import UpstreamServiceError
from artemis.models import DeepResearchRun, SearchResult, TokenUsage
from artemis.researcher import (
    _deduplicate_results,
    _merge_usage,
    format_results_for_synthesis,
    select_relevant_results,
    generate_outline,
    generate_subqueries_for_section,
    deep_research,
)


class MergeUsageTestCase(unittest.TestCase):
    def test_accumulates_tokens(self) -> None:
        total = TokenUsage()
        _merge_usage(total, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        _merge_usage(total, {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30})
        self.assertEqual(total.input_tokens, 30)
        self.assertEqual(total.output_tokens, 15)
        self.assertEqual(total.total_tokens, 45)

    def test_none_usage_skipped(self) -> None:
        total = TokenUsage(input_tokens=5)
        _merge_usage(total, None)
        self.assertEqual(total.input_tokens, 5)

    def test_missing_keys_default_zero(self) -> None:
        total = TokenUsage()
        _merge_usage(total, {"input_tokens": 10})
        self.assertEqual(total.input_tokens, 10)
        self.assertEqual(total.output_tokens, 0)


class DeduplicateResultsTestCase(unittest.TestCase):
    def test_removes_duplicates_by_url(self) -> None:
        results = [
            SearchResult(title="A", url="https://a.com", snippet="s1"),
            SearchResult(title="B", url="https://b.com", snippet="s2"),
            SearchResult(title="A again", url="https://a.com", snippet="s3"),
        ]
        deduped = _deduplicate_results(results)
        self.assertEqual(len(deduped), 2)
        urls = [r.url for r in deduped]
        self.assertIn("https://a.com", urls)
        self.assertIn("https://b.com", urls)

    def test_preserves_first_occurrence(self) -> None:
        results = [
            SearchResult(title="First", url="https://x.com", snippet="s1"),
            SearchResult(title="Second", url="https://x.com", snippet="s2"),
        ]
        deduped = _deduplicate_results(results)
        self.assertEqual(deduped[0].title, "First")

    def test_empty_input(self) -> None:
        self.assertEqual(_deduplicate_results([]), [])


class FormatResultsTestCase(unittest.TestCase):
    def test_basic_formatting(self) -> None:
        results = [
            SearchResult(title="Page 1", url="https://a.com", snippet="Content A"),
        ]
        text = format_results_for_synthesis(results)
        self.assertIn("Title: Page 1", text)
        self.assertIn("URL: https://a.com", text)
        self.assertIn("Content: Content A", text)

    def test_content_map_overrides_snippet(self) -> None:
        results = [
            SearchResult(title="Page 1", url="https://a.com", snippet="Short snippet"),
        ]
        content_map = {"https://a.com": "Full extracted page content from Playwright"}
        text = format_results_for_synthesis(results, content_map=content_map)
        self.assertIn("Full extracted page content", text)
        self.assertNotIn("Short snippet", text)

    def test_empty_results(self) -> None:
        self.assertEqual(format_results_for_synthesis([]), "")

    def test_multiple_results_separated(self) -> None:
        results = [
            SearchResult(title="A", url="https://a.com", snippet="s1"),
            SearchResult(title="B", url="https://b.com", snippet="s2"),
        ]
        text = format_results_for_synthesis(results)
        self.assertIn("---", text)


class SelectRelevantResultsTestCase(unittest.IsolatedAsyncioTestCase):
    def _make_results(self, n: int) -> list[SearchResult]:
        return [
            SearchResult(title=f"Result {i}", url=f"https://r{i}.com", snippet=f"Snippet {i}")
            for i in range(n)
        ]

    async def test_returns_all_when_under_max(self) -> None:
        results = self._make_results(3)
        selected, usage = await select_relevant_results(
            topic="test", section="S", description="D", results=results, max_results=5
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(usage.total_tokens, 0)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_selects_by_llm_indices(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": "[0, 2, 4]",
            "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
        }
        results = self._make_results(10)
        selected, usage = await select_relevant_results(
            topic="test", section="S", description="D", results=results, max_results=3
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0].title, "Result 0")
        self.assertEqual(selected[1].title, "Result 2")
        self.assertEqual(selected[2].title, "Result 4")

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_fallback_on_llm_failure(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = UpstreamServiceError("LLM down")
        results = self._make_results(10)
        selected, usage = await select_relevant_results(
            topic="test", section="S", description="D", results=results, max_results=3
        )
        self.assertEqual(len(selected), 3)
        # Should fall back to first N
        self.assertEqual(selected[0].title, "Result 0")

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_handles_invalid_indices(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": "[0, 999, -1, 2]",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        results = self._make_results(5)
        selected, usage = await select_relevant_results(
            topic="test", section="S", description="D", results=results, max_results=3
        )
        # Only indices 0 and 2 are valid
        self.assertEqual(len(selected), 2)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_handles_markdown_wrapped_response(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": "```json\n[1, 3]\n```",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        results = self._make_results(5)
        selected, _ = await select_relevant_results(
            topic="test", section="S", description="D", results=results, max_results=3
        )
        self.assertEqual(selected[0].title, "Result 1")
        self.assertEqual(selected[1].title, "Result 3")

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_deduplicates_indices(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": "[0, 0, 1, 1]",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        results = self._make_results(5)
        selected, _ = await select_relevant_results(
            topic="test", section="S", description="D", results=results, max_results=3
        )
        self.assertEqual(len(selected), 2)


class GenerateOutlineTestCase(unittest.IsolatedAsyncioTestCase):
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_parses_outline_from_llm(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": '[{"section": "Intro", "description": "Introduction"}, {"section": "Body", "description": "Main body"}]',
            "usage": {"input_tokens": 30, "output_tokens": 20, "total_tokens": 50},
        }
        outline, usage = await generate_outline("Test topic", num_sections=2)
        self.assertEqual(len(outline), 2)
        self.assertEqual(outline[0]["section"], "Intro")
        self.assertEqual(usage.total_tokens, 50)


class GenerateSubqueriesTestCase(unittest.IsolatedAsyncioTestCase):
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_generates_queries(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": '["query one", "query two"]',
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        }
        queries, usage = await generate_subqueries_for_section(
            topic="AI", section="History", description="Historical overview",
            num_queries=2
        )
        self.assertEqual(queries, ["query one", "query two"])
        self.assertEqual(usage.total_tokens, 30)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_includes_existing_queries_in_prompt(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": '["new query"]',
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        }
        await generate_subqueries_for_section(
            topic="AI", section="S", description="D", num_queries=1,
            existing_queries=["old query 1", "old query 2"]
        )
        # Existing queries appear in the user message
        messages = mock_llm.call_args[1]["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        self.assertIn("old query 1", user_msg)


class DeepResearchOrchestrationTestCase(unittest.IsolatedAsyncioTestCase):
    """Test the deep_research function with all dependencies mocked."""

    def _mock_search_results(self, n: int = 3) -> list[SearchResult]:
        # Snippets must contain keywords from the topic/section to pass relevance filtering
        return [
            SearchResult(title=f"Result {i} about test topic", url=f"https://r{i}.com", snippet=f"Relevant test topic content {i}")
            for i in range(n)
        ]

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_full_pipeline(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        # Outline generation
        mock_llm.side_effect = [
            # generate_outline
            {"content": '[{"section": "Overview", "description": "Topic overview"}]',
             "usage": {"input_tokens": 30, "output_tokens": 20, "total_tokens": 50}},
            # generate_subqueries_for_section
            {"content": '["search query 1", "search query 2"]',
             "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}},
            # synthesize_essay_with_outline
            {"content": "This is the final essay about the topic.",
             "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}},
        ]
        mock_search.return_value = self._mock_search_results(3)
        mock_select.return_value = (self._mock_search_results(2), TokenUsage())
        mock_enrich.return_value = {"https://r0.com": "Full content"}

        result = await deep_research("test topic", stages=1, passes=1)

        self.assertIsInstance(result, DeepResearchRun)
        self.assertEqual(result.essay, "This is the final essay about the topic.")
        self.assertGreater(len(result.results), 0)
        self.assertGreater(len(result.sub_queries), 0)
        self.assertEqual(result.stages_completed, 1)
        self.assertGreater(result.usage.total_tokens, 0)
        self.assertGreater(result.usage.search_requests, 0)

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_custom_outline_skips_generation(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        custom_outline = [{"section": "Custom", "description": "Custom section"}]
        mock_llm.side_effect = [
            # generate_subqueries_for_section (no outline generation needed)
            {"content": '["query 1"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            # synthesize_essay_with_outline
            {"content": "Custom essay.",
             "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150}},
        ]
        mock_search.return_value = self._mock_search_results(2)
        mock_select.return_value = (self._mock_search_results(1), TokenUsage())
        mock_enrich.return_value = {}

        result = await deep_research("test", stages=1, passes=1, outline=custom_outline)

        self.assertEqual(result.essay, "Custom essay.")
        # LLM should have been called twice (subqueries + synthesis), not 3 times
        self.assertEqual(mock_llm.call_count, 2)

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_progress_callback_invoked(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        mock_llm.side_effect = [
            {"content": '[{"section": "S1", "description": "D1"}]',
             "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}},
            {"content": '["q1"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            {"content": "Essay text.",
             "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150}},
        ]
        mock_search.return_value = self._mock_search_results(2)
        mock_select.return_value = (self._mock_search_results(1), TokenUsage())
        mock_enrich.return_value = {}

        progress_calls: list[tuple[str, str]] = []

        def progress_cb(stage: str, msg: str):
            progress_calls.append((stage, msg))

        await deep_research("test", stages=1, passes=1, progress_callback=progress_cb)

        stages_seen = [s for s, _ in progress_calls]
        self.assertIn("start", stages_seen)
        self.assertIn("outline", stages_seen)
        self.assertIn("complete", stages_seen)

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_outline_generation_fallback(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        """If outline generation fails, fallback outline is used."""
        call_count = 0

        async def llm_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Outline generation fails
                raise UpstreamServiceError("LLM down")
            elif call_count == 2:
                # Subqueries for fallback sections
                return {"content": '["query"]',
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
            else:
                # Synthesis
                return {"content": "Fallback essay.",
                        "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150}}

        mock_llm.side_effect = llm_side_effect
        mock_search.return_value = self._mock_search_results(2)
        mock_select.return_value = (self._mock_search_results(1), TokenUsage())
        mock_enrich.return_value = {}

        result = await deep_research("test", stages=1, passes=1)

        self.assertEqual(result.essay, "Fallback essay.")
        # Fallback outline should have been generated
        self.assertEqual(result.stages_completed, 1)
