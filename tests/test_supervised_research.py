"""Unit tests for the supervised (hierarchical) research architecture."""

import unittest
from unittest.mock import AsyncMock, patch, MagicMock

from artemis.models import DeepResearchRun, SearchResult, TokenUsage
from artemis.researcher import (
    _run_researcher,
    supervised_deep_research,
    _RESEARCHER_TOOL_DEFINITIONS,
)


# Standard mock response for generate_research_brief (prepend to side_effect lists)
_BRIEF_RESPONSE = {
    "content": '{"research_question": "test topic", "scope": "", "search_guidance": ""}',
    "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
}


class ResearcherToolDefinitionsTestCase(unittest.TestCase):
    """Verify the tool definitions are well-formed."""

    def test_has_three_tools(self) -> None:
        self.assertEqual(len(_RESEARCHER_TOOL_DEFINITIONS), 3)

    def test_tool_names(self) -> None:
        names = {td["function"]["name"] for td in _RESEARCHER_TOOL_DEFINITIONS}
        self.assertEqual(names, {"web_search", "read_page", "note"})

    def test_all_have_required_fields(self) -> None:
        for td in _RESEARCHER_TOOL_DEFINITIONS:
            self.assertEqual(td["type"], "function")
            fn = td["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            self.assertIn("parameters", fn)
            self.assertIn("required", fn["parameters"])


class RunResearcherTestCase(unittest.IsolatedAsyncioTestCase):
    """Test a single researcher agent."""

    def _mock_search_results(self, n: int = 3) -> list[SearchResult]:
        return [
            SearchResult(
                title=f"Result {i}",
                url=f"https://r{i}.com",
                snippet=f"Snippet about the topic {i}",
            )
            for i in range(n)
        ]

    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    async def test_researcher_returns_findings(
        self,
        mock_agentic: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
    ) -> None:
        mock_agentic.return_value = {
            "content": "These are my findings about the section.",
            "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
            "tool_calls_made": 3,
        }

        result = await _run_researcher(
            topic="AI safety",
            section="Overview",
            description="General overview of AI safety",
            results_per_query=5,
            max_tool_rounds=10,
            content_max_chars=3000,
        )

        self.assertEqual(result["findings"], "These are my findings about the section.")
        self.assertIsInstance(result["results"], list)
        self.assertIsInstance(result["queries"], list)
        self.assertIsInstance(result["content_map"], dict)
        self.assertEqual(result["usage"]["total_tokens"], 300)

    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    async def test_researcher_passes_tool_definitions(
        self,
        mock_agentic: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
    ) -> None:
        mock_agentic.return_value = {
            "content": "Findings.",
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            "tool_calls_made": 0,
        }

        await _run_researcher(
            topic="test",
            section="S",
            description="D",
            results_per_query=5,
            max_tool_rounds=10,
            content_max_chars=3000,
        )

        call_kwargs = mock_agentic.call_args[1]
        self.assertIn("tool_definitions", call_kwargs)
        self.assertEqual(len(call_kwargs["tool_definitions"]), 3)
        tool_names = {td["function"]["name"] for td in call_kwargs["tool_definitions"]}
        self.assertEqual(tool_names, {"web_search", "read_page", "note"})

    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    async def test_researcher_passes_three_tool_handlers(
        self,
        mock_agentic: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
    ) -> None:
        mock_agentic.return_value = {
            "content": "Findings.",
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            "tool_calls_made": 0,
        }

        await _run_researcher(
            topic="test",
            section="S",
            description="D",
            results_per_query=5,
            max_tool_rounds=10,
            content_max_chars=3000,
        )

        call_kwargs = mock_agentic.call_args[1]
        handlers = call_kwargs["tool_handlers"]
        self.assertIn("web_search", handlers)
        self.assertIn("read_page", handlers)
        self.assertIn("note", handlers)

    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    async def test_researcher_fallback_on_failure(
        self,
        mock_agentic: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
    ) -> None:
        """When the agentic loop fails, researcher falls back gracefully."""
        from artemis.errors import UpstreamServiceError

        mock_agentic.side_effect = UpstreamServiceError("LLM down")

        result = await _run_researcher(
            topic="test",
            section="Section A",
            description="Description",
            results_per_query=5,
            max_tool_rounds=10,
            content_max_chars=3000,
        )

        # Should return a fallback message (no results gathered either)
        self.assertIn("could not be completed", result["findings"])

    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    async def test_progress_callback_invoked(
        self,
        mock_agentic: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
    ) -> None:
        mock_agentic.return_value = {
            "content": "Findings.",
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            "tool_calls_made": 1,
        }

        progress_calls: list[tuple[str, str]] = []

        def cb(stage: str, msg: str):
            progress_calls.append((stage, msg))

        await _run_researcher(
            topic="test",
            section="Overview",
            description="D",
            results_per_query=5,
            max_tool_rounds=10,
            content_max_chars=3000,
            progress_callback=cb,
        )

        # Should have at least the "Done" progress message
        stages = [s for s, _ in progress_calls]
        self.assertIn("researcher", stages)
        messages = [m for _, m in progress_calls]
        self.assertTrue(any("Done" in m for m in messages))


class SupervisedDeepResearchTestCase(unittest.IsolatedAsyncioTestCase):
    """Test the full supervised research pipeline."""

    def _mock_search_results(self, n: int = 3) -> list[SearchResult]:
        return [
            SearchResult(
                title=f"Result {i}",
                url=f"https://r{i}.com",
                snippet=f"Content about the topic {i}",
            )
            for i in range(n)
        ]

    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_full_supervised_pipeline(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
        mock_agentic: AsyncMock,
    ) -> None:
        # chat_completion is called for: brief + outline generation + synthesis
        mock_llm.side_effect = [
            # generate_research_brief
            _BRIEF_RESPONSE,
            # generate_outline
            {
                "content": '[{"section": "Overview", "description": "Topic overview"}]',
                "usage": {"input_tokens": 30, "output_tokens": 20, "total_tokens": 50},
            },
            # synthesize (non-agentic path)
            {
                "content": "This is the final synthesized essay.",
                "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
            },
        ]
        # agentic_chat_completion is called for each researcher
        mock_agentic.return_value = {
            "content": "Researcher findings for this section.",
            "usage": {"input_tokens": 50, "output_tokens": 80, "total_tokens": 130},
            "tool_calls_made": 2,
        }

        result = await supervised_deep_research("test topic", stages=1)

        self.assertIsInstance(result, DeepResearchRun)
        self.assertEqual(result.essay, "This is the final synthesized essay.")
        self.assertGreater(result.usage.total_tokens, 0)

    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_custom_outline_skips_generation(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
        mock_agentic: AsyncMock,
    ) -> None:
        custom_outline = [{"section": "Custom", "description": "Custom section"}]
        # brief + synthesis (no outline generation)
        mock_llm.side_effect = [
            _BRIEF_RESPONSE,
            {
                "content": "Custom essay.",
                "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
            },
        ]
        mock_agentic.return_value = {
            "content": "Researcher findings.",
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "tool_calls_made": 1,
        }

        result = await supervised_deep_research(
            "test", stages=1, outline=custom_outline
        )

        self.assertEqual(result.essay, "Custom essay.")
        # chat_completion called twice: brief + synthesis (no outline)
        self.assertEqual(mock_llm.call_count, 2)

    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_progress_callbacks(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
        mock_agentic: AsyncMock,
    ) -> None:
        mock_llm.side_effect = [
            _BRIEF_RESPONSE,
            {
                "content": '[{"section": "S1", "description": "D1"}]',
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            },
            {
                "content": "Essay.",
                "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
            },
        ]
        mock_agentic.return_value = {
            "content": "Findings.",
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            "tool_calls_made": 1,
        }

        progress_calls: list[tuple[str, str]] = []

        def cb(stage: str, msg: str):
            progress_calls.append((stage, msg))

        await supervised_deep_research("test", stages=1, progress_callback=cb)

        stages = [s for s, _ in progress_calls]
        self.assertIn("start", stages)
        self.assertIn("outline", stages)
        self.assertIn("researchers", stages)
        self.assertIn("synthesis", stages)
        self.assertIn("complete", stages)

    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_researcher_failure_handled_gracefully(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
        mock_agentic: AsyncMock,
    ) -> None:
        """If a researcher fails, the pipeline continues with other sections."""
        mock_llm.side_effect = [
            _BRIEF_RESPONSE,
            {
                "content": '[{"section": "S1", "description": "D1"}, {"section": "S2", "description": "D2"}]',
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            },
            {
                "content": "Essay from partial findings.",
                "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
            },
        ]
        # First researcher succeeds, second fails
        mock_agentic.side_effect = [
            {
                "content": "Good findings.",
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
                "tool_calls_made": 1,
            },
            Exception("Researcher crashed"),
        ]

        result = await supervised_deep_research("test", stages=2)

        self.assertEqual(result.essay, "Essay from partial findings.")
        self.assertEqual(result.stages_completed, 2)

    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_multiple_sections_run_parallel(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
        mock_agentic: AsyncMock,
    ) -> None:
        """Verify multiple researchers are spawned for multiple sections."""
        mock_llm.side_effect = [
            _BRIEF_RESPONSE,
            {
                "content": '[{"section": "S1", "description": "D1"}, {"section": "S2", "description": "D2"}, {"section": "S3", "description": "D3"}]',
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            },
            {
                "content": "Final essay.",
                "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
            },
        ]
        mock_agentic.return_value = {
            "content": "Section findings.",
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            "tool_calls_made": 1,
        }

        result = await supervised_deep_research("test", stages=3)

        # agentic_chat_completion should be called once per section
        self.assertEqual(mock_agentic.call_count, 3)
        self.assertEqual(result.stages_completed, 3)
