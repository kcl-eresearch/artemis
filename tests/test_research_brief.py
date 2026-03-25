"""Unit tests for the research brief generation feature."""

import unittest
from unittest.mock import AsyncMock, patch

from artemis.errors import UpstreamServiceError
from artemis.models import SearchResult, TokenUsage
from artemis.researcher import _parse_research_brief, generate_research_brief


class ParseResearchBriefTestCase(unittest.TestCase):
    """Test parsing of the LLM research brief JSON response."""

    def test_valid_json_all_fields(self) -> None:
        raw = '{"research_question": "How does X work?", "scope": "2020-2025", "search_guidance": "academic papers"}'
        result = _parse_research_brief(raw, "original")
        self.assertIn("How does X work?", result)
        self.assertIn("Scope: 2020-2025", result)
        self.assertIn("Search guidance: academic papers", result)

    def test_question_only(self) -> None:
        raw = '{"research_question": "What is quantum computing?", "scope": "", "search_guidance": ""}'
        result = _parse_research_brief(raw, "original")
        self.assertEqual(result, "What is quantum computing?")

    def test_question_and_scope_no_guidance(self) -> None:
        raw = '{"research_question": "Q?", "scope": "global", "search_guidance": ""}'
        result = _parse_research_brief(raw, "original")
        self.assertEqual(result, "Q?\nScope: global")

    def test_markdown_code_block(self) -> None:
        raw = '```json\n{"research_question": "Brief Q", "scope": "narrow", "search_guidance": ""}\n```'
        result = _parse_research_brief(raw, "original")
        self.assertIn("Brief Q", result)
        self.assertIn("Scope: narrow", result)

    def test_json_embedded_in_text(self) -> None:
        raw = 'Here is the brief: {"research_question": "Embedded Q", "scope": "", "search_guidance": ""} done.'
        result = _parse_research_brief(raw, "original")
        self.assertIn("Embedded Q", result)

    def test_invalid_json_returns_original(self) -> None:
        result = _parse_research_brief("this is not json", "fallback query")
        self.assertEqual(result, "fallback query")

    def test_non_object_returns_original(self) -> None:
        result = _parse_research_brief('["an", "array"]', "fallback")
        self.assertEqual(result, "fallback")

    def test_empty_question_returns_original(self) -> None:
        raw = '{"research_question": "", "scope": "something", "search_guidance": ""}'
        result = _parse_research_brief(raw, "fallback")
        self.assertEqual(result, "fallback")

    def test_missing_question_key_returns_original(self) -> None:
        raw = '{"scope": "wide", "search_guidance": "news"}'
        result = _parse_research_brief(raw, "fallback")
        self.assertEqual(result, "fallback")

    def test_whitespace_only_question_returns_original(self) -> None:
        raw = '{"research_question": "   ", "scope": "", "search_guidance": ""}'
        result = _parse_research_brief(raw, "fallback")
        self.assertEqual(result, "fallback")

    def test_truncated_json_returns_original(self) -> None:
        raw = '{"research_question": "Truncat'
        result = _parse_research_brief(raw, "fallback")
        self.assertEqual(result, "fallback")

    def test_whitespace_stripped(self) -> None:
        raw = '{"research_question": "  Q  ", "scope": "  S  ", "search_guidance": "  G  "}'
        result = _parse_research_brief(raw, "fallback")
        self.assertIn("Q", result)
        self.assertIn("Scope: S", result)
        self.assertIn("Search guidance: G", result)


class GenerateResearchBriefTestCase(unittest.IsolatedAsyncioTestCase):
    """Test the generate_research_brief async function."""

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_returns_parsed_brief_and_usage(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": '{"research_question": "How does AI impact healthcare?", "scope": "2020-2025", "search_guidance": "academic papers"}',
            "usage": {"input_tokens": 40, "output_tokens": 30, "total_tokens": 70},
        }

        brief, usage = await generate_research_brief("AI healthcare")

        self.assertIn("How does AI impact healthcare?", brief)
        self.assertIn("Scope: 2020-2025", brief)
        self.assertEqual(usage.total_tokens, 70)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_llm_failure_returns_original_query(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = UpstreamServiceError("LLM down")

        brief, usage = await generate_research_brief("my query")

        self.assertEqual(brief, "my query")
        self.assertEqual(usage.total_tokens, 0)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_unparseable_response_returns_original(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": "I cannot generate a brief right now.",
            "usage": {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25},
        }

        brief, usage = await generate_research_brief("my query")

        self.assertEqual(brief, "my query")
        # Usage is still tracked even when parsing falls back
        self.assertEqual(usage.total_tokens, 25)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_prompt_includes_query(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": '{"research_question": "Q", "scope": "", "search_guidance": ""}',
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        }

        await generate_research_brief("quantum computing applications")

        messages = mock_llm.call_args[1]["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        self.assertIn("quantum computing applications", user_msg)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_null_usage_handled(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": '{"research_question": "Q", "scope": "", "search_guidance": ""}',
            "usage": None,
        }

        brief, usage = await generate_research_brief("test")

        self.assertEqual(brief, "Q")
        self.assertEqual(usage.total_tokens, 0)


class ResearchBriefConfigTestCase(unittest.IsolatedAsyncioTestCase):
    """Test that the research_brief_enabled config toggle works."""

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    @patch("artemis.researcher.get_settings")
    async def test_brief_disabled_skips_generation(
        self,
        mock_settings,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        """When research_brief_enabled=False, no brief LLM call is made."""
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
            research_brief_enabled=False,
            progressive_summarization=False,
            progressive_summary_max_chars=800,
            progressive_summary_max_tokens=500,
            researcher_min_relevant_sources=3,
            researcher_overlap_threshold=0.6,
        )
        mock_settings.return_value = settings

        mock_llm.side_effect = [
            # generate_outline (no brief!)
            {"content": '[{"section": "S1", "description": "D1"}]',
             "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}},
            # generate_subqueries
            {"content": '["query 1"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            # synthesize_essay
            {"content": "Essay without brief.",
             "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}},
        ]
        mock_search.return_value = [
            SearchResult(title="R1", url="https://r1.com", snippet="Content")
        ]
        mock_select.return_value = (mock_search.return_value, TokenUsage())
        mock_enrich.return_value = {}

        result = await deep_research("test topic", stages=1, passes=1)

        self.assertEqual(result.essay, "Essay without brief.")
        # Only 3 LLM calls: outline + subqueries + synthesis (no brief)
        self.assertEqual(mock_llm.call_count, 3)
