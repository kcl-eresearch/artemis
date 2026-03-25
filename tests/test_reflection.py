"""Unit tests for the think-tool reflection pattern."""

import unittest
from unittest.mock import AsyncMock, patch

from artemis.errors import UpstreamServiceError
from artemis.models import DeepResearchRun, SearchResult, TokenUsage
from artemis.researcher import (
    _parse_reflection,
    reflect_on_findings,
    deep_research,
)


class ParseReflectionTestCase(unittest.TestCase):
    """Test parsing of the LLM reflection JSON response."""

    def test_valid_json(self) -> None:
        raw = '''{
            "section_assessments": {
                "Overview": {"coverage": "good", "gaps": [], "sufficient": true},
                "Analysis": {"coverage": "partial", "gaps": ["missing data on X"], "sufficient": false}
            },
            "should_continue": true,
            "focus_areas": ["data on X", "recent studies"]
        }'''
        result = _parse_reflection(raw)
        self.assertEqual(len(result["section_assessments"]), 2)
        self.assertTrue(result["should_continue"])
        self.assertEqual(result["focus_areas"], ["data on X", "recent studies"])

    def test_should_continue_false(self) -> None:
        raw = '{"section_assessments": {}, "should_continue": false, "focus_areas": []}'
        result = _parse_reflection(raw)
        self.assertFalse(result["should_continue"])

    def test_should_continue_string_false(self) -> None:
        raw = '{"section_assessments": {}, "should_continue": "false", "focus_areas": []}'
        result = _parse_reflection(raw)
        self.assertFalse(result["should_continue"])

    def test_should_continue_string_no(self) -> None:
        raw = '{"section_assessments": {}, "should_continue": "no", "focus_areas": []}'
        result = _parse_reflection(raw)
        self.assertFalse(result["should_continue"])

    def test_markdown_code_block(self) -> None:
        raw = '```json\n{"section_assessments": {}, "should_continue": true, "focus_areas": ["X"]}\n```'
        result = _parse_reflection(raw)
        self.assertTrue(result["should_continue"])
        self.assertEqual(result["focus_areas"], ["X"])

    def test_json_embedded_in_text(self) -> None:
        raw = 'Here is my assessment: {"section_assessments": {}, "should_continue": false, "focus_areas": []} end.'
        result = _parse_reflection(raw)
        self.assertFalse(result["should_continue"])

    def test_invalid_json_returns_default(self) -> None:
        result = _parse_reflection("this is not json at all")
        self.assertTrue(result["should_continue"])
        self.assertEqual(result["section_assessments"], {})
        self.assertEqual(result["focus_areas"], [])

    def test_non_object_returns_default(self) -> None:
        result = _parse_reflection('["an", "array"]')
        self.assertTrue(result["should_continue"])

    def test_missing_keys_use_defaults(self) -> None:
        result = _parse_reflection('{"section_assessments": {}}')
        self.assertTrue(result["should_continue"])
        self.assertEqual(result["focus_areas"], [])

    def test_non_list_focus_areas_normalised(self) -> None:
        raw = '{"section_assessments": {}, "should_continue": true, "focus_areas": "a string"}'
        result = _parse_reflection(raw)
        self.assertEqual(result["focus_areas"], [])

    def test_non_dict_assessments_normalised(self) -> None:
        raw = '{"section_assessments": "bad", "should_continue": true, "focus_areas": []}'
        result = _parse_reflection(raw)
        self.assertEqual(result["section_assessments"], {})

    def test_truncated_json_returns_default(self) -> None:
        raw = '{"section_assessments": {"S1": {"coverage": "good"'
        result = _parse_reflection(raw)
        # Should return default (continue) rather than crash
        self.assertTrue(result["should_continue"])


class ReflectOnFindingsTestCase(unittest.IsolatedAsyncioTestCase):
    """Test the reflect_on_findings async function."""

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_returns_parsed_reflection(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": '{"section_assessments": {"S1": {"coverage": "good", "gaps": [], "sufficient": true}}, "should_continue": false, "focus_areas": []}',
            "usage": {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
        }

        outline = [{"section": "S1", "description": "D1"}]
        section_results = {"S1": [
            SearchResult(title="R1", url="https://r1.com", snippet="Content"),
        ]}

        reflection, usage = await reflect_on_findings(
            topic="test", outline=outline, section_results=section_results,
            pass_num=1, total_passes=2,
        )

        self.assertFalse(reflection["should_continue"])
        self.assertIn("S1", reflection["section_assessments"])
        self.assertEqual(usage.total_tokens, 80)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_llm_failure_returns_continue(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = UpstreamServiceError("LLM down")

        outline = [{"section": "S1", "description": "D1"}]
        section_results = {"S1": []}

        reflection, usage = await reflect_on_findings(
            topic="test", outline=outline, section_results=section_results,
            pass_num=1, total_passes=2,
        )

        # Should gracefully continue
        self.assertTrue(reflection["should_continue"])
        self.assertEqual(usage.total_tokens, 0)

    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_prompt_includes_section_results(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": '{"section_assessments": {}, "should_continue": true, "focus_areas": []}',
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        }

        outline = [{"section": "History", "description": "Historical context"}]
        section_results = {"History": [
            SearchResult(title="History of X", url="https://h.com", snippet="Founded in 1990"),
        ]}

        await reflect_on_findings(
            topic="Topic X", outline=outline, section_results=section_results,
            pass_num=1, total_passes=3,
        )

        # Verify findings were included in the prompt
        call_args = mock_llm.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        all_content = " ".join(m["content"] for m in messages)
        self.assertIn("History of X", all_content)
        self.assertIn("pass 1 of 3", all_content)


class DeepResearchReflectionTestCase(unittest.IsolatedAsyncioTestCase):
    """Test reflection integration in the deep_research pipeline."""

    def _mock_search_results(self, n: int = 3) -> list[SearchResult]:
        return [
            SearchResult(
                title=f"Result {i} about test topic",
                url=f"https://r{i}.com",
                snippet=f"Relevant test topic content {i}",
            )
            for i in range(n)
        ]

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_reflection_runs_on_multipass(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        """With passes=2, reflection should run after each pass."""
        mock_llm.side_effect = [
            # generate_outline
            {"content": '[{"section": "S1", "description": "D1"}]',
             "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}},
            # generate_subqueries (pass 1)
            {"content": '["query 1"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            # reflect_on_findings (after pass 1) — says continue
            {"content": '{"section_assessments": {"S1": {"coverage": "partial", "gaps": ["missing detail X"], "sufficient": false}}, "should_continue": true, "focus_areas": ["detail X"]}',
             "usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}},
            # generate_refined_queries (pass 2)
            {"content": '["refined query"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            # reflect_on_findings (after pass 2)
            {"content": '{"section_assessments": {"S1": {"coverage": "good", "gaps": [], "sufficient": true}}, "should_continue": false, "focus_areas": []}',
             "usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}},
            # synthesize_essay
            {"content": "Final essay.",
             "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}},
        ]
        mock_search.return_value = self._mock_search_results(3)
        mock_select.return_value = (self._mock_search_results(2), TokenUsage())
        mock_enrich.return_value = {}

        result = await deep_research("test topic", stages=1, passes=2)

        self.assertEqual(result.essay, "Final essay.")
        # LLM should have been called 6 times (outline + subqueries + reflect + refined + reflect + synthesis)
        self.assertEqual(mock_llm.call_count, 6)

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_early_stopping_on_sufficient_coverage(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        """If reflection says should_continue=false, remaining passes are skipped."""
        mock_llm.side_effect = [
            # generate_outline
            {"content": '[{"section": "S1", "description": "D1"}]',
             "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}},
            # generate_subqueries (pass 1)
            {"content": '["query 1"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            # reflect_on_findings (after pass 1) — says STOP
            {"content": '{"section_assessments": {"S1": {"coverage": "good", "gaps": [], "sufficient": true}}, "should_continue": false, "focus_areas": []}',
             "usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}},
            # synthesize_essay (pass 2 and 3 are skipped!)
            {"content": "Essay from early stop.",
             "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}},
        ]
        mock_search.return_value = self._mock_search_results(3)
        mock_select.return_value = (self._mock_search_results(2), TokenUsage())
        mock_enrich.return_value = {}

        result = await deep_research("test topic", stages=1, passes=3)

        self.assertEqual(result.essay, "Essay from early stop.")
        # Only 4 LLM calls: outline + subqueries + reflection + synthesis
        # Passes 2 and 3 were skipped
        self.assertEqual(mock_llm.call_count, 4)

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_gap_analysis_fed_to_refinement(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        """Gaps from reflection should appear in the refined query prompt."""
        mock_llm.side_effect = [
            # generate_outline
            {"content": '[{"section": "S1", "description": "D1"}]',
             "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}},
            # generate_subqueries (pass 1)
            {"content": '["query 1"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            # reflect_on_findings — identifies specific gaps
            {"content": '{"section_assessments": {"S1": {"coverage": "partial", "gaps": ["missing recent 2025 data", "no comparison with competitors"], "sufficient": false}}, "should_continue": true, "focus_areas": ["2025 market data"]}',
             "usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}},
            # generate_refined_queries (pass 2) — should contain gap context
            {"content": '["2025 data query"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            # reflect_on_findings (after pass 2)
            {"content": '{"section_assessments": {}, "should_continue": false, "focus_areas": []}',
             "usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}},
            # synthesize_essay
            {"content": "Essay with gaps filled.",
             "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}},
        ]
        mock_search.return_value = self._mock_search_results(3)
        mock_select.return_value = (self._mock_search_results(2), TokenUsage())
        mock_enrich.return_value = {}

        await deep_research("test topic", stages=1, passes=2)

        # The 4th LLM call is generate_refined_queries — check that gap context is in the prompt
        refined_call = mock_llm.call_args_list[3]
        messages = refined_call[1]["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user" and "Generate" in m["content"])
        self.assertIn("missing recent 2025 data", user_msg)
        self.assertIn("no comparison with competitors", user_msg)
        self.assertIn("2025 market data", user_msg)

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_single_pass_skips_reflection(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        """With passes=1, no reflection step should occur."""
        mock_llm.side_effect = [
            # generate_outline
            {"content": '[{"section": "S1", "description": "D1"}]',
             "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}},
            # generate_subqueries
            {"content": '["query 1"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            # synthesize_essay (no reflection!)
            {"content": "Single pass essay.",
             "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}},
        ]
        mock_search.return_value = self._mock_search_results(3)
        mock_select.return_value = (self._mock_search_results(2), TokenUsage())
        mock_enrich.return_value = {}

        result = await deep_research("test topic", stages=1, passes=1)

        self.assertEqual(result.essay, "Single pass essay.")
        # Only 3 LLM calls — no reflection
        self.assertEqual(mock_llm.call_count, 3)

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_reflection_progress_callbacks(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        """Reflection should emit progress callbacks."""
        mock_llm.side_effect = [
            # generate_outline
            {"content": '[{"section": "S1", "description": "D1"}]',
             "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}},
            # generate_subqueries (pass 1)
            {"content": '["query 1"]',
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            # reflect_on_findings — partial coverage
            {"content": '{"section_assessments": {"S1": {"coverage": "partial", "gaps": ["X"], "sufficient": false}}, "should_continue": false, "focus_areas": []}',
             "usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}},
            # synthesize_essay
            {"content": "Essay.",
             "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}},
        ]
        mock_search.return_value = self._mock_search_results(3)
        mock_select.return_value = (self._mock_search_results(2), TokenUsage())
        mock_enrich.return_value = {}

        progress_calls: list[tuple[str, str]] = []

        def cb(stage: str, msg: str):
            progress_calls.append((stage, msg))

        await deep_research("test topic", stages=1, passes=2, progress_callback=cb)

        stages = [s for s, _ in progress_calls]
        self.assertIn("reflection", stages)

    @patch("artemis.researcher.enrich_results", new_callable=AsyncMock)
    @patch("artemis.researcher.select_relevant_results", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.chat_completion", new_callable=AsyncMock)
    async def test_reflection_failure_continues_pipeline(
        self,
        mock_llm: AsyncMock,
        mock_search: AsyncMock,
        mock_select: AsyncMock,
        mock_enrich: AsyncMock,
    ) -> None:
        """If the reflection LLM call fails, the pipeline should continue."""
        call_count = 0

        async def llm_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"content": '[{"section": "S1", "description": "D1"}]',
                        "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}}
            elif call_count == 2:
                return {"content": '["query 1"]',
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
            elif call_count == 3:
                # Reflection fails
                raise UpstreamServiceError("LLM temporarily unavailable")
            elif call_count == 4:
                # Pass 2 refined queries (should still run because failure = continue)
                return {"content": '["refined query"]',
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
            elif call_count == 5:
                # Pass 2 reflection
                return {"content": '{"section_assessments": {}, "should_continue": false, "focus_areas": []}',
                        "usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}}
            else:
                return {"content": "Essay despite reflection failure.",
                        "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}}

        mock_llm.side_effect = llm_side_effect
        mock_search.return_value = self._mock_search_results(3)
        mock_select.return_value = (self._mock_search_results(2), TokenUsage())
        mock_enrich.return_value = {}

        result = await deep_research("test topic", stages=1, passes=2)

        self.assertEqual(result.essay, "Essay despite reflection failure.")


class SupervisedResearcherNoteToolTestCase(unittest.IsolatedAsyncioTestCase):
    """Test the enhanced note tool in supervised researchers."""

    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    async def test_note_tool_prompt_in_system_message(
        self,
        mock_agentic: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
    ) -> None:
        """Researcher system prompt should mandate note usage."""
        from artemis.researcher import _run_researcher

        mock_agentic.return_value = {
            "content": "Findings.",
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            "tool_calls_made": 0,
        }

        await _run_researcher(
            topic="test", section="S", description="D",
            results_per_query=5, max_tool_rounds=10, content_max_chars=3000,
        )

        messages = mock_agentic.call_args[1]["messages"]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        self.assertIn("MUST call the note tool", system_msg)
        self.assertIn("What did I find", system_msg)
        self.assertIn("What specific gaps", system_msg)
