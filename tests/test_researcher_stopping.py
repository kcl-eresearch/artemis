"""Unit tests for researcher stopping heuristics."""

import json
import unittest
from unittest.mock import AsyncMock, patch

from artemis.models import SearchResult, TokenUsage
from artemis.researcher import _ResearcherState, _run_researcher


class ResearcherStateTestCase(unittest.TestCase):
    """Test the _ResearcherState stopping heuristics."""

    def test_should_stop_returns_none_initially(self) -> None:
        state = _ResearcherState()
        self.assertIsNone(state.should_stop())

    def test_should_stop_returns_none_with_insufficient_sources(self) -> None:
        state = _ResearcherState(min_relevant_sources=3)
        state.record_page_read("A" * 300)
        state.record_page_read("B" * 300)
        self.assertIsNone(state.should_stop())

    def test_should_stop_triggers_on_sufficient_sources(self) -> None:
        state = _ResearcherState(min_relevant_sources=3)
        for i in range(3):
            state.record_page_read("Content " * 50)
        reason = state.should_stop()
        self.assertIsNotNone(reason)
        self.assertIn("Sufficient sources", reason)
        self.assertEqual(state.last_stop_reason, reason)

    def test_short_content_not_counted_as_relevant(self) -> None:
        state = _ResearcherState(min_relevant_sources=3)
        for _ in range(5):
            state.record_page_read("short")  # < 200 chars
        self.assertIsNone(state.should_stop())
        self.assertEqual(state.relevant_source_count, 0)

    def test_none_content_not_counted(self) -> None:
        state = _ResearcherState(min_relevant_sources=3)
        state.record_page_read(None)
        self.assertEqual(state.relevant_source_count, 0)

    def test_custom_min_sources_threshold(self) -> None:
        state = _ResearcherState(min_relevant_sources=5)
        for _ in range(4):
            state.record_page_read("X" * 300)
        self.assertIsNone(state.should_stop())
        state.record_page_read("X" * 300)
        self.assertIsNotNone(state.should_stop())

    def test_overlap_not_triggered_with_one_search(self) -> None:
        state = _ResearcherState(overlap_threshold=0.6)
        results = [SearchResult(title="R", url="https://a.com", snippet="S")]
        state.record_search(results)
        self.assertFalse(state.last_two_searches_overlap())

    def test_overlap_triggered_with_identical_searches(self) -> None:
        state = _ResearcherState(overlap_threshold=0.6)
        results = [
            SearchResult(title="R1", url="https://a.com", snippet="S1"),
            SearchResult(title="R2", url="https://b.com", snippet="S2"),
        ]
        state.record_search(results)
        state.record_search(results)  # Same URLs
        self.assertTrue(state.last_two_searches_overlap())

    def test_overlap_not_triggered_with_diverse_searches(self) -> None:
        state = _ResearcherState(overlap_threshold=0.6)
        results1 = [
            SearchResult(title="R1", url="https://a.com", snippet="S1"),
            SearchResult(title="R2", url="https://b.com", snippet="S2"),
        ]
        results2 = [
            SearchResult(title="R3", url="https://c.com", snippet="S3"),
            SearchResult(title="R4", url="https://d.com", snippet="S4"),
        ]
        state.record_search(results1)
        state.record_search(results2)
        self.assertFalse(state.last_two_searches_overlap())

    def test_overlap_partial_above_threshold(self) -> None:
        state = _ResearcherState(overlap_threshold=0.5)
        results1 = [
            SearchResult(title="R1", url="https://a.com", snippet="S"),
            SearchResult(title="R2", url="https://b.com", snippet="S"),
            SearchResult(title="R3", url="https://c.com", snippet="S"),
        ]
        # 2 out of 3 overlap = 2/4 union = 0.5, meets threshold
        results2 = [
            SearchResult(title="R1", url="https://a.com", snippet="S"),
            SearchResult(title="R2", url="https://b.com", snippet="S"),
            SearchResult(title="R4", url="https://d.com", snippet="S"),
        ]
        state.record_search(results1)
        state.record_search(results2)
        self.assertTrue(state.last_two_searches_overlap())

    def test_overlap_triggers_should_stop(self) -> None:
        """Overlap detection feeds into should_stop()."""
        state = _ResearcherState(min_relevant_sources=10, overlap_threshold=0.6)
        results = [SearchResult(title="R", url="https://a.com", snippet="S")]
        state.record_search(results)
        state.record_search(results)
        reason = state.should_stop()
        self.assertIsNotNone(reason)
        self.assertIn("Diminishing returns", reason)

    def test_empty_search_results_handled(self) -> None:
        state = _ResearcherState()
        state.record_search([])
        state.record_search([])
        # Both sets empty — overlap check should not trigger
        self.assertFalse(state.last_two_searches_overlap())


class AgenticShouldStopTestCase(unittest.IsolatedAsyncioTestCase):
    """Test that agentic_chat_completion respects the should_stop callback."""

    @patch("artemis.llm._post_completion", new_callable=AsyncMock)
    async def test_forces_text_after_should_stop(self, mock_post: AsyncMock) -> None:
        from artemis.llm import agentic_chat_completion

        call_count = 0
        bodies_sent: list[dict] = []

        async def fake_post(client, url, body):
            nonlocal call_count
            bodies_sent.append(body)
            call_count += 1
            if call_count == 1:
                # Round 0: tool call returned
                return {
                    "choices": [{
                        "message": {
                            "content": None,
                            "tool_calls": [{
                                "id": f"call_{call_count}",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": json.dumps({"query": "test 1"}),
                                },
                            }],
                        }
                    }],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            else:
                # Round 1: text response (forced by should_stop)
                return {
                    "choices": [{
                        "message": {"content": "Final findings."}
                    }],
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                }

        mock_post.side_effect = fake_post

        triggered = False

        def should_stop():
            if triggered:
                return "Test stop reason"
            return None

        async def search_handler(query: str) -> str:
            nonlocal triggered
            triggered = True  # Trigger stop after first tool call
            return "search results"

        result = await agentic_chat_completion(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            max_tokens=1000,
            tool_handlers={"web_search": search_handler},
            max_tool_rounds=10,
            should_stop=should_stop,
        )

        self.assertEqual(result["content"], "Final findings.")
        self.assertEqual(len(bodies_sent), 2)
        # First request: tool_choice="auto"
        self.assertEqual(bodies_sent[0]["tool_choice"], "auto")
        # Second request: tool_choice="none" (forced by should_stop)
        self.assertEqual(bodies_sent[1]["tool_choice"], "none")


class RunResearcherStoppingTestCase(unittest.IsolatedAsyncioTestCase):
    """Test stopping heuristics integration in _run_researcher()."""

    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    async def test_stop_reason_in_output(
        self,
        mock_agentic: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
    ) -> None:
        mock_agentic.return_value = {
            "content": "Findings text.",
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "tool_calls_made": 2,
        }

        result = await _run_researcher(
            topic="test",
            section="Overview",
            description="General overview",
            results_per_query=5,
            max_tool_rounds=10,
            content_max_chars=3000,
        )

        self.assertIn("stop_reason", result)
        # No tool calls actually executed (mocked), so default reason
        self.assertEqual(result["stop_reason"], "max_rounds")

    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    async def test_should_stop_callback_passed_to_agentic(
        self,
        mock_agentic: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
    ) -> None:
        """Verify _run_researcher passes should_stop callback."""
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
        self.assertIn("should_stop", call_kwargs)
        self.assertIsNotNone(call_kwargs["should_stop"])
        # The callback should be callable
        self.assertTrue(callable(call_kwargs["should_stop"]))

    @patch("artemis.researcher.fetch_and_extract", new_callable=AsyncMock)
    @patch("artemis.researcher.search_searxng", new_callable=AsyncMock)
    @patch("artemis.researcher.agentic_chat_completion", new_callable=AsyncMock)
    async def test_system_prompt_mentions_early_stopping(
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

        messages = mock_agentic.call_args[1]["messages"]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        self.assertIn("system may ask you to write your findings early", system_msg)
