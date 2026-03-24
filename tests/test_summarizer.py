"""Tests for the summarizer module."""

import unittest
from unittest.mock import AsyncMock, patch

from artemis.models import SearchResult
from artemis.summarizer import summarize_results


class SummarizeResultsTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_empty_results_returns_none(self) -> None:
        result = await summarize_results("test query", [], model="gpt-4")
        self.assertIsNone(result["summary"])
        self.assertIsNone(result["usage"])

    @patch("artemis.summarizer.chat_completion", new_callable=AsyncMock)
    async def test_calls_llm_with_tool_messages(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {
            "content": "Summary of results",
            "usage": {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
        }
        results = [
            SearchResult(title="Result 1", url="https://a.com", snippet="Snippet 1"),
            SearchResult(title="Result 2", url="https://b.com", snippet="Snippet 2"),
        ]

        output = await summarize_results("test query", results, model="gpt-4", max_tokens=512)

        self.assertEqual(output["summary"], "Summary of results")
        self.assertEqual(output["usage"]["total_tokens"], 80)
        mock_llm.assert_awaited_once()

        # Verify messages structure: system, user (with search_results)
        messages = mock_llm.call_args[1]["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("test query", messages[1]["content"])
        self.assertIn("<search_results>", messages[1]["content"])
        # Search results content should include result data
        self.assertIn("Result 1", messages[1]["content"])
        self.assertIn("Snippet 2", messages[1]["content"])

    @patch("artemis.summarizer.chat_completion", new_callable=AsyncMock)
    async def test_passes_model_and_max_tokens(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = {"content": "Summary", "usage": None}
        results = [SearchResult(title="R", url="https://r.com", snippet="S")]

        await summarize_results("q", results, model="claude-3", max_tokens=2048)

        self.assertEqual(mock_llm.call_args[1]["model"], "claude-3")
        self.assertEqual(mock_llm.call_args[1]["max_tokens"], 2048)
