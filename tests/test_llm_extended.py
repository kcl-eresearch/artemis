"""Tests for the LLM chat_completion function and client management."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from artemis.errors import UpstreamServiceError
from artemis.llm import chat_completion, _normalize_usage


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


def _valid_llm_response(content: str = "Hello!", usage: dict | None = None) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class ChatCompletionTestCase(unittest.IsolatedAsyncioTestCase):
    @patch("artemis.llm._get_client")
    async def test_successful_completion(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(_valid_llm_response("Test answer"))
        mock_get_client.return_value = mock_client

        result = await chat_completion("test prompt", "gpt-4", 1000)
        self.assertEqual(result["content"], "Test answer")
        self.assertIsNotNone(result["usage"])

    @patch("artemis.llm._get_client")
    async def test_timeout_raises_upstream_error(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", "gpt-4", 1000)
        self.assertIn("timed out", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_http_error_raises_upstream_error(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        response = MagicMock()
        response.status_code = 500
        mock_client.post.return_value = response
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=response
        )
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", "gpt-4", 1000)
        self.assertIn("500", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_connection_error_raises_upstream_error(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", "gpt-4", 1000)
        self.assertIn("request failed", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_invalid_json_raises_upstream_error(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.side_effect = ValueError("invalid json")
        mock_client.post.return_value = resp
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", "gpt-4", 1000)
        self.assertIn("invalid JSON", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_empty_choices_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response({"choices": []})
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", "gpt-4", 1000)
        self.assertIn("no completion choices", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_missing_choices_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response({"result": "bad"})
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError):
            await chat_completion("test", "gpt-4", 1000)

    @patch("artemis.llm._get_client")
    async def test_empty_content_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": ""}}]}
        )
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", "gpt-4", 1000)
        self.assertIn("empty content", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_think_blocks_stripped(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        content = "<think>internal reasoning</think>Actual answer here"
        mock_client.post.return_value = _mock_response(_valid_llm_response(content))
        mock_get_client.return_value = mock_client

        result = await chat_completion("test", "gpt-4", 1000)
        self.assertEqual(result["content"], "Actual answer here")
        self.assertNotIn("<think>", result["content"])

    @patch("artemis.llm._get_client")
    async def test_only_think_block_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        content = "<think>only reasoning, no answer</think>"
        mock_client.post.return_value = _mock_response(_valid_llm_response(content))
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", "gpt-4", 1000)
        self.assertIn("empty content", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_model_name_split(self, mock_get_client: MagicMock) -> None:
        """Model name with provider prefix is split correctly."""
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(_valid_llm_response())
        mock_get_client.return_value = mock_client

        await chat_completion("test", "openai/gpt-4", 1000)
        call_body = mock_client.post.call_args[1]["json"]
        self.assertEqual(call_body["model"], "gpt-4")

    @patch("artemis.llm._get_client")
    async def test_response_format_passed(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(_valid_llm_response('{"key": "val"}'))
        mock_get_client.return_value = mock_client

        await chat_completion("test", "gpt-4", 1000, response_format={"type": "json_object"})
        call_body = mock_client.post.call_args[1]["json"]
        self.assertEqual(call_body["response_format"], {"type": "json_object"})

    @patch("artemis.llm._get_client")
    async def test_invalid_choice_type_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response({"choices": ["not a dict"]})
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", "gpt-4", 1000)
        self.assertIn("invalid choice", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_invalid_message_type_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(
            {"choices": [{"message": "not a dict"}]}
        )
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", "gpt-4", 1000)
        self.assertIn("invalid message", str(ctx.exception))
