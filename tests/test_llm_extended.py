"""Tests for the LLM chat_completion function and client management."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from artemis.errors import UpstreamServiceError
from artemis.llm import (
    chat_completion,
    build_tool_messages,
    sanitize_content,
    _normalize_usage,
)


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

        result = await chat_completion("test prompt", model="gpt-4", max_tokens=1000)
        self.assertEqual(result["content"], "Test answer")
        self.assertIsNotNone(result["usage"])

    @patch("artemis.llm._get_client")
    async def test_timeout_raises_upstream_error(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", model="gpt-4", max_tokens=1000)
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
            await chat_completion("test", model="gpt-4", max_tokens=1000)
        self.assertIn("500", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_connection_error_raises_upstream_error(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", model="gpt-4", max_tokens=1000)
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
            await chat_completion("test", model="gpt-4", max_tokens=1000)
        self.assertIn("invalid JSON", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_empty_choices_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response({"choices": []})
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", model="gpt-4", max_tokens=1000)
        self.assertIn("no completion choices", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_missing_choices_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response({"result": "bad"})
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError):
            await chat_completion("test", model="gpt-4", max_tokens=1000)

    @patch("artemis.llm._get_client")
    async def test_empty_content_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(
            {"choices": [{"message": {"content": ""}}]}
        )
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", model="gpt-4", max_tokens=1000)
        self.assertIn("empty content", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_think_blocks_stripped(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        content = "<think>internal reasoning</think>Actual answer here"
        mock_client.post.return_value = _mock_response(_valid_llm_response(content))
        mock_get_client.return_value = mock_client

        result = await chat_completion("test", model="gpt-4", max_tokens=1000)
        self.assertEqual(result["content"], "Actual answer here")
        self.assertNotIn("<think>", result["content"])

    @patch("artemis.llm._get_client")
    async def test_only_think_block_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        content = "<think>only reasoning, no answer</think>"
        mock_client.post.return_value = _mock_response(_valid_llm_response(content))
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", model="gpt-4", max_tokens=1000)
        self.assertIn("empty content", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_model_name_split(self, mock_get_client: MagicMock) -> None:
        """Model name with provider prefix is split correctly."""
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(_valid_llm_response())
        mock_get_client.return_value = mock_client

        await chat_completion("test", model="openai/gpt-4", max_tokens=1000)
        call_body = mock_client.post.call_args[1]["json"]
        self.assertEqual(call_body["model"], "gpt-4")

    @patch("artemis.llm._get_client")
    async def test_response_format_passed(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(_valid_llm_response('{"key": "val"}'))
        mock_get_client.return_value = mock_client

        await chat_completion("test", model="gpt-4", max_tokens=1000, response_format={"type": "json_object"})
        call_body = mock_client.post.call_args[1]["json"]
        self.assertEqual(call_body["response_format"], {"type": "json_object"})

    @patch("artemis.llm._get_client")
    async def test_invalid_choice_type_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response({"choices": ["not a dict"]})
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", model="gpt-4", max_tokens=1000)
        self.assertIn("invalid choice", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_invalid_message_type_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(
            {"choices": [{"message": "not a dict"}]}
        )
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await chat_completion("test", model="gpt-4", max_tokens=1000)
        self.assertIn("invalid message", str(ctx.exception))

    @patch("artemis.llm._get_client")
    async def test_messages_interface(self, mock_get_client: MagicMock) -> None:
        """Messages list is sent directly to the API."""
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(_valid_llm_response("reply"))
        mock_get_client.return_value = mock_client

        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = await chat_completion(messages=msgs, model="gpt-4", max_tokens=500)
        self.assertEqual(result["content"], "reply")
        call_body = mock_client.post.call_args[1]["json"]
        self.assertEqual(call_body["messages"], msgs)
        self.assertNotIn("tools", call_body)

    @patch("artemis.llm._get_client")
    async def test_tool_messages_include_tools(self, mock_get_client: MagicMock) -> None:
        """When messages contain tool calls, tools definition is included."""
        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(_valid_llm_response("answer"))
        mock_get_client.return_value = mock_client

        msgs = build_tool_messages(
            system="Instructions",
            user="Query",
            tool_content="search results here",
        )
        await chat_completion(messages=msgs, model="gpt-4", max_tokens=500)
        call_body = mock_client.post.call_args[1]["json"]
        self.assertIn("tools", call_body)
        self.assertEqual(call_body["tools"][0]["function"]["name"], "web_search")

    async def test_both_prompt_and_messages_raises(self) -> None:
        with self.assertRaises(ValueError):
            await chat_completion(
                "prompt",
                messages=[{"role": "user", "content": "msg"}],
                model="gpt-4",
                max_tokens=100,
            )

    async def test_neither_prompt_nor_messages_raises(self) -> None:
        with self.assertRaises(ValueError):
            await chat_completion(model="gpt-4", max_tokens=100)


class SanitizeContentTestCase(unittest.TestCase):
    def test_strips_control_characters(self) -> None:
        self.assertEqual(sanitize_content("hello\x00world\x07"), "helloworld")

    def test_preserves_newlines_tabs(self) -> None:
        self.assertEqual(sanitize_content("line1\nline2\ttab"), "line1\nline2\ttab")

    def test_max_length(self) -> None:
        self.assertEqual(sanitize_content("abcdefghij", max_length=5), "abcde")

    def test_empty_string(self) -> None:
        self.assertEqual(sanitize_content(""), "")


class BuildToolMessagesTestCase(unittest.TestCase):
    def test_structure(self) -> None:
        msgs = build_tool_messages(
            system="sys", user="usr", tool_content="data"
        )
        self.assertEqual(len(msgs), 4)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "sys")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"], "usr")
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertIsNotNone(msgs[2]["tool_calls"])
        self.assertEqual(msgs[3]["role"], "tool")
        self.assertEqual(msgs[3]["content"], "data")

    def test_tool_call_ids_match(self) -> None:
        msgs = build_tool_messages(
            system="s", user="u", tool_content="c"
        )
        call_id = msgs[2]["tool_calls"][0]["id"]
        self.assertEqual(msgs[3]["tool_call_id"], call_id)

    def test_custom_tool_name(self) -> None:
        msgs = build_tool_messages(
            system="s", user="u", tool_content="c", tool_name="fetch_page"
        )
        self.assertEqual(
            msgs[2]["tool_calls"][0]["function"]["name"], "fetch_page"
        )
