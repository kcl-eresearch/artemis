"""Shared LiteLLM-compatible client helpers.

This module provides a thin wrapper around LiteLLM-compatible LLM endpoints.
It handles authentication, request formatting, response parsing, and error handling
for chat completion API calls.

The client works with any LiteLLM-compatible API (OpenAI, Anthropic, local models, etc.)
and normalizes token usage tracking across different provider formats.

A module-level httpx.AsyncClient is used for connection pooling across requests.

Content isolation
-----------------
Untrusted web content (search results, extracted pages) should be delivered to the
LLM via tool-call messages rather than user messages.  Models treat tool responses
as returned data, not instructions, providing an architectural privilege boundary
against prompt injection.  Use :func:`build_tool_messages` to construct the
correct conversation structure.
"""

import logging
import re
import uuid
from typing import Any

import httpx

from artemis.config import get_settings
from artemis.errors import UpstreamServiceError

logger = logging.getLogger(__name__)

# Default HTTP headers for all LLM API requests
_DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Artemis/0.2.0",
}

# Module-level client for connection pooling (created lazily)
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient, creating it lazily on first use."""
    global _client
    if _client is None or _client.is_closed:
        settings = get_settings()
        headers = dict(_DEFAULT_HEADERS)
        if settings.litellm_api_key:
            headers["Authorization"] = f"Bearer {settings.litellm_api_key}"
        _client = httpx.AsyncClient(
            timeout=settings.llm_timeout_seconds,
            follow_redirects=False,
            headers=headers,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
        )
    return _client


async def close_client() -> None:
    """Close the shared httpx client (called during app shutdown)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python, no numpy)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def embed(text: str, model: str) -> list[float]:
    """Compute an embedding vector via the LiteLLM-compatible embeddings endpoint.

    Args:
        text: The text to embed
        model: The embedding model identifier

    Returns:
        Embedding vector as a list of floats

    Raises:
        UpstreamServiceError: If the embedding endpoint is unreachable or errors
    """
    settings = get_settings()
    client = _get_client()
    body = {"model": model, "input": text}

    try:
        response = await client.post(
            f"{settings.litellm_base_url}/embeddings", json=body
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise UpstreamServiceError("The embedding endpoint timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise UpstreamServiceError(
            f"The embedding endpoint returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise UpstreamServiceError("The embedding request failed.") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise UpstreamServiceError(
            "The embedding endpoint returned invalid JSON."
        ) from exc

    try:
        return data["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise UpstreamServiceError(
            "The embedding response has an unexpected structure."
        ) from exc


def _normalize_usage(usage: Any) -> dict[str, int] | None:
    """Normalize token usage from various LLM provider formats.

    Different LLM providers use different field names for token counts.
    This function normalizes them to a consistent format.

    Handles:
    - prompt_tokens / input_tokens -> input_tokens
    - completion_tokens / output_tokens -> output_tokens
    - total_tokens

    Args:
        usage: Raw usage dictionary from LLM response

    Returns:
        Normalized usage dict or None if input is invalid
    """
    if not isinstance(usage, dict):
        return None

    raw_input = usage.get("input_tokens")
    if raw_input is None:
        raw_input = usage.get("prompt_tokens", 0)
    input_tokens = int(raw_input or 0)

    raw_output = usage.get("output_tokens")
    if raw_output is None:
        raw_output = usage.get("completion_tokens", 0)
    output_tokens = int(raw_output or 0)

    raw_total = usage.get("total_tokens")
    total_tokens = int(raw_total) if raw_total is not None else (input_tokens + output_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Content isolation helpers
# ---------------------------------------------------------------------------

_SANITIZE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_content(text: str, max_length: int | None = None) -> str:
    """Strip control characters from untrusted text.

    This is a defence-in-depth measure applied before content enters any
    message role.  It removes ASCII control characters (except newline,
    carriage return, and tab) that could confuse tokenisers or hide
    injected instructions.

    Args:
        text: Raw untrusted string (title, snippet, extracted page, etc.)
        max_length: Optional hard truncation limit.

    Returns:
        Cleaned string.
    """
    cleaned = _SANITIZE_RE.sub("", text)
    if max_length is not None:
        cleaned = cleaned[:max_length]
    return cleaned


def build_tool_messages(
    *,
    system: str,
    user: str,
    tool_content: str,
    tool_name: str = "web_search",
) -> list[dict[str, Any]]:
    """Build a message list that delivers untrusted content via a tool response.

    The returned conversation looks like::

        1. system   – LLM instructions
        2. user     – the user's query / task
        3. assistant – a synthetic tool_call requesting *tool_name*
        4. tool     – the untrusted web content as the tool's return value

    Models treat message 4 as returned data rather than instructions,
    providing an architectural privilege boundary against prompt injection.

    Args:
        system: System-level instructions for the LLM.
        user: The user query or task description.
        tool_content: Untrusted content (search results, extracted pages, etc.).
            Should already be passed through :func:`sanitize_content`.
        tool_name: Name to use for the synthetic tool call.

    Returns:
        List of message dicts ready for the chat completions API.
    """
    call_id = f"call_{uuid.uuid4().hex[:24]}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": tool_content,
        },
    ]


async def chat_completion(
    prompt: str | None = None,
    *,
    messages: list[dict[str, Any]] | None = None,
    model: str,
    max_tokens: int,
    response_format: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Send a chat completion request to the LLM backend.

    Accepts either a simple ``prompt`` string (wrapped as a single user
    message for backward compatibility) or a pre-built ``messages`` list
    for full control over role separation and tool-call isolation.

    Args:
        prompt: Simple prompt string (legacy). Mutually exclusive with *messages*.
        messages: Pre-built message list. Use :func:`build_tool_messages` to
            construct conversations that isolate untrusted content.
        model: Model identifier (e.g., "gpt-4", "claude-3-opus").
        max_tokens: Maximum tokens in the response.
        response_format: Optional response format hint (e.g. {"type": "json_object"}).

    Returns:
        Dictionary with:
        - "content": Generated text response
        - "usage": Normalized token usage dict

    Raises:
        UpstreamServiceError: If the LLM request fails or returns invalid data
        ValueError: If both or neither of *prompt* and *messages* are provided
    """
    if (prompt is None) == (messages is None):
        raise ValueError("Provide exactly one of 'prompt' or 'messages'.")

    if prompt is not None:
        messages = [{"role": "user", "content": prompt}]
    settings = get_settings()
    model_name = model.split("/")[-1] if "/" in model else model
    client = _get_client()

    # Determine whether tool definitions are needed (when messages contain
    # a tool-call turn we must declare the tool so the API accepts it).
    has_tool_call = any(
        m.get("role") == "assistant" and m.get("tool_calls")
        for m in messages  # type: ignore[union-attr]
    )

    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if has_tool_call:
        # Extract unique tool names from assistant tool_calls
        tool_names: set[str] = set()
        for m in messages:  # type: ignore[union-attr]
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {})
                if fn.get("name"):
                    tool_names.add(fn["name"])
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Retrieve information from the web.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in sorted(tool_names)
        ]
        # Tool defs are only declared so the API accepts the tool-call
        # history — we never want the model to call tools in its response.
        body["tool_choice"] = "none"
    if response_format is not None:
        body["response_format"] = response_format
    # Ask the backend to suppress <think> reasoning blocks if supported
    # (works with sglang, vllm, and other OpenAI-compatible backends)
    body["include_reasoning"] = False

    try:
        response = await client.post(
            f"{settings.litellm_base_url}/chat/completions",
            json=body,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise UpstreamServiceError("The LLM backend timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise UpstreamServiceError(
            f"The LLM backend returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise UpstreamServiceError("The LLM backend request failed.") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise UpstreamServiceError("The LLM backend returned invalid JSON.") from exc

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamServiceError("The LLM backend returned no completion choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise UpstreamServiceError(
            "The LLM backend returned an invalid choice payload."
        )

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise UpstreamServiceError(
            "The LLM backend returned an invalid message payload."
        )

    content = message.get("content")
    # Some APIs return content as a list of blocks, e.g. [{"type":"text","text":"..."}]
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if not isinstance(content, str) or not content.strip():
        raise UpstreamServiceError("The LLM backend returned empty content.")

    # Strip <think>...</think> reasoning blocks some models emit.
    # Also handle orphaned tags: leading content before a lone </think>
    # (opening tag was outside this response) or trailing <think> without close.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"^.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL)
    content = content.strip()
    if not content:
        raise UpstreamServiceError("The LLM backend returned empty content.")

    return {
        "content": content,
        "usage": _normalize_usage(data.get("usage")),
    }
