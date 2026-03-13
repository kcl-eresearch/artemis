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

import json
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


def _extract_message_content(
    data: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract the assistant message and its text content from a /chat/completions response.

    Returns ``(message, content)`` where *content* is ``None`` when the
    response contains tool calls instead of text (or is otherwise empty).

    Raises:
        UpstreamServiceError: If the response structure is fundamentally invalid.
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamServiceError("The LLM backend returned no completion choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise UpstreamServiceError("The LLM backend returned an invalid choice payload.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise UpstreamServiceError("The LLM backend returned an invalid message payload.")

    content = message.get("content")
    # Some APIs return content as a list of blocks, e.g. [{"type":"text","text":"..."}]
    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if not isinstance(content, str) or not content.strip():
        return message, None
    return message, content


async def _post_completion(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    """POST to a /chat/completions endpoint and return the parsed JSON response."""
    try:
        response = await client.post(url, json=body)
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
        return response.json()
    except ValueError as exc:
        raise UpstreamServiceError("The LLM backend returned invalid JSON.") from exc


def _strip_llm_artifacts(content: str) -> str:
    """Strip non-content artifacts that models sometimes emit as plain text.

    Handles:
    - ``<think>…</think>`` reasoning blocks (and orphaned open/close tags)
    - Text-based tool-call blocks that some models emit instead of using the
      structured ``tool_calls`` response field, e.g.::

          <minimax:tool_call>…</minimax:tool_call>
          <tool_call>…</tool_call>
          <|tool_call|>…<|/tool_call|>

    Returns the cleaned string (may be empty).
    """
    # Think / reasoning blocks (including orphaned tags)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"^.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL)

    # Text-based tool calls: <prefix:tool_call>…</prefix:tool_call> and
    # <tool_call>…</tool_call> with optional pipe delimiters
    content = re.sub(
        r"<\|?[\w:]*tool_call\|?>.*?<\|?/[\w:]*tool_call\|?>",
        "",
        content,
        flags=re.DOTALL,
    )
    # Orphaned opening tag at the end (model hit token limit mid-call)
    content = re.sub(
        r"<\|?[\w:]*tool_call\|?>.*$",
        "",
        content,
        flags=re.DOTALL,
    )

    return content.strip()


async def agentic_chat_completion(
    *,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    tool_handlers: dict[str, Any],
    max_tool_rounds: int = 5,
    response_format: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Chat completion with a live tool-execution loop.

    Declares ``tool_handlers`` as callable tools and executes any tool calls
    the model makes, feeding results back until it produces a text response.

    If ``messages`` already contain tool-call history (e.g. from
    :func:`build_tool_messages`), those tool names are also declared so the
    API accepts the conversation.  When both a historical call and an active
    handler share the same name, the handler's parameter schema wins.

    Args:
        messages: Conversation so far (may include pre-loaded tool-call history).
        model: LLM model identifier.
        max_tokens: Maximum tokens in the final response.
        tool_handlers: Mapping of tool name → async callable.  Each callable
            receives keyword arguments extracted from the model's tool-call
            arguments JSON and must return a ``str`` result.
        max_tool_rounds: Maximum number of tool-call rounds before forcing a
            text response with ``tool_choice="none"``.
        response_format: Optional response format hint.

    Returns:
        Dictionary with:
        - ``"content"``: Generated text response
        - ``"usage"``: Accumulated normalised token usage across all rounds
        - ``"tool_calls_made"``: Total number of individual tool calls executed
    """
    settings = get_settings()
    model_name = model.split("/")[-1] if "/" in model else model
    client = _get_client()
    url = f"{settings.litellm_base_url}/chat/completions"

    # Collect all tool names: ones we can execute + historical ones already in messages
    active_names: set[str] = set(tool_handlers.keys())
    history_names: set[str] = set()
    for m in messages:
        for tc in m.get("tool_calls", []):
            name = tc.get("function", {}).get("name")
            if name:
                history_names.add(name)

    tools: list[dict[str, Any]] = []
    for name in sorted(active_names | history_names):
        if name in active_names:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "Search the web for information.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query string",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            )
        else:
            # Historical-only: declared so the API accepts the existing turn
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "Retrieve information from the web.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            )

    current_messages = list(messages)
    accumulated_usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    total_tool_calls = 0

    for round_num in range(max_tool_rounds + 1):
        # On the last allowed round force a text response
        tool_choice = "none" if round_num >= max_tool_rounds else "auto"

        body: dict[str, Any] = {
            "model": model_name,
            "messages": current_messages,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
            "include_reasoning": False,
        }
        if response_format is not None:
            body["response_format"] = response_format

        data = await _post_completion(client, url, body)

        round_usage = _normalize_usage(data.get("usage"))
        if round_usage:
            for k in accumulated_usage:
                accumulated_usage[k] += round_usage.get(k, 0)

        message, content = _extract_message_content(data)

        if content is not None:
            content = _strip_llm_artifacts(content)
            if content:
                return {
                    "content": content,
                    "usage": accumulated_usage,
                    "tool_calls_made": total_tool_calls,
                }

        tool_calls_in_response = message.get("tool_calls") if message else None
        if not tool_calls_in_response:
            raise UpstreamServiceError("The LLM backend returned empty content.")

        if round_num >= max_tool_rounds:
            raise UpstreamServiceError(
                f"Model made tool calls across all {max_tool_rounds} rounds "
                "without producing a text response."
            )

        logger.info(
            "Synthesis tool round %d/%d: %d call(s)",
            round_num + 1,
            max_tool_rounds,
            len(tool_calls_in_response),
        )

        current_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_in_response,
            }
        )

        for tc in tool_calls_in_response:
            tc_id = tc.get("id", "")
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            try:
                tool_args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                tool_args = {}

            handler = tool_handlers.get(tool_name)
            if handler:
                try:
                    result = await handler(**tool_args)
                    total_tool_calls += 1
                except Exception as exc:
                    logger.warning("Tool handler %r failed: %s", tool_name, exc)
                    result = f"Search failed: {exc}"
            else:
                result = f"Unknown tool: {tool_name}"

            current_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result,
                }
            )

    raise UpstreamServiceError(
        "Agentic loop exhausted without producing a text response."
    )


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

    data = await _post_completion(client, f"{settings.litellm_base_url}/chat/completions", body)

    message, content = _extract_message_content(data)

    # Handle models that ignore tool_choice="none" and return tool calls instead
    # of text content.  Fulfil each call with an empty result and retry once so
    # the model writes its actual response.
    if content is None:
        tool_calls_in_response = message.get("tool_calls") if message else None
        if not tool_calls_in_response:
            raise UpstreamServiceError("The LLM backend returned empty content.")

        logger.warning(
            "Model returned %d tool call(s) instead of text content; "
            "fulfilling with empty results and retrying",
            len(tool_calls_in_response),
        )
        retry_messages = list(messages)  # type: ignore[arg-type]
        retry_messages.append(
            {"role": "assistant", "content": None, "tool_calls": tool_calls_in_response}
        )
        for tc in tool_calls_in_response:
            retry_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": (
                        "No additional results are available. "
                        "Write your response using only the previously provided information."
                    ),
                }
            )

        # Collect all tool names now present in the conversation
        all_tool_names: set[str] = set()
        for m in retry_messages:
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {})
                if fn.get("name"):
                    all_tool_names.add(fn["name"])

        retry_body: dict[str, Any] = {
            "model": model_name,
            "messages": retry_messages,
            "max_tokens": max_tokens,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "Retrieve information from the web.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in sorted(all_tool_names)
            ],
            "tool_choice": "none",
            "include_reasoning": False,
        }
        if response_format is not None:
            retry_body["response_format"] = response_format

        try:
            data = await _post_completion(
                client, f"{settings.litellm_base_url}/chat/completions", retry_body
            )
        except UpstreamServiceError as exc:
            raise UpstreamServiceError(
                str(exc).replace("backend", "backend (retry)")
            ) from exc

        _, content = _extract_message_content(data)
        if content is None:
            raise UpstreamServiceError(
                "The LLM backend returned empty content even after fulfilling tool calls."
            )

    content = _strip_llm_artifacts(content)
    if not content:
        raise UpstreamServiceError("The LLM backend returned empty content.")

    return {
        "content": content,
        "usage": _normalize_usage(data.get("usage")),
    }
