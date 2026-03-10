"""Shared LiteLLM-compatible client helpers.

This module provides a thin wrapper around LiteLLM-compatible LLM endpoints.
It handles authentication, request formatting, response parsing, and error handling
for chat completion API calls.

The client works with any LiteLLM-compatible API (OpenAI, Anthropic, local models, etc.)
and normalizes token usage tracking across different provider formats.

A module-level httpx.AsyncClient is used for connection pooling across requests.
"""

import logging
import re
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

    input_tokens = int(
        usage.get("input_tokens", 0)
        or usage.get("prompt_tokens", 0)
        or 0
    )
    output_tokens = int(
        usage.get("output_tokens", 0)
        or usage.get("completion_tokens", 0)
        or 0
    )
    total_tokens = int(usage.get("total_tokens", 0) or 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens or (input_tokens + output_tokens),
    }


async def chat_completion(
    prompt: str,
    model: str,
    max_tokens: int,
    response_format: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Send a chat completion request to the LLM backend.

    Makes a POST request to the configured LiteLLM-compatible endpoint
    with the given prompt and returns the generated content along with
    token usage information.

    Args:
        prompt: The user prompt to send to the LLM
        model: Model identifier (e.g., "gpt-4", "claude-3-opus")
        max_tokens: Maximum tokens in the response
        response_format: Optional response format hint (e.g. {"type": "json_object"})

    Returns:
        Dictionary with:
        - "content": Generated text response
        - "usage": Normalized token usage dict

    Raises:
        UpstreamServiceError: If the LLM request fails or returns invalid data
    """
    settings = get_settings()
    model_name = model.split("/")[-1] if "/" in model else model
    client = _get_client()

    body: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format

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
    if not isinstance(content, str) or not content.strip():
        raise UpstreamServiceError("The LLM backend returned empty content.")

    # Strip <think>...</think> reasoning blocks some models emit
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if not content:
        raise UpstreamServiceError("The LLM backend returned empty content.")

    return {
        "content": content,
        "usage": _normalize_usage(data.get("usage")),
    }
