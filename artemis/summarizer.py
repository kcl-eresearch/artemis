"""Summarization functionality for Artemis using LiteLLM-compatible endpoints.

This module provides LLM-powered summarization of search results. Given a
search query and a list of results, it uses an LLM to generate a concise
summary that addresses the user's information needs.

Search result content (titles, URLs, snippets) is delivered to the LLM via
tool-call messages so that untrusted web content is treated as data rather
than instructions, mitigating prompt injection risks.
"""

from typing import Any

from artemis.models import SearchResult
from artemis.llm import build_tool_messages, chat_completion, sanitize_content


async def summarize_results(
    query: str,
    results: list[SearchResult],
    model: str,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Generate an LLM-powered summary of search results.

    Takes a search query and list of results, formats them into a prompt,
    and sends to the LLM for summarization. The LLM is instructed to produce
    a concise, informative summary that addresses the query.

    Search results are placed in a tool-response message so the model treats
    them as retrieved data rather than user instructions.

    Args:
        query: The original search query
        results: List of SearchResult objects to summarize
        model: LLM model identifier to use
        max_tokens: Maximum tokens in the summary (default: 1024)

    Returns:
        Dictionary with:
        - "summary": Generated summary text
        - "usage": Token usage information from the LLM

    Note:
        Returns {"summary": None, "usage": None} if results list is empty.
    """

    if not results:
        return {"summary": None, "usage": None}

    context_parts = []
    for i, r in enumerate(results, 1):
        title = sanitize_content(r.title, max_length=200)
        snippet = sanitize_content(r.snippet, max_length=500)
        context_parts.append(f"{i}. Title: {title}\n   URL: {r.url}\n   {snippet}")

    tool_content = "\n\n".join(context_parts)

    system = (
        "You are a research assistant. Based on the search results returned by the "
        "web_search tool, provide a concise, informative summary that answers the "
        "user's query.\n\n"
        "Please provide a well-structured summary that:\n"
        "1. Directly addresses the search query\n"
        "2. Summarizes the key information from the results\n"
        "3. Cites sources where relevant\n"
        "4. Is informative and comprehensive but concise"
    )

    messages = build_tool_messages(
        system=system,
        user=query,
        tool_content=tool_content,
    )

    completion = await chat_completion(
        messages=messages, model=model, max_tokens=max_tokens
    )

    return {
        "summary": completion["content"],
        "usage": completion["usage"],
    }
