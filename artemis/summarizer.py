"""Summarization functionality for Artemis using LiteLLM-compatible endpoints.

This module provides LLM-powered summarization of search results. Given a
search query and a list of results, it uses an LLM to generate a concise
summary that addresses the user's information needs.

The summarization prompt is designed to produce well-structured responses
that cite sources and directly answer the original query.
"""

from typing import Any

from artemis.models import SearchResult
from artemis.llm import chat_completion


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

    context_parts = [f"Search query: {query}\n\nSearch results:\n"]
    for i, r in enumerate(results, 1):
        context_parts.append(f"{i}. **{r.title}**\n   URL: {r.url}\n   {r.snippet}\n")

    context = "\n".join(context_parts)

    prompt = f"""You are a research assistant. Based on the search results below, provide a concise, informative summary that answers the user's query. 

{context}

Please provide a well-structured summary that:
1. Directly addresses the search query
2. Summarizes the key information from the results
3. Cites sources where relevant
4. Is informative and comprehensive but concise

Summary:"""

    completion = await chat_completion(
        prompt=prompt, model=model, max_tokens=max_tokens
    )

    return {
        "summary": completion["content"],
        "usage": completion["usage"],
    }
