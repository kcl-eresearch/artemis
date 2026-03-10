"""Test script for Artemis API."""

import asyncio
import httpx
from artemis.searcher import search_searxng
from artemis.summarizer import summarize_results
from artemis.models import SearchResult
from artemis.config import SUMMARY_MODEL

SEARXNG_API_BASE = "http://localhost:8888"


async def test_search():
    """Test direct SearXNG search."""
    print("Testing SearXNG search...")

    results = await search_searxng(
        query="Python async programming", max_results=3, language="en"
    )

    print(f"Got {len(results)} results:")
    for r in results:
        print(f"  - {r.title}")
        print(f"    {r.url}")
        print(f"    {r.snippet[:100]}...")
        print()

    return results


async def test_summarize():
    """Test LLM summarization."""
    print("Testing summarization...")

    results = [
        SearchResult(
            title="Python asyncio documentation",
            url="https://docs.python.org/3/library/asyncio.html",
            snippet="asyncio is a library to write concurrent code using the async/await syntax.",
            date="2024-01-15",
        ),
        SearchResult(
            title="Real Python - Async IO in Python",
            url="https://realpython.com/async-io-python/",
            snippet="A complete guide to asynchronous I/O in Python with asyncio and async/await.",
            date="2024-02-20",
        ),
        SearchResult(
            title="Python Concurrency Tutorial",
            url="https://www.geeksforgeeks.org/python-concurrency/",
            snippet="Learn about threading, multiprocessing, and asyncio in Python.",
            date="2024-03-01",
        ),
    ]

    summary = await summarize_results(
        query="Python async programming",
        results=results,
        model=SUMMARY_MODEL,
        max_tokens=500,
    )

    print("Summary:")
    print(summary)
    print()

    return summary


async def test_api_server():
    """Test the FastAPI server."""
    print("Testing API server...")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get("http://localhost:8000/health")
        print(f"Health check: {response.status_code} - {response.json()}")

        # Test /search (Perplexity format)
        response = await client.post(
            "http://localhost:8000/search",
            json={
                "query": "latest Python 3.12 features",
                "max_results": 3,
            },
        )

        print(f"Search: {response.status_code}")
        data = response.json()

        print(f"Results count: {len(data.get('results', []))}")
        print(f"Has id: {'id' in data}")
        if data.get("summary"):
            print(f"Summary length: {len(data['summary'])} chars")

        # Test /v1/responses (basic)
        response = await client.post(
            "http://localhost:8000/v1/responses",
            json={
                "input": "What is FastAPI?",
                "preset": "fast-search",
            },
        )

        print(f"\nResponses (fast-search): {response.status_code}")
        data = response.json()
        print(f"Has output: {'output' in data}")
        if data.get("output"):
            for item in data["output"]:
                print(f"  Output type: {item.get('type')}")


async def main():
    print("=" * 50)
    print("Artemis Tests")
    print("=" * 50)
    print()

    try:
        await test_search()
    except Exception as e:
        print(f"Search test failed: {e}")

    print("-" * 50)

    try:
        await test_summarize()
    except Exception as e:
        print(f"Summarize test failed: {e}")

    print("-" * 50)

    try:
        await test_api_server()
    except Exception as e:
        print(f"API server test failed: {e}")

    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
