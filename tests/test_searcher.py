"""Searcher unit tests for Artemis."""

import unittest
from unittest.mock import AsyncMock, patch

from artemis.searcher import search_searxng


class _ResponseStub:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class SearcherTestCase(unittest.IsolatedAsyncioTestCase):
    @patch("artemis.searcher._get_client")
    async def test_domain_filter_is_applied_before_result_limit(
        self, mock_get_client
    ) -> None:
        payload = {
            "results": [
                {
                    "title": "Blocked result",
                    "url": "https://blocked.example/article",
                    "content": "Should be filtered out",
                },
                {
                    "title": "Allowed result",
                    "url": "https://docs.allowed.example/article",
                    "content": "Should be kept",
                },
            ]
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_ResponseStub(payload))
        mock_get_client.return_value = mock_client

        results = await search_searxng(
            query="example",
            max_results=1,
            domain_filter=["allowed.example"],
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://docs.allowed.example/article")

    @patch("artemis.searcher._get_client")
    async def test_invalid_urls_are_skipped(self, mock_get_client) -> None:
        payload = {
            "results": [
                {
                    "title": "Unsafe result",
                    "url": "javascript:alert(1)",
                    "content": "Unsafe",
                },
                {
                    "title": "Safe result",
                    "url": "https://example.com/article",
                    "content": "Safe",
                },
            ]
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_ResponseStub(payload))
        mock_get_client.return_value = mock_client

        results = await search_searxng(query="example", max_results=10)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/article")
