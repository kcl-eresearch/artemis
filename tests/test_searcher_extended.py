"""Tests for the searcher module — domain helpers and error paths."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from artemis.errors import UpstreamServiceError
from artemis.models import SearchResult
from artemis.searcher import (
    _domain_matches,
    _normalize_domain,
    _normalize_domain_filters,
    search_searxng,
)


class NormalizeDomainTestCase(unittest.TestCase):
    def test_lowercase(self) -> None:
        self.assertEqual(_normalize_domain("Example.COM"), "example.com")

    def test_strip_whitespace(self) -> None:
        self.assertEqual(_normalize_domain("  example.com  "), "example.com")

    def test_strip_leading_dots(self) -> None:
        self.assertEqual(_normalize_domain(".example.com."), "example.com")

    def test_combined(self) -> None:
        self.assertEqual(_normalize_domain("  .Example.Com.  "), "example.com")


class DomainMatchesTestCase(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertTrue(_domain_matches("example.com", "example.com"))

    def test_subdomain_match(self) -> None:
        self.assertTrue(_domain_matches("www.example.com", "example.com"))

    def test_deep_subdomain(self) -> None:
        self.assertTrue(_domain_matches("a.b.example.com", "example.com"))

    def test_no_match(self) -> None:
        self.assertFalse(_domain_matches("notexample.com", "example.com"))

    def test_partial_no_match(self) -> None:
        self.assertFalse(_domain_matches("myexample.com", "example.com"))


class NormalizeDomainFiltersTestCase(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(_normalize_domain_filters(None), [])
        self.assertEqual(_normalize_domain_filters([]), [])

    def test_plain_domains(self) -> None:
        result = _normalize_domain_filters(["Example.com", "test.org"])
        self.assertEqual(result, ["example.com", "test.org"])

    def test_full_urls_extract_hostname(self) -> None:
        result = _normalize_domain_filters(["https://example.com/path?q=1"])
        self.assertEqual(result, ["example.com"])

    def test_deduplication(self) -> None:
        result = _normalize_domain_filters(["example.com", "EXAMPLE.COM"])
        self.assertEqual(result, ["example.com"])

    def test_empty_values_skipped(self) -> None:
        result = _normalize_domain_filters(["", "  ", "example.com"])
        self.assertEqual(result, ["example.com"])


class _ResponseStub:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            resp = MagicMock()
            resp.status_code = self.status_code
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=MagicMock(), response=resp
            )

    def json(self) -> dict:
        return self._payload


class SearchErrorPathsTestCase(unittest.IsolatedAsyncioTestCase):
    @patch("artemis.searcher._get_client")
    async def test_timeout_raises_upstream_error(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await search_searxng(query="test")
        self.assertIn("timed out", str(ctx.exception))

    @patch("artemis.searcher._get_client")
    async def test_http_500_raises_upstream_error(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.get.return_value = _ResponseStub({}, status_code=500)
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await search_searxng(query="test")
        self.assertIn("500", str(ctx.exception))

    @patch("artemis.searcher._get_client")
    async def test_invalid_json_raises_upstream_error(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.side_effect = ValueError("bad json")
        mock_client.get.return_value = resp
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await search_searxng(query="test")
        self.assertIn("invalid JSON", str(ctx.exception))

    @patch("artemis.searcher._get_client")
    async def test_invalid_results_type_raises(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.get.return_value = _ResponseStub({"results": "not a list"})
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await search_searxng(query="test")
        self.assertIn("invalid results", str(ctx.exception))

    @patch("artemis.searcher._get_client")
    async def test_empty_results_returns_empty_list(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.get.return_value = _ResponseStub({"results": []})
        mock_get_client.return_value = mock_client

        results = await search_searxng(query="test")
        self.assertEqual(results, [])

    @patch("artemis.searcher._get_client")
    async def test_connection_error_raises_upstream_error(self, mock_get_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError) as ctx:
            await search_searxng(query="test")
        self.assertIn("request failed", str(ctx.exception))

    @patch("artemis.searcher._get_client")
    async def test_date_fields_parsed(self, mock_get_client: MagicMock) -> None:
        payload = {
            "results": [
                {"title": "R1", "url": "https://a.com", "content": "S1", "publishedDate": "2025-01-01"},
                {"title": "R2", "url": "https://b.com", "content": "S2", "date": "2025-02-01"},
            ]
        }
        mock_client = AsyncMock()
        mock_client.get.return_value = _ResponseStub(payload)
        mock_get_client.return_value = mock_client

        results = await search_searxng(query="test", max_results=10)
        self.assertEqual(results[0].date, "2025-01-01")
        self.assertEqual(results[1].date, "2025-02-01")

    @patch("artemis.searcher._get_client")
    async def test_missing_title_uses_url(self, mock_get_client: MagicMock) -> None:
        payload = {"results": [{"url": "https://a.com", "content": "S1"}]}
        mock_client = AsyncMock()
        mock_client.get.return_value = _ResponseStub(payload)
        mock_get_client.return_value = mock_client

        results = await search_searxng(query="test")
        self.assertEqual(results[0].title, "https://a.com")
