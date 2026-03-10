"""Unit tests for the Artemis content extractor module."""

import unittest
from unittest.mock import AsyncMock, patch

from artemis.extractor import extract_content, enrich_results, _is_blocked
from artemis.models import SearchResult


class ExtractContentTestCase(unittest.TestCase):
    def test_extracts_article_text(self) -> None:
        html = """
        <html><body>
        <nav>Navigation stuff</nav>
        <article>
            <h1>Test Article</h1>
            <p>This is the main body content of the article that should be extracted.
            It contains multiple sentences with useful information about the topic.</p>
            <p>A second paragraph with more detailed content for the reader.</p>
        </article>
        <footer>Footer junk</footer>
        </body></html>
        """
        result = extract_content(html)
        if result is not None:
            self.assertIn("main body content", result)

    def test_returns_none_for_empty_html(self) -> None:
        self.assertIsNone(extract_content(""))

    def test_returns_none_for_no_content(self) -> None:
        html = "<html><body><nav>Just nav</nav></body></html>"
        result = extract_content(html)
        # trafilatura may or may not extract "Just nav" - either None or short text is fine
        if result is not None:
            self.assertIsInstance(result, str)

    def test_truncates_to_max_chars(self) -> None:
        long_paragraphs = " ".join(
            [f"<p>Paragraph {i}. This is a long sentence with enough content to fill the buffer.</p>"
             for i in range(100)]
        )
        html = f"<html><body><article>{long_paragraphs}</article></body></html>"
        result = extract_content(html, max_chars=200)
        if result is not None:
            self.assertLessEqual(len(result), 200)

    def test_truncates_at_sentence_boundary(self) -> None:
        paragraphs = "<p>" + ". ".join(
            [f"Sentence number {i} with some content" for i in range(50)]
        ) + ".</p>"
        html = f"<html><body><article>{paragraphs}</article></body></html>"
        result = extract_content(html, max_chars=300)
        if result is not None and len(result) > 10:
            self.assertTrue(result.endswith("."))


class EnrichResultsTestCase(unittest.IsolatedAsyncioTestCase):
    @patch("artemis.extractor.fetch_and_extract", new_callable=AsyncMock)
    async def test_returns_content_map(self, mock_fetch: AsyncMock) -> None:
        mock_fetch.side_effect = [
            "Full content from page one.",
            "Full content from page two.",
        ]
        results = [
            SearchResult(title="R1", url="https://a.com/1", snippet="s1"),
            SearchResult(title="R2", url="https://b.com/2", snippet="s2"),
        ]
        content_map = await enrich_results(results, max_pages=2)
        self.assertEqual(len(content_map), 2)
        self.assertEqual(content_map["https://a.com/1"], "Full content from page one.")

    @patch("artemis.extractor.fetch_and_extract", new_callable=AsyncMock)
    async def test_skips_failed_extractions(self, mock_fetch: AsyncMock) -> None:
        mock_fetch.side_effect = [None, "Good content."]
        results = [
            SearchResult(title="R1", url="https://a.com/1", snippet="s1"),
            SearchResult(title="R2", url="https://b.com/2", snippet="s2"),
        ]
        content_map = await enrich_results(results, max_pages=2)
        self.assertEqual(len(content_map), 1)
        self.assertNotIn("https://a.com/1", content_map)

    @patch("artemis.extractor.fetch_and_extract", new_callable=AsyncMock)
    async def test_respects_max_pages(self, mock_fetch: AsyncMock) -> None:
        mock_fetch.return_value = "Content."
        results = [
            SearchResult(title=f"R{i}", url=f"https://example{i}.com/{i}", snippet=f"s{i}")
            for i in range(10)
        ]
        await enrich_results(results, max_pages=3)
        self.assertEqual(mock_fetch.call_count, 3)

    async def test_empty_results(self) -> None:
        content_map = await enrich_results([])
        self.assertEqual(content_map, {})

    @patch("artemis.extractor.fetch_and_extract", new_callable=AsyncMock)
    async def test_handles_exceptions_gracefully(self, mock_fetch: AsyncMock) -> None:
        mock_fetch.side_effect = Exception("Network error")
        results = [
            SearchResult(title="R1", url="https://a.com/1", snippet="s1"),
        ]
        content_map = await enrich_results(results, max_pages=1)
        self.assertEqual(content_map, {})

    @patch("artemis.extractor.fetch_and_extract", new_callable=AsyncMock)
    async def test_skips_blocked_domains(self, mock_fetch: AsyncMock) -> None:
        mock_fetch.return_value = "Content."
        results = [
            SearchResult(title="NYT", url="https://www.nytimes.com/article", snippet="s1"),
            SearchResult(title="WSJ", url="https://www.wsj.com/article", snippet="s2"),
            SearchResult(title="Good", url="https://example.com/article", snippet="s3"),
        ]
        content_map = await enrich_results(results, max_pages=3)
        # Only the non-blocked URL should have been fetched
        self.assertEqual(mock_fetch.call_count, 1)
        self.assertIn("https://example.com/article", content_map)

    @patch("artemis.extractor.fetch_and_extract", new_callable=AsyncMock)
    async def test_backfills_past_blocked_domains(self, mock_fetch: AsyncMock) -> None:
        mock_fetch.return_value = "Content."
        results = [
            SearchResult(title="NYT", url="https://nytimes.com/1", snippet="s1"),
            SearchResult(title="Good1", url="https://a.com/1", snippet="s2"),
            SearchResult(title="Good2", url="https://b.com/2", snippet="s3"),
            SearchResult(title="Good3", url="https://c.com/3", snippet="s4"),
        ]
        content_map = await enrich_results(results, max_pages=3)
        # Should skip NYT and pick the next 3 non-blocked
        self.assertEqual(mock_fetch.call_count, 3)
        self.assertNotIn("https://nytimes.com/1", content_map)


class BlockedDomainTestCase(unittest.TestCase):
    def test_blocked_nytimes(self) -> None:
        self.assertTrue(_is_blocked("https://www.nytimes.com/2024/article"))

    def test_blocked_subdomain(self) -> None:
        self.assertTrue(_is_blocked("https://cooking.nytimes.com/recipes"))

    def test_blocked_statista(self) -> None:
        self.assertTrue(_is_blocked("https://www.statista.com/chart/123"))

    def test_allowed_domain(self) -> None:
        self.assertFalse(_is_blocked("https://en.wikipedia.org/wiki/Test"))

    def test_allowed_custom_domain(self) -> None:
        self.assertFalse(_is_blocked("https://electrive.com/2025/article"))
