"""Tests for the async TTL cache with request coalescing."""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, patch

from artemis.cache import AsyncTTLCache, CacheStats
from artemis.models import SearchResult


class CacheStatsTestCase(unittest.TestCase):
    def test_total(self) -> None:
        stats = CacheStats(hits=3, misses=7)
        self.assertEqual(stats.total, 10)

    def test_hit_rate(self) -> None:
        stats = CacheStats(hits=3, misses=7)
        self.assertAlmostEqual(stats.hit_rate, 0.3)

    def test_hit_rate_zero_total(self) -> None:
        stats = CacheStats()
        self.assertEqual(stats.hit_rate, 0.0)


class BasicCacheOperationsTestCase(unittest.TestCase):
    def test_get_missing_returns_none(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)
        self.assertIsNone(cache.get("missing"))

    def test_put_and_get(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)
        cache.put("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

    def test_size(self) -> None:
        cache: AsyncTTLCache[int] = AsyncTTLCache(ttl_seconds=60)
        self.assertEqual(cache.size, 0)
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(cache.size, 2)

    def test_clear(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)
        cache.put("key1", "value1")
        cache.stats.hits = 5
        cache.clear()
        self.assertEqual(cache.size, 0)
        self.assertIsNone(cache.get("key1"))
        self.assertEqual(cache.stats.hits, 0)


class TTLExpiryTestCase(unittest.TestCase):
    def test_expired_entry_returns_none(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=0.01)
        cache.put("key", "value")
        time.sleep(0.02)
        self.assertIsNone(cache.get("key"))

    def test_non_expired_entry_returns_value(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)
        cache.put("key", "value")
        self.assertEqual(cache.get("key"), "value")


class EvictionTestCase(unittest.TestCase):
    def test_max_entries_evicts_oldest(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60, max_entries=3)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")
        cache.put("d", "4")  # Should evict "a"
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), "2")
        self.assertEqual(cache.get("d"), "4")
        self.assertEqual(cache.size, 3)

    def test_periodic_cleanup_evicts_expired(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=0.01)
        cache.put("a", "1")
        cache.put("b", "2")
        time.sleep(0.02)
        evicted = cache.periodic_cleanup()
        self.assertEqual(evicted, 2)
        self.assertEqual(cache.size, 0)


class GetOrFetchTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_cache_miss_calls_factory(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)
        factory = AsyncMock(return_value="fetched_value")

        result = await cache.get_or_fetch("key", factory)

        self.assertEqual(result, "fetched_value")
        factory.assert_awaited_once()
        self.assertEqual(cache.stats.misses, 1)
        self.assertEqual(cache.stats.hits, 0)

    async def test_cache_hit_skips_factory(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)
        cache.put("key", "cached_value")
        factory = AsyncMock(return_value="new_value")

        result = await cache.get_or_fetch("key", factory)

        self.assertEqual(result, "cached_value")
        factory.assert_not_awaited()
        self.assertEqual(cache.stats.hits, 1)

    async def test_factory_result_is_cached(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        r1 = await cache.get_or_fetch("key", factory)
        r2 = await cache.get_or_fetch("key", factory)

        self.assertEqual(r1, "value_1")
        self.assertEqual(r2, "value_1")  # Cached, not re-fetched
        self.assertEqual(call_count, 1)

    async def test_factory_exception_propagates(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)

        async def factory():
            raise ValueError("fetch failed")

        with self.assertRaises(ValueError):
            await cache.get_or_fetch("key", factory)

        # Key should not be cached after failure
        self.assertIsNone(cache.get("key"))

    async def test_different_keys_call_factory_separately(self) -> None:
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        r1 = await cache.get_or_fetch("key1", factory)
        r2 = await cache.get_or_fetch("key2", factory)

        self.assertEqual(r1, "value_1")
        self.assertEqual(r2, "value_2")
        self.assertEqual(call_count, 2)


class RequestCoalescingTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_coalesce(self) -> None:
        """Multiple concurrent requests for the same key only fetch once."""
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)
        call_count = 0
        fetch_started = asyncio.Event()

        async def slow_factory():
            nonlocal call_count
            call_count += 1
            fetch_started.set()
            await asyncio.sleep(0.1)
            return "result"

        # Launch 5 concurrent requests for the same key
        tasks = [
            asyncio.create_task(cache.get_or_fetch("key", slow_factory))
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)

        self.assertEqual(call_count, 1)  # Only one fetch
        self.assertTrue(all(r == "result" for r in results))
        self.assertGreater(cache.stats.coalesced, 0)

    async def test_coalesced_requests_get_same_exception(self) -> None:
        """All waiters see the same exception if the fetch fails."""
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)

        async def failing_factory():
            await asyncio.sleep(0.05)
            raise RuntimeError("boom")

        tasks = [
            asyncio.create_task(cache.get_or_fetch("key", failing_factory))
            for _ in range(3)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        self.assertTrue(all(isinstance(r, RuntimeError) for r in results))
        # Key should not be cached
        self.assertIsNone(cache.get("key"))

    async def test_different_keys_fetch_independently(self) -> None:
        """Requests for different keys are not coalesced."""
        cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60)
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return f"result_{call_count}"

        tasks = [
            asyncio.create_task(cache.get_or_fetch(f"key{i}", factory))
            for i in range(3)
        ]
        await asyncio.gather(*tasks)

        self.assertEqual(call_count, 3)  # Each key fetched independently
        self.assertEqual(cache.stats.coalesced, 0)


class SearchCacheIntegrationTestCase(unittest.IsolatedAsyncioTestCase):
    """Test cache integration in searcher module."""

    @patch("artemis.searcher._search_searxng_uncached", new_callable=AsyncMock)
    async def test_search_cache_hit(self, mock_search: AsyncMock) -> None:
        from artemis.searcher import search_searxng, _get_search_cache

        mock_search.return_value = [
            SearchResult(title="R1", url="https://r1.com", snippet="S1"),
        ]

        cache = _get_search_cache()
        if cache is not None:
            cache.clear()

        # First call — cache miss
        r1 = await search_searxng(query="test query", max_results=5)
        # Second call — cache hit (same params)
        r2 = await search_searxng(query="test query", max_results=5)

        self.assertEqual(r1, r2)

        cache = _get_search_cache()
        if cache is not None:
            # With cache enabled, only 1 actual search
            self.assertEqual(mock_search.await_count, 1)
            self.assertEqual(cache.stats.hits, 1)
            self.assertEqual(cache.stats.misses, 1)
            cache.clear()
        else:
            # Without cache, both calls go through
            self.assertEqual(mock_search.await_count, 2)

    @patch("artemis.searcher._search_searxng_uncached", new_callable=AsyncMock)
    async def test_different_params_cache_miss(self, mock_search: AsyncMock) -> None:
        from artemis.searcher import search_searxng, _get_search_cache

        mock_search.return_value = [
            SearchResult(title="R1", url="https://r1.com", snippet="S1"),
        ]

        cache = _get_search_cache()
        if cache is not None:
            cache.clear()

        await search_searxng(query="query one", max_results=5)
        await search_searxng(query="query two", max_results=5)

        # Different queries = different cache keys = 2 fetches
        self.assertEqual(mock_search.await_count, 2)

        if cache is not None:
            cache.clear()


class ContentCacheIntegrationTestCase(unittest.IsolatedAsyncioTestCase):
    """Test cache integration in extractor module."""

    @patch("artemis.extractor._fetch_and_extract_uncached", new_callable=AsyncMock)
    async def test_content_cache_hit(self, mock_extract: AsyncMock) -> None:
        from artemis.extractor import fetch_and_extract, _get_content_cache

        mock_extract.return_value = "Extracted page content"

        cache = _get_content_cache()
        if cache is not None:
            cache.clear()

        # First call — cache miss
        r1 = await fetch_and_extract("https://example.com/page", max_chars=3000)
        # Second call — cache hit
        r2 = await fetch_and_extract("https://example.com/page", max_chars=3000)

        self.assertEqual(r1, "Extracted page content")
        self.assertEqual(r1, r2)

        cache = _get_content_cache()
        if cache is not None:
            self.assertEqual(mock_extract.await_count, 1)
            self.assertEqual(cache.stats.hits, 1)
            cache.clear()
        else:
            self.assertEqual(mock_extract.await_count, 2)


# ---------------------------------------------------------------------------
# Embedding / cosine similarity tests
# ---------------------------------------------------------------------------

class CosineSimilarityTestCase(unittest.TestCase):
    """Test the pure-Python cosine similarity helper."""

    def test_identical_vectors(self) -> None:
        from artemis.llm import cosine_similarity
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=6)

    def test_orthogonal_vectors(self) -> None:
        from artemis.llm import cosine_similarity
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=6)

    def test_opposite_vectors(self) -> None:
        from artemis.llm import cosine_similarity
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0, places=6)

    def test_zero_vector_returns_zero(self) -> None:
        from artemis.llm import cosine_similarity
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        self.assertEqual(cosine_similarity(a, b), 0.0)

    def test_known_similarity(self) -> None:
        from artemis.llm import cosine_similarity
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        # cos(45°) ≈ 0.7071
        self.assertAlmostEqual(cosine_similarity(a, b), 0.7071, places=3)


class EmbedEndpointTestCase(unittest.IsolatedAsyncioTestCase):
    """Test the embed() function."""

    @patch("artemis.llm._get_client")
    async def test_embed_returns_vector(self, mock_get_client: AsyncMock) -> None:
        from artemis.llm import embed
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = await embed("test query", "text-embedding-3-small")
        self.assertEqual(result, [0.1, 0.2, 0.3])

        call_args = mock_client.post.call_args
        self.assertIn("/embeddings", call_args[0][0])
        body = call_args[1]["json"]
        self.assertEqual(body["input"], "test query")
        self.assertEqual(body["model"], "text-embedding-3-small")

    @patch("artemis.llm._get_client")
    async def test_embed_timeout_raises(self, mock_get_client: AsyncMock) -> None:
        from artemis.llm import embed
        from artemis.errors import UpstreamServiceError
        import httpx

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError):
            await embed("test query", "text-embedding-3-small")

    @patch("artemis.llm._get_client")
    async def test_embed_bad_structure_raises(self, mock_get_client: AsyncMock) -> None:
        from artemis.llm import embed
        from artemis.errors import UpstreamServiceError
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": []}  # No embeddings
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        with self.assertRaises(UpstreamServiceError):
            await embed("test query", "text-embedding-3-small")


class SemanticSearchCacheTestCase(unittest.IsolatedAsyncioTestCase):
    """Test semantic query deduplication in the searcher."""

    def setUp(self) -> None:
        import artemis.searcher as s
        self._searcher = s
        # Reset embedding index
        s._embedding_index.clear()

    def tearDown(self) -> None:
        self._searcher._embedding_index.clear()
        cache = self._searcher._get_search_cache()
        if cache is not None:
            cache.clear()

    @patch("artemis.searcher.get_settings")
    @patch("artemis.llm.embed", new_callable=AsyncMock)
    @patch("artemis.searcher._search_searxng_uncached", new_callable=AsyncMock)
    async def test_semantic_hit_skips_search(
        self, mock_search: AsyncMock, mock_embed: AsyncMock, mock_settings
    ) -> None:
        from artemis.searcher import search_searxng, _get_search_cache
        from artemis.config import Settings

        # Configure with embedding model enabled
        settings = Settings(
            searxng_api_base="http://localhost:8888",
            searxng_timeout_seconds=30.0,
            litellm_base_url="http://localhost:11434/api",
            litellm_api_key=None,
            llm_timeout_seconds=120.0,
            summary_model="qwen3.5:9b",
            summary_max_tokens=1024,
            enable_summary=True,
            deep_research_stages=2,
            deep_research_passes=1,
            deep_research_subqueries=5,
            deep_research_results_per_query=10,
            deep_research_max_tokens=4000,
            deep_research_content_extraction=True,
            deep_research_pages_per_section=3,
            deep_research_content_max_chars=3000,
            shallow_research_stages=1,
            shallow_research_passes=1,
            shallow_research_subqueries=3,
            shallow_research_results_per_query=5,
            shallow_research_max_tokens=2500,
            shallow_research_content_extraction=False,
            shallow_research_pages_per_section=2,
            shallow_research_content_max_chars=1500,
            allowed_origins=tuple(),
            artemis_api_key=None,
            log_level="INFO",
            cache_enabled=True,
            search_cache_ttl_seconds=3600,
            content_cache_ttl_seconds=86400,
            cache_max_entries=1000,
            embedding_model="text-embedding-3-small",
            semantic_similarity_threshold=0.90,
            log_format="text",
            playwright_context_recycle_pages=50,
            playwright_max_html_bytes=5242880,
            synthesis_tool_rounds=0,
            supervised_research=False,
            researcher_max_tool_rounds=15,
        )
        mock_settings.return_value = settings

        expected_results = [
            SearchResult(title="R1", url="https://r1.com", snippet="S1"),
        ]
        mock_search.return_value = expected_results

        # Both queries return very similar embeddings (cosine sim > 0.90)
        emb_a = [0.9, 0.1, 0.0]
        emb_b = [0.89, 0.12, 0.01]  # Very similar to emb_a
        mock_embed.side_effect = [emb_a, emb_b]

        cache = _get_search_cache()
        if cache is not None:
            cache.clear()
        self._searcher._embedding_index.clear()

        # First search — cache miss, search executed
        r1 = await search_searxng(query="quantum computing advances")
        self.assertEqual(mock_search.await_count, 1)

        # Second search — semantically similar, should hit
        r2 = await search_searxng(query="recent quantum computing progress")
        # Should still be 1 search (semantic hit)
        self.assertEqual(mock_search.await_count, 1)
        self.assertEqual(r1, r2)

    @patch("artemis.searcher.get_settings")
    @patch("artemis.llm.embed", new_callable=AsyncMock)
    @patch("artemis.searcher._search_searxng_uncached", new_callable=AsyncMock)
    async def test_dissimilar_queries_no_semantic_hit(
        self, mock_search: AsyncMock, mock_embed: AsyncMock, mock_settings
    ) -> None:
        from artemis.searcher import search_searxng, _get_search_cache
        from artemis.config import Settings

        settings = Settings(
            searxng_api_base="http://localhost:8888",
            searxng_timeout_seconds=30.0,
            litellm_base_url="http://localhost:11434/api",
            litellm_api_key=None,
            llm_timeout_seconds=120.0,
            summary_model="qwen3.5:9b",
            summary_max_tokens=1024,
            enable_summary=True,
            deep_research_stages=2,
            deep_research_passes=1,
            deep_research_subqueries=5,
            deep_research_results_per_query=10,
            deep_research_max_tokens=4000,
            deep_research_content_extraction=True,
            deep_research_pages_per_section=3,
            deep_research_content_max_chars=3000,
            shallow_research_stages=1,
            shallow_research_passes=1,
            shallow_research_subqueries=3,
            shallow_research_results_per_query=5,
            shallow_research_max_tokens=2500,
            shallow_research_content_extraction=False,
            shallow_research_pages_per_section=2,
            shallow_research_content_max_chars=1500,
            allowed_origins=tuple(),
            artemis_api_key=None,
            log_level="INFO",
            cache_enabled=True,
            search_cache_ttl_seconds=3600,
            content_cache_ttl_seconds=86400,
            cache_max_entries=1000,
            embedding_model="text-embedding-3-small",
            semantic_similarity_threshold=0.90,
            log_format="text",
            playwright_context_recycle_pages=50,
            playwright_max_html_bytes=5242880,
            synthesis_tool_rounds=0,
            supervised_research=False,
            researcher_max_tool_rounds=15,
        )
        mock_settings.return_value = settings

        mock_search.return_value = [
            SearchResult(title="R1", url="https://r1.com", snippet="S1"),
        ]

        # Very different embeddings (orthogonal)
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0]
        mock_embed.side_effect = [emb_a, emb_b]

        cache = _get_search_cache()
        if cache is not None:
            cache.clear()
        self._searcher._embedding_index.clear()

        await search_searxng(query="quantum computing")
        await search_searxng(query="best pizza recipes")

        # Different topics, both should have searched
        self.assertEqual(mock_search.await_count, 2)

    @patch("artemis.searcher.get_settings")
    @patch("artemis.searcher._search_searxng_uncached", new_callable=AsyncMock)
    async def test_no_embedding_model_skips_semantic(
        self, mock_search: AsyncMock, mock_settings
    ) -> None:
        from artemis.searcher import search_searxng, _get_search_cache
        from artemis.config import Settings

        settings = Settings(
            searxng_api_base="http://localhost:8888",
            searxng_timeout_seconds=30.0,
            litellm_base_url="http://localhost:11434/api",
            litellm_api_key=None,
            llm_timeout_seconds=120.0,
            summary_model="qwen3.5:9b",
            summary_max_tokens=1024,
            enable_summary=True,
            deep_research_stages=2,
            deep_research_passes=1,
            deep_research_subqueries=5,
            deep_research_results_per_query=10,
            deep_research_max_tokens=4000,
            deep_research_content_extraction=True,
            deep_research_pages_per_section=3,
            deep_research_content_max_chars=3000,
            shallow_research_stages=1,
            shallow_research_passes=1,
            shallow_research_subqueries=3,
            shallow_research_results_per_query=5,
            shallow_research_max_tokens=2500,
            shallow_research_content_extraction=False,
            shallow_research_pages_per_section=2,
            shallow_research_content_max_chars=1500,
            allowed_origins=tuple(),
            artemis_api_key=None,
            log_level="INFO",
            cache_enabled=True,
            search_cache_ttl_seconds=3600,
            content_cache_ttl_seconds=86400,
            cache_max_entries=1000,
            embedding_model=None,  # Disabled
            semantic_similarity_threshold=0.92,
            log_format="text",
            playwright_context_recycle_pages=50,
            playwright_max_html_bytes=5242880,
            synthesis_tool_rounds=0,
            supervised_research=False,
            researcher_max_tool_rounds=15,
        )
        mock_settings.return_value = settings

        mock_search.return_value = [
            SearchResult(title="R1", url="https://r1.com", snippet="S1"),
        ]

        cache = _get_search_cache()
        if cache is not None:
            cache.clear()

        await search_searxng(query="query one")
        await search_searxng(query="query two")

        # No semantic dedup, both go through (different exact keys)
        self.assertEqual(mock_search.await_count, 2)
        # No embeddings indexed
        self.assertEqual(len(self._searcher._embedding_index), 0)
