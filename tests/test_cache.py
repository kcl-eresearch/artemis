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
