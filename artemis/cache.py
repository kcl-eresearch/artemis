"""In-memory async TTL cache with request coalescing.

Provides :class:`AsyncTTLCache`, a generic cache designed for concurrent async
workloads.  When multiple coroutines request the same key simultaneously, only
one fetch is executed — the others await the same ``asyncio.Future`` (request
coalescing).  This is critical when many researchers hit the service at once.

Usage::

    cache: AsyncTTLCache[list[SearchResult]] = AsyncTTLCache(
        ttl_seconds=3600, max_entries=500, name="search"
    )

    async def fetch(key: str) -> list[SearchResult]:
        return await search_searxng(query=key)

    results = await cache.get_or_fetch("quantum computing", fetch)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, TypeVar

_MISSING = object()  # sentinel to distinguish "not in cache" from cached None

logger = logging.getLogger(__name__)

V = TypeVar("V")


@dataclass
class _CacheEntry(Generic[V]):
    """A single cached value with its expiry timestamp."""
    value: V
    expires_at: float
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class CacheStats:
    """Counters for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    coalesced: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class AsyncTTLCache(Generic[V]):
    """In-memory TTL cache with request coalescing for async workloads.

    Args:
        ttl_seconds: Time-to-live for cached entries.
        max_entries: Maximum number of entries before oldest are evicted.
        name: Human-readable name for logging.
    """

    def __init__(
        self,
        ttl_seconds: float = 3600,
        max_entries: int = 1000,
        name: str = "cache",
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._name = name
        self._data: dict[str, _CacheEntry[V]] = {}
        # In-flight futures for request coalescing
        self._in_flight: dict[str, asyncio.Future[V]] = {}
        self._lock = asyncio.Lock()
        self.stats = CacheStats()

    @property
    def size(self) -> int:
        """Number of entries currently in the cache (including expired)."""
        return len(self._data)

    def _is_expired(self, entry: _CacheEntry[V]) -> bool:
        return time.monotonic() > entry.expires_at

    def _evict_expired(self) -> None:
        """Remove all expired entries."""
        now = time.monotonic()
        expired = [k for k, v in self._data.items() if now > v.expires_at]
        for k in expired:
            del self._data[k]
        if expired:
            self.stats.evictions += len(expired)

    def _evict_oldest(self) -> None:
        """Evict oldest entries until under max_entries."""
        if len(self._data) <= self._max_entries:
            return
        # Sort by created_at, evict oldest
        to_evict = len(self._data) - self._max_entries
        sorted_keys = sorted(
            self._data, key=lambda k: self._data[k].created_at
        )
        for k in sorted_keys[:to_evict]:
            del self._data[k]
        self.stats.evictions += to_evict

    def _get_value(self, key: str) -> object:
        """Return cached value or _MISSING sentinel if not present/expired."""
        entry = self._data.get(key)
        if entry is None:
            return _MISSING
        if self._is_expired(entry):
            del self._data[key]
            return _MISSING
        return entry.value

    def get(self, key: str) -> V | None:
        """Return cached value if present and not expired, else None."""
        result = self._get_value(key)
        if result is _MISSING:
            return None
        return result  # type: ignore[return-value]

    def put(self, key: str, value: V) -> None:
        """Store a value with TTL-based expiry."""
        now = time.monotonic()
        self._data[key] = _CacheEntry(
            value=value,
            expires_at=now + self._ttl,
            created_at=now,
        )
        self._evict_oldest()

    async def get_or_fetch(
        self,
        key: str,
        factory: Callable[[], Awaitable[V]],
    ) -> V:
        """Return cached value or fetch it, coalescing concurrent requests.

        If *key* is in the cache and not expired, return immediately.  If
        another coroutine is already fetching the same key, wait for that
        result instead of making a duplicate request.  Otherwise, call
        *factory* and cache the result.

        Args:
            key: Cache key string.
            factory: Async callable that produces the value on a cache miss.

        Returns:
            The cached or freshly-fetched value.

        Raises:
            Any exception raised by *factory* (propagated to all waiters).
        """
        # Fast path — no lock needed for reads
        cached = self._get_value(key)
        if cached is not _MISSING:
            self.stats.hits += 1
            return cached  # type: ignore[return-value]

        # Determine if we should fetch or wait for an in-flight request
        should_fetch = False
        async with self._lock:
            # Double-check after acquiring lock
            cached = self._get_value(key)
            if cached is not _MISSING:
                self.stats.hits += 1
                return cached  # type: ignore[return-value]

            future = self._in_flight.get(key)
            if future is not None:
                # Another coroutine is already fetching — coalesce
                self.stats.coalesced += 1
                logger.debug("[%s] coalescing request for key=%s", self._name, key[:80])
            else:
                # We are the fetcher — create a future for others to await
                self.stats.misses += 1
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                self._in_flight[key] = future
                should_fetch = True

        assert future is not None

        if should_fetch:
            try:
                value = await factory()
                self.put(key, value)
                future.set_result(value)
                return value
            except BaseException as exc:
                future.set_exception(exc)
                raise
            finally:
                async with self._lock:
                    self._in_flight.pop(key, None)
        else:
            # Wait for the in-flight fetch to complete
            return await future

    def clear(self) -> None:
        """Remove all entries and reset stats."""
        self._data.clear()
        self._in_flight.clear()
        self.stats = CacheStats()
        logger.info("[%s] cache cleared", self._name)

    def periodic_cleanup(self) -> int:
        """Remove expired entries. Returns number evicted."""
        before = len(self._data)
        self._evict_expired()
        evicted = before - len(self._data)
        if evicted:
            logger.debug(
                "[%s] evicted %d expired entries (%d remaining)",
                self._name, evicted, len(self._data),
            )
        return evicted
