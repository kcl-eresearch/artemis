"""Content extraction for Artemis deep research.

This module fetches web pages using Playwright (headless Chromium) and
extracts their main body text using trafilatura. Using a real browser
avoids 403s from TLS fingerprinting and JS-challenge bot detection.

The module provides:
- fetch_and_extract(): Fetch a single URL and extract its content
- enrich_results(): Batch-enrich a list of SearchResult URLs

Extracted content is cached in-memory with request coalescing — if
multiple researchers request the same URL concurrently, only one
Playwright fetch is performed.
"""

import asyncio
import hashlib
import logging
from typing import Optional
from urllib.parse import urlparse

import trafilatura
from playwright.async_api import async_playwright, Browser, BrowserContext

from artemis.cache import AsyncTTLCache
from artemis.config import get_settings
from artemis.models import SearchResult

logger = logging.getLogger(__name__)

# Module-level Playwright state (created lazily)
_browser: Browser | None = None
_context: BrowserContext | None = None
_pw_instance = None  # Playwright instance holder
_launch_lock = asyncio.Lock()

# Module-level content cache (created lazily based on config)
_content_cache: AsyncTTLCache[Optional[str]] | None = None

# Domains known to require login (paywall) — not even a real browser helps
_BLOCKED_DOMAINS: frozenset[str] = frozenset({
    "nytimes.com",
    "wsj.com",
    "bloomberg.com",
    "ft.com",
    "economist.com",
    "washingtonpost.com",
    "theathletic.com",
    "thetimes.co.uk",
    "barrons.com",
    "statista.com",
    "seekingalpha.com",
    "hbr.org",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
})


def _is_blocked(url: str) -> bool:
    """Check if a URL belongs to a known paywalled/blocked domain."""
    try:
        hostname = urlparse(url).hostname or ""
    except ValueError:
        return True
    parts = hostname.lower().split(".")
    for i in range(len(parts) - 1):
        domain = ".".join(parts[i:])
        if domain in _BLOCKED_DOMAINS:
            return True
    return False


async def _get_context() -> BrowserContext:
    """Return a shared Playwright browser context, launching Chromium lazily.

    Uses an asyncio.Lock to prevent concurrent callers from launching
    duplicate browser instances.
    """
    global _browser, _context, _pw_instance
    if _context is not None:
        return _context
    async with _launch_lock:
        # Double-check after acquiring the lock
        if _context is not None:
            return _context
        logger.info("Launching Playwright headless Chromium browser")
        _pw_instance = await async_playwright().start()
        _browser = await _pw_instance.chromium.launch(headless=True)
        logger.info("Playwright Chromium browser launched successfully")
        _context = await _browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        return _context


async def close_client() -> None:
    """Shut down the Playwright browser and clear cache (called during app shutdown)."""
    global _browser, _context, _pw_instance, _content_cache
    logger.info("Shutting down Playwright browser")
    if _context is not None:
        await _context.close()
        _context = None
    if _browser is not None:
        await _browser.close()
        _browser = None
    if _pw_instance is not None:
        await _pw_instance.stop()
        _pw_instance = None
    if _content_cache is not None:
        logger.info(
            "Content cache stats: hits=%d misses=%d coalesced=%d hit_rate=%.1f%%",
            _content_cache.stats.hits,
            _content_cache.stats.misses,
            _content_cache.stats.coalesced,
            _content_cache.stats.hit_rate * 100,
        )
        _content_cache.clear()
        _content_cache = None
    logger.info("Playwright browser shut down")


def _get_content_cache() -> AsyncTTLCache[Optional[str]] | None:
    """Return the content cache if caching is enabled, creating lazily."""
    global _content_cache
    settings = get_settings()
    if not settings.cache_enabled:
        return None
    if _content_cache is None:
        _content_cache = AsyncTTLCache(
            ttl_seconds=settings.content_cache_ttl_seconds,
            max_entries=settings.cache_max_entries,
            name="content",
        )
    return _content_cache


def _make_content_cache_key(url: str, max_chars: int) -> str:
    """Build a deterministic cache key for content extraction."""
    return hashlib.sha256(f"{url}|{max_chars}".encode()).hexdigest()


async def fetch_page(url: str, timeout: float = 15.0) -> Optional[str]:
    """Fetch raw HTML from a URL using headless Chromium.

    Skips known paywalled domains. Uses a real browser to avoid
    TLS-fingerprint and JS-challenge bot detection.

    Args:
        url: The URL to fetch
        timeout: Navigation timeout in seconds

    Returns:
        Raw HTML string, or None on failure
    """
    if _is_blocked(url):
        logger.debug("Skipping blocked domain: %s", url)
        return None
    try:
        context = await _get_context()
        page = await context.new_page()
        logger.debug("Playwright: navigating to %s", url)
        try:
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout * 1000,
            )
            if response is None or response.status >= 400:
                logger.debug("Playwright: HTTP %s for %s", response.status if response else "?", url)
                return None
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return None
            logger.debug("Playwright: successfully fetched %s", url)
            return await page.content()
        finally:
            await page.close()
    except Exception as exc:
        logger.warning("Playwright: failed to fetch %s: %s", url, exc)
        return None


def extract_content(html: str, max_chars: int = 3000) -> Optional[str]:
    """Extract main body text from HTML using trafilatura.

    Args:
        html: Raw HTML string
        max_chars: Maximum characters to return

    Returns:
        Extracted text truncated to max_chars, or None if extraction fails
    """
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_recall=True,
    )
    if not text or not text.strip():
        return None
    text = text.strip()
    if len(text) > max_chars:
        # Truncate at last sentence boundary within budget
        truncated = text[:max_chars]
        last_period = truncated.rfind(". ")
        if last_period > max_chars // 2:
            truncated = truncated[: last_period + 1]
        text = truncated
    return text


async def fetch_and_extract(
    url: str, max_chars: int = 3000, timeout: float = 15.0
) -> Optional[str]:
    """Fetch a URL and extract its main content.

    Results are cached by (url, max_chars) with request coalescing — if
    multiple coroutines request the same URL concurrently, only one
    Playwright fetch is performed.

    Args:
        url: The URL to fetch and extract
        max_chars: Maximum characters of extracted text
        timeout: Request timeout in seconds

    Returns:
        Extracted text, or None on failure
    """
    cache = _get_content_cache()
    if cache is not None:
        cache_key = _make_content_cache_key(url, max_chars)

        async def _fetch() -> Optional[str]:
            return await _fetch_and_extract_uncached(url, max_chars, timeout)

        return await cache.get_or_fetch(cache_key, _fetch)

    return await _fetch_and_extract_uncached(url, max_chars, timeout)


async def _fetch_and_extract_uncached(
    url: str, max_chars: int = 3000, timeout: float = 15.0
) -> Optional[str]:
    """Fetch a URL and extract its main content (no caching)."""
    html = await fetch_page(url, timeout=timeout)
    if html is None:
        return None
    return extract_content(html, max_chars=max_chars)


async def enrich_results(
    results: list[SearchResult],
    max_pages: int = 3,
    max_chars_per_page: int = 3000,
    timeout: float = 15.0,
) -> dict[str, str]:
    """Fetch and extract content for the top N search results.

    Pre-filters blocked domains and backfills with the next eligible URLs
    so we don't waste slots on sites that will 403. Fetches in parallel.

    Args:
        results: Search results to enrich
        max_pages: Maximum number of pages to fetch
        max_chars_per_page: Maximum extracted chars per page
        timeout: Per-request timeout

    Returns:
        Dict mapping URL to extracted content (only successful extractions)
    """
    # Pick the first max_pages non-blocked results
    targets: list[SearchResult] = []
    for r in results:
        if len(targets) >= max_pages:
            break
        if not _is_blocked(r.url):
            targets.append(r)

    if not targets:
        return {}

    tasks = [
        fetch_and_extract(r.url, max_chars=max_chars_per_page, timeout=timeout)
        for r in targets
    ]
    extracted = await asyncio.gather(*tasks, return_exceptions=True)

    content_map: dict[str, str] = {}
    for result, content in zip(targets, extracted):
        if isinstance(content, Exception):
            logger.warning("Extraction failed for %s: %s", result.url, content)
            continue
        if content:
            content_map[result.url] = content

    logger.info(
        "Extracted content from %d/%d pages", len(content_map), len(targets)
    )
    return content_map
