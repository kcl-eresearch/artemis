"""Search functionality for Artemis.

This module provides the core search integration with SearXNG. It handles:
- Building search query parameters
- Making async HTTP requests to SearXNG
- Parsing and normalizing search results
- Domain filtering functionality
- Caching with request coalescing to avoid duplicate searches
- Semantic query deduplication via embeddings (when EMBEDDING_MODEL is set)

The main entry point is search_searxng(), which returns a list of SearchResult
objects ready for use in API responses or further processing.

A module-level httpx.AsyncClient is used for connection pooling across requests.
"""

import hashlib
import json
import logging
import time
from typing import Optional
from urllib.parse import urlparse

import httpx

from artemis.cache import AsyncTTLCache
from artemis.config import get_settings
from artemis.errors import UpstreamServiceError
from artemis.models import SearchResult

logger = logging.getLogger(__name__)

# Module-level client for connection pooling (created lazily)
_client: httpx.AsyncClient | None = None

# Module-level search cache (created lazily based on config)
_search_cache: AsyncTTLCache[list[SearchResult]] | None = None

# Semantic index: maps cache_key -> (embedding, params_hash, created_at)
# Only populated when EMBEDDING_MODEL is configured.
_embedding_index: dict[str, tuple[list[float], str, float]] = {}


def _get_client() -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient, creating it lazily on first use."""
    global _client
    if _client is None or _client.is_closed:
        settings = get_settings()
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.searxng_timeout_seconds),
            follow_redirects=False,
            headers={"User-Agent": "Artemis/0.2.0"},
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
        )
    return _client


async def close_client() -> None:
    """Close the shared httpx client and clear cache (called during app shutdown)."""
    global _client, _search_cache, _embedding_index
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
    if _search_cache is not None:
        logger.info(
            "Search cache stats: hits=%d misses=%d coalesced=%d hit_rate=%.1f%%",
            _search_cache.stats.hits,
            _search_cache.stats.misses,
            _search_cache.stats.coalesced,
            _search_cache.stats.hit_rate * 100,
        )
        _search_cache.clear()
        _search_cache = None
    if _embedding_index:
        logger.info("Clearing %d embedding index entries", len(_embedding_index))
        _embedding_index.clear()


def _get_search_cache() -> AsyncTTLCache[list[SearchResult]] | None:
    """Return the search cache if caching is enabled, creating lazily."""
    global _search_cache
    settings = get_settings()
    if not settings.cache_enabled:
        return None
    if _search_cache is None:
        _search_cache = AsyncTTLCache(
            ttl_seconds=settings.search_cache_ttl_seconds,
            max_entries=settings.cache_max_entries,
            name="search",
        )
    return _search_cache


def _make_search_cache_key(
    query: str,
    categories: Optional[str],
    engines: Optional[str],
    language: Optional[str],
    pageno: Optional[int],
    time_range: Optional[str],
    safesearch: Optional[int],
    max_results: Optional[int],
    domain_filter: Optional[list[str]],
) -> str:
    """Build a deterministic cache key from search parameters."""
    # Normalize domain_filter to sorted tuple for consistent hashing
    df = tuple(sorted(domain_filter)) if domain_filter else ()
    key_data = json.dumps(
        [query, categories, engines, language, pageno, time_range,
         safesearch, max_results, df],
        sort_keys=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()


def _make_params_hash(
    categories: Optional[str],
    engines: Optional[str],
    language: Optional[str],
    pageno: Optional[int],
    time_range: Optional[str],
    safesearch: Optional[int],
    max_results: Optional[int],
    domain_filter: Optional[list[str]],
) -> str:
    """Hash of non-query params — semantic matches must share identical params."""
    df = tuple(sorted(domain_filter)) if domain_filter else ()
    key_data = json.dumps(
        [categories, engines, language, pageno, time_range,
         safesearch, max_results, df],
        sort_keys=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()


def _find_semantic_match(
    query_embedding: list[float],
    params_hash: str,
    threshold: float,
    cache: AsyncTTLCache[list[SearchResult]],
) -> list[SearchResult] | None:
    """Scan the embedding index for a semantically similar cached query.

    Only considers entries whose non-query params match exactly (same
    categories, engines, language, etc.) and whose cache entry is still
    live (not expired).

    Returns cached results if cosine similarity >= threshold, else None.
    """
    from artemis.llm import cosine_similarity

    best_sim = 0.0
    best_key: str | None = None

    for cache_key, (emb, p_hash, _created) in _embedding_index.items():
        if p_hash != params_hash:
            continue
        # Only consider entries still live in the TTL cache
        if cache.get(cache_key) is None:
            continue
        sim = cosine_similarity(query_embedding, emb)
        if sim > best_sim:
            best_sim = sim
            best_key = cache_key

    if best_key is not None and best_sim >= threshold:
        logger.info(
            "Semantic cache hit: similarity=%.3f (threshold=%.2f)",
            best_sim, threshold,
        )
        return cache.get(best_key)

    return None


def _normalize_domain(domain: str) -> str:
    """Normalize a domain string for consistent comparison.

    Converts to lowercase, strips whitespace, and removes leading dots.
    This ensures "example.com", "Example.com", and ".example.com" all
    compare equally.

    Args:
        domain: Raw domain string

    Returns:
        Normalized domain string
    """
    return domain.strip().lower().strip(".")


def _normalize_domain_filters(domain_filter: Optional[list[str]]) -> list[str]:
    """Parse and normalize domain filter list.

    Accepts raw domain strings or full URLs, extracts the hostname,
    and normalizes each for filtering. Removes duplicates and empty values.

    Args:
        domain_filter: List of domains or URLs to filter by

    Returns:
        List of normalized domain strings
    """
    if not domain_filter:
        return []

    normalized_filters: list[str] = []
    for raw_domain in domain_filter:
        candidate = raw_domain.strip().lower()
        if not candidate:
            continue

        # If full URL provided, extract hostname
        if "://" in candidate:
            parsed = urlparse(candidate)
            candidate = parsed.hostname or ""

        candidate = _normalize_domain(candidate)
        if candidate and candidate not in normalized_filters:
            normalized_filters.append(candidate)

    return normalized_filters


def _domain_matches(domain: str, allowed_domain: str) -> bool:
    """Check if a domain matches an allowed domain pattern.

    Performs exact match and subdomain matching. For example,
    "www.example.com" matches "example.com" but "notexample.com" does not.

    Args:
        domain: The domain to check
        allowed_domain: The allowed domain pattern

    Returns:
        True if domain matches the allowed pattern
    """
    return domain == allowed_domain or domain.endswith(f".{allowed_domain}")


async def search_searxng(
    query: str,
    categories: Optional[str] = None,
    engines: Optional[str] = None,
    language: Optional[str] = "en",
    pageno: Optional[int] = 1,
    time_range: Optional[str] = None,
    format: str = "json",
    safesearch: Optional[int] = None,
    image_proxy: Optional[bool] = None,
    autocomplete: Optional[str] = None,
    results_on_new_tab: Optional[int] = None,
    max_results: Optional[int] = 10,
    domain_filter: Optional[list[str]] = None,
) -> list[SearchResult]:
    """Execute a search against the configured SearXNG instance.

    Makes an async HTTP GET request to the SearXNG search endpoint and
    parses the JSON response into SearchResult objects. Supports various
    SearXNG-specific parameters and optional domain filtering.

    Args:
        query: Search query string
        categories: SearXNG categories to search (e.g., "general,images")
        engines: Specific search engines to use
        language: Language code for results (default: "en")
        pageno: Page number for pagination (default: 1)
        time_range: Time range filter (e.g., "day", "month", "year")
        format: Response format (default: "json")
        safesearch: Safe search level (0=off, 1=moderate, 2=strict)
        image_proxy: Whether to proxy images through SearXNG
        autocomplete: Autocomplete service to use
        results_on_new_tab: Open results in new tab (1=yes, 0=no)
        max_results: Maximum results to return (default: 10)
        domain_filter: Optional list of domains to restrict results

    Returns:
        List of SearchResult objects (may be fewer than max_results)

    Raises:
        UpstreamServiceError: If SearXNG is unreachable or returns invalid data
    """
    settings = get_settings()

    # Check cache first
    cache = _get_search_cache()
    cache_key = _make_search_cache_key(
        query, categories, engines, language, pageno,
        time_range, safesearch, max_results, domain_filter,
    ) if cache is not None else ""

    if cache is not None:
        # Exact cache hit handled by get_or_fetch; but first try semantic match
        exact = cache.get(cache_key)
        if exact is not None:
            cache.stats.hits += 1
            return exact

        # Semantic deduplication: if an embedding model is configured,
        # check whether a semantically similar query is already cached.
        if settings.embedding_model:
            try:
                from artemis.llm import embed
                query_emb = await embed(query, settings.embedding_model)
                params_hash = _make_params_hash(
                    categories, engines, language, pageno,
                    time_range, safesearch, max_results, domain_filter,
                )
                semantic_hit = _find_semantic_match(
                    query_emb, params_hash,
                    settings.semantic_similarity_threshold, cache,
                )
                if semantic_hit is not None:
                    return semantic_hit
            except Exception:
                logger.debug("Semantic lookup failed, falling back to exact match",
                             exc_info=True)
                query_emb = None
                params_hash = ""
        else:
            query_emb = None
            params_hash = ""

        async def _fetch() -> list[SearchResult]:
            return await _search_searxng_uncached(
                query, categories, engines, language, pageno,
                time_range, format, safesearch, image_proxy,
                autocomplete, results_on_new_tab, max_results, domain_filter,
            )

        results = await cache.get_or_fetch(cache_key, _fetch)

        # Index the embedding for future semantic lookups
        if query_emb is not None and params_hash:
            _embedding_index[cache_key] = (query_emb, params_hash, time.monotonic())

        return results

    return await _search_searxng_uncached(
        query, categories, engines, language, pageno,
        time_range, format, safesearch, image_proxy,
        autocomplete, results_on_new_tab, max_results, domain_filter,
    )


async def _search_searxng_uncached(
    query: str,
    categories: Optional[str] = None,
    engines: Optional[str] = None,
    language: Optional[str] = "en",
    pageno: Optional[int] = 1,
    time_range: Optional[str] = None,
    format: str = "json",
    safesearch: Optional[int] = None,
    image_proxy: Optional[bool] = None,
    autocomplete: Optional[str] = None,
    results_on_new_tab: Optional[int] = None,
    max_results: Optional[int] = 10,
    domain_filter: Optional[list[str]] = None,
) -> list[SearchResult]:
    """Execute a search against SearXNG without caching."""
    settings = get_settings()
    params = {
        "q": query,
        "format": format,
        "lang": language or "en",
        "pageno": pageno or 1,
    }

    max_results = max_results or 10

    if categories:
        params["categories"] = categories
    if engines:
        params["engines"] = engines
    if time_range:
        params["time_range"] = time_range
    if safesearch is not None:
        params["safesearch"] = safesearch
    if image_proxy is not None:
        params["image_proxy"] = str(image_proxy).lower()
    if autocomplete:
        params["autocomplete"] = autocomplete
    if results_on_new_tab is not None:
        params["results_on_new_tab"] = results_on_new_tab

    try:
        client = _get_client()
        response = await client.get(
            f"{settings.searxng_api_base}/search", params=params
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise UpstreamServiceError("The search backend timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise UpstreamServiceError(
            f"The search backend returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise UpstreamServiceError("The search backend request failed.") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise UpstreamServiceError("The search backend returned invalid JSON.") from exc

    results: list[SearchResult] = []
    raw_results = data.get("results", [])
    if not isinstance(raw_results, list):
        raise UpstreamServiceError(
            "The search backend returned an invalid results payload."
        )
    normalized_filters = _normalize_domain_filters(domain_filter)

    for item in raw_results:
        url = item.get("url", "")
        parsed_url = urlparse(url)
        if (
            parsed_url.scheme.lower() not in {"http", "https"}
            or not parsed_url.hostname
        ):
            continue

        if normalized_filters:
            domain = _normalize_domain(parsed_url.hostname)
            if not any(
                _domain_matches(domain, allowed_domain)
                for allowed_domain in normalized_filters
            ):
                continue

        snippet = item.get("content", "") or item.get("snippet", "")
        title = item.get("title") or url

        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                date=item.get("publishedDate") or item.get("date"),
            )
        )

        if len(results) >= max_results:
            break

    return results
