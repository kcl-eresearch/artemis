"""Search functionality for Artemis.

This module provides the core search integration with SearXNG. It handles:
- Building search query parameters
- Making async HTTP requests to SearXNG
- Parsing and normalizing search results
- Domain filtering functionality

The main entry point is search_searxng(), which returns a list of SearchResult
objects ready for use in API responses or further processing.

A module-level httpx.AsyncClient is used for connection pooling across requests.
"""

import logging
from typing import Optional
from urllib.parse import urlparse

import httpx

from artemis.config import get_settings
from artemis.errors import UpstreamServiceError
from artemis.models import SearchResult

logger = logging.getLogger(__name__)

# Module-level client for connection pooling (created lazily)
_client: httpx.AsyncClient | None = None


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
    """Close the shared httpx client (called during app shutdown)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


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
