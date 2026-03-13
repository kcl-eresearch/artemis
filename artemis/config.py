"""Configuration loading and validation for Artemis.

This module loads configuration from environment variables with type validation
and sensible defaults. It provides a centralized Settings dataclass that is
cached (via lru_cache) to avoid repeated parsing overhead.

Environment variables are loaded from .env files via python-dotenv. All config
values are validated at startup, raising ConfigError for invalid values.

The module exposes:
- Settings: Frozen dataclass holding all validated configuration
- get_settings(): Cached function returning the Settings singleton
- refresh_settings(): Clears cache and reloads configuration
"""

from dataclasses import dataclass
from functools import lru_cache
import os
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

# Valid boolean string values (case-insensitive)
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
# Valid Python logging levels
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(ValueError):
    """Raised when Artemis receives invalid configuration.

    This exception is raised at startup if any environment variable contains
    an invalid value that fails validation (e.g., invalid URL, out-of-range
    integer, unknown boolean string).
    """


def _parse_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable.

    Accepts common string representations of boolean values (case-insensitive).
    Returns the default if the variable is not set.

    Args:
        name: Environment variable name
        default: Default value if variable is not set

    Returns:
        Parsed boolean value

    Raises:
        ConfigError: If the value is not a recognized boolean string
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be a boolean value.")


def _parse_int(
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Parse an integer environment variable with bounds checking.

    Args:
        name: Environment variable name
        default: Default value if variable is not set
        minimum: Minimum allowed value (inclusive)
        maximum: Maximum allowed value (inclusive), or None for no upper bound

    Returns:
        Parsed integer value

    Raises:
        ConfigError: If the value is not an integer or is outside allowed range
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc

    if parsed < minimum:
        raise ConfigError(f"{name} must be greater than or equal to {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ConfigError(f"{name} must be less than or equal to {maximum}.")

    return parsed


def _parse_float(
    name: str,
    default: float,
    *,
    minimum: float = 0.1,
    maximum: float | None = None,
) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number.") from exc

    if parsed < minimum:
        raise ConfigError(f"{name} must be greater than or equal to {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ConfigError(f"{name} must be less than or equal to {maximum}.")

    return parsed


def _parse_optional_str(name: str) -> str | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None

    stripped = raw_value.strip()
    return stripped or None


def _validate_url(name: str, value: str) -> str:
    """Validate that a value is a proper HTTP/HTTPS URL.

    Args:
        name: Environment variable name (for error messages)
        value: The URL string to validate

    Returns:
        Validated URL with trailing slashes stripped

    Raises:
        ConfigError: If the URL is invalid or doesn't use http/https
    """
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"{name} must be a valid http or https URL.")
    return value.rstrip("/")


def _parse_allowed_origins(name: str = "ALLOWED_ORIGINS") -> tuple[str, ...]:
    """Parse comma-separated CORS origins from environment variable.

    Supports:
    - Empty string: returns empty tuple (no origins allowed)
    - "*": wildcard (allows all origins)
    - Comma-separated list: e.g., "https://example.com,https://app.example.com"

    Each origin is normalized to scheme://netloc format and duplicates are removed.

    Args:
        name: Environment variable name (default: ALLOWED_ORIGINS)

    Returns:
        Tuple of validated origin strings

    Raises:
        ConfigError: If any origin is invalid
    """
    raw_value = os.getenv(name, "")
    if not raw_value.strip():
        return tuple()

    if raw_value.strip() == "*":
        return ("*",)

    origins: list[str] = []
    for part in raw_value.split(","):
        origin = part.strip()
        if not origin:
            continue

        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigError(
                f"{name} must contain valid comma-separated http or https origins."
            )

        normalized_origin = f"{parsed.scheme}://{parsed.netloc}"
        if normalized_origin not in origins:
            origins.append(normalized_origin)

    return tuple(origins)


def _parse_log_level(name: str, default: str = "INFO") -> str:
    """Parse Python logging level from environment variable.

    Args:
        name: Environment variable name
        default: Default level if variable is not set (default: INFO)

    Returns:
        Validated logging level string

    Raises:
        ConfigError: If the value is not a valid logging level
    """
    raw_value = os.getenv(name, default).strip().upper()
    if raw_value not in _LOG_LEVELS:
        allowed_levels = ", ".join(sorted(_LOG_LEVELS))
        raise ConfigError(f"{name} must be one of: {allowed_levels}.")
    return raw_value


_LOG_FORMATS = {"text", "json"}


def _parse_log_format(name: str, default: str = "text") -> str:
    """Parse log format from environment variable.

    Args:
        name: Environment variable name
        default: Default format if variable is not set (default: text)

    Returns:
        Validated log format string ("text" or "json")

    Raises:
        ConfigError: If the value is not a valid log format
    """
    raw_value = os.getenv(name, default).strip().lower()
    if raw_value not in _LOG_FORMATS:
        raise ConfigError(f"{name} must be one of: text, json.")
    return raw_value


@dataclass(frozen=True)
class Settings:
    """Immutable configuration container for Artemis.

    This frozen dataclass holds all validated configuration values for the
    Artemis service. All fields are populated at initialization from environment
    variables with sensible defaults.

    Attributes:
        searxng_api_base: Base URL of the SearXNG instance
        searxng_timeout_seconds: Timeout for SearXNG API calls
        litellm_base_url: Base URL for LLM API (LiteLLM-compatible)
        litellm_api_key: API key for LLM service (optional)
        llm_timeout_seconds: Timeout for LLM API calls
        summary_model: Model identifier for result summarization
        summary_max_tokens: Max tokens for summary generation
        enable_summary: Whether LLM summarization is enabled
        deep_research_stages: Number of outline sections in deep research mode
        deep_research_passes: Number of research passes (refines queries between passes)
        deep_research_subqueries: Subqueries generated per stage
        deep_research_results_per_query: Search results per subquery
        deep_research_max_tokens: Max tokens for final essay
        deep_research_content_extraction: Whether to fetch full page content for synthesis
        deep_research_pages_per_section: Pages to fetch per outline section
        deep_research_content_max_chars: Max characters of extracted content per page
        shallow_research_stages: Number of outline sections in shallow research mode
        shallow_research_passes: Number of research passes in shallow research mode
        shallow_research_subqueries: Subqueries generated per stage in shallow research mode
        shallow_research_results_per_query: Search results per subquery in shallow research mode
        shallow_research_max_tokens: Max tokens for shallow research synthesis
        shallow_research_content_extraction: Whether shallow research extracts page content
        shallow_research_pages_per_section: Pages to fetch per outline section in shallow research mode
        shallow_research_content_max_chars: Max characters of extracted content per page in shallow research mode
        allowed_origins: CORS-allowed origins (empty = browser clients blocked)
        artemis_api_key: Bearer token for API authentication (optional)
        log_level: Python logging level
        cache_enabled: Whether in-memory caching is enabled
        search_cache_ttl_seconds: TTL for cached search results
        content_cache_ttl_seconds: TTL for cached extracted page content
        cache_max_entries: Maximum entries per cache before oldest are evicted
        embedding_model: Model for query embeddings (enables semantic dedup when set)
        semantic_similarity_threshold: Cosine similarity threshold for semantic cache hits
        log_format: Log output format ("text" or "json")
        playwright_context_recycle_pages: Recycle Playwright context after this many pages
        playwright_max_html_bytes: Maximum HTML bytes to extract per page
        synthesis_tool_rounds: Max rounds of live web_search tool calls during synthesis
            (0 = disabled; model writes from pre-loaded results only)
    """

    searxng_api_base: str
    searxng_timeout_seconds: float
    litellm_base_url: str
    litellm_api_key: str | None
    llm_timeout_seconds: float
    summary_model: str
    summary_max_tokens: int
    enable_summary: bool
    deep_research_stages: int
    deep_research_passes: int
    deep_research_subqueries: int
    deep_research_results_per_query: int
    deep_research_max_tokens: int
    deep_research_content_extraction: bool
    deep_research_pages_per_section: int
    deep_research_content_max_chars: int
    shallow_research_stages: int
    shallow_research_passes: int
    shallow_research_subqueries: int
    shallow_research_results_per_query: int
    shallow_research_max_tokens: int
    shallow_research_content_extraction: bool
    shallow_research_pages_per_section: int
    shallow_research_content_max_chars: int
    allowed_origins: tuple[str, ...]
    artemis_api_key: str | None
    log_level: str
    cache_enabled: bool
    search_cache_ttl_seconds: int
    content_cache_ttl_seconds: int
    cache_max_entries: int
    embedding_model: str | None
    semantic_similarity_threshold: float
    log_format: str
    playwright_context_recycle_pages: int
    playwright_max_html_bytes: int
    synthesis_tool_rounds: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the cached application settings.

    This function uses lru_cache to ensure configuration is only parsed once.
    On first call, all environment variables are read and validated. Subsequent
    calls return the cached Settings instance.

    Returns:
        The cached Settings instance with all configuration values

    Note:
        Any invalid configuration will raise ConfigError at first call.
        Use refresh_settings() to clear cache and reload configuration.
    """
    return Settings(
        searxng_api_base=_validate_url(
            "SEARXNG_API_BASE",
            os.getenv("SEARXNG_API_BASE", "http://localhost:8888"),
        ),
        searxng_timeout_seconds=_parse_float(
            "SEARXNG_TIMEOUT_SECONDS", 60.0, minimum=0.5, maximum=300.0
        ),
        litellm_base_url=_validate_url(
            "LITELLM_BASE_URL",
            os.getenv("LITELLM_BASE_URL", "http://localhost:11434/api"),
        ),
        litellm_api_key=_parse_optional_str("LITELLM_API_KEY")
        or _parse_optional_str("OPENAI_API_KEY"),
        llm_timeout_seconds=_parse_float(
            "LLM_TIMEOUT_SECONDS", 120.0, minimum=1.0, maximum=600.0
        ),
        summary_model=os.getenv("SUMMARY_MODEL", "qwen3.5:27b").strip() or "qwen3.5:27b",
        summary_max_tokens=_parse_int(
            "SUMMARY_MAX_TOKENS", 4000, minimum=512, maximum=16384
        ),
        enable_summary=_parse_bool("ENABLE_SUMMARY", True),
        deep_research_stages=_parse_int(
            "DEEP_RESEARCH_STAGES", 2, minimum=1, maximum=10
        ),
        deep_research_passes=_parse_int(
            "DEEP_RESEARCH_PASSES", 1, minimum=1, maximum=5
        ),
        deep_research_subqueries=_parse_int(
            "DEEP_RESEARCH_SUBQUERIES", 5, minimum=1, maximum=10
        ),
        deep_research_results_per_query=_parse_int(
            "DEEP_RESEARCH_RESULTS_PER_QUERY", 10, minimum=1, maximum=25
        ),
        deep_research_max_tokens=_parse_int(
            "DEEP_RESEARCH_MAX_TOKENS", 8000, minimum=256, maximum=32768
        ),
        deep_research_content_extraction=_parse_bool(
            "DEEP_RESEARCH_CONTENT_EXTRACTION", True
        ),
        deep_research_pages_per_section=_parse_int(
            "DEEP_RESEARCH_PAGES_PER_SECTION", 3, minimum=1, maximum=10
        ),
        deep_research_content_max_chars=_parse_int(
            "DEEP_RESEARCH_CONTENT_MAX_CHARS", 3000, minimum=500, maximum=10000
        ),
        shallow_research_stages=_parse_int(
            "SHALLOW_RESEARCH_STAGES", 1, minimum=1, maximum=10
        ),
        shallow_research_passes=_parse_int(
            "SHALLOW_RESEARCH_PASSES", 1, minimum=1, maximum=5
        ),
        shallow_research_subqueries=_parse_int(
            "SHALLOW_RESEARCH_SUBQUERIES", 3, minimum=1, maximum=10
        ),
        shallow_research_results_per_query=_parse_int(
            "SHALLOW_RESEARCH_RESULTS_PER_QUERY", 5, minimum=1, maximum=25
        ),
        shallow_research_max_tokens=_parse_int(
            "SHALLOW_RESEARCH_MAX_TOKENS", 4000, minimum=256, maximum=32768
        ),
        shallow_research_content_extraction=_parse_bool(
            "SHALLOW_RESEARCH_CONTENT_EXTRACTION", False
        ),
        shallow_research_pages_per_section=_parse_int(
            "SHALLOW_RESEARCH_PAGES_PER_SECTION", 2, minimum=1, maximum=10
        ),
        shallow_research_content_max_chars=_parse_int(
            "SHALLOW_RESEARCH_CONTENT_MAX_CHARS", 2000, minimum=500, maximum=10000
        ),
        allowed_origins=_parse_allowed_origins(),
        artemis_api_key=_parse_optional_str("ARTEMIS_API_KEY"),
        log_level=_parse_log_level("LOG_LEVEL"),
        cache_enabled=_parse_bool("CACHE_ENABLED", True),
        search_cache_ttl_seconds=_parse_int(
            "SEARCH_CACHE_TTL_SECONDS", 3600, minimum=0, maximum=86400
        ),
        content_cache_ttl_seconds=_parse_int(
            "CONTENT_CACHE_TTL_SECONDS", 86400, minimum=0, maximum=604800
        ),
        cache_max_entries=_parse_int(
            "CACHE_MAX_ENTRIES", 1000, minimum=10, maximum=100000
        ),
        embedding_model=_parse_optional_str("EMBEDDING_MODEL"),
        semantic_similarity_threshold=_parse_float(
            "SEMANTIC_SIMILARITY_THRESHOLD", 0.92, minimum=0.5, maximum=1.0
        ),
        log_format=_parse_log_format("LOG_FORMAT"),
        playwright_context_recycle_pages=_parse_int(
            "PLAYWRIGHT_CONTEXT_RECYCLE_PAGES", 50, minimum=1, maximum=10000
        ),
        playwright_max_html_bytes=_parse_int(
            "PLAYWRIGHT_MAX_HTML_BYTES", 5 * 1024 * 1024, minimum=65536, maximum=50 * 1024 * 1024
        ),
        synthesis_tool_rounds=_parse_int(
            "SYNTHESIS_TOOL_ROUNDS", 0, minimum=0, maximum=10
        ),
    )


def refresh_settings() -> Settings:
    """Clear cached settings and reload configuration.

    Clears the lru_cache and re-parses all environment variables.
    Primarily useful for testing or hot-reloading configuration.

    Returns:
        The newly loaded Settings instance

    Raises:
        ConfigError: If any environment variable has an invalid value
    """
    get_settings.cache_clear()
    return get_settings()
