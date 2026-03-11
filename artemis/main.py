"""FastAPI application entrypoint for Artemis.

This module provides the FastAPI application with two API patterns:

1. /search endpoint - Simple search with optional LLM summarization
2. /v1/responses endpoint - Perplexity-compatible API with deep research support

The application includes:
- Bearer token authentication (optional, configured via ARTEMIS_API_KEY)
- CORS middleware for browser access
- Circuit breaker pattern for LLM summarization failures
- Comprehensive error handling for upstream service failures
"""

from dataclasses import dataclass
import asyncio
import contextvars
import json as _json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from artemis.config import (
    Settings,
    get_settings,
)
from artemis.errors import UpstreamServiceError
from artemis.extractor import close_client as close_extractor_client
from artemis.llm import close_client as close_llm_client
from artemis.models import (
    AssistantMessage,
    OutputText,
    ResponsesAPIResponse,
    ResponsesRequest,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchResultItem,
    SearchResultsBlock,
    TokenUsage,
)
from artemis.researcher import deep_research
from artemis.searcher import search_searxng
from artemis.searcher import close_client as close_searxng_client
from artemis.summarizer import summarize_results

# ---------------------------------------------------------------------------
# Request ID context variable — set by middleware, read by log formatter
# ---------------------------------------------------------------------------
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class _RequestIdFilter(logging.Filter):
    """Inject request_id from context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("-")  # type: ignore[attr-defined]
        return True


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            obj["exception"] = self.formatException(record.exc_info)
        return _json.dumps(obj)


def _configure_logging(settings_obj: Settings) -> None:
    """Configure root logger with request ID support."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings_obj.log_level, logging.INFO))
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())

    if settings_obj.log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] [%(request_id)s] %(message)s"
        ))
    root.addHandler(handler)


# Load settings at module import time to fail fast on config errors
initial_settings = get_settings()

# Configure structured logging
_configure_logging(initial_settings)
logger = logging.getLogger(__name__)

# Log security warnings at startup
if not initial_settings.artemis_api_key:
    logger.warning("ARTEMIS_API_KEY is not set; API endpoints are unauthenticated.")
if not initial_settings.allowed_origins:
    logger.info("ALLOWED_ORIGINS is empty; browser clients will be blocked by CORS.")

# HTTP Bearer security scheme (auto_error=False allows optional auth)
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    yield
    await close_llm_client()
    await close_searxng_client()
    await close_extractor_client()


@dataclass
class SummaryCircuitState:
    """State for the summarization circuit breaker.

    Tracks consecutive failures to temporarily disable summarization
    after repeated upstream LLM failures, preventing cascading errors.

    Attributes:
        consecutive_failures: Number of recent LLM failures
        opened_until: Unix timestamp when circuit closes (0 = closed)
    """

    consecutive_failures: int = 0
    opened_until: float = 0.0


# Circuit breaker configuration
SUMMARY_CIRCUIT_FAILURE_THRESHOLD = 3  # Failures before opening circuit
SUMMARY_CIRCUIT_BACKOFF_SECONDS = 300  # 5 minute cooldown
summary_circuit = SummaryCircuitState()


@dataclass(frozen=True)
class ResearchPresetConfig:
    stages: int
    passes: int
    subqueries: int
    results_per_query: int
    max_tokens: int
    content_extraction: bool
    pages_per_section: int
    content_max_chars: int
    model_name: str


def _research_preset_config(settings: Settings, preset: str) -> ResearchPresetConfig:
    if preset == "shallow-research":
        return ResearchPresetConfig(
            stages=settings.shallow_research_stages,
            passes=settings.shallow_research_passes,
            subqueries=settings.shallow_research_subqueries,
            results_per_query=settings.shallow_research_results_per_query,
            max_tokens=settings.shallow_research_max_tokens,
            content_extraction=settings.shallow_research_content_extraction,
            pages_per_section=settings.shallow_research_pages_per_section,
            content_max_chars=settings.shallow_research_content_max_chars,
            model_name="artemis-shallow-research",
        )

    return ResearchPresetConfig(
        stages=settings.deep_research_stages,
        passes=settings.deep_research_passes,
        subqueries=settings.deep_research_subqueries,
        results_per_query=settings.deep_research_results_per_query,
        max_tokens=settings.deep_research_max_tokens,
        content_extraction=settings.deep_research_content_extraction,
        pages_per_section=settings.deep_research_pages_per_section,
        content_max_chars=settings.deep_research_content_max_chars,
        model_name="artemis-deep-research",
    )


app = FastAPI(
    title="Artemis",
    description="SearXNG API - Perplexity-compatible interface",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(initial_settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique request ID to every request for log correlation."""
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    token = request_id_ctx.set(rid)
    try:
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response
    finally:
        request_id_ctx.reset(token)


@app.exception_handler(UpstreamServiceError)
async def upstream_service_error_handler(_, exc: UpstreamServiceError) -> JSONResponse:
    logger.error("Upstream service failure: %s", exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc)},
    )


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    """Verify API key from Authorization header.

    If ARTEMIS_API_KEY is configured, validates the Bearer token.
    Uses constant-time comparison (secrets.compare_digest) to prevent timing attacks.
    If no API key is configured, always allows access (unauthenticated).

    Args:
        credentials: HTTP Authorization credentials from request

    Raises:
        HTTPException: 401 if credentials are invalid or missing when auth is required
    """
    api_key = get_settings().artemis_api_key
    if not api_key:
        return

    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not secrets.compare_digest(credentials.credentials, api_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _created_timestamp() -> int:
    """Get current Unix timestamp for response metadata.

    Returns:
        Current Unix timestamp (seconds since epoch)
    """
    return int(time.time())


def _result_items(results: list[SearchResult]) -> list[SearchResultItem]:
    """Convert internal SearchResult objects to API response format.

    Args:
        results: List of internal SearchResult objects

    Returns:
        List of SearchResultItem for Perplexity-compatible responses
    """
    return [
        SearchResultItem(
            id=index,
            url=result.url,
            title=result.title,
            snippet=result.snippet,
            date=result.date,
        )
        for index, result in enumerate(results)
    ]


def _message_output(text: str) -> AssistantMessage:
    """Create an AssistantMessage for Perplexity-style responses.

    Args:
        text: The message content

    Returns:
        AssistantMessage with the given text
    """
    return AssistantMessage(
        id=str(uuid.uuid4()),
        content=[OutputText(text=text)],
    )


def _fallback_text(results: list[SearchResult]) -> str:
    """Generate fallback text from search result snippets.

    Used when LLM summarization is unavailable. Concatenates snippets
    from the first 3 results.

    Args:
        results: List of search results

    Returns:
        Space-joined snippets or "No results found."
    """
    snippets = [
        result.snippet.strip() for result in results[:3] if result.snippet.strip()
    ]
    return " ".join(snippets) if snippets else "No results found."


def _reset_summary_circuit() -> None:
    """Reset the summarization circuit after successful summary."""
    summary_circuit.consecutive_failures = 0
    summary_circuit.opened_until = 0.0


def _record_summary_failure() -> list[str]:
    """Record a summarization failure and potentially open the circuit.

    Increments failure counter. If threshold is reached, opens the circuit
    for a cooldown period.

    Returns:
        Warning message to include in API response
    """
    summary_circuit.consecutive_failures += 1
    if summary_circuit.consecutive_failures >= SUMMARY_CIRCUIT_FAILURE_THRESHOLD:
        summary_circuit.opened_until = time.time() + SUMMARY_CIRCUIT_BACKOFF_SECONDS
        logger.error(
            "Summary circuit opened after %s consecutive failures.",
            summary_circuit.consecutive_failures,
        )
        return [
            "LLM summarization is temporarily disabled after repeated upstream failures."
        ]

    return ["LLM summarization is unavailable; returning search results only."]



async def _stream_responses(request: ResponsesRequest) -> StreamingResponse:
    """Stream response as plain text - thinking + final output.

    Uses an asyncio.Queue so progress events from deep_research are
    yielded to the client in real-time rather than collected and replayed.
    """
    async def event_generator():
        settings = get_settings()

        yield f"[Starting research on: {request.input}]\n\n"

        if request.preset in {"deep-research", "shallow-research"}:
            queue: asyncio.Queue[str] = asyncio.Queue()
            preset_config = _research_preset_config(settings, request.preset)

            def progress_cb(stage: str, msg: str):
                queue.put_nowait(f"[{stage.upper()}] {msg}")

            research_task = asyncio.create_task(deep_research(
                request.input,
                stages=request.max_steps or preset_config.stages,
                passes=preset_config.passes,
                sub_queries_per_stage=preset_config.subqueries,
                results_per_query=preset_config.results_per_query,
                max_tokens=preset_config.max_tokens,
                outline=request.outline,
                content_extraction=preset_config.content_extraction,
                pages_per_section=preset_config.pages_per_section,
                content_max_chars=preset_config.content_max_chars,
                progress_callback=progress_cb,
            ))

            # Yield progress events as they arrive
            while not research_task.done():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield msg + "\n\n"
                except asyncio.TimeoutError:
                    continue

            # Drain any remaining queued messages
            while not queue.empty():
                yield queue.get_nowait() + "\n\n"

            # Propagate any exception from the research task
            research_run = research_task.result()

            yield research_run.essay
            yield f"\n\n[Found {len(research_run.results)} sources]"
            yield f"\n[USAGE] {research_run.usage.model_dump_json()}"
        else:
            yield "[Searching...]\n"
            results = await search_searxng(query=request.input, max_results=10)
            yield f"[Found {len(results)} results]\n\n"

            summary, usage, warnings = await _build_summary(request.input, results)
            yield summary or _fallback_text(results)

            response_usage = usage or TokenUsage()
            response_usage.search_requests = 1
            citation_chars = sum(len(r.snippet or "") + len(r.title) for r in results)
            response_usage.citation_tokens = citation_chars // 4
            yield f"\n[USAGE] {response_usage.model_dump_json()}"

    return StreamingResponse(event_generator(), media_type="text/plain")


async def _build_summary(
    query: str, results: list[SearchResult]
) -> tuple[str | None, TokenUsage | None, list[str]]:
    """Build an LLM summary of search results.

    Handles the summarization workflow including:
    - Checking if summarization is enabled
    - Circuit breaker check (skips if circuit is open)
    - Calling the summarizer
    - Error handling and circuit management

    Args:
        query: Original search query
        results: Search results to summarize

    Returns:
        Tuple of (summary text, token usage, warnings)
    """
    settings = get_settings()
    if not settings.enable_summary or not results:
        return None, None, []

    if summary_circuit.opened_until > time.time():
        return (
            None,
            None,
            [
                "LLM summarization is temporarily disabled after repeated upstream failures."
            ],
        )

    try:
        summarize_output = await summarize_results(
            query=query,
            results=results,
            model=settings.summary_model,
            max_tokens=settings.summary_max_tokens,
        )
    except UpstreamServiceError as exc:
        logger.warning("Summary generation failed for %r: %s", query, exc)
        return (None, None, _record_summary_failure())

    usage = summarize_output.get("usage")
    _reset_summary_circuit()
    return (
        summarize_output.get("summary"),
        TokenUsage.model_validate(usage or {}) if usage is not None else None,
        [],
    )


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint returning service information."""
    return {"message": "Artemis API - SearXNG with Perplexity-compatible interface"}


@app.get("/health")
async def health() -> dict[str, object]:
    """Health check endpoint.

    Returns service status and configuration info including:
    - Whether summarization is enabled
    - Whether authentication is required
    - Whether the summary circuit is open (due to failures)
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "summary_enabled": settings.enable_summary,
        "auth_enabled": bool(settings.artemis_api_key),
        "summary_circuit_open": summary_circuit.opened_until > time.time(),
    }


@app.post(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(verify_api_key)],
)
async def search(request: SearchRequest) -> SearchResponse:
    """Simple search endpoint.

    Executes a search against SearXNG and optionally generates an LLM summary.

    Request:
        - query: Search query string
        - max_results: Maximum results to return (1-50)
        - search_domain_filter: Optional domains to restrict search

    Response:
        - id: Unique response identifier
        - results: List of search results
        - summary: LLM-generated summary (if enabled)
        - usage: Token usage for summarization
        - warnings: Any operational warnings
    """
    results = await search_searxng(
        query=request.query,
        max_results=request.max_results,
        domain_filter=request.search_domain_filter,
    )

    summary, usage, warnings = await _build_summary(request.query, results)

    # Enrich usage with search-specific metrics
    if usage is not None:
        usage.search_requests = 1
        citation_chars = sum(len(r.snippet or "") + len(r.title) for r in results)
        usage.citation_tokens = citation_chars // 4

    return SearchResponse(
        id=str(uuid.uuid4()),
        results=results,
        summary=summary,
        usage=usage,
        warnings=warnings,
    )


@app.post(
    "/v1/responses",
    response_model=None,
    dependencies=[Depends(verify_api_key)],
)
async def responses(request: ResponsesRequest) -> ResponsesAPIResponse | StreamingResponse:
    """Perplexity-compatible responses endpoint.

    Supports streaming via SSE when streaming=true.
    """
    if request.streaming:
        return await _stream_responses(request)
    
    if request.preset in {"deep-research", "shallow-research"}:
        settings = get_settings()
        preset_config = _research_preset_config(settings, request.preset)
        research_run = await deep_research(
            request.input,
            stages=request.max_steps or preset_config.stages,
            passes=preset_config.passes,
            sub_queries_per_stage=preset_config.subqueries,
            results_per_query=preset_config.results_per_query,
            max_tokens=preset_config.max_tokens,
            outline=request.outline,
            content_extraction=preset_config.content_extraction,
            pages_per_section=preset_config.pages_per_section,
            content_max_chars=preset_config.content_max_chars,
        )
        return ResponsesAPIResponse(
            id=str(uuid.uuid4()),
            created=_created_timestamp(),
            model=preset_config.model_name,
            output=[
                _message_output(research_run.essay),
                SearchResultsBlock(results=_result_items(research_run.results)),
            ],
            usage=research_run.usage,
        )

    results = await search_searxng(query=request.input, max_results=10)
    summary, usage, warnings = await _build_summary(request.input, results)

    response_usage = usage or TokenUsage()
    response_usage.search_requests = 1
    citation_chars = sum(len(r.snippet or "") + len(r.title) for r in results)
    response_usage.citation_tokens = citation_chars // 4

    return ResponsesAPIResponse(
        id=str(uuid.uuid4()),
        created=_created_timestamp(),
        model="artemis-search",
        output=[
            _message_output(summary or _fallback_text(results)),
            SearchResultsBlock(results=_result_items(results)),
        ],
        usage=response_usage,
        warnings=warnings,
    )


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application with uvicorn
    # Bind to all interfaces on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
