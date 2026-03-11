# Copilot Instructions — Artemis

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python -m artemis.main

# Run all tests (247 unit tests, unittest-based)
python3 -m unittest discover -s tests -p 'test_*.py'

# Run a single test file
python3 -m unittest tests.test_searcher -v

# Run a single test case
python3 -m unittest tests.test_searcher.TestDomainFiltering.test_normalize_domain -v

# Integration test (requires live SearXNG + running app)
python3 -m tests.test

# Docker
docker-compose up -d
```

No linter or formatter is configured.

## Architecture

Artemis is a FastAPI service that wraps SearXNG (a meta-search engine) behind a Perplexity-compatible API. It sits behind LiteLLM as a custom LLM provider — LiteLLM reads `prompt_tokens`, `completion_tokens`, `search_requests`, and `citation_tokens` from responses for billing.

**Request flow:**

- `/search` → `searcher.search_searxng()` → optional `summarizer.summarize_results()` → response
- `/v1/responses` with `preset: "deep-research"` → `researcher.deep_research()` → multi-stage outline-driven research with query generation, search, optional content extraction, and essay synthesis
- `/v1/responses` without deep-research preset → single search + summarize (fast path)

**Module roles:**

| Module | Responsibility |
|---|---|
| `main.py` | FastAPI app, endpoints, auth, circuit breaker, streaming, request ID middleware |
| `config.py` | Frozen `Settings` dataclass from env vars, cached via `get_settings()` |
| `models.py` | Pydantic request/response models with LiteLLM billing fields |
| `searcher.py` | SearXNG HTTP client with domain filtering and caching |
| `summarizer.py` | Single-query LLM summarization |
| `researcher.py` | Multi-stage deep research orchestration (most complex module) |
| `extractor.py` | Playwright + trafilatura content extraction with domain blocking and context recycling |
| `llm.py` | LiteLLM-compatible chat completion and embedding client |
| `cache.py` | Generic async TTL cache with request coalescing |
| `errors.py` | `ArtemisError` → `UpstreamServiceError` hierarchy |
| `cli.py` | One-shot CLI for deep research without the HTTP server (JSON/MD/DOCX output) |

## Key Conventions

### Async HTTP client pooling

Every module that makes HTTP calls (`searcher`, `llm`, `extractor`) follows this pattern:

```python
_client: httpx.AsyncClient | None = None

def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(...)
    return _client

async def close_client():
    global _client
    if _client:
        await _client.aclose()
        _client = None
```

All clients are closed in `main.py`'s lifespan handler. New HTTP modules must follow this pattern.

### Configuration access

Always use `get_settings()` — never read env vars directly or hardcode values:

```python
from artemis.config import get_settings
settings = get_settings()
```

`Settings` is a frozen dataclass with validation. All parsing and bounds-checking happens in `config.py`.

### Error handling for external services

All calls to SearXNG, LLM, or Playwright follow this pattern:

```python
try:
    response = await client.get(...)
    response.raise_for_status()
except httpx.TimeoutException:
    raise UpstreamServiceError("The X timed out.")
except httpx.HTTPStatusError as exc:
    raise UpstreamServiceError(f"The X returned HTTP {exc.response.status_code}.")
except httpx.HTTPError:
    raise UpstreamServiceError("The X request failed.")
```

`UpstreamServiceError` is caught by a global exception handler in `main.py` and returns a JSON error with the appropriate status code.

### Token usage tracking

All LLM operations return `TokenUsage`. Usage is accumulated across operations via `_merge_usage(total, usage_dict)`. The final response always includes aggregated usage for LiteLLM billing. `TokenUsage` exposes both `input_tokens`/`output_tokens` (native) and `prompt_tokens`/`completion_tokens` (LiteLLM aliases) via Pydantic computed fields.

### Graceful degradation

- Summary circuit breaker: opens after 3 consecutive LLM failures, 5-minute cooldown
- Content extraction failures are logged and skipped, never fatal
- LLM relevance selection falls back to first-N results on failure
- Outline generation falls back to a default outline on failure
- Blocked domains (paywalled sites) are skipped without error

### Streaming and request cancellation

`/v1/responses` supports `streaming: true`. Implementation uses `asyncio.Queue` — `deep_research()` runs as an `asyncio.create_task()` and pushes progress via a callback; the streaming generator reads from the queue. A `[USAGE]` JSON line is emitted at the end for LiteLLM billing.

If the client disconnects mid-request, in-flight research tasks are cancelled automatically:
- Streaming: `try/finally` in the generator cancels the task on teardown
- Non-streaming: polls `request.is_disconnected()` and cancels with HTTP 499

### Testing patterns

- `unittest` with `IsolatedAsyncioTestCase` for async tests
- `AsyncMock` and `unittest.mock.patch` for mocking external services
- `TestClient` from `fastapi.testclient` for endpoint tests
- Tests mock at the module boundary (e.g., patch `artemis.main.search_searxng`)

### Playwright browser management

The Playwright browser context is shared globally and protected by `asyncio.Lock` with double-check locking to prevent concurrent launches. `extractor.py` maintains a blocklist of paywalled domains as a `frozenset`.

The context is recycled after a configurable number of pages (`PLAYWRIGHT_CONTEXT_RECYCLE_PAGES`, default 50) to prevent memory leaks — the browser process is kept alive, only the context is recreated. Heavy resource types (images, fonts, stylesheets, media) are blocked via `page.route()`, and HTML size is capped (`PLAYWRIGHT_MAX_HTML_BYTES`, default 5MB).

### Observability

Structured logging with request ID correlation is built in:
- Request ID middleware reads `x-request-id` from upstream (nginx/LiteLLM) or generates a UUID
- Request ID is stored in a `contextvars.ContextVar` and injected into every log record
- `LOG_FORMAT=json` emits single-line JSON logs for production aggregation
- `/health` probes SearXNG and LLM connectivity, returning per-check status and overall `healthy`/`degraded`

### Caching

Search results and extracted page content are cached in-memory via `AsyncTTLCache` (in `cache.py`). The cache supports request coalescing — concurrent requests for the same key share a single fetch. When `EMBEDDING_MODEL` is configured, the searcher also performs semantic query deduplication via cosine similarity on embeddings.
