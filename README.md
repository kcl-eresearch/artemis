# Artemis

Artemis provides an LLM model layer incorporating search via SearXNG, content extraction via Playwright and LLM synthesis via LiteLLM.
The end result is a researched document answering the query.

## Features

- **Perplexity-compatible API**
- **SearXNG backend** - Your own search instance, no external dependencies
- **Deep research mode** - Multi-stage adaptive research with essay synthesis
- **Content extraction** - Fetches full page content via Playwright (headless Chromium) + trafilatura, with LLM-based relevance selection to extract only the most useful pages
- **Streaming** - Real-time progress events via `streaming: true` on `/v1/responses`
- **LLM summarization** - Optional AI-powered summary of search results
- **Production controls** - Optional bearer auth, validated config, configurable CORS, circuit breaker for LLM failures, and upstream timeout handling

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Required variables:
- `SEARXNG_API_BASE` - Your SearXNG instance URL

Optional variables:
- `ARTEMIS_API_KEY` - Require `Authorization: Bearer <key>` for protected endpoints
- `ALLOWED_ORIGINS` - Comma-separated CORS allowlist for browser clients
- `SUMMARY_MODEL` - Model for summarization (default: `arc:apex`)
- `ENABLE_SUMMARY` - Enable LLM summarization (default: `true`)
- `SUMMARY_MAX_TOKENS` - Max completion tokens for summaries
- `SEARXNG_TIMEOUT_SECONDS` - Upstream search timeout in seconds
- `LLM_TIMEOUT_SECONDS` - Upstream LLM timeout in seconds
- `LOG_LEVEL` - Python log level (default: `INFO`)
- `LOG_FORMAT` - Log output format: `text` (default) or `json`
- `CACHE_ENABLED` - Enable in-memory caching (default: `true`)
- `SEARCH_CACHE_TTL_SECONDS` - TTL for cached search results (default: 3600)
- `CONTENT_CACHE_TTL_SECONDS` - TTL for cached extracted page content (default: 86400)
- `CACHE_MAX_ENTRIES` - Maximum cache entries before oldest are evicted (default: 1000)
- `EMBEDDING_MODEL` - Model for query embeddings; enables semantic dedup when set
- `SEMANTIC_SIMILARITY_THRESHOLD` - Cosine similarity threshold for semantic cache hits (default: 0.92)
- `PLAYWRIGHT_CONTEXT_RECYCLE_PAGES` - Recycle browser context after N pages (default: 50)
- `PLAYWRIGHT_MAX_HTML_BYTES` - Max HTML bytes to extract per page (default: 5MB)

Production recommendation:
- Set `ARTEMIS_API_KEY`
- Set a specific `ALLOWED_ORIGINS` allowlist
- Run behind TLS / a reverse proxy
- Use a managed secret store instead of committing `.env`

### 3. Run the server

```bash
python -m artemis.main
```

Or with uvicorn:

```bash
uvicorn artemis.main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Search (`POST /search`)

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "query": "latest AI developments 2024",
    "max_results": 5
  }'
```

Response:

```json
{
  "id": "abc-123",
  "results": [
    {
      "title": "Example Result",
      "url": "https://example.com",
      "snippet": "This is the content snippet...",
      "date": "2024-01-15"
    }
  ],
  "summary": "LLM-generated summary..."
}
```

### Deep Research (`POST /v1/responses`)

```bash
# Basic deep research (uses env config for stages/passes)
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "input": "Research topic: the future of AI in healthcare",
    "preset": "deep-research"
  }'

# With max_steps to override DEEP_RESEARCH_STAGES
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "input": "Research topic: the future of AI in healthcare",
    "preset": "deep-research",
    "max_steps": 5
  }'

# With custom outline (overrides DEEP_RESEARCH_STAGES)
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "input": "What is quantum computing?",
    "preset": "deep-research",
    "outline": [
      {"section": "Definition", "description": "What is quantum computing and how does it work?"},
      {"section": "History", "description": "Evolution and key milestones"},
      {"section": "Applications", "description": "Real-world use cases"},
      {"section": "Future", "description": "What's next for quantum computing"}
    ]
  }'
```

### Shallow Research (`POST /v1/responses`)

```bash
# Faster research pass with shallower env-configured defaults
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "input": "Research topic: the future of AI in healthcare",
    "preset": "shallow-research"
  }'
```

### Streaming (`POST /v1/responses` with `streaming: true`)

```bash
curl -N -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "input": "Research topic: the future of AI in healthcare",
    "preset": "deep-research",
    "streaming": true
  }'
```

Progress events are streamed in real-time as plain text, followed by the final essay.

For non-streaming requests, the response includes the essay and all sources as JSON.

## Request Parameters

### /search

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `max_results` | integer | No | Max results (default: 10) |
| `search_domain_filter` | array | No | Domains to filter |

### /v1/responses

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | string | Yes | Research query |
| `preset` | string | No | `fast-search` (default), `shallow-research`, or `deep-research` |
| `max_steps` | integer | No | Override the preset's configured research stages per-request (1-10, research presets only) |
| `outline` | array | No | Custom outline sections. Each: `{"section": "Name", "description": "..."}`. Overrides `DEEP_RESEARCH_STAGES`. |
| `streaming` | boolean | No | Stream progress events in real-time (default: `false`) |

Deep research config via environment:
- `DEEP_RESEARCH_STAGES` - Number of outline sections (default: 2)
- `DEEP_RESEARCH_PASSES` - Research passes, refines queries each pass (default: 1)
- `DEEP_RESEARCH_SUBQUERIES` - Queries per section per pass (default: 5)
- `DEEP_RESEARCH_RESULTS_PER_QUERY` - Search results per subquery (default: 10)
- `DEEP_RESEARCH_MAX_TOKENS` - Max tokens for final essay (default: 8000)
- `DEEP_RESEARCH_CONTENT_EXTRACTION` - Fetch full page content via Playwright + trafilatura (default: `true`)
- `DEEP_RESEARCH_PAGES_PER_SECTION` - Pages to fetch per section (default: 3)
- `DEEP_RESEARCH_CONTENT_MAX_CHARS` - Max extracted chars per page (default: 3000)

Shallow research config via environment:
- `SHALLOW_RESEARCH_STAGES` - Number of outline sections (default: 1)
- `SHALLOW_RESEARCH_PASSES` - Research passes (default: 1)
- `SHALLOW_RESEARCH_SUBQUERIES` - Queries per section per pass (default: 3)
- `SHALLOW_RESEARCH_RESULTS_PER_QUERY` - Search results per query (default: 5)
- `SHALLOW_RESEARCH_MAX_TOKENS` - Max tokens for final essay (default: 4000)
- `SHALLOW_RESEARCH_CONTENT_EXTRACTION` - Fetch full page content via Playwright + trafilatura (default: `false`)
- `SHALLOW_RESEARCH_PAGES_PER_SECTION` - Pages to fetch per section (default: 2)
- `SHALLOW_RESEARCH_CONTENT_MAX_CHARS` - Max extracted chars per page (default: 2000)

If `ARTEMIS_API_KEY` is set, both `POST /search` and `POST /v1/responses` require `Authorization: Bearer <key>`.

## SearXNG Setup

This tool requires a running SearXNG instance with JSON format enabled.

### Using a public instance

Find public instances at https://searx.space/

### Self-hosting

See [SearXNG installation](https://docs.searxng.org/admin/installation.html)

Make sure JSON format is enabled in `settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

## Testing

```bash
# Run isolated unit tests
python3 -m unittest discover -s tests -p 'test_*.py'

# Optional manual integration script (requires live SearXNG + app)
python3 -m tests.test
```

## CLI

Run deep research directly from the command line without the HTTP server:

```bash
# JSON output (default)
python cli.py "quantum computing advances"

# Markdown output
python cli.py "climate change" --format md

# DOCX report, shallow preset
python cli.py "rust vs go" --format docx --preset shallow

# Write to stdout
python cli.py "LLM architectures" --format md --output -
```

## Operational Notes

- `/health` probes SearXNG and LLM connectivity, returning per-check status and overall `healthy`/`degraded`
- Summary failures degrade gracefully via a circuit breaker pattern — after 3 consecutive LLM failures, summarization is temporarily disabled for 5 minutes
- Deep research supports per-request `max_steps` (1-10) and reports aggregated token usage
- If a client disconnects mid-request, in-flight research tasks are cancelled to avoid wasting LLM/Playwright resources
- Search results and extracted page content are cached in-memory with request coalescing; optional semantic query deduplication via embeddings
- Content extraction uses LLM-based relevance selection to pick the best results per section before running expensive Playwright page fetches
- Playwright browser context is recycled after a configurable number of pages; heavy resource types (images, fonts, stylesheets) are blocked to save memory
- Structured logging with request ID correlation (`x-request-id` header); use `LOG_FORMAT=json` for production log aggregation
- The container runs as a non-root user and includes a Docker healthcheck

## License

MIT
