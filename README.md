# Artemis

Artemis provides a Perplexity-compatible API interface over SearXNG. Use it as a drop-in replacement for Perplexity's API while running your own search infrastructure.

## Features

- **Perplexity-compatible API** - Drop-in replacement for Perplexity clients
- **SearXNG backend** - Your own search instance, no external dependencies
- **Deep research mode** - Multi-stage adaptive research with essay synthesis
- **LLM summarization** - Optional AI-powered summary of search results
- **Production controls** - Optional bearer auth, validated config, configurable CORS, and upstream timeout handling

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

Response includes the essay and all sources.

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
| `preset` | string | No | `fast-search` (default) or `deep-research` |
| `max_steps` | integer | No | Override `DEEP_RESEARCH_STAGES` per-request (1-10, deep-research only) |
| `outline` | array | No | Custom outline sections. Each: `{"section": "Name", "description": "..."}`. Overrides `DEEP_RESEARCH_STAGES`. |

Deep research config via environment:
- `DEEP_RESEARCH_STAGES` - Number of outline sections (default: 2)
- `DEEP_RESEARCH_PASSES` - Research passes, refines queries each pass (default: 1)
- `DEEP_RESEARCH_SUBQUERIES` - Queries per section per pass (default: 5)
- `DEEP_RESEARCH_MAX_TOKENS` - Max tokens for final essay (default: 8000)
- `DEEP_RESEARCH_CONTENT_EXTRACTION` - Fetch full page content via Playwright + trafilatura (default: `true`)
- `DEEP_RESEARCH_PAGES_PER_SECTION` - Pages to fetch per section (default: 3)
- `DEEP_RESEARCH_CONTENT_MAX_CHARS` - Max extracted chars per page (default: 3000)

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

## Operational Notes

- `/health` reports service health plus whether auth and summarization are enabled
- Summary failures degrade gracefully and return warnings instead of leaking raw upstream errors
- Deep research supports per-request `max_steps` (1-10) and reports aggregated token usage
- The container runs as a non-root user and includes a Docker healthcheck

## License

MIT
