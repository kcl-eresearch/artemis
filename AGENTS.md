# AGENTS.md - AI Agent Context

This document provides context for AI agents working on the Artemis project.

## Project Overview

Artemis is a FastAPI wrapper around SearXNG with a Perplexity-compatible API. Provides search and deep research capabilities using your own SearXNG instance.

**Tech Stack:**
- FastAPI (web framework)
- httpx (async HTTP client)
- Pydantic (validation)
- python-dotenv (config)

## Project Structure

```
artemis/
├── __init__.py       # Package initialization
├── main.py           # FastAPI app and endpoints (Perplexity-compatible API)
├── config.py         # Configuration
├── models.py         # Pydantic models
├── searcher.py       # search_searxng() - SearXNG API client
├── summarizer.py     # summarize_results() - LLM summarization
├── researcher.py     # deep_research() - Multi-stage research
├── extractor.py      # fetch_and_extract() - Page content extraction via trafilatura
├── llm.py            # chat_completion() - LiteLLM-compatible client
└── errors.py         # Custom exception hierarchy

tests/
├── __init__.py
└── test.py           # Test script

searxng.yml           # SearXNG configuration
docker-compose.yml    # Docker Compose setup
Dockerfile            # Container definition
requirements.txt      # Python dependencies
.env.example          # Environment template
README.md             # Documentation
```

## Key Files

### artemis/main.py
- **FastAPI app** (`app`): Main application instance
- **Endpoints** (Perplexity-compatible):
  - `GET /` - Root endpoint
  - `GET /health` - Health check
  - `POST /search` - Search endpoint
  - `POST /v1/responses` - Deep research endpoint

### artemis/searcher.py
- `search_searxng()` - Calls SearXNG API, parses results

### artemis/summarizer.py
- `summarize_results()` - Uses LLM to summarize results

### artemis/researcher.py
- `deep_research()` - Multi-stage adaptive research

### artemis/config.py
- `SEARXNG_API_BASE` - SearXNG instance URL
- `SUMMARY_MODEL` - Model for summarization
- `ENABLE_SUMMARY` - Toggle summarization on/off

## Development Commands

```bash
# Run the server
python -m artemis.main

# Or with uvicorn
uvicorn artemis.main:app --host 0.0.0.0 --port 8000

# Install dependencies
pip install -r requirements.txt

# Run with Docker
docker-compose up -d
```

## Testing

```bash
# Run test script
python -m tests.test

# Test search endpoint
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "max_results": 5}'

# Test deep research
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "research topic", "preset": "deep-research"}'

# Health check
curl http://localhost:8000/health
```

## Code Style

- Async/await for I/O operations (httpx)
- Pydantic models for validation
- Type hints throughout
- Use `httpx.AsyncClient` for async HTTP calls
