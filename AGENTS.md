# AGENTS.md - AI Agent Context

This document provides context for AI agents working on the Artemis project.

## Project Overview

Artemis is a FastAPI wrapper around SearXNG with a Perplexity-compatible API. Provides search and deep research capabilities using your own SearXNG instance. It sits behind LiteLLM as a custom LLM provider.

**Tech Stack:**
- FastAPI (web framework)
- httpx (async HTTP client)
- Pydantic (validation)
- Playwright + trafilatura (content extraction)
- python-dotenv (config)

## Project Structure

```
artemis/
├── __init__.py       # Package initialization
├── main.py           # FastAPI app, endpoints, auth, middleware, streaming
├── config.py         # Frozen Settings dataclass from env vars
├── models.py         # Pydantic request/response models
├── searcher.py       # SearXNG HTTP client with domain filtering and caching
├── summarizer.py     # Single-query LLM summarization
├── researcher.py     # Multi-stage deep research orchestration (most complex module)
├── extractor.py      # Playwright + trafilatura content extraction
├── llm.py            # LiteLLM-compatible chat completion and embedding client
├── cache.py          # Generic async TTL cache with request coalescing
└── errors.py         # ArtemisError → UpstreamServiceError hierarchy

cli.py                # One-shot CLI for deep research (JSON/MD/DOCX output)

tests/
├── __init__.py
├── test.py                    # Manual integration test (requires live services)
├── test_api.py                # Core API endpoint tests
├── test_api_extended.py       # Health, auth, streaming, helper tests
├── test_cache.py              # TTL cache, coalescing, semantic dedup tests
├── test_circuit.py            # Circuit breaker tests
├── test_config.py             # Config parsing tests
├── test_config_extended.py    # Config edge case tests
├── test_extractor.py          # Content extraction tests
├── test_llm.py                # LLM client tests
├── test_llm_extended.py       # LLM edge case tests
├── test_models.py             # Pydantic model tests
├── test_researcher.py         # Deep research tests
├── test_researcher_extended.py # Research edge case tests
├── test_searcher.py           # SearXNG client tests
├── test_searcher_extended.py  # Searcher edge case tests
└── test_summarizer.py         # Summarizer tests
```

## Development Commands

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

## Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check with dependency probes (SearXNG + LLM)
- `POST /search` - Search with optional LLM summarization
- `POST /v1/responses` - Perplexity-compatible: fast-search, shallow-research, or deep-research presets

## Key Architecture Patterns

- **Config**: Always use `get_settings()` from `config.py` — never read env vars directly
- **HTTP clients**: Module-level lazy singletons, all closed in `main.py`'s lifespan handler
- **Error handling**: `UpstreamServiceError` for all external failures, caught by global handler
- **Caching**: `AsyncTTLCache` with request coalescing; optional semantic dedup via embeddings
- **Streaming**: `asyncio.Queue` with progress callback; tasks cancelled on client disconnect
- **Playwright**: Context recycled every N pages; heavy resources blocked; HTML size capped
- **Observability**: Request ID middleware (`x-request-id`), `contextvars` propagation, JSON log format option
- **Content isolation**: Untrusted web content delivered to LLM via tool-call messages (not user messages) to mitigate prompt injection
