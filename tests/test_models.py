"""Tests for Pydantic models — validation, computed fields, defaults."""

import unittest

from pydantic import ValidationError

from artemis.models import (
    SearchRequest,
    SearchResult,
    TokenUsage,
    ResponsesRequest,
    ResponsesAPIResponse,
    AssistantMessage,
    OutputText,
    SearchResultItem,
    SearchResultsBlock,
)


class TokenUsageTestCase(unittest.TestCase):
    def test_defaults_to_zero(self) -> None:
        u = TokenUsage()
        self.assertEqual(u.input_tokens, 0)
        self.assertEqual(u.output_tokens, 0)
        self.assertEqual(u.total_tokens, 0)
        self.assertEqual(u.search_requests, 0)
        self.assertEqual(u.citation_tokens, 0)

    def test_prompt_tokens_alias(self) -> None:
        u = TokenUsage(input_tokens=42)
        self.assertEqual(u.prompt_tokens, 42)

    def test_completion_tokens_alias(self) -> None:
        u = TokenUsage(output_tokens=17)
        self.assertEqual(u.completion_tokens, 17)

    def test_serialization_includes_computed_fields(self) -> None:
        u = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        data = u.model_dump()
        self.assertEqual(data["prompt_tokens"], 10)
        self.assertEqual(data["completion_tokens"], 5)

    def test_json_serialization(self) -> None:
        u = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, search_requests=3)
        import json
        data = json.loads(u.model_dump_json())
        self.assertEqual(data["prompt_tokens"], 10)
        self.assertEqual(data["search_requests"], 3)


class SearchRequestValidationTestCase(unittest.TestCase):
    def test_valid_request(self) -> None:
        req = SearchRequest(query="test query")
        self.assertEqual(req.max_results, 10)

    def test_empty_query_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SearchRequest(query="")

    def test_query_too_long_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SearchRequest(query="x" * 501)

    def test_max_results_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            SearchRequest(query="test", max_results=0)
        with self.assertRaises(ValidationError):
            SearchRequest(query="test", max_results=51)

    def test_domain_filter_accepted(self) -> None:
        req = SearchRequest(query="test", search_domain_filter=["example.com"])
        self.assertEqual(req.search_domain_filter, ["example.com"])


class ResponsesRequestValidationTestCase(unittest.TestCase):
    def test_defaults(self) -> None:
        req = ResponsesRequest(input="test")
        self.assertEqual(req.preset, "fast-search")
        self.assertFalse(req.streaming)
        self.assertIsNone(req.outline)
        self.assertIsNone(req.max_steps)

    def test_empty_input_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ResponsesRequest(input="")

    def test_input_too_long_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ResponsesRequest(input="x" * 4001)

    def test_invalid_preset_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ResponsesRequest(input="test", preset="invalid")

    def test_shallow_research_preset_accepted(self) -> None:
        req = ResponsesRequest(input="test", preset="shallow-research")
        self.assertEqual(req.preset, "shallow-research")

    def test_max_steps_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            ResponsesRequest(input="test", max_steps=0)
        with self.assertRaises(ValidationError):
            ResponsesRequest(input="test", max_steps=11)


class ResponsesAPIResponseTestCase(unittest.TestCase):
    def test_defaults(self) -> None:
        resp = ResponsesAPIResponse(
            id="test-id",
            created=1000,
            model="artemis-search",
            output=[],
        )
        self.assertEqual(resp.object, "response")
        self.assertEqual(resp.status, "completed")
        self.assertEqual(resp.usage.total_tokens, 0)
        self.assertEqual(resp.warnings, [])

    def test_full_response_serialization(self) -> None:
        resp = ResponsesAPIResponse(
            id="id-1",
            created=1234567890,
            model="artemis-search",
            output=[
                AssistantMessage(
                    id="msg-1",
                    content=[OutputText(text="Hello")],
                ),
                SearchResultsBlock(
                    results=[SearchResultItem(id=0, url="https://a.com", title="A", snippet="S")],
                ),
            ],
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        data = resp.model_dump()
        self.assertEqual(data["output"][0]["type"], "message")
        self.assertEqual(data["output"][1]["type"], "search_results")
        self.assertEqual(data["usage"]["prompt_tokens"], 10)
