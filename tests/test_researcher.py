"""Unit tests for the Artemis researcher module."""

import unittest

from artemis.errors import UpstreamServiceError
from artemis.researcher import (
    _parse_outline,
    _parse_query_list,
    filter_results_by_relevance_sync,
)
from artemis.models import SearchResult


class ParseQueryListTestCase(unittest.TestCase):
    def test_valid_json_array(self) -> None:
        raw = '["query one", "query two"]'
        result = _parse_query_list(raw)
        self.assertEqual(result, ["query one", "query two"])

    def test_json_in_markdown_code_block(self) -> None:
        raw = '```json\n["query one", "query two"]\n```'
        result = _parse_query_list(raw)
        self.assertEqual(result, ["query one", "query two"])

    def test_json_with_surrounding_text(self) -> None:
        raw = 'Here are queries: ["query one", "query two"] hope that helps'
        result = _parse_query_list(raw)
        self.assertEqual(result, ["query one", "query two"])

    def test_empty_array_raises(self) -> None:
        with self.assertRaises(UpstreamServiceError):
            _parse_query_list("[]")

    def test_non_string_items_return_fallback(self) -> None:
        result = _parse_query_list('[123, 456]')
        self.assertEqual(result, ["research topic analysis"])

    def test_no_json_raises(self) -> None:
        with self.assertRaises(UpstreamServiceError):
            _parse_query_list("no json here at all")

    def test_duplicates_removed(self) -> None:
        raw = '["query one", "query one", "query two"]'
        result = _parse_query_list(raw)
        self.assertEqual(result, ["query one", "query two"])

    def test_truncated_array_salvaged(self) -> None:
        raw = '["complete query one", "complete query two", "truncat'
        result = _parse_query_list(raw)
        self.assertEqual(result, ["complete query one", "complete query two"])

    def test_wrapped_object_queries_key(self) -> None:
        raw = '{"queries": ["query one", "query two"]}'
        result = _parse_query_list(raw)
        self.assertEqual(result, ["query one", "query two"])

    def test_wrapped_object_search_queries_key(self) -> None:
        raw = '{"search_queries": ["query one"]}'
        result = _parse_query_list(raw)
        self.assertEqual(result, ["query one"])

    def test_wrapped_object_any_key(self) -> None:
        raw = '{"whatever_key": ["query one", "query two"]}'
        result = _parse_query_list(raw)
        self.assertEqual(result, ["query one", "query two"])

    def test_wrapped_object_no_array_raises(self) -> None:
        with self.assertRaises(UpstreamServiceError):
            _parse_query_list('{"foo": "bar"}')

    def test_non_string_items_skipped(self) -> None:
        raw = '["valid query", 123, "another valid"]'
        result = _parse_query_list(raw)
        self.assertEqual(result, ["valid query", "another valid"])


class ParseOutlineTestCase(unittest.TestCase):
    def test_valid_outline(self) -> None:
        raw = '[{"section": "Intro", "description": "An introduction"}]'
        result = _parse_outline(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["section"], "Intro")

    def test_missing_section_skipped(self) -> None:
        raw = '[{"description": "No section key"}, {"section": "Valid", "description": "ok"}]'
        result = _parse_outline(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["section"], "Valid")

    def test_empty_section_skipped(self) -> None:
        raw = '[{"section": "", "description": "empty"}, {"section": "Valid", "description": "ok"}]'
        result = _parse_outline(raw)
        self.assertEqual(len(result), 1)

    def test_code_block_wrapped(self) -> None:
        raw = '```json\n[{"section": "A", "description": "B"}]\n```'
        result = _parse_outline(raw)
        self.assertEqual(len(result), 1)

    def test_unwraps_object_wrapper(self) -> None:
        raw = '{"sections": [{"section": "A", "description": "B"}]}'
        result = _parse_outline(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["section"], "A")

    def test_unwraps_any_key_wrapper(self) -> None:
        raw = '{"outline": [{"section": "X", "description": "Y"}, {"section": "Z", "description": "W"}]}'
        result = _parse_outline(raw)
        self.assertEqual(len(result), 2)

    def test_invalid_json_returns_fallback(self) -> None:
        result = _parse_outline("not json")
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["section"], "Overview")

    def test_empty_list_raises(self) -> None:
        with self.assertRaises(UpstreamServiceError):
            _parse_outline("[]")


class FilterResultsSyncTestCase(unittest.TestCase):
    def _make_result(self, title: str, snippet: str) -> SearchResult:
        return SearchResult(title=title, url="https://example.com", snippet=snippet)

    def test_filters_by_keywords(self) -> None:
        results = [
            self._make_result("Python tutorial", "Learn Python programming"),
            self._make_result("Cooking recipes", "Best pasta recipes"),
        ]
        filtered = filter_results_by_relevance_sync(
            results, keywords=["python"], min_matches=1
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].title, "Python tutorial")

    def test_empty_results(self) -> None:
        self.assertEqual(filter_results_by_relevance_sync([], ["test"]), [])

    def test_max_results_respected(self) -> None:
        results = [
            self._make_result(f"Result {i}", "keyword match") for i in range(10)
        ]
        filtered = filter_results_by_relevance_sync(
            results, keywords=["keyword"], max_results=3
        )
        self.assertEqual(len(filtered), 3)
