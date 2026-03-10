"""Unit tests for the Artemis LLM client module."""

import unittest

from artemis.llm import _normalize_usage


class NormalizeUsageTestCase(unittest.TestCase):
    def test_openai_format(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        result = _normalize_usage(usage)
        self.assertEqual(result, {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30})

    def test_anthropic_format(self) -> None:
        usage = {"input_tokens": 15, "output_tokens": 25, "total_tokens": 40}
        result = _normalize_usage(usage)
        self.assertEqual(result, {"input_tokens": 15, "output_tokens": 25, "total_tokens": 40})

    def test_total_tokens_computed_when_missing(self) -> None:
        usage = {"input_tokens": 10, "output_tokens": 20}
        result = _normalize_usage(usage)
        assert result is not None
        self.assertEqual(result["total_tokens"], 30)

    def test_none_input_returns_none(self) -> None:
        self.assertIsNone(_normalize_usage(None))

    def test_non_dict_returns_none(self) -> None:
        self.assertIsNone(_normalize_usage("not a dict"))

    def test_empty_dict_returns_zeros(self) -> None:
        result = _normalize_usage({})
        self.assertEqual(result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})


class TestConnectionPooling(unittest.TestCase):
    def test_get_client_returns_client(self) -> None:
        """Verify _get_client creates a client lazily."""
        from artemis.llm import _get_client
        client = _get_client()
        self.assertFalse(client.is_closed)
