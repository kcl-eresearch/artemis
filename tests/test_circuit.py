"""Unit tests for the Artemis circuit breaker and helpers in main.py."""

import time
import unittest

from artemis.main import (
    SUMMARY_CIRCUIT_BACKOFF_SECONDS,
    SUMMARY_CIRCUIT_FAILURE_THRESHOLD,
    _fallback_text,
    _record_summary_failure,
    _reset_summary_circuit,
    summary_circuit,
)
from artemis.models import SearchResult


class CircuitBreakerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        summary_circuit.consecutive_failures = 0
        summary_circuit.opened_until = 0.0

    def test_circuit_opens_after_threshold(self) -> None:
        for _ in range(SUMMARY_CIRCUIT_FAILURE_THRESHOLD):
            _record_summary_failure()
        self.assertGreater(summary_circuit.opened_until, time.time())

    def test_circuit_stays_closed_below_threshold(self) -> None:
        for _ in range(SUMMARY_CIRCUIT_FAILURE_THRESHOLD - 1):
            _record_summary_failure()
        self.assertEqual(summary_circuit.opened_until, 0.0)

    def test_reset_clears_state(self) -> None:
        for _ in range(SUMMARY_CIRCUIT_FAILURE_THRESHOLD):
            _record_summary_failure()
        _reset_summary_circuit()
        self.assertEqual(summary_circuit.consecutive_failures, 0)
        self.assertEqual(summary_circuit.opened_until, 0.0)

    def test_backoff_duration(self) -> None:
        for _ in range(SUMMARY_CIRCUIT_FAILURE_THRESHOLD):
            _record_summary_failure()
        expected_min = time.time() + SUMMARY_CIRCUIT_BACKOFF_SECONDS - 1
        self.assertGreater(summary_circuit.opened_until, expected_min - 2)


class FallbackTextTestCase(unittest.TestCase):
    def test_with_results(self) -> None:
        results = [
            SearchResult(title="T1", url="https://a.com", snippet="First snippet."),
            SearchResult(title="T2", url="https://b.com", snippet="Second snippet."),
        ]
        text = _fallback_text(results)
        self.assertIn("First snippet.", text)
        self.assertIn("Second snippet.", text)

    def test_with_empty_results(self) -> None:
        self.assertEqual(_fallback_text([]), "No results found.")

    def test_limits_to_three_results(self) -> None:
        results = [
            SearchResult(title=f"T{i}", url=f"https://{i}.com", snippet=f"Snippet {i}.")
            for i in range(5)
        ]
        text = _fallback_text(results)
        self.assertIn("Snippet 0.", text)
        self.assertIn("Snippet 2.", text)
        self.assertNotIn("Snippet 3.", text)
