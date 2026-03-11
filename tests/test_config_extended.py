"""Extended configuration parsing tests."""

import os
import unittest
from unittest.mock import patch

from artemis.config import (
    ConfigError,
    _parse_bool,
    _parse_float,
    _parse_int,
    _parse_log_level,
    _parse_optional_str,
    _validate_url,
    _parse_allowed_origins,
    get_settings,
    refresh_settings,
)


class ParseBoolTestCase(unittest.TestCase):
    def test_true_values(self) -> None:
        for val in ("1", "true", "True", "TRUE", "yes", "YES", "on", "ON"):
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                self.assertTrue(_parse_bool("TEST_BOOL", False))

    def test_false_values(self) -> None:
        for val in ("0", "false", "False", "FALSE", "no", "NO", "off", "OFF"):
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                self.assertFalse(_parse_bool("TEST_BOOL", True))

    def test_default_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_BOOL_UNSET", None)
            self.assertTrue(_parse_bool("TEST_BOOL_UNSET", True))
            self.assertFalse(_parse_bool("TEST_BOOL_UNSET", False))

    def test_invalid_value_raises(self) -> None:
        with patch.dict(os.environ, {"TEST_BOOL": "maybe"}):
            with self.assertRaises(ConfigError):
                _parse_bool("TEST_BOOL", False)

    def test_whitespace_stripped(self) -> None:
        with patch.dict(os.environ, {"TEST_BOOL": "  true  "}):
            self.assertTrue(_parse_bool("TEST_BOOL", False))


class ParseIntTestCase(unittest.TestCase):
    def test_valid_integer(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            self.assertEqual(_parse_int("TEST_INT", 10), 42)

    def test_default_when_unset(self) -> None:
        os.environ.pop("TEST_INT_UNSET", None)
        self.assertEqual(_parse_int("TEST_INT_UNSET", 99), 99)

    def test_non_integer_raises(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "abc"}):
            with self.assertRaises(ConfigError):
                _parse_int("TEST_INT", 10)

    def test_below_minimum_raises(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "0"}):
            with self.assertRaises(ConfigError):
                _parse_int("TEST_INT", 10, minimum=1)

    def test_above_maximum_raises(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "100"}):
            with self.assertRaises(ConfigError):
                _parse_int("TEST_INT", 10, maximum=50)

    def test_at_bounds_accepted(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "5"}):
            self.assertEqual(_parse_int("TEST_INT", 10, minimum=5, maximum=5), 5)


class ParseFloatTestCase(unittest.TestCase):
    def test_valid_float(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            self.assertAlmostEqual(_parse_float("TEST_FLOAT", 1.0), 3.14)

    def test_default_when_unset(self) -> None:
        os.environ.pop("TEST_FLOAT_UNSET", None)
        self.assertAlmostEqual(_parse_float("TEST_FLOAT_UNSET", 2.5), 2.5)

    def test_non_numeric_raises(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": "abc"}):
            with self.assertRaises(ConfigError):
                _parse_float("TEST_FLOAT", 1.0)

    def test_below_minimum_raises(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": "0.01"}):
            with self.assertRaises(ConfigError):
                _parse_float("TEST_FLOAT", 1.0, minimum=0.1)

    def test_above_maximum_raises(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": "500.0"}):
            with self.assertRaises(ConfigError):
                _parse_float("TEST_FLOAT", 1.0, maximum=300.0)


class ValidateUrlTestCase(unittest.TestCase):
    def test_valid_http(self) -> None:
        self.assertEqual(_validate_url("X", "http://localhost:8888"), "http://localhost:8888")

    def test_valid_https(self) -> None:
        self.assertEqual(_validate_url("X", "https://api.example.com/v1"), "https://api.example.com/v1")

    def test_trailing_slash_stripped(self) -> None:
        self.assertEqual(_validate_url("X", "http://localhost:8888/"), "http://localhost:8888")

    def test_invalid_scheme_raises(self) -> None:
        with self.assertRaises(ConfigError):
            _validate_url("X", "ftp://example.com")

    def test_no_scheme_raises(self) -> None:
        with self.assertRaises(ConfigError):
            _validate_url("X", "just-a-hostname")

    def test_empty_netloc_raises(self) -> None:
        with self.assertRaises(ConfigError):
            _validate_url("X", "http://")


class ParseLogLevelTestCase(unittest.TestCase):
    def test_valid_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            with patch.dict(os.environ, {"LOG_LVL": level}):
                self.assertEqual(_parse_log_level("LOG_LVL"), level)

    def test_case_insensitive(self) -> None:
        with patch.dict(os.environ, {"LOG_LVL": "debug"}):
            self.assertEqual(_parse_log_level("LOG_LVL"), "DEBUG")

    def test_invalid_level_raises(self) -> None:
        with patch.dict(os.environ, {"LOG_LVL": "TRACE"}):
            with self.assertRaises(ConfigError):
                _parse_log_level("LOG_LVL")

    def test_default_used(self) -> None:
        os.environ.pop("LOG_LVL_UNSET", None)
        self.assertEqual(_parse_log_level("LOG_LVL_UNSET", "WARNING"), "WARNING")


class ParseOptionalStrTestCase(unittest.TestCase):
    def test_returns_value(self) -> None:
        with patch.dict(os.environ, {"OPT": "hello"}):
            self.assertEqual(_parse_optional_str("OPT"), "hello")

    def test_empty_returns_none(self) -> None:
        with patch.dict(os.environ, {"OPT": ""}):
            self.assertIsNone(_parse_optional_str("OPT"))

    def test_whitespace_only_returns_none(self) -> None:
        with patch.dict(os.environ, {"OPT": "   "}):
            self.assertIsNone(_parse_optional_str("OPT"))

    def test_unset_returns_none(self) -> None:
        os.environ.pop("OPT_MISSING", None)
        self.assertIsNone(_parse_optional_str("OPT_MISSING"))


class ParseAllowedOriginsTestCase(unittest.TestCase):
    def test_wildcard(self) -> None:
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "*"}):
            self.assertEqual(_parse_allowed_origins(), ("*",))

    def test_empty_string(self) -> None:
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": ""}):
            self.assertEqual(_parse_allowed_origins(), ())

    def test_deduplication(self) -> None:
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "https://a.com,https://a.com"}):
            self.assertEqual(_parse_allowed_origins(), ("https://a.com",))


class GetSettingsDefaultsTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        refresh_settings()

    def test_defaults_load_without_env(self) -> None:
        """Settings load with sensible defaults when minimal env is set."""
        with patch.dict(os.environ, {
            "SEARXNG_API_BASE": "http://localhost:8888",
            "LITELLM_BASE_URL": "http://localhost:11434/api",
        }, clear=True):
            refresh_settings()
            s = get_settings()
            self.assertEqual(s.searxng_api_base, "http://localhost:8888")
            self.assertEqual(s.deep_research_stages, 2)
            self.assertEqual(s.deep_research_passes, 1)
            self.assertEqual(s.shallow_research_stages, 1)
            self.assertEqual(s.shallow_research_passes, 1)
            self.assertEqual(s.shallow_research_subqueries, 3)
            self.assertEqual(s.shallow_research_results_per_query, 5)
            self.assertEqual(s.shallow_research_max_tokens, 4000)
            self.assertFalse(s.shallow_research_content_extraction)
            self.assertEqual(s.shallow_research_pages_per_section, 2)
            self.assertEqual(s.shallow_research_content_max_chars, 2000)
            self.assertTrue(s.enable_summary)
            self.assertIsNone(s.artemis_api_key)

    def test_shallow_research_settings_can_be_overridden(self) -> None:
        with patch.dict(os.environ, {
            "SEARXNG_API_BASE": "http://localhost:8888",
            "LITELLM_BASE_URL": "http://localhost:11434/api",
            "SHALLOW_RESEARCH_STAGES": "3",
            "SHALLOW_RESEARCH_PASSES": "2",
            "SHALLOW_RESEARCH_SUBQUERIES": "4",
            "SHALLOW_RESEARCH_RESULTS_PER_QUERY": "6",
            "SHALLOW_RESEARCH_MAX_TOKENS": "5000",
            "SHALLOW_RESEARCH_CONTENT_EXTRACTION": "true",
            "SHALLOW_RESEARCH_PAGES_PER_SECTION": "4",
            "SHALLOW_RESEARCH_CONTENT_MAX_CHARS": "2500",
        }, clear=True):
            refresh_settings()
            s = get_settings()
            self.assertEqual(s.shallow_research_stages, 3)
            self.assertEqual(s.shallow_research_passes, 2)
            self.assertEqual(s.shallow_research_subqueries, 4)
            self.assertEqual(s.shallow_research_results_per_query, 6)
            self.assertEqual(s.shallow_research_max_tokens, 5000)
            self.assertTrue(s.shallow_research_content_extraction)
            self.assertEqual(s.shallow_research_pages_per_section, 4)
            self.assertEqual(s.shallow_research_content_max_chars, 2500)

    def test_shallow_research_integer_bounds_are_validated(self) -> None:
        invalid_cases = [
            ("SHALLOW_RESEARCH_STAGES", "0"),
            ("SHALLOW_RESEARCH_STAGES", "11"),
            ("SHALLOW_RESEARCH_PASSES", "0"),
            ("SHALLOW_RESEARCH_PASSES", "6"),
            ("SHALLOW_RESEARCH_SUBQUERIES", "0"),
            ("SHALLOW_RESEARCH_SUBQUERIES", "11"),
            ("SHALLOW_RESEARCH_RESULTS_PER_QUERY", "0"),
            ("SHALLOW_RESEARCH_RESULTS_PER_QUERY", "26"),
            ("SHALLOW_RESEARCH_MAX_TOKENS", "255"),
            ("SHALLOW_RESEARCH_PAGES_PER_SECTION", "0"),
            ("SHALLOW_RESEARCH_PAGES_PER_SECTION", "11"),
            ("SHALLOW_RESEARCH_CONTENT_MAX_CHARS", "499"),
            ("SHALLOW_RESEARCH_CONTENT_MAX_CHARS", "10001"),
        ]

        for name, value in invalid_cases:
            with self.subTest(name=name, value=value):
                with patch.dict(
                    os.environ,
                    {
                        "SEARXNG_API_BASE": "http://localhost:8888",
                        "LITELLM_BASE_URL": "http://localhost:11434/api",
                        name: value,
                    },
                    clear=True,
                ):
                    with self.assertRaises(ConfigError):
                        refresh_settings()
