"""Configuration unit tests for Artemis."""

import os
import unittest
from unittest.mock import patch

from artemis import config


class ConfigTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        config.refresh_settings()

    def test_invalid_allowed_origins_raises(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SEARXNG_API_BASE": "http://localhost:8888",
                "LITELLM_BASE_URL": "http://localhost:11434/api",
                "ALLOWED_ORIGINS": "not-a-url",
            },
            clear=True,
        ):
            config.get_settings.cache_clear()
            with self.assertRaises(config.ConfigError):
                config.get_settings()

    def test_settings_parse_security_options(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SEARXNG_API_BASE": "http://localhost:8888",
                "LITELLM_BASE_URL": "http://localhost:11434/api",
                "ALLOWED_ORIGINS": "https://app.example.com, https://api.example.com/",
                "ARTEMIS_API_KEY": "secret-token",
                "ENABLE_SUMMARY": "false",
            },
            clear=True,
        ):
            config.get_settings.cache_clear()
            settings = config.get_settings()

        self.assertEqual(
            settings.allowed_origins,
            ("https://app.example.com", "https://api.example.com"),
        )
        self.assertEqual(settings.artemis_api_key, "secret-token")
        self.assertFalse(settings.enable_summary)
