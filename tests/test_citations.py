"""Tests for the deterministic citation pipeline.

The synthesis stage previously asked the LLM to invent both inline
citation numbers and a References section, which produced hallucinated
DOIs/URLs and a `[1] Various sources` catch-all. These tests cover the
replacement: pre-numbered sources, fabricated-citation stripping, and
deterministic References generation from real SearchResult records.
"""

import unittest

from artemis.models import SearchResult
from artemis.researcher import (
    _build_source_registry,
    _extract_used_citations,
    _finalize_essay_citations,
    _replace_url_citations,
    _strip_invalid_citations,
    _strip_llm_references_section,
    format_results_for_synthesis,
)


def _make(url: str, title: str = "T", snippet: str = "s", date: str | None = None) -> SearchResult:
    return SearchResult(title=title, url=url, snippet=snippet, date=date)


class BuildSourceRegistryTestCase(unittest.TestCase):
    def test_assigns_1_based_ids_in_outline_order(self) -> None:
        outline = [
            {"section": "A", "description": "a"},
            {"section": "B", "description": "b"},
        ]
        section_results = {
            "A": [_make("https://a1.com"), _make("https://a2.com")],
            "B": [_make("https://b1.com")],
        }
        url_to_id, ordered = _build_source_registry(outline, section_results)
        self.assertEqual(url_to_id["https://a1.com"], 1)
        self.assertEqual(url_to_id["https://a2.com"], 2)
        self.assertEqual(url_to_id["https://b1.com"], 3)
        self.assertEqual([s.url for s in ordered], ["https://a1.com", "https://a2.com", "https://b1.com"])

    def test_dedupes_urls_across_sections(self) -> None:
        outline = [{"section": "A", "description": ""}, {"section": "B", "description": ""}]
        shared = _make("https://shared.com", title="First")
        section_results = {
            "A": [shared],
            "B": [_make("https://shared.com", title="Second"), _make("https://b.com")],
        }
        url_to_id, ordered = _build_source_registry(outline, section_results)
        self.assertEqual(url_to_id["https://shared.com"], 1)
        self.assertEqual(url_to_id["https://b.com"], 2)
        self.assertEqual(len(ordered), 2)
        # Keeps the first occurrence's title
        self.assertEqual(ordered[0].title, "First")

    def test_skips_blank_urls(self) -> None:
        outline = [{"section": "A", "description": ""}]
        section_results = {"A": [_make(""), _make("https://ok.com")]}
        url_to_id, ordered = _build_source_registry(outline, section_results)
        self.assertEqual(url_to_id, {"https://ok.com": 1})
        self.assertEqual(len(ordered), 1)

    def test_extra_results_appended_after_outline(self) -> None:
        outline = [{"section": "A", "description": ""}]
        section_results = {"A": [_make("https://a.com")]}
        extras = [_make("https://extra.com"), _make("https://a.com")]
        url_to_id, ordered = _build_source_registry(outline, section_results, extra_results=extras)
        self.assertEqual(url_to_id["https://a.com"], 1)
        self.assertEqual(url_to_id["https://extra.com"], 2)
        self.assertEqual(len(ordered), 2)


class FormatResultsNumberingTestCase(unittest.TestCase):
    def test_prefixes_block_with_number(self) -> None:
        results = [_make("https://a.com", title="Article A")]
        text = format_results_for_synthesis(results, numbering={"https://a.com": 7})
        self.assertIn("[7] Title: Article A", text)
        self.assertIn("URL: https://a.com", text)

    def test_omits_prefix_when_url_not_in_numbering(self) -> None:
        results = [_make("https://unknown.com", title="X")]
        text = format_results_for_synthesis(results, numbering={"https://other.com": 1})
        self.assertNotIn("[1]", text)
        self.assertIn("Title: X", text)

    def test_includes_date_when_present(self) -> None:
        results = [_make("https://a.com", title="A", date="2024")]
        text = format_results_for_synthesis(results, numbering={"https://a.com": 1})
        self.assertIn("Date: 2024", text)

    def test_back_compat_without_numbering(self) -> None:
        # Old callers (reflection, refined queries) pass no numbering.
        results = [_make("https://a.com", title="A", snippet="snip")]
        text = format_results_for_synthesis(results)
        self.assertNotIn("[1]", text)
        self.assertIn("Title: A", text)


class ExtractUsedCitationsTestCase(unittest.TestCase):
    def test_finds_simple_markers(self) -> None:
        essay = "Foo [1]. Bar [2]. Baz [3]."
        self.assertEqual(_extract_used_citations(essay), {1, 2, 3})

    def test_finds_grouped_markers(self) -> None:
        essay = "Foo [1, 2, 3]. Bar [4,5]."
        self.assertEqual(_extract_used_citations(essay), {1, 2, 3, 4, 5})

    def test_ignores_non_numeric_brackets(self) -> None:
        essay = "See [Smith 2024] and [https://x.com] but cite [3]."
        self.assertEqual(_extract_used_citations(essay), {3})


class ReplaceUrlCitationsTestCase(unittest.TestCase):
    def test_replaces_known_urls(self) -> None:
        text = "A finding (Smith, 2024) [https://a.com]. Another [https://b.com]."
        url_to_id = {"https://a.com": 1, "https://b.com": 2}
        out = _replace_url_citations(text, url_to_id)
        self.assertIn("[1]", out)
        self.assertIn("[2]", out)
        self.assertNotIn("https://a.com]", out)

    def test_drops_unknown_urls(self) -> None:
        text = "Bogus [https://nowhere.com] real [https://a.com]."
        out = _replace_url_citations(text, {"https://a.com": 1})
        self.assertNotIn("https://nowhere.com", out)
        self.assertIn("[1]", out)

    def test_handles_trailing_punctuation_inside_brackets(self) -> None:
        text = "[https://a.com.]"
        url_to_id = {"https://a.com": 1}
        out = _replace_url_citations(text, url_to_id)
        self.assertEqual(out, "[1]")


class StripLLMReferencesTestCase(unittest.TestCase):
    def test_strips_markdown_heading(self) -> None:
        essay = "Body.\n\n## References\n\n[1] Foo bar.\n[2] Baz.\n"
        self.assertEqual(_strip_llm_references_section(essay), "Body.")

    def test_strips_plain_heading(self) -> None:
        essay = "Body.\n\nReferences:\n\n[1] Foo bar.\n"
        self.assertEqual(_strip_llm_references_section(essay), "Body.")

    def test_no_references_section_unchanged(self) -> None:
        essay = "Body without any references list."
        self.assertEqual(_strip_llm_references_section(essay), essay)


class StripInvalidCitationsTestCase(unittest.TestCase):
    def test_drops_fabricated_numbers(self) -> None:
        essay = "Real [1]. Fake [99]. Mixed [1, 99, 2]."
        cleaned, dropped = _strip_invalid_citations(essay, {1, 2})
        self.assertNotIn("[99]", cleaned)
        self.assertIn("[1]", cleaned)
        self.assertIn("[1, 2]", cleaned)
        self.assertEqual(dropped, {99})

    def test_no_dropped_when_all_valid(self) -> None:
        essay = "Foo [1] bar [2]."
        cleaned, dropped = _strip_invalid_citations(essay, {1, 2})
        self.assertEqual(cleaned, essay)
        self.assertEqual(dropped, set())

    def test_tidies_punctuation_after_removal(self) -> None:
        essay = "Claim [99] ."
        cleaned, _ = _strip_invalid_citations(essay, {1})
        self.assertNotIn("  ", cleaned)
        self.assertNotIn(" .", cleaned)


class FinalizeEssayCitationsTestCase(unittest.TestCase):
    def test_appends_references_only_for_used_ids(self) -> None:
        sources = [
            _make("https://a.com", title="Source A", date="2024"),
            _make("https://b.com", title="Source B"),
            _make("https://c.com", title="Source C"),
        ]
        essay = "Body cites [1] and [3]."
        final = _finalize_essay_citations(essay, sources)
        self.assertIn("## References", final)
        self.assertIn('[1] "Source A" — https://a.com (2024)', final)
        self.assertIn('[3] "Source C" — https://c.com', final)
        # [2] was not cited so it must not appear in References
        self.assertNotIn("[2]", final)
        self.assertNotIn("Source B", final)

    def test_strips_llm_written_references_section(self) -> None:
        sources = [_make("https://a.com", title="Real Source")]
        essay = (
            "Claim [1].\n\n"
            "## References\n\n"
            '[1] "Hallucinated Title" — https://fake.example/doi/123 — 2023\n'
            '[2] "Made up" — https://invented.com\n'
        )
        final = _finalize_essay_citations(essay, sources)
        self.assertNotIn("Hallucinated", final)
        self.assertNotIn("invented.com", final)
        self.assertIn("https://a.com", final)

    def test_drops_fabricated_inline_citations(self) -> None:
        sources = [_make("https://a.com", title="A")]
        essay = "Real [1]. Fake [42]. Range [1, 42]."
        final = _finalize_essay_citations(essay, sources)
        self.assertNotIn("[42]", final)
        self.assertIn("[1]", final)

    def test_no_references_when_no_citations_used(self) -> None:
        sources = [_make("https://a.com")]
        essay = "Body with no citations."
        final = _finalize_essay_citations(essay, sources)
        self.assertNotIn("## References", final)

    def test_various_sources_catchall_no_longer_possible(self) -> None:
        # The pre-fix failure mode: model wrote `[1] Various sources` in the
        # References. The deterministic builder uses the real SearchResult
        # records so this fabrication can no longer reach the user.
        sources = [
            _make("https://nature.com/x", title="A real paper", date="2024"),
        ]
        essay = (
            "Major progress on detectors [1].\n\n"
            "## References\n\n"
            "[1] Various sources — Research synthesis on detector technology.\n"
        )
        final = _finalize_essay_citations(essay, sources)
        self.assertNotIn("Various sources", final)
        self.assertIn("A real paper", final)
        self.assertIn("https://nature.com/x", final)


if __name__ == "__main__":
    unittest.main()
