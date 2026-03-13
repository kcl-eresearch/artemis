#!/usr/bin/env python3
"""One-shot CLI for deep research without the HTTP server.

Usage examples::

    # Basic research (outputs JSON by default)
    python cli.py "quantum computing advances"

    # DOCX report with auto-generated filename
    python cli.py "climate change mitigation" --format docx

    # Markdown to specific file, shallow preset
    python cli.py "rust vs go" --format md --output comparison.md --preset shallow

    # Write to stdout
    python cli.py "LLM architectures" --format md --output -
"""

import asyncio
import argparse
import re
import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from artemis.writers import write_json, write_markdown, md_to_docx  # noqa: E402

_FORMATS = ("json", "md", "docx")
_PRESETS = ("deep", "shallow")


def _slugify(text: str, max_len: int = 48) -> str:
    """Convert a query string into a filename-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:max_len]


def _default_output_path(query: str, fmt: str) -> str:
    """Generate an output filename from query and date."""
    ext = {"json": "json", "md": "md", "docx": "docx"}[fmt]
    slug = _slugify(query)
    today = date.today().isoformat()
    return f"{slug}-{today}.{ext}"


def _progress_callback(quiet: bool):
    """Return a progress callback (or None if quiet)."""
    if quiet:
        return None
    icons = {
        "start": "🎯",
        "outline": "📋",
        "pass": "🔄",
        "search": "🔍",
        "synthesis": "✍️",
        "complete": "✅",
    }

    def show_progress(stage: str, message: str) -> None:
        icon = icons.get(stage, "•")
        print(f"{icon} {message}", file=sys.stderr)

    return show_progress


async def run_outline(query: str, stages: int | None, preset: str) -> None:
    import json
    from artemis.config import get_settings
    from artemis.researcher import generate_outline

    settings = get_settings()
    if preset == "shallow":
        num_sections = stages or settings.shallow_research_stages
    else:
        num_sections = stages or settings.deep_research_stages

    outline, _ = await generate_outline(query, num_sections=num_sections)
    print(json.dumps(outline, indent=2))


async def run_research(
    query: str,
    fmt: str,
    output: str | None,
    preset: str,
    stages: int | None,
    passes: int | None,
    quiet: bool,
) -> None:
    from artemis.config import get_settings
    from artemis.researcher import deep_research

    settings = get_settings()

    # Resolve preset defaults
    if preset == "shallow":
        stages = stages or settings.shallow_research_stages
        passes = passes or settings.shallow_research_passes
        sub_queries = settings.shallow_research_subqueries
        results_per_query = settings.shallow_research_results_per_query
        max_tokens = settings.shallow_research_max_tokens
        content_extraction = settings.shallow_research_content_extraction
        pages_per_section = settings.shallow_research_pages_per_section
        content_max_chars = settings.shallow_research_content_max_chars
    else:
        stages = stages or settings.deep_research_stages
        passes = passes or settings.deep_research_passes
        sub_queries = settings.deep_research_subqueries
        results_per_query = settings.deep_research_results_per_query
        max_tokens = settings.deep_research_max_tokens
        content_extraction = settings.deep_research_content_extraction
        pages_per_section = settings.deep_research_pages_per_section
        content_max_chars = settings.deep_research_content_max_chars

    stdout = output == "-"

    # Resolve output path
    if output is None:
        output = _default_output_path(query, fmt)
    elif stdout and fmt == "docx":
        print("Error: DOCX format cannot be written to stdout.", file=sys.stderr)
        sys.exit(1)

    progress = _progress_callback(quiet or stdout)

    if not quiet and not stdout:
        print(f"Research: {query}", file=sys.stderr)
        print(f"Preset: {preset} | Stages: {stages} | Passes: {passes}", file=sys.stderr)
        print("-" * 50, file=sys.stderr)

    try:
        result = await deep_research(
            query=query,
            stages=stages,
            passes=passes,
            sub_queries_per_stage=sub_queries,
            results_per_query=results_per_query,
            max_tokens=max_tokens,
            content_extraction=content_extraction,
            pages_per_section=pages_per_section,
            content_max_chars=content_max_chars,
            progress_callback=progress,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Write output
    usage_dict = result.usage.model_dump() if result.usage else None
    if fmt == "json":
        write_json(output, query, result.essay, result.results, usage_dict, stdout=stdout)
    elif fmt == "md":
        write_markdown(output, result.essay, result.results, stdout=stdout)
    elif fmt == "docx":
        md_to_docx(output, result.essay, title=query, results=result.results)

    if not stdout and not quiet:
        print(f"\n✅ Saved to {output}", file=sys.stderr)
        print(
            f"   {len(result.essay)} chars | {len(result.results)} sources",
            file=sys.stderr,
        )
        if result.usage:
            print(
                f"   Tokens: {result.usage.total_tokens} "
                f"(in={result.usage.input_tokens} out={result.usage.output_tokens})",
                file=sys.stderr,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Artemis — one-shot deep research from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  %(prog)s "quantum computing"\n'
            '  %(prog)s "climate change" --format docx\n'
            '  %(prog)s "rust vs go" --format md -o comparison.md --preset shallow\n'
            '  %(prog)s "LLM architectures" --format md -o -    # stdout\n'
        ),
    )
    parser.add_argument("query", help="Research query or question")
    parser.add_argument(
        "--format", "-f",
        choices=_FORMATS,
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=None,
        help='Output file path (default: auto-generated). Use "-" for stdout.',
    )
    parser.add_argument(
        "--preset",
        choices=_PRESETS,
        default="deep",
        help="Research preset (default: deep)",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=None,
        help="Number of outline sections (overrides preset)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=None,
        help="Number of research passes (overrides preset)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--outline-only",
        action="store_true",
        help="Generate and print the research outline as JSON, then exit",
    )
    args = parser.parse_args()

    if args.outline_only:
        asyncio.run(run_outline(query=args.query, stages=args.stages, preset=args.preset))
        return

    asyncio.run(
        run_research(
            query=args.query,
            fmt=args.format,
            output=args.output,
            preset=args.preset,
            stages=args.stages,
            passes=args.passes,
            quiet=args.quiet,
        )
    )


if __name__ == "__main__":
    main()
