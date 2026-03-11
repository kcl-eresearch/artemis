#!/usr/bin/env python3
"""Convert Artemis research output between formats.

Supports converting:
- Artemis JSON response files → DOCX or Markdown
- Markdown files → DOCX

Usage examples::

    # Artemis JSON response → DOCX
    python convert.py research_output.txt -f docx

    # Artemis JSON response → Markdown
    python convert.py research_output.txt -f md

    # Markdown file → DOCX
    python convert.py essay.md -f docx

    # Custom output path
    python convert.py research_output.txt -f docx -o report.docx

    # Custom title for DOCX
    python convert.py essay.md -f docx --title "My Research Report"
"""

import argparse
import json
import re
import sys
from pathlib import Path

from artemis.writers import write_json, write_markdown, md_to_docx


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks and orphaned tags."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_from_artemis_json(data: dict) -> tuple[str, str | None, list, dict | None]:
    """Extract essay, title, results, and usage from an Artemis API response.

    Returns:
        Tuple of (essay, title, results, usage)
    """
    essay = None
    results = []

    for item in data.get("output", []):
        if item.get("type") == "message" and essay is None:
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    essay = block["text"]
                    break
        elif item.get("type") == "search_results":
            results = item.get("results", [])

    if essay is None:
        print("Error: Could not find essay text in the response.", file=sys.stderr)
        sys.exit(1)

    # Try to extract query as title from the input field
    title = data.get("input")

    usage = data.get("usage")

    return essay, title, results, usage


def _load_input(path: str) -> tuple[str, str | None, list, dict | None]:
    """Load input file and extract essay content.

    Returns:
        Tuple of (essay, title, results, usage)
    """
    input_path = Path(path)
    if not input_path.exists():
        print(f"Error: {path} not found.", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text()

    # Try parsing as Artemis JSON response
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Artemis API response format (has "output" key)
            if "output" in data:
                essay, title, results, usage = _extract_from_artemis_json(data)
                return _strip_think_tags(essay), title, results, usage
            # CLI JSON format (has "essay" key)
            if "essay" in data:
                return (
                    _strip_think_tags(data["essay"]),
                    data.get("query"),
                    data.get("results", []),
                    data.get("usage"),
                )
    except (json.JSONDecodeError, ValueError):
        pass

    # Treat as raw Markdown
    title = input_path.stem.replace("-", " ").replace("_", " ").title()
    return _strip_think_tags(text), title, [], None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Artemis research output between formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  %(prog)s research_output.txt -f docx\n'
            '  %(prog)s research_output.txt -f md\n'
            '  %(prog)s essay.md -f docx --title "My Report"\n'
            '  %(prog)s essay.md -f md -o -    # stdout\n'
        ),
    )
    parser.add_argument("input", help="Input file (Artemis JSON response, CLI JSON, or Markdown)")
    parser.add_argument(
        "--format", "-f",
        choices=("md", "docx", "json"),
        required=True,
        help="Output format",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=None,
        help='Output file path (default: input stem + new extension). Use "-" for stdout (md/json only).',
    )
    parser.add_argument(
        "--title", "-t",
        default=None,
        help="Document title (default: extracted from input or filename)",
    )
    args = parser.parse_args()

    essay, auto_title, results, usage = _load_input(args.input)
    title = args.title or auto_title

    stdout = args.output == "-"

    # Resolve output path
    if args.output is None:
        ext = {"md": "md", "docx": "docx", "json": "json"}[args.format]
        output = str(Path(args.input).with_suffix(f".{ext}"))
    elif stdout and args.format == "docx":
        print("Error: DOCX format cannot be written to stdout.", file=sys.stderr)
        sys.exit(1)
    else:
        output = args.output

    if args.format == "json":
        write_json(output, title or "", essay, results, usage, stdout=stdout)
    elif args.format == "md":
        write_markdown(output, essay, results, stdout=stdout)
    elif args.format == "docx":
        md_to_docx(output, essay, title=title, results=results)

    if not stdout:
        print(f"Saved to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
