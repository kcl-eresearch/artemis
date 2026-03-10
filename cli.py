#!/usr/bin/env python3
"""One-shot CLI for deep research without HTTP server."""

import asyncio
import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


async def run_research(query: str, stages: int = 3, passes: int = 2):
    from artemis.config import get_settings
    from artemis.researcher import deep_research
    from artemis.models import DeepResearchRun
    import json
    
    settings = get_settings()
    
    # Progress callback for real-time updates
    def show_progress(stage: str, message: str):
        icons = {
            "start": "🎯",
            "outline": "📋",
            "pass": "🔄",
            "search": "🔍",
            "synthesis": "✍️",
            "complete": "✅",
        }
        icon = icons.get(stage, "•")
        print(f"{icon} {message}")
    
    # Progress callback
    def show_progress(stage: str, message: str):
        icons = {"start": "🎯", "outline": "📋", "pass": "🔄", "search": "🔍", "synthesis": "✍️", "complete": "✅"}
        print(f"{icons.get(stage, '•')} {message}")
    
    print(f"Running deep research on: {query}")
    print(f"Stages: {stages}, Passes: {passes}")
    print("-" * 50)
    
    try:
        result: DeepResearchRun = await deep_research(
            query=query,
            stages=stages,
            passes=passes,
            progress_callback=show_progress,
        )
        
        print(f"\n=== RESULTS ===")
        print(f"Essay length: {len(result.essay)} chars")
        print(f"Sources: {len(result.results)}")
        print(f"Token usage: {result.usage}")
        print(f"\n=== ESSAY ===\n")
        print(result.essay[:5000])
        
        output = {
            "query": query,
            "essay": result.essay,
            "results": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in result.results],
            "usage": result.usage.model_dump() if result.usage else None,
        }
        
        with open("research_output.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n\nFull output saved to: research_output.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="One-shot deep research")
    parser.add_argument("query", help="Research query")
    parser.add_argument("--stages", type=int, default=3, help="Number of outline sections")
    parser.add_argument("--passes", type=int, default=2, help="Number of research passes")
    args = parser.parse_args()
    
    asyncio.run(run_research(args.query, args.stages, args.passes))


if __name__ == "__main__":
    main()
