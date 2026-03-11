"""Deep research orchestration for Artemis.

This module provides multi-stage research capabilities. The deep_research()
function performs adaptive research by:

1. Generating a research outline (sections to cover)
2. Multiple passes: each pass generates queries and searches
3. Later passes refine queries based on findings from earlier passes
4. Synthesizes a comprehensive essay following the outline structure
"""

import asyncio
import json
import logging
import re
from typing import Callable

from artemis.config import (
    get_settings,
)
from artemis.errors import UpstreamServiceError
from artemis.extractor import enrich_results
from artemis.llm import build_tool_messages, chat_completion, sanitize_content
from artemis.models import DeepResearchRun, SearchResult, TokenUsage
from artemis.searcher import search_searxng

logger = logging.getLogger(__name__)

# Maximum length for outline section titles and descriptions
_MAX_OUTLINE_SECTION_LEN = 200
_MAX_OUTLINE_DESCRIPTION_LEN = 1000


def _merge_usage(total: TokenUsage, usage: dict[str, int] | None) -> None:
    """Accumulate token usage from a single operation into a total."""
    if usage is None:
        return
    total.input_tokens += usage.get("input_tokens", 0)
    total.output_tokens += usage.get("output_tokens", 0)
    total.total_tokens += usage.get("total_tokens", 0)


def _parse_query_list(raw_content: str) -> list[str]:
    """Parse LLM response into a list of search queries.

    Handles common LLM output quirks: markdown fencing, truncated arrays,
    and wrapped objects like {"queries": [...]}.
    """
    candidate = raw_content.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start < 0:
            raise UpstreamServiceError("The LLM returned an invalid subquery payload.")
        if end <= start:
            # Truncated array — try to salvage by closing it
            parsed = _try_salvage_truncated_array(candidate[start:])
        else:
            try:
                parsed = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError as exc:
                raise UpstreamServiceError(
                    "The LLM returned an invalid subquery payload."
                ) from exc
    else:
        # Some models wrap the array: {"queries": [...]}
        if isinstance(parsed, dict):
            parsed = _unwrap_list_from_object(parsed)

    if not isinstance(parsed, list) or not parsed:
        raise UpstreamServiceError("The LLM returned an empty subquery payload.")

    queries: list[str] = []
    for item in parsed:
        if isinstance(item, str) and item.strip():
            normalized = item.strip()
            if normalized not in queries:
                queries.append(normalized)
    if not queries:
        logger.warning("Query list empty after parsing: %s", raw_content[:200])
        # Return fallback queries
        return ["research topic analysis"]
    return queries


def _unwrap_list_from_object(parsed: dict) -> list:
    """Extract the first list value from a JSON object wrapper.

    When response_format=json_object is used, LLMs must return a root
    object, so they wrap arrays like {"sections": [...]} or {"queries": [...]}.
    This finds and returns the inner list regardless of the key name.
    """
    for value in parsed.values():
        if isinstance(value, list):
            return value
    raise UpstreamServiceError(
        "The LLM returned a JSON object with no array field."
    )


def _try_salvage_truncated_array(fragment: str) -> list[str]:
    """Attempt to extract valid strings from a truncated JSON array.

    When the LLM hits the token limit mid-array, we get something like:
      ["query one", "query two", "query thr
    This extracts all complete quoted strings.
    """
    strings = re.findall(r'"([^"]+)"', fragment)
    if not strings:
        raise UpstreamServiceError(
            "The LLM returned a truncated payload with no recoverable queries."
        )
    return strings


def _parse_outline(raw_content: str) -> list[dict[str, str]]:
    """Parse LLM response into a research outline."""
    logger.debug("LLM outline response: %s", raw_content[:500])
    candidate = raw_content.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start < 0 or end <= start:
            logger.warning(f"Outline parsing failed: {raw_content[:200]}"); return [{"section": "Overview", "description": "General overview"}, {"section": "Analysis", "description": "Detailed analysis"}, {"section": "Conclusion", "description": "Summary"}]
        try:
            parsed = json.loads(candidate[start:end+1])
        except json.JSONDecodeError as exc:
            logger.warning("Outline parsing failed, using fallback"); return [{"section": "Overview", "description": "Overview"}, {"section": "Analysis", "description": "Analysis"}, {"section": "Conclusion", "description": "Conclusion"}]

    # Unwrap object wrappers like {"sections": [...]}
    if isinstance(parsed, dict):
        parsed = _unwrap_list_from_object(parsed)

    if not isinstance(parsed, list) or not parsed:
        raise UpstreamServiceError("The LLM returned an empty outline.")

    outline: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        section = item.get("section", "").strip()
        description = item.get("description", "").strip()
        if section:
            outline.append({"section": section, "description": description})
    return outline


async def generate_outline(topic: str, num_sections: int = 5) -> tuple[list[dict[str, str]], TokenUsage]:
    """Generate a research outline with sections to cover."""
    settings = get_settings()
    system = "You are a research planning specialist. Create an outline for a comprehensive research report on the topic provided by the user."
    user = f"""Topic: {topic}

Generate exactly {num_sections} main sections that need to be covered for a thorough research report.
Each section should cover a distinct aspect of the topic.

Respond with ONLY a JSON array of objects with 'section' and 'description' keys, nothing else.

Example:
[
  {{"section": "Introduction and Definition", "description": "Define the topic and explain its importance"}},
  {{"section": "Historical Context", "description": "Discuss the history and evolution"}}
]"""

    completion = await chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=settings.summary_model,
        max_tokens=4000,
    )
    outline = _parse_outline(completion["content"])
    usage = TokenUsage.model_validate(completion["usage"] or {})
    return outline, usage


async def generate_subqueries_for_section(
    topic: str,
    section: str,
    description: str,
    num_queries: int,
    existing_queries: list[str] | None = None,
    results_summary: str | None = None,
) -> tuple[list[str], TokenUsage]:
    """Generate focused search queries for a specific section."""
    settings = get_settings()
    existing_str = f"\nAlready explored: {', '.join(existing_queries)}" if existing_queries else ""
    results_context = f"\nWhat we've found so far:\n{results_summary}" if results_summary else ""

    system = "You are a research query decomposition specialist. Generate search queries to research a specific section of a report."
    user = f"""Overall Topic: {topic}
Section: {section}
Section Description: {description}
{existing_str}{results_context}

Generate {num_queries} search queries that will help gather information for this specific section.
Each query should be specific and search-friendly.
Respond with ONLY a JSON array of strings, nothing else.

Example: ["specific aspect query", "another aspect query"]"""

    completion = await chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=settings.summary_model,
        max_tokens=2000,
    )
    queries = _parse_query_list(completion["content"])
    usage = TokenUsage.model_validate(completion["usage"] or {})
    return queries, usage


async def generate_refined_queries(
    topic: str,
    section: str,
    current_results: list[SearchResult],
    num_queries: int,
    existing_queries: list[str],
) -> tuple[list[str], TokenUsage]:
    """Generate refined queries based on current findings."""
    settings = get_settings()
    
    # Summarize current findings (untrusted web content)
    findings = format_results_for_synthesis(current_results[:10])
    
    system = "You are a research query refinement specialist. Based on initial findings returned by the web_search tool, generate better follow-up queries."
    user = f"""Topic: {topic}
Section: {section}

Already Explored Queries: {', '.join(existing_queries)}

Generate {num_queries} NEW queries that explore aspects NOT yet covered by existing queries.
Focus on gaps, unanswered questions, or deeper exploration of promising leads.
Respond with ONLY a JSON array of strings.

Example: ["deeper aspect query", "contradiction check query"]"""

    messages = build_tool_messages(
        system=system,
        user=user,
        tool_content=findings,
    )

    completion = await chat_completion(
        messages=messages,
        model=settings.summary_model,
        max_tokens=2000,
    )
    queries = _parse_query_list(completion["content"])
    usage = TokenUsage.model_validate(completion["usage"] or {})
    return queries, usage


async def search_and_collect(query: str, max_results: int) -> list[SearchResult]:
    """Execute a single search query and return results."""
    return await search_searxng(query=query, max_results=max_results)


def format_results_for_synthesis(
    results: list[SearchResult],
    content_map: dict[str, str] | None = None,
) -> str:
    """Format search results into a text block for LLM consumption.

    When content_map is provided, uses extracted page content instead of
    the short search snippet, giving the LLM much richer source material.

    All untrusted fields are sanitised to strip control characters.
    """
    parts = []
    for r in results:
        body = (content_map or {}).get(r.url, r.snippet)
        title = sanitize_content(r.title, max_length=200)
        body = sanitize_content(body)
        parts.append(f"Title: {title}\nURL: {r.url}\nContent: {body}\n")
    return "\n---\n".join(parts)


async def synthesize_essay_with_outline(
    topic: str,
    outline: list[dict[str, str]],
    section_results: dict[str, list[SearchResult]],
    max_tokens: int,
    content_map: dict[str, str] | None = None,
) -> tuple[str, TokenUsage]:
    """Synthesize research into essay following the outline structure."""
    settings = get_settings()
    
    sections_text = []
    for item in outline:
        section = item["section"]
        description = item["description"]
        results = section_results.get(section, [])
        results_text = format_results_for_synthesis(results, content_map=content_map)
        sections_text.append(
            f"## Section: {section}\nDescription: {description}\nFindings:\n{results_text}"
        )
    
    sections_block = "\n\n".join(sections_text)
    outline_str = "\n".join(f"- {item['section']}: {item['description']}" for item in outline)

    system = """You are a research synthesis specialist. Write a comprehensive, well-structured research report following the provided outline.

Write a thorough research report that:
1. Has a clear introduction explaining the topic
2. Follows the outline structure with dedicated sections
3. Synthesizes information from multiple sources within each section
4. Uses inline citations like [1], [2], etc. referencing the URLs
5. Has substantive content in each section
6. Ends with a conclusion summarizing key findings
7. Note any conflicting information or disagreements between sources

The report should be detailed and comprehensive.
Make it as long as necessary to cover all the information gathered."""

    user = f"Topic: {topic}\n\nReport Outline:\n{outline_str}"

    messages = build_tool_messages(
        system=system,
        user=user,
        tool_content=sections_block,
    )

    completion = await chat_completion(messages=messages, model=settings.summary_model, max_tokens=max_tokens)
    usage = TokenUsage.model_validate(completion["usage"] or {})
    return completion["content"], usage



def _deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate results based on URL."""
    unique_results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for result in results:
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)
        unique_results.append(result)
    return unique_results


async def deep_research(
    query: str,
    stages: int | None = None,
    sub_queries_per_stage: int | None = None,
    results_per_query: int | None = None,
    max_tokens: int | None = None,
    passes: int | None = None,
    outline: list[dict[str, str]] | None = None,
    content_extraction: bool | None = None,
    pages_per_section: int | None = None,
    content_max_chars: int | None = None,
    enable_filtering: bool = True,
    progress_callback: Callable[[str, str], None] | None = None,
) -> DeepResearchRun:
    """Execute a multi-pass deep research operation.

    Flow:
    1. Generate outline (sections to cover)
    2. For each pass: generate queries → search → gather results
    3. Later passes refine queries based on earlier findings
    4. Synthesize final essay
    
    Args:
        query: The research topic or question
        stages: Number of sections in outline (default from config)
        sub_queries_per_stage: Queries per section per pass (default from config)
        results_per_query: Results per sub-query (default from config)
        max_tokens: Max tokens for final essay (default from config)
        passes: Number of research passes (default 1, more = deeper research)
    outline: Optional custom outline. If provided, use this instead of generating one.
    enable_filtering: Enable keyword-based relevance filtering (default True).
    """
    settings = get_settings()
    stages = settings.deep_research_stages if stages is None else stages
    sub_queries_per_stage = (
        settings.deep_research_subqueries
        if sub_queries_per_stage is None
        else sub_queries_per_stage
    )
    results_per_query = (
        settings.deep_research_results_per_query
        if results_per_query is None
        else results_per_query
    )
    max_tokens = settings.deep_research_max_tokens if max_tokens is None else max_tokens
    passes = 1 if passes is None else passes
    content_extraction = (
        settings.deep_research_content_extraction
        if content_extraction is None
        else content_extraction
    )
    pages_per_section = (
        settings.deep_research_pages_per_section
        if pages_per_section is None
        else pages_per_section
    )
    content_max_chars = (
        settings.deep_research_content_max_chars
        if content_max_chars is None
        else content_max_chars
    )

    total_usage = TokenUsage()
    all_results: list[SearchResult] = []
    all_sub_queries: list[str] = []
    section_results: dict[str, list[SearchResult]] = {}
    total_search_requests = 0

    # Progress callback helper
    def _progress(stage: str, message: str):
        if progress_callback:
            progress_callback(stage, message)

    _progress("start", f"Starting research on: {query}")

    # Step 1: Use provided outline or generate one
    if outline:
        # Validate provided outline format
        validated_outline = []
        for item in outline:
            if isinstance(item, dict) and "section" in item:
                section = str(item["section"]).strip()[:_MAX_OUTLINE_SECTION_LEN]
                description = str(item.get("description", "")).strip()[:_MAX_OUTLINE_DESCRIPTION_LEN]
                if section:
                    validated_outline.append({
                        "section": section,
                        "description": description,
                    })
        if validated_outline:
            outline = validated_outline
        else:
            outline = None

    if not outline:
        try:
            outline, usage = await generate_outline(query, num_sections=stages)
            _merge_usage(total_usage, usage.model_dump())
        except Exception as exc:
            logger.warning("Outline generation failed, using fallback: %s", exc)
            outline = [
                {"section": f"Section {i + 1}", "description": f"Research aspect {i + 1} of {query}"}
                for i in range(stages)
            ]

    # Initialize section results
    for item in outline:
        section_results[item["section"]] = []
    _progress("outline", f"Research outline ready: {len(outline)} sections")

    # Step 2: Multiple passes
    for pass_num in range(1, passes + 1):
        _progress("pass", f"Starting research pass {pass_num}/{passes}")
        is_first_pass = pass_num == 1
        
        # Get results summary from previous pass for refinement
        previous_findings = None
        if not is_first_pass:
            # Summarize what we found in previous passes
            sample_results = all_results[:15]
            previous_findings = format_results_for_synthesis(sample_results)
        
        # 2a: Generate all queries in parallel
        query_generation_tasks = []
        for section_item in outline:
            if is_first_pass:
                task = generate_subqueries_for_section(
                    topic=query,
                    section=section_item["section"],
                    description=section_item["description"],
                    num_queries=sub_queries_per_stage,
                    existing_queries=all_sub_queries,
                    results_summary=None,
                )
            else:
                # Refine queries based on findings
                task = generate_refined_queries(
                    topic=query,
                    section=section_item["section"],
                    current_results=section_results[section_item["section"]],
                    num_queries=sub_queries_per_stage,
                    existing_queries=all_sub_queries,
                )
            query_generation_tasks.append(task)
        
        query_results = await asyncio.gather(
            *query_generation_tasks, return_exceptions=True
        )

        # Filter out failed tasks and log errors
        valid_query_results = []
        for section_item, result in zip(outline, query_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Query generation failed for section %r: %s",
                    section_item["section"],
                    result,
                )
                if not is_first_pass:
                    # Retry with initial query generation as fallback
                    try:
                        fallback = await generate_subqueries_for_section(
                            topic=query,
                            section=section_item["section"],
                            description=section_item["description"],
                            num_queries=sub_queries_per_stage,
                            existing_queries=all_sub_queries,
                        )
                        valid_query_results.append(fallback)
                        continue
                    except Exception as fallback_exc:
                        logger.warning(
                            "Fallback query generation also failed for %r: %s",
                            section_item["section"],
                            fallback_exc,
                        )
                valid_query_results.append(([], TokenUsage()))
            else:
                valid_query_results.append(result)
        
        # Collect all queries and organize by section
        section_queries: dict[str, list[str]] = {}
        for section_item, (queries, usage) in zip(outline, valid_query_results):
            section = section_item["section"]
            section_queries[section] = queries
            _merge_usage(total_usage, usage.model_dump())
            
            for q in queries:
                if q not in all_sub_queries:
                    all_sub_queries.append(q)
        
        # 2b: Execute all searches in parallel
        all_queries = []
        query_to_section: dict[str, str] = {}
        for section, queries in section_queries.items():
            for q in queries:
                if q not in query_to_section:
                    query_to_section[q] = section
                    all_queries.append(q)
        
        if all_queries:
            total_search_requests += len(all_queries)
            search_tasks = [search_and_collect(q, results_per_query) for q in all_queries]
            all_search_results = await asyncio.gather(
                *search_tasks, return_exceptions=True
            )
            
            # Organize results by section and aggregate
            for query_str, results in zip(all_queries, all_search_results):
                if isinstance(results, Exception):
                    logger.warning("Search failed for query %r: %s", query_str, results)
                    continue
                section = query_to_section[query_str]
                
                # Apply relevance filtering if enabled (fast keyword-based)
                if enable_filtering and results:
                    # Extract keywords from section description for filtering
                    section_desc = next((item["description"] for item in outline if item["section"] == section), "")
                    keywords = query.split() + section.split() + section_desc.split()
                    results = filter_results_by_relevance_sync(
                        results, keywords, min_matches=1, max_results=results_per_query
                    )
                
                section_results[section].extend(results)
                all_results.extend(results)

        # List URLs found in this pass
        unique_urls = list(set(r.url for r in all_results if r.url))
        urls_msg = f"Found {len(unique_urls)} unique URLs: " + ", ".join(unique_urls[:5])
        if len(unique_urls) > 5:
            urls_msg += f"... (+{len(unique_urls)-5} more)"
        _progress("search", urls_msg)

    # Deduplicate and synthesize
    unique_results = _deduplicate_results(all_results)

    # Organize results by section for synthesis
    final_section_results: dict[str, list[SearchResult]] = {}
    for item in outline:
        section = item["section"]
        section_results_list = section_results.get(section, [])
        final_section_results[section] = _deduplicate_results(section_results_list)

    # Step 3: LLM-based relevance selection + content extraction
    content_map: dict[str, str] = {}
    section_result_counts = {s: len(r) for s, r in final_section_results.items()}
    logger.info(
        "Content extraction: enabled=%s, sections=%s",
        content_extraction,
        section_result_counts,
    )
    if content_extraction:
        _progress("extraction", "Selecting most relevant results for content extraction")

        # Use LLM to pick the best results per section (parallel)
        selection_tasks = []
        sections_for_extraction: list[str] = []
        for item in outline:
            section = item["section"]
            section_res = final_section_results.get(section, [])
            if section_res:
                sections_for_extraction.append(section)
                selection_tasks.append(
                    select_relevant_results(
                        topic=query,
                        section=section,
                        description=item["description"],
                        results=section_res,
                        max_results=pages_per_section,
                    )
                )

        extraction_targets: dict[str, list[SearchResult]] = {}
        if selection_tasks:
            selection_outcomes = await asyncio.gather(
                *selection_tasks, return_exceptions=True
            )
            for section, outcome in zip(sections_for_extraction, selection_outcomes):
                if isinstance(outcome, Exception):
                    logger.warning(
                        "LLM result selection failed for %r, using first N: %s",
                        section, outcome,
                    )
                    extraction_targets[section] = final_section_results[section][:pages_per_section]
                else:
                    selected, usage = outcome
                    _merge_usage(total_usage, usage.model_dump())
                    extraction_targets[section] = selected

        # Extract content from selected results (parallel)
        extraction_tasks_list = []
        extraction_sections: list[str] = []
        for section, targets in extraction_targets.items():
            if targets:
                extraction_sections.append(section)
                extraction_tasks_list.append(
                    enrich_results(
                        targets,
                        max_pages=len(targets),
                        max_chars_per_page=content_max_chars,
                    )
                )

        if extraction_tasks_list:
            extracted_maps = await asyncio.gather(
                *extraction_tasks_list, return_exceptions=True
            )
            for section, result in zip(extraction_sections, extracted_maps):
                if isinstance(result, Exception):
                    logger.warning("Content extraction failed for %r: %s", section, result)
                    continue
                content_map.update(result)

        if content_map:
            logger.info(
                "Enriched synthesis with full content from %d pages",
                len(content_map),
            )

    # Step 4: Synthesize with outline
    essay, usage = await synthesize_essay_with_outline(
        topic=query,
        outline=outline,
        section_results=final_section_results,
        max_tokens=max_tokens,
        content_map=content_map or None,
    )
    _merge_usage(total_usage, usage.model_dump())

    # Estimate citation tokens from search content fed to the LLM
    citation_chars = 0
    for item in outline:
        section = item["section"]
        results = final_section_results.get(section, [])
        citation_chars += len(
            format_results_for_synthesis(results, content_map=content_map or None)
        )
    total_usage.citation_tokens = citation_chars // 4 if citation_chars else 0
    total_usage.search_requests = total_search_requests

    _progress("complete", f"Research complete! Generated {len(essay)} char essay from {len(unique_results)} sources")
    return DeepResearchRun(
        essay=essay,
        results=unique_results,
        sub_queries=all_sub_queries,
        stages_completed=stages * passes,
        usage=total_usage,
    )


async def select_relevant_results(
    topic: str,
    section: str,
    description: str,
    results: list[SearchResult],
    max_results: int = 5,
) -> tuple[list[SearchResult], TokenUsage]:
    """Use LLM to select the most relevant results for content extraction.

    Makes a single batch LLM call to pick the best candidates, rather than
    scoring each result individually. This is used as a gate before the
    expensive Playwright content extraction step.

    Args:
        topic: Research topic
        section: Outline section name
        description: Section description
        results: Candidate search results
        max_results: How many to select

    Returns:
        Tuple of (selected results, token usage)
    """
    if len(results) <= max_results:
        return results, TokenUsage()

    settings = get_settings()

    results_text = "\n".join(
        f"{i}. Title: {sanitize_content(r.title, max_length=200)}\n   URL: {r.url}\n   Snippet: {sanitize_content(r.snippet, max_length=500)}"
        for i, r in enumerate(results)
    )

    system = "You are a research result selection specialist. Select the most relevant search results for content extraction based on the topic and section described by the user."
    user = f"""Select the {max_results} most relevant search results for content extraction.

Topic: {topic}
Section: {section}
Description: {description}

Return ONLY a JSON array of the result indices (0-based) for the most relevant results.
Example: [0, 2, 5]"""

    messages = build_tool_messages(
        system=system,
        user=user,
        tool_content=results_text,
    )

    try:
        completion = await chat_completion(
            messages=messages,
            model=settings.summary_model,
            max_tokens=200,
        )
        usage = TokenUsage.model_validate(completion["usage"] or {})

        raw = completion["content"].strip()
        # Extract array from potential markdown fencing
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()

        indices = json.loads(raw)
        if not isinstance(indices, list):
            return results[:max_results], usage

        selected: list[SearchResult] = []
        seen: set[int] = set()
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(results) and idx not in seen:
                seen.add(idx)
                selected.append(results[idx])
        return (selected[:max_results] if selected else results[:max_results]), usage
    except (UpstreamServiceError, json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("LLM result selection failed, using first N: %s", exc)
        return results[:max_results], TokenUsage()


def filter_results_by_relevance_sync(
    results: list[SearchResult],
    keywords: list[str],
    min_matches: int = 1,
    max_results: int = 30,
) -> list[SearchResult]:
    """Simple keyword-based relevance filtering (sync version)."""
    if not results:
        return []
    
    keyword_set = set(k.lower() for k in keywords)
    
    scored_results = []
    for result in results:
        text = (result.title + " " + (result.snippet or "")).lower()
        matches = sum(1 for kw in keyword_set if kw in text)
        if matches >= min_matches:
            scored_results.append((result, matches))
    
    scored_results.sort(key=lambda x: (-x[1], results.index(x[0])))
    return [r for r, _ in scored_results[:max_results]]
