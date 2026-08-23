from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Any

import httpx

from app.config import get_settings
from app.models import PaperCandidate
from app.tools.context import ToolContext
from app.tools.http import request_with_retries
from app.tools.intent import build_search_queries, extract_concept_candidates

ATOM = "http://www.w3.org/2005/Atom"
NS = {"atom": ATOM}

_STOPWORDS = {
    "and",
    "for",
    "or",
    "the",
    "about",
    "after",
    "also",
    "between",
    "from",
    "into",
    "more",
    "over",
    "than",
    "that",
    "their",
    "these",
    "this",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
}

_ARXIV_REQUEST_LOCK = asyncio.Lock()
_ARXIV_LAST_REQUEST = 0.0
_MAX_ARXIV_SEARCH_CALLS = 6
_MAX_ARXIV_RESULTS = 10
_MAX_ABSTRACT_CHARS = 1600


def build_arxiv_query(query: str) -> str:
    """Render natural or phrase-planned text as a fielded arXiv query."""

    if '"' not in query and not re.search(r"\b(?:AND|OR|ANDNOT)\b", query):
        concepts = extract_concept_candidates(query)
    else:
        concepts = []
        for match in re.finditer(r'"([^"]+)"|([a-zA-Z][a-zA-Z0-9-]*)', query):
            value = match.group(1) or match.group(2)
            if value.casefold() in {"and", "or", "andnot"}:
                continue
            concepts.append(value)

    def normalise_tokens(value: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", value.casefold())
            if len(token) >= 3
            and token not in _STOPWORDS
            and token
            not in {
                "date",
                "dates",
                "last",
                "only",
                "papers",
                "query",
                "recent",
                "years",
            }
            and not token.isdigit()
            and not (len(token) == 4 and token.startswith(("19", "20")))
        ]

    def token_clause(token: str) -> str:
        variants = [token]
        if token.endswith("ies") and len(token) > 4:
            variants.append(f"{token[:-3]}y")
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 4:
            variants.append(token[:-1])
        fields = [field for term in variants for field in (f"ti:{term}", f"abs:{term}")]
        return f"({' OR '.join(fields)})"

    clauses: list[str] = []
    for concept in concepts:
        tokens = normalise_tokens(concept)
        if not tokens:
            continue
        if len(tokens) == 1:
            clause = token_clause(tokens[0])
        else:
            phrase = " ".join(tokens)
            fallback = " AND ".join(token_clause(token) for token in tokens)
            clause = f'(ti:"{phrase}" OR abs:"{phrase}" OR ({fallback}))'
        if clause not in clauses:
            clauses.append(clause)

    if not clauses:
        raise ValueError("arXiv query must contain at least one searchable term")
    return " AND ".join(clauses)


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def parse_arxiv_feed(xml_payload: str) -> list[PaperCandidate]:
    """Parse an arXiv Atom response into validated paper candidates."""

    root = ET.fromstring(xml_payload)
    papers: list[PaperCandidate] = []
    for entry in root.findall("atom:entry", NS):
        entry_id = _clean_text(entry.findtext("atom:id", default="", namespaces=NS))
        arxiv_id = entry_id.rsplit("/abs/", 1)[-1]
        if not arxiv_id:
            continue

        published_raw = _clean_text(
            entry.findtext("atom:published", default="", namespaces=NS)
        )
        published_at = date.fromisoformat(published_raw[:10])
        updated_raw = _clean_text(
            entry.findtext("atom:updated", default="", namespaces=NS)
        )
        links = entry.findall("atom:link", NS)
        pdf_url = next(
            (
                link.attrib.get("href")
                for link in links
                if link.attrib.get("title") == "pdf"
            ),
            None,
        )
        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=NS))
            for author in entry.findall("atom:author", NS)
        ]
        papers.append(
            PaperCandidate(
                arxiv_id=arxiv_id,
                title=_clean_text(
                    entry.findtext("atom:title", default="", namespaces=NS)
                ),
                abstract=_clean_text(
                    entry.findtext("atom:summary", default="", namespaces=NS)
                ),
                authors=[author for author in authors if author],
                arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_url,
                published_at=published_at,
                updated_at=date.fromisoformat(updated_raw[:10])
                if updated_raw
                else None,
            )
        )
    return papers


async def search_arxiv(
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    max_results: int = 15,
    tool_context: ToolContext | None = None,
) -> list[dict[str, Any]]:
    """Search arXiv and return normalized paper metadata.

    This is an ADK-compatible tool. Date filtering is optional and repeated
    locally when enabled because source APIs can return incomplete or inconsistent
    date data.
    """

    search_index = 0
    if tool_context is not None:
        search_index = tool_context.state.get("arxiv_search_calls", 0)
        queries = tool_context.state.get("search_queries", [])
        if not queries:
            intent = tool_context.state.get("research_intent", {})
            if isinstance(intent, dict):
                queries = intent.get("search_queries", []) or build_search_queries(
                    intent.get("core_concepts", []) or intent.get("keywords", []),
                    intent.get("refinement_concepts", []),
                )
        if queries:
            query = queries[search_index % len(queries)]
            tool_context.state["active_search_query"] = query
    arxiv_query = build_arxiv_query(query)
    effective_max_results = min(max_results, _MAX_ARXIV_RESULTS)
    settings = get_settings()
    policy_end = date.today()
    policy_start = policy_end - timedelta(days=settings.recent_days)
    filter_enabled = settings.enforce_recent_filter
    # Date filtering is opt-in. When disabled, arXiv results from every
    # publication date remain eligible and freshness is used only for ranking.
    effective_start_date = policy_start.isoformat() if filter_enabled else None
    effective_end_date = policy_end.isoformat() if filter_enabled else None
    cache_key = (
        f"{arxiv_query}|{effective_start_date}|{effective_end_date}|"
        f"{effective_max_results}"
    )
    if tool_context is not None:
        cache = tool_context.state.setdefault("arxiv_query_cache", {})
        if cache_key in cache:
            return cache[cache_key]
        calls = tool_context.state.get("arxiv_search_calls", 0)
        if calls >= _MAX_ARXIV_SEARCH_CALLS:
            tool_context.state["arxiv_search_exhausted"] = True
            return []
        tool_context.state["arxiv_search_calls"] = calls + 1

    params = {
        "search_query": arxiv_query,
        "start": 0,
        "max_results": effective_max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    timeout = httpx.Timeout(connect=15.0, read=60.0, write=15.0, pool=15.0)
    headers = {"User-Agent": "agentic-literature-researcher/0.1"}
    global _ARXIV_LAST_REQUEST
    async with _ARXIV_REQUEST_LOCK:
        wait_for = 3.5 - (time.monotonic() - _ARXIV_LAST_REQUEST)
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        _ARXIV_LAST_REQUEST = time.monotonic()
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await request_with_retries(
                client, "GET", settings.arxiv_api_url, params=params
            )
            response.raise_for_status()

    papers = parse_arxiv_feed(response.text)
    if filter_enabled:
        lower_bound = date.fromisoformat(effective_start_date)
        upper_bound = date.fromisoformat(effective_end_date)
        papers = [
            paper
            for paper in papers
            if lower_bound <= paper.published_at <= upper_bound
        ]
    result = [
        paper.model_copy(
            update={"abstract": paper.abstract[:_MAX_ABSTRACT_CHARS]}
        ).model_dump(mode="json")
        for paper in papers
    ]
    if tool_context is not None:
        tool_context.state["research_window"] = {
            "filter_enabled": filter_enabled,
            "start_date": effective_start_date,
            "end_date": effective_end_date,
            "recent_days": settings.recent_days,
        }
        existing = {
            paper.get("arxiv_id") for paper in tool_context.state.get("candidates", [])
        }
        tool_context.state["candidates"] = [
            *tool_context.state.get("candidates", []),
            *[paper for paper in result if paper.get("arxiv_id") not in existing],
        ]
        tool_context.state.setdefault("arxiv_query_cache", {})[cache_key] = result
    return result
