from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from app.config import get_settings
from app.tools.context import ToolContext, canonicalize_papers
from app.tools.http import request_with_retries

API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = (
    "paperId,externalIds,title,year,publicationDate,citationCount,"
    "influentialCitationCount"
)

_SEMANTIC_SCHOLAR_REQUEST_LOCK = asyncio.Lock()
_SEMANTIC_SCHOLAR_LAST_REQUEST = 0.0
_AUTHENTICATED_MIN_INTERVAL = 1.1
_UNAUTHENTICATED_MIN_INTERVAL = 3.5


def _without_arxiv_version(arxiv_id: str) -> str:
    """Semantic Scholar expects arXiv IDs without the v1/v2 suffix."""

    return re.sub(r"v[0-9]+\Z", "", arxiv_id)


async def enrich_with_semantic_scholar(
    papers: list[dict[str, Any]],
    tool_context: ToolContext | None = None,
) -> list[dict[str, Any]]:
    """Attach Semantic Scholar citation metadata to arXiv candidates.

    Citation metadata is supplementary: a temporary Semantic Scholar rate limit
    must not discard otherwise valid arXiv candidates or stop the workflow.
    Results are cached in ADK session state so loop iterations do not request the
    same paper repeatedly.
    """

    if not papers:
        return []

    settings = get_settings()
    try:
        candidates = canonicalize_papers(papers, tool_context)
    except ValueError:
        state_candidates = (
            tool_context.state.get("candidates", []) if tool_context is not None else []
        )
        if not state_candidates:
            raise
        candidates = canonicalize_papers(state_candidates, tool_context)
    identifiers = [
        f"ARXIV:{_without_arxiv_version(paper.arxiv_id)}" for paper in candidates
    ]
    cache = (
        tool_context.state.setdefault("semantic_scholar_cache", {})
        if tool_context is not None
        else {}
    )
    uncached_identifiers = [
        identifier
        for identifier in dict.fromkeys(identifiers)
        if identifier not in cache
    ]
    headers = {"x-api-key": settings.semantic_scholar_api_key}
    headers = {key: value for key, value in headers.items() if value}
    if uncached_identifiers:
        timeout = httpx.Timeout(connect=15.0, read=60.0, write=15.0, pool=15.0)
        min_interval = (
            _AUTHENTICATED_MIN_INTERVAL
            if settings.semantic_scholar_api_key
            else _UNAUTHENTICATED_MIN_INTERVAL
        )
        global _SEMANTIC_SCHOLAR_LAST_REQUEST
        async with _SEMANTIC_SCHOLAR_REQUEST_LOCK:
            wait_for = min_interval - (
                time.monotonic() - _SEMANTIC_SCHOLAR_LAST_REQUEST
            )
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            _SEMANTIC_SCHOLAR_LAST_REQUEST = time.monotonic()
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await request_with_retries(
                    client,
                    "POST",
                    API_URL,
                    retries=1,
                    params={"fields": FIELDS},
                    json={"ids": uncached_identifiers},
                    headers=headers,
                )

        if response.status_code == 429:
            if tool_context is not None:
                tool_context.state["semantic_scholar_rate_limited"] = True
            metadata = [None] * len(uncached_identifiers)
        else:
            response.raise_for_status()
            metadata = response.json()

        cache.update(dict(zip(uncached_identifiers, metadata, strict=False)))

    enriched: list[dict[str, Any]] = []
    retrieved_at = datetime.now(UTC)
    for candidate, identifier in zip(candidates, identifiers, strict=True):
        scholar_data = cache.get(identifier)
        if scholar_data:
            candidate.semantic_scholar_id = scholar_data.get("paperId")
            candidate.citation_count = scholar_data.get("citationCount")
            candidate.influential_citation_count = scholar_data.get(
                "influentialCitationCount"
            )
            candidate.citation_source = "semantic_scholar"
            candidate.citation_retrieved_at = retrieved_at
            candidate.doi = (scholar_data.get("externalIds") or {}).get("DOI")
        enriched.append(candidate.model_dump(mode="json"))
    if tool_context is not None:
        by_id = {
            paper.get("arxiv_id"): paper
            for paper in tool_context.state.get("candidates", [])
            if isinstance(paper, dict)
        }
        by_id.update({paper["arxiv_id"]: paper for paper in enriched})
        tool_context.state["candidates"] = list(by_id.values())
    return enriched
