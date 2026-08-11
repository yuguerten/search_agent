from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from app.config import get_settings
from app.models import PaperCandidate

API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = (
    "paperId,externalIds,title,year,publicationDate,citationCount,"
    "influentialCitationCount"
)


async def enrich_with_semantic_scholar(
    papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach Semantic Scholar citation metadata to arXiv candidates."""

    if not papers:
        return []

    settings = get_settings()
    identifiers = [f"ARXIV:{paper['arxiv_id']}" for paper in papers]
    headers = {"x-api-key": settings.semantic_scholar_api_key}
    headers = {key: value for key, value in headers.items() if value}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            API_URL,
            params={"fields": FIELDS},
            json={"ids": identifiers},
            headers=headers,
        )
        response.raise_for_status()
        metadata = response.json()

    enriched: list[dict[str, Any]] = []
    retrieved_at = datetime.now(UTC)
    for paper, scholar_data in zip(papers, metadata, strict=False):
        candidate = PaperCandidate.model_validate(paper)
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
    return enriched
