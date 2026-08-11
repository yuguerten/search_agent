from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any

from app.models import PaperCandidate


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _normalise(values: Iterable[float]) -> list[float]:
    values = list(values)
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def recent_papers(
    papers: list[PaperCandidate],
    as_of: date,
    recent_days: int = 730,
) -> list[PaperCandidate]:
    """Apply the strict rolling date policy before citation ranking."""

    lower_bound = as_of - timedelta(days=recent_days)
    return [paper for paper in papers if lower_bound <= paper.published_at <= as_of]


def rank_papers(
    papers: list[PaperCandidate],
    keywords: list[str],
    as_of: date,
    recent_days: int = 730,
) -> list[PaperCandidate]:
    """Rank recent candidates using relevance, age-adjusted citations, and freshness."""

    candidates = recent_papers(papers, as_of=as_of, recent_days=recent_days)
    keyword_tokens = _tokens(" ".join(keywords))
    relevance_values: list[float] = []
    citation_values: list[float] = []
    freshness_values: list[float] = []
    metadata_values: list[float] = []

    for paper in candidates:
        title_tokens = _tokens(paper.title)
        abstract_tokens = _tokens(paper.abstract)
        title_overlap = len(keyword_tokens & title_tokens) / max(len(keyword_tokens), 1)
        abstract_overlap = len(keyword_tokens & abstract_tokens) / max(
            len(keyword_tokens), 1
        )
        relevance_values.append(0.7 * title_overlap + 0.3 * abstract_overlap)

        age_years = max((as_of - paper.published_at).days / 365.25, 0.25)
        citations = max(paper.citation_count or 0, 0)
        citation_values.append(math.log1p(citations) / age_years)

        age_days = max((as_of - paper.published_at).days, 0)
        freshness_values.append(1.0 - age_days / max(recent_days, 1))
        metadata_values.append(
            sum(
                bool(value)
                for value in (
                    paper.abstract,
                    paper.authors,
                    paper.pdf_url,
                    paper.citation_source,
                )
            )
            / 4
        )

    relevance_scores = _normalise(relevance_values)
    citation_scores = _normalise(citation_values)
    freshness_scores = _normalise(freshness_values)

    ranked: list[PaperCandidate] = []
    for index, paper in enumerate(candidates):
        relevance = relevance_scores[index]
        citation = citation_scores[index]
        freshness = freshness_scores[index]
        metadata = metadata_values[index]
        final_score = (
            0.55 * relevance + 0.20 * citation + 0.15 * freshness + 0.10 * metadata
        )
        ranked.append(
            paper.model_copy(
                update={
                    "relevance_score": relevance,
                    "citation_score": citation,
                    "recency_score": freshness,
                    "metadata_score": metadata,
                    "final_score": final_score,
                }
            )
        )
    return sorted(ranked, key=lambda paper: paper.final_score, reverse=True)


def rank_papers_tool(
    papers: list[dict[str, Any]],
    keywords: list[str],
    as_of: str,
    recent_days: int = 730,
) -> list[dict[str, Any]]:
    """ADK tool that filters and ranks candidates deterministically."""

    ranked = rank_papers(
        [PaperCandidate.model_validate(paper) for paper in papers],
        keywords=keywords,
        as_of=date.fromisoformat(as_of),
        recent_days=recent_days,
    )
    return [paper.model_dump(mode="json") for paper in ranked]
