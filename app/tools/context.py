"""Compatibility types and paper-state helpers for ADK tools."""

import re
from difflib import SequenceMatcher
from typing import Any

from pydantic import ValidationError

from app.models import PaperCandidate

try:
    from google.adk.tools.tool_context import ToolContext
except ImportError:  # pragma: no cover - used only when ADK is not installed

    class ToolContext:  # type: ignore[no-redef]
        state: dict
        actions: object


def _normalize_title(value: object) -> str:
    """Normalize full or model-truncated titles for safe state matching."""

    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def canonicalize_papers(
    papers: list[dict[str, Any]], tool_context: ToolContext | None = None
) -> list[PaperCandidate]:
    """Validate papers or recover shortened objects from session state.

    Local language models sometimes resend a display object instead of the full
    search result. Identity fields are always taken from session state; only
    ranking scores supplied by the model may be merged into a matching paper.
    """

    state_papers: list[dict[str, Any]] = []
    if tool_context is not None:
        for key in ("candidates", "ranked_papers", "approved_papers"):
            value = tool_context.state.get(key, [])
            if isinstance(value, list):
                state_papers.extend(item for item in value if isinstance(item, dict))

    by_id = {
        item.get("arxiv_id"): item for item in state_papers if item.get("arxiv_id")
    }
    by_title = {
        _normalize_title(item.get("title")): item
        for item in state_papers
        if item.get("title")
    }
    result: list[PaperCandidate] = []
    score_fields = (
        "relevance_score",
        "citation_score",
        "recency_score",
        "metadata_score",
        "final_score",
    )
    for raw in papers:
        try:
            result.append(PaperCandidate.model_validate(raw))
            continue
        except ValidationError:
            raw_id = raw.get("arxiv_id")
            title = _normalize_title(raw.get("title", ""))
            canonical = by_id.get(raw_id) if raw_id else by_title.get(title)
            if canonical is None and title:
                matches = []
                for candidate_title, candidate in by_title.items():
                    if (
                        title.startswith(candidate_title)
                        or candidate_title.startswith(title)
                        or SequenceMatcher(None, title, candidate_title).ratio() >= 0.72
                    ):
                        matches.append(candidate)
                if len(matches) == 1:
                    canonical = matches[0]
            if canonical is None and len(state_papers) == 1:
                canonical = state_papers[0]
            if canonical is None:
                raise ValueError(
                    "Paper metadata is incomplete. Pass the exact objects returned "
                    "by search_arxiv; do not reconstruct papers manually."
                ) from None
            merged = dict(canonical)
            for key in score_fields:
                if key in raw:
                    merged[key] = raw[key]
            result.append(PaperCandidate.model_validate(merged))
    return result
