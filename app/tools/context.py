"""Compatibility types and paper-state helpers for ADK tools."""

import json
import re
from collections.abc import Mapping, Sequence
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


def _coerce_paper_payloads(papers: object) -> list[dict[str, Any]]:
    """Normalize common LLM/ADK argument shapes before model validation."""

    payloads: list[dict[str, Any]] = []
    pending: list[object] = [papers]
    while pending:
        value = pending.pop(0)
        if isinstance(value, PaperCandidate):
            payloads.append(value.model_dump(mode="json"))
            continue
        if isinstance(value, Mapping):
            payloads.append(dict(value))
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, (Mapping, list)):
                pending.insert(0, decoded)
                continue
            if re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?", text):
                payloads.append({"arxiv_id": text})
            else:
                payloads.append({"title": text})
            continue
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            pending[0:0] = list(value)
            continue
        raise ValueError(
            "Paper input must be a paper object, a list of paper objects, "
            "or JSON encoding one of those shapes."
        )
    return payloads


def canonicalize_papers(
    papers: object, tool_context: ToolContext | None = None
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
    for raw in _coerce_paper_payloads(papers):
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
