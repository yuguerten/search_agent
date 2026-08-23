from __future__ import annotations

import re
from typing import Any

from app.models import PaperCandidate

_STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "between",
    "from",
    "into",
    "more",
    "only",
    "that",
    "the",
    "their",
    "this",
    "what",
    "where",
    "which",
    "with",
}


def _tokens(value: object) -> set[str]:
    tokens = re.findall(r"[a-z0-9]{3,}", str(value).casefold())
    normalized = set()
    for token in tokens:
        if token in _STOPWORDS:
            continue
        normalized.add(token[:-1] if token.endswith("s") and len(token) > 4 else token)
    return normalized


def _requested_keywords(intent: dict[str, Any] | None) -> set[str]:
    if not intent:
        return set()
    core_concepts = intent.get("core_concepts", [])
    if core_concepts:
        return _tokens(" ".join(str(concept) for concept in core_concepts))
    keywords = intent.get("keywords", [])
    if keywords:
        return _tokens(" ".join(str(keyword) for keyword in keywords))
    return _tokens(
        " ".join(
            str(intent.get(field, ""))
            for field in (
                "original_question",
                "clarified_question",
                "inclusion_criteria",
            )
        )
    )


def evaluate_scope(
    paper: PaperCandidate, intent: dict[str, Any] | None
) -> tuple[bool, list[str]]:
    """Check topical alignment using only the current user intent.

    This function is domain-agnostic: it does not know about scaling laws, LLMs,
    medicine, or any other subject. It only compares the user's extracted intent
    keywords with the paper title and abstract.
    """

    requested = _requested_keywords(intent)
    if not requested:
        return True, []

    title_tokens = _tokens(paper.title)
    evidence_tokens = title_tokens | _tokens(paper.abstract)
    matched = requested & evidence_tokens
    match_ratio = len(matched) / len(requested)
    minimum_ratio = 0.35 if len(requested) >= 4 else 0.5
    if not matched:
        return False, ["no user-intent keywords found in title or abstract"]
    if match_ratio < minimum_ratio:
        return False, [
            f"insufficient intent overlap ({len(matched)}/{len(requested)} keywords)"
        ]
    return True, []
