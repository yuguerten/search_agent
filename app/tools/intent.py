from __future__ import annotations

import re
from datetime import date, timedelta

from app.config import get_settings
from app.models import ResearchIntent

_STOP_WORDS = {
    "about",
    "after",
    "and",
    "are",
    "from",
    "into",
    "that",
    "the",
    "their",
    "what",
    "where",
    "which",
    "with",
}


def extract_keyword_candidates(text: str, limit: int = 12) -> list[str]:
    """Extract a deterministic fallback keyword list from a question or answer."""

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
    keywords: list[str] = []
    for word in words:
        if word not in _STOP_WORDS and word not in keywords:
            keywords.append(word)
        if len(keywords) == limit:
            break
    return keywords


def update_intent(
    original_question: str,
    clarification_answers: list[str],
    target_paper_count: int = 5,
) -> dict:
    """Build a serializable research intent from the current conversation."""

    context = " ".join([original_question, *clarification_answers]).strip()
    keywords = extract_keyword_candidates(context)
    clarified_question = " ".join(clarification_answers).strip() or None
    settings = get_settings()
    end_date = date.today()
    start_date = end_date - timedelta(days=settings.recent_days)
    intent = ResearchIntent(
        original_question=original_question,
        clarified_question=clarified_question,
        keywords=keywords,
        target_paper_count=target_paper_count,
        start_date=start_date,
        end_date=end_date,
    )
    return intent.model_dump(mode="json")
