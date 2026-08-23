from __future__ import annotations

import re
from datetime import date, timedelta

from app.config import get_settings
from app.models import ResearchIntent
from app.tools.context import ToolContext

_STOP_WORDS = {
    "about",
    "according",
    "after",
    "agent",
    "answers",
    "ask",
    "begin",
    "call",
    "clarification",
    "clarifier",
    "context",
    "content",
    "delegate",
    "every",
    "final",
    "for",
    "any",
    "art",
    "aspect",
    "and",
    "are",
    "from",
    "focus",
    "focusing",
    "find",
    "exploring",
    "interested",
    "interest",
    "into",
    "instructions",
    "that",
    "the",
    "know",
    "latest",
    "moment",
    "need",
    "once",
    "open",
    "their",
    "important",
    "underlying",
    "paper",
    "perform",
    "probably",
    "problem",
    "report",
    "research",
    "researcher",
    "researchers",
    "specific",
    "state",
    "says",
    "said",
    "session",
    "synthesis",
    "synthesizer",
    "thought",
    "topic",
    "transfer",
    "turn",
    "until",
    "user",
    "want",
    "we",
    "workflow",
    "you",
    "what",
    "when",
    "where",
    "whether",
    "which",
    "with",
    "should",
    "this",
}


def extract_keyword_candidates(text: str, limit: int = 12) -> list[str]:
    """Extract a deterministic fallback keyword list from a question or answer."""

    # A clarification such as "I am open to any domain or industry" expresses
    # no topical constraint. Remove that non-restrictive phrase while keeping
    # ``domain`` and ``industry`` available when they are actual topics.
    text = re.sub(
        r"\b(?:i am )?open to any domain(?: or industry)?(?: not important)?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bnot important\b", " ", text, flags=re.IGNORECASE)

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
    keywords: list[str] = []
    for word in words:
        if word not in _STOP_WORDS and word not in keywords:
            keywords.append(word)
        if len(keywords) == limit:
            break
    return keywords


def build_search_queries(keywords: list[str]) -> list[str]:
    """Build generic topic queries from the current user's keywords."""

    terms: list[str] = []
    for keyword in keywords:
        for token in re.findall(r"[a-z0-9]+", keyword.casefold()):
            if len(token) >= 3 and token not in _STOP_WORDS and token not in terms:
                terms.append(token)

    if not terms:
        return []
    variants = [
        terms[:6],
        terms[::2][:5],
        terms[1:7] if len(terms) > 1 else terms[:6],
    ]
    queries: list[str] = []
    for variant in variants:
        query = " ".join(dict.fromkeys(variant))
        if query and query not in queries:
            queries.append(query)
    return queries


def update_intent(
    original_question: str,
    clarification_answers: list[str],
    target_paper_count: int = 5,
    tool_context: ToolContext | None = None,
) -> dict:
    """Build a serializable research intent from the current conversation."""

    context = " ".join([original_question, *clarification_answers]).strip()
    keywords = extract_keyword_candidates(context)
    clarified_question = " ".join(clarification_answers).strip() or None
    settings = get_settings()
    end_date = date.today() if settings.enforce_recent_filter else None
    start_date = (
        end_date - timedelta(days=settings.recent_days)
        if end_date is not None
        else None
    )
    intent = ResearchIntent(
        original_question=original_question,
        clarified_question=clarified_question,
        keywords=keywords,
        search_queries=build_search_queries(keywords),
        target_paper_count=target_paper_count,
        start_date=start_date,
        end_date=end_date,
    )
    result = intent.model_dump(mode="json")
    if tool_context is not None:
        previous_intent = tool_context.state.get("research_intent")
        is_new_request = not isinstance(previous_intent, dict) or (
            previous_intent.get("original_question") != original_question
        )
        if is_new_request:
            # A session may contain several independent research requests. Do
            # not let papers, approvals, caches, or loop flags leak between
            # them. The dispatcher intent is the boundary for a new run.
            # ADK's State intentionally supports mapping reads and updates,
            # but not deletion. Reset values to their neutral types so this
            # works for both ADK State and the dict used by unit tests.
            tool_context.state.update(
                {
                    "candidates": [],
                    "ranked_papers": [],
                    "approved_papers": [],
                    "critic_decisions": [],
                    "arxiv_search_calls": 0,
                    "arxiv_query_cache": {},
                    "arxiv_search_exhausted": False,
                    "active_search_query": None,
                    "research_status": None,
                    "critic_status": None,
                    "researcher_status": None,
                    "synthesizer_status": None,
                    "loop_complete": False,
                    "report": None,
                }
            )
        tool_context.state["research_intent"] = result
        tool_context.state["search_queries"] = result["search_queries"]
    return result
