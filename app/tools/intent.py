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
    "all",
    "am",
    "an",
    "and",
    "answers",
    "any",
    "apply",
    "applying",
    "are",
    "art",
    "ask",
    "aspect",
    "at",
    "be",
    "been",
    "begin",
    "being",
    "but",
    "call",
    "can",
    "clarification",
    "clarifier",
    "content",
    "context",
    "could",
    "delegate",
    "does",
    "doing",
    "dont",
    "every",
    "exploring",
    "final",
    "find",
    "focus",
    "focusing",
    "for",
    "from",
    "give",
    "go",
    "had",
    "has",
    "have",
    "how",
    "ideal",
    "important",
    "include",
    "includes",
    "including",
    "instructions",
    "interest",
    "interested",
    "into",
    "is",
    "just",
    "know",
    "latest",
    "like",
    "may",
    "me",
    "might",
    "moment",
    "need",
    "no",
    "not",
    "now",
    "of",
    "on",
    "once",
    "open",
    "paper",
    "perform",
    "please",
    "prefer",
    "preferred",
    "probably",
    "problem",
    "recent",
    "report",
    "research",
    "researcher",
    "researchers",
    "said",
    "says",
    "session",
    "should",
    "specific",
    "start",
    "state",
    "synthesis",
    "synthesizer",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "this",
    "thought",
    "to",
    "topic",
    "transfer",
    "turn",
    "underlying",
    "until",
    "use",
    "user",
    "using",
    "want",
    "we",
    "what",
    "when",
    "where",
    "whether",
    "which",
    "with",
    "workflow",
    "would",
    "you",
}

_GENERIC_EDGE_WORDS = {
    "approach",
    "approaches",
    "area",
    "areas",
    "constraint",
    "constraints",
    "example",
    "examples",
    "goal",
    "goals",
    "method",
    "methods",
    "paper",
    "papers",
    "research",
    "study",
    "studies",
    "technique",
    "techniques",
    "topic",
    "topics",
    "work",
}

_CLAUSE_BOUNDARY = re.compile(
    r"[.,;:!?()\n]+|\b(?:about|across|affect|affects|against|among|and|between|"
    r"compare|compared|comparing|concerning|during|for|from|impact|impacts|in|"
    r"into|of|"
    r"on|or|over|through|to|until|versus|via|with|without)\b",
    flags=re.IGNORECASE,
)

_CONVERSATIONAL_PREFIX = re.compile(
    r"^\s*(?:(?:hello|hi|hey)\b[,!\s]*)?"
    r"(?:(?:i|we)\s+(?:am\s+|are\s+|['\u2019]m\s+)?)?"
    r"(?:curious|interested)\s+(?:to\s+)?"
    r"(?:know|learn|find\s+out|read)\s+(?:more\s+)?(?:about|on)\s+",
    flags=re.IGNORECASE,
)

_NON_RESTRICTIVE_CHOICE = re.compile(
    r"\b(?:all|any|both|either)\b.*"
    r"\b(?:cannot|can['\u2019]?t|not|no|unable)\b.*"
    r"\b(?:choose|decide|pick|select|preference)\w*\b",
    flags=re.IGNORECASE,
)


def _normalise_phrase(value: str) -> str:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{1,}", value.casefold())
    return " ".join(tokens)


def _concept_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{1,}", value.casefold())
        if token not in _STOP_WORDS
        and not token.isdigit()
        and not (len(token) == 4 and token.startswith(("19", "20")))
    ]


def extract_concept_candidates(text: str, limit: int = 8) -> list[str]:
    """Extract domain-agnostic scientific phrases from literal user text."""

    text = _CONVERSATIONAL_PREFIX.sub("", text)
    text = re.sub(
        r"\bapplied\s+(?=(?:for|on|to)\b)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:state[- ]of[- ]the[- ]art|not important|at the moment)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:i am )?open to any domain(?: or industry)?(?: not important)?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    negative_constraint = re.search(
        r"\b(?:no|not(?:\s+(?:a|an))?)\s+(?:\w+\s+){0,3}"
        r"(?:constraint|preference|requirement)s?\b",
        text,
        flags=re.IGNORECASE,
    )
    if negative_constraint:
        text = text[: negative_constraint.start()]

    concepts: list[str] = []
    for clause in _CLAUSE_BOUNDARY.split(text):
        tokens = _concept_tokens(clause)
        while tokens and tokens[0] in _GENERIC_EDGE_WORDS:
            tokens.pop(0)
        while tokens and tokens[-1] in _GENERIC_EDGE_WORDS:
            tokens.pop()
        if not tokens:
            continue

        phrases = [tokens]
        if len(tokens) > 5:
            phrases = [tokens[:4], tokens[-4:]]
        for phrase_tokens in phrases:
            phrase = " ".join(phrase_tokens)
            if phrase and phrase not in concepts:
                concepts.append(phrase)
            if len(concepts) == limit:
                return concepts
    return concepts


def extract_keyword_candidates(text: str, limit: int = 20) -> list[str]:
    """Flatten extracted concepts into deterministic ranking keywords."""

    keywords: list[str] = []
    for concept in extract_concept_candidates(text, limit=limit):
        for word in re.findall(r"[a-z0-9]+", concept):
            if len(word) >= 3 and word not in keywords:
                keywords.append(word)
            if len(keywords) == limit:
                return keywords
    return keywords


def build_search_queries(
    core_concepts: list[str],
    refinement_concepts: list[str] | None = None,
    limit: int = 3,
) -> list[str]:
    """Build phrase-aware queries from arbitrary structured research concepts."""

    core = list(
        dict.fromkeys(
            phrase for value in core_concepts if (phrase := _normalise_phrase(value))
        )
    )
    refinements = list(
        dict.fromkeys(
            phrase
            for value in (refinement_concepts or [])
            if (phrase := _normalise_phrase(value)) and phrase not in core
        )
    )
    if not core:
        return []

    base = core[:2]

    def format_query(concepts: list[str]) -> str:
        return " AND ".join(f'"{concept}"' for concept in concepts)

    queries = [format_query(base)]
    ordered_refinements = sorted(
        enumerate(refinements),
        key=lambda item: (-len(item[1].split()), item[0]),
    )
    for _, refinement in ordered_refinements:
        query = format_query([*base, refinement])
        if query not in queries:
            queries.append(query)
        if len(queries) == limit:
            return queries

    for concept in core:
        query = format_query([concept])
        if query not in queries:
            queries.append(query)
        if len(queries) == limit:
            return queries

        tokens = concept.split()
        if len(tokens) >= 3:
            for relaxed in (tokens[:-1], tokens[1:]):
                query = format_query([" ".join(relaxed)])
                if query not in queries:
                    queries.append(query)
                if len(queries) == limit:
                    return queries
    return queries


def _merge_concepts(texts: list[str], excluded: list[str], limit: int = 8) -> list[str]:
    excluded_keys = {_normalise_phrase(value) for value in excluded}
    concepts: list[str] = []
    for text in texts:
        if _NON_RESTRICTIVE_CHOICE.search(text):
            continue
        # A standalone "Any ..." clarification means that dimension is not a
        # search constraint. Removing the whole sentence avoids turning it
        # into a false positive concept while remaining domain-independent.
        text = re.sub(
            r"(?:^|(?<=[.!?]))\s*any\b[^.!?\n]*[.!?]?",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        extracted = extract_concept_candidates(text, limit=limit)
        if 1 < len(extracted) <= 3 and all(
            len(concept.split()) == 1 for concept in extracted
        ):
            extracted = [" ".join(extracted)]
        for concept in extracted:
            key = _normalise_phrase(concept)
            if key and key not in excluded_keys and concept not in concepts:
                concepts.append(concept)
                excluded_keys.add(key)
            if len(concepts) == limit:
                return concepts
    return concepts


def update_intent(
    original_question: str,
    clarification_answers: list[str],
    target_paper_count: int = 5,
    tool_context: ToolContext | None = None,
) -> dict:
    """Build a serializable research intent from the current conversation."""

    if tool_context is not None:
        captured_question = tool_context.state.get("original_question")
        captured_answers = tool_context.state.get("clarification_answers")
        if isinstance(captured_question, str) and captured_question.strip():
            # Literal Context.user_content captured by the clarifier is more
            # trustworthy than arguments reconstructed by the dispatcher LLM.
            original_question = captured_question.strip()
            if isinstance(captured_answers, list) and all(
                isinstance(answer, str) for answer in captured_answers
            ):
                clarification_answers = [
                    answer.strip() for answer in captured_answers if answer.strip()
                ]

    core_concepts = extract_concept_candidates(original_question, limit=4)
    refinement_concepts = _merge_concepts(
        clarification_answers, excluded=core_concepts, limit=8
    )
    if not core_concepts and refinement_concepts:
        core_concepts = refinement_concepts[:2]
        refinement_concepts = refinement_concepts[2:]
    concepts = [*core_concepts, *refinement_concepts]
    keywords = extract_keyword_candidates("; ".join(concepts))
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
        core_concepts=core_concepts,
        refinement_concepts=refinement_concepts,
        keywords=keywords,
        search_queries=build_search_queries(core_concepts, refinement_concepts),
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
