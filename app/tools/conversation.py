from __future__ import annotations

import re

from google.adk.agents.context import Context
from google.genai import types

_NON_RESEARCH_UTTERANCES = {
    "begin",
    "continue",
    "go",
    "go ahead",
    "hello",
    "hey",
    "hi",
    "please continue",
    "please proceed",
    "proceed",
    "search",
    "start",
    "start search",
    "start the search",
}


def _content_text(content: types.Content | None) -> str:
    """Return only literal text parts from the invocation's user content."""

    if content is None:
        return ""
    return " ".join(
        part.text.strip()
        for part in (content.parts or [])
        if part.text and part.text.strip()
    ).strip()


def _is_non_research_utterance(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    return normalized in _NON_RESEARCH_UTTERANCES


def capture_user_research_input(*, callback_context: Context) -> None:
    """Persist exact user turns for deterministic downstream intent creation.

    ``Context.user_content`` is the user content that started the current ADK
    invocation. Recording it here prevents a downstream model from rebuilding
    the research question from agent narration, tool calls, or transcript text.
    """

    invocation_id = str(callback_context.invocation_id)
    captured_invocations = list(
        callback_context.state.get("research_input_invocation_ids", [])
    )
    if invocation_id in captured_invocations:
        return None

    text = _content_text(callback_context.user_content)
    if not text or _is_non_research_utterance(text):
        return None

    # A completed report marks the next substantive user turn as a new research
    # request in the same session.
    if callback_context.state.get("final_report") is not None:
        callback_context.state["original_question"] = text
        callback_context.state["clarification_answers"] = []
        callback_context.state["final_report"] = None
    elif not callback_context.state.get("original_question"):
        callback_context.state["original_question"] = text
        callback_context.state["clarification_answers"] = []
    else:
        answers = list(callback_context.state.get("clarification_answers", []))
        answers.append(text)
        callback_context.state["clarification_answers"] = answers

    captured_invocations.append(invocation_id)
    callback_context.state["research_input_invocation_ids"] = captured_invocations
    return None
