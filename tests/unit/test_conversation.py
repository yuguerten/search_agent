from types import SimpleNamespace

from google.genai import types

from app.tools.conversation import capture_user_research_input


def callback_context(text: str, invocation_id: str, state: dict) -> SimpleNamespace:
    return SimpleNamespace(
        invocation_id=invocation_id,
        state=state,
        user_content=types.Content(
            role="user",
            parts=[types.Part(text=text)],
        ),
    )


def test_capture_uses_literal_user_content_and_ignores_control_turns() -> None:
    state: dict = {}
    turns = [
        ("hello", "1"),
        ("I want to apply distillation on vision language models how?", "2"),
        ("any distillation technique", "3"),
        ("target model size", "4"),
        ("number of parameters, FLOPs", "5"),
        ("go", "6"),
    ]

    for text, invocation_id in turns:
        capture_user_research_input(
            callback_context=callback_context(text, invocation_id, state)
        )

    assert state["original_question"] == (
        "I want to apply distillation on vision language models how?"
    )
    assert state["clarification_answers"] == [
        "any distillation technique",
        "target model size",
        "number of parameters, FLOPs",
    ]


def test_capture_records_an_invocation_only_once() -> None:
    state: dict = {}
    context = callback_context("vision language model distillation", "same-id", state)

    capture_user_research_input(callback_context=context)
    capture_user_research_input(callback_context=context)

    assert state["original_question"] == "vision language model distillation"
    assert state["clarification_answers"] == []
    assert state["research_input_invocation_ids"] == ["same-id"]


def test_capture_starts_new_intent_after_completed_report() -> None:
    state = {
        "original_question": "old question",
        "clarification_answers": ["old answer"],
        "final_report": "completed report",
    }

    capture_user_research_input(
        callback_context=callback_context("new research topic", "new-id", state)
    )

    assert state["original_question"] == "new research topic"
    assert state["clarification_answers"] == []
    assert state["final_report"] is None

    capture_user_research_input(
        callback_context=callback_context("new constraint", "answer-id", state)
    )

    assert state["original_question"] == "new research topic"
    assert state["clarification_answers"] == ["new constraint"]
