from types import SimpleNamespace

from app.agents.synthesizer import (
    _synthesizer_instruction,
    guard_without_approved_papers,
    publish_final_status,
)


def approved(arxiv_id: str) -> dict[str, object]:
    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "abstract": "Verified abstract",
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def test_incomplete_synthesis_receives_approved_evidence() -> None:
    context = SimpleNamespace(state={"approved_papers": [approved("2401.12345")]})

    prompt = _synthesizer_instruction(context)

    assert "incomplete" in prompt
    assert "2401.12345" in prompt
    assert "do not pretend the target was reached" in prompt


def test_synthesizer_guard_allows_partial_evidence() -> None:
    context = SimpleNamespace(state={"approved_papers": [approved("2401.12345")]})

    assert guard_without_approved_papers(callback_context=context) is None


def test_final_status_is_state_derived() -> None:
    context = SimpleNamespace(
        state={"approved_papers": [approved("2401.12345")]},
        output="model claimed five papers",
    )

    result = publish_final_status(callback_context=context)
    text = result.parts[0].text

    assert text.startswith("Research status: incomplete (1/5")
    assert "model claimed five papers" in text
