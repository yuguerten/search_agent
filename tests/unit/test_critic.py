from datetime import date
from types import SimpleNamespace

from app.agents.critic import run_deterministic_critic
from app.models import PaperCandidate
from app.tools.critic import evaluate_candidates


def test_critic_rejects_missing_abstract() -> None:
    paper = PaperCandidate(
        arxiv_id="2401.12345",
        title="A paper",
        arxiv_url="https://arxiv.org/abs/2401.12345",
        published_at=date(2025, 1, 1),
        relevance_score=0.8,
    )

    result = evaluate_candidates([paper.model_dump(mode="json")])

    assert result["approved_count"] == 0
    assert result["decisions"][0]["status"] == "rejected"


def complete_paper(arxiv_id: str) -> PaperCandidate:
    return PaperCandidate(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="A relevant abstract.",
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=date(2025, 1, 1),
        relevance_score=0.8,
    )


def test_critic_accumulates_approvals_across_iterations() -> None:
    context = SimpleNamespace(state={})

    first = evaluate_candidates(
        [complete_paper("2401.12345").model_dump(mode="json")],
        tool_context=context,
    )
    second = evaluate_candidates(
        [complete_paper("2401.12346").model_dump(mode="json")],
        tool_context=context,
    )

    assert first["approved_count"] == 1
    assert second["approved_count"] == 2
    assert {paper["arxiv_id"] for paper in context.state["approved_papers"]} == {
        "2401.12345",
        "2401.12346",
    }


def test_critic_keeps_previous_approval_when_reseen() -> None:
    context = SimpleNamespace(state={})
    paper = complete_paper("2401.12347")

    evaluate_candidates([paper.model_dump(mode="json")], tool_context=context)
    paper.relevance_score = 0.0
    result = evaluate_candidates([paper.model_dump(mode="json")], tool_context=context)

    assert result["approved_count"] == 1
    assert result["approved_papers"][0]["critic_status"] == "approved"


def test_critic_uses_configured_threshold_instead_of_model_argument() -> None:
    context = SimpleNamespace(state={})
    paper = complete_paper("2401.12348")
    paper.relevance_score = 0.4

    result = evaluate_candidates(
        [paper.model_dump(mode="json")],
        target_count=3,
        min_relevance_score=0.8,
        tool_context=context,
    )

    assert result["approved_count"] == 1
    assert result["target_count"] == 5

def test_deterministic_critic_publishes_configured_status_without_llm() -> None:
    context = SimpleNamespace(
        state={"ranked_papers": [complete_paper("2401.12349").model_dump(mode="json")]},
        actions=SimpleNamespace(escalate=False),
    )

    result = run_deterministic_critic(callback_context=context)
    status = result.parts[0].text

    assert '"min_relevance_score":0.35' in status
    assert '"target_count":5' in status
    assert '"approved_count":1' in status
    assert "0.85" not in status
