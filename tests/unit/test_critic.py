from datetime import date

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
