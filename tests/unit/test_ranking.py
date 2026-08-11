from datetime import date

from app.models import PaperCandidate
from app.tools.ranking import rank_papers, recent_papers


def paper(arxiv_id: str, published_at: date, citations: int) -> PaperCandidate:
    return PaperCandidate(
        arxiv_id=arxiv_id,
        title="Agentic research workflow",
        abstract="Research agents retrieve and critique papers.",
        authors=["Author"],
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=published_at,
        citation_count=citations,
    )


def test_recent_filter_excludes_old_papers() -> None:
    papers = [
        paper("new", date(2025, 1, 1), 2),
        paper("old", date(2023, 1, 1), 100),
    ]

    result = recent_papers(papers, as_of=date(2026, 1, 1), recent_days=730)

    assert [item.arxiv_id for item in result] == ["new"]


def test_rank_papers_returns_descending_scores() -> None:
    papers = [
        paper("a", date(2025, 1, 1), 10),
        paper("b", date(2025, 6, 1), 1),
    ]

    result = rank_papers(
        papers,
        keywords=["agentic", "research", "workflow"],
        as_of=date(2026, 1, 1),
        recent_days=730,
    )

    assert len(result) == 2
    assert result[0].final_score >= result[1].final_score
    assert all(0.0 <= item.final_score <= 1.0 for item in result)
