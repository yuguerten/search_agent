from datetime import date
from types import SimpleNamespace

from app.models import PaperCandidate
from app.tools.ranking import rank_papers, rank_papers_tool, recent_papers


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


def test_rank_tool_uses_dispatcher_keywords_from_state() -> None:
    papers = [
        paper("a", date(2025, 1, 1), 10).model_dump(mode="json"),
        paper("b", date(2025, 6, 1), 1).model_dump(mode="json"),
    ]
    context = SimpleNamespace(
        state={"research_intent": {"keywords": ["unrelated-term"]}}
    )

    result = rank_papers_tool(
        papers,
        keywords=["agentic", "research", "workflow"],
        as_of="2026-01-01",
        tool_context=context,
    )

    assert len(result) == 2
    assert context.state["ranked_papers"] == result


def test_rank_tool_uses_core_topic_instead_of_optional_refinements() -> None:
    papers = [paper("a", date(2025, 1, 1), 10).model_dump(mode="json")]
    context = SimpleNamespace(
        state={
            "research_intent": {
                "core_concepts": ["agentic systems"],
                "refinement_concepts": ["observability", "failure detection"],
                "keywords": ["unrelated-term"],
            }
        }
    )

    result = rank_papers_tool(
        papers,
        keywords=["unrelated-term"],
        as_of="2026-01-01",
        tool_context=context,
    )

    assert result[0]["relevance_score"] > 0


def test_rank_tool_ignores_reconstructed_papers_when_candidates_are_in_state() -> None:
    candidate = paper("state-paper", date(2025, 1, 1), 10).model_dump(mode="json")
    context = SimpleNamespace(
        state={
            "candidates": [candidate],
            "research_intent": {"core_concepts": ["agentic systems"]},
        }
    )

    result = rank_papers_tool(
        papers=["not a paper object"],
        tool_context=context,
    )

    assert [item["arxiv_id"] for item in result] == ["state-paper"]


def test_relevance_is_zero_when_no_keywords_match() -> None:
    result = rank_papers(
        [paper("irrelevant", date(2025, 1, 1), 10)],
        keywords=["quantum", "biology"],
        as_of=date(2026, 1, 1),
        recent_days=730,
    )

    assert result[0].relevance_score == 0.0


def test_recent_filter_can_be_disabled() -> None:
    papers = [paper("old", date(2020, 1, 1), 100)]

    result = recent_papers(papers, as_of=date(2026, 1, 1), recent_days=None)

    assert [item.arxiv_id for item in result] == ["old"]
