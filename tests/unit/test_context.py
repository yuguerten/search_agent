from types import SimpleNamespace

from app.tools.context import canonicalize_papers


def full_paper(arxiv_id: str, title: str) -> dict[str, object]:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": "Abstract",
        "authors": ["Author"],
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        "published_at": "2025-01-01",
    }


def test_canonicalize_matches_truncated_title_from_session_state() -> None:
    context = SimpleNamespace(
        state={
            "candidates": [
                full_paper("2401.12345", "Scaling Laws in Vision and Language Models"),
                full_paper("2401.12346", "Optimization for Neural Networks"),
            ]
        }
    )

    result = canonicalize_papers(
        [{"title": "Scaling Laws in Vision...", "relevance_score": 0.95}],
        context,
    )

    assert result[0].arxiv_id == "2401.12345"
    assert result[0].relevance_score == 0.95
