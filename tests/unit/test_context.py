from types import SimpleNamespace

import pytest

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


def test_canonicalize_accepts_single_mapping_instead_of_list() -> None:
    payload = full_paper("2401.12345", "Scaling Laws")

    result = canonicalize_papers(payload)

    assert [paper.arxiv_id for paper in result] == ["2401.12345"]


def test_canonicalize_decodes_json_list() -> None:
    context = SimpleNamespace(
        state={"candidates": [full_paper("2401.12345", "Scaling Laws")]}
    )

    result = canonicalize_papers('[{"arxiv_id": "2401.12345"}]', context)

    assert [paper.title for paper in result] == ["Scaling Laws"]


def test_canonicalize_recovers_string_id_and_title_from_state() -> None:
    context = SimpleNamespace(
        state={
            "candidates": [
                full_paper("2401.12345", "Scaling Laws"),
                full_paper("2401.12346", "Optimization for Neural Networks"),
            ]
        }
    )

    result = canonicalize_papers(
        ["2401.12345", "Optimization for Neural Networks"], context
    )

    assert [paper.arxiv_id for paper in result] == ["2401.12345", "2401.12346"]


def test_canonicalize_rejects_invalid_container_with_clear_error() -> None:
    with pytest.raises(ValueError, match="Paper input must be"):
        canonicalize_papers(42)
