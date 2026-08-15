from datetime import date

from app.models import PaperCandidate
from app.tools.scope import evaluate_scope


def paper(title: str, abstract: str) -> PaperCandidate:
    return PaperCandidate(
        arxiv_id="2401.12345",
        title=title,
        abstract=abstract,
        arxiv_url="https://arxiv.org/abs/2401.12345",
        published_at=date(2025, 1, 1),
    )


def test_scope_is_domain_agnostic_for_an_arbitrary_topic() -> None:
    intent = {
        "original_question": "protein folding algorithms",
        "keywords": ["protein", "folding", "algorithm"],
    }

    accepted, reasons = evaluate_scope(
        paper(
            "Efficient protein folding algorithms",
            "We compare algorithms for predicting protein structure.",
        ),
        intent,
    )

    assert accepted is True
    assert reasons == []


def test_scope_rejects_low_overlap_without_subject_hardcoding() -> None:
    intent = {
        "original_question": "protein folding algorithms",
        "keywords": ["protein", "folding", "algorithm"],
    }

    accepted, reasons = evaluate_scope(
        paper("Solar energy forecasting", "We study weather prediction models."),
        intent,
    )

    assert accepted is False
    assert reasons == ["no user-intent keywords found in title or abstract"]


def test_scope_accepts_scaling_example_without_special_case() -> None:
    intent = {
        "original_question": "scaling laws in large language model training",
        "keywords": ["scaling", "laws", "large", "language", "model", "training"],
    }

    accepted, reasons = evaluate_scope(
        paper(
            "Scaling laws for language model training",
            "We study training compute, model size, and performance in large language models.",
        ),
        intent,
    )

    assert accepted is True
    assert reasons == []
