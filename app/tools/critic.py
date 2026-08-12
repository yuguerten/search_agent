from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.models import CriticDecision
from app.tools.context import ToolContext, canonicalize_papers


def evaluate_candidates(
    papers: list[dict[str, Any]],
    target_count: int = 5,
    min_relevance_score: float | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Apply deterministic critic checks before semantic LLM review."""

    threshold = min_relevance_score
    if threshold is None:
        threshold = get_settings().min_relevance_score

    canonical_papers = canonicalize_papers(papers, tool_context)
    decisions: list[CriticDecision] = []
    for paper in canonical_papers:
        reasons: list[str] = []
        if not paper.abstract:
            reasons.append("missing abstract")
        if paper.relevance_score < threshold:
            reasons.append("below relevance threshold")
        status = "rejected" if reasons else "approved"
        decisions.append(
            CriticDecision(
                arxiv_id=paper.arxiv_id,
                status=status,
                relevance_score=paper.relevance_score,
                reasons=reasons or ["passed deterministic checks"],
            )
        )

    approved_ids = {
        decision.arxiv_id for decision in decisions if decision.status == "approved"
    }
    approved = [
        paper.model_copy(
            update={
                "critic_status": "approved",
                "critic_reasons": ["passed deterministic checks"],
            }
        ).model_dump(mode="json")
        for paper in canonical_papers
        if paper.arxiv_id in approved_ids
    ][:target_count]
    decisions_json = [decision.model_dump(mode="json") for decision in decisions]
    if tool_context is not None:
        tool_context.state["critic_decisions"] = decisions_json
        tool_context.state["approved_papers"] = approved

    return {
        "decisions": decisions_json,
        "approved_papers": approved,
        "approved_count": len(approved),
        "ready": len(approved) >= target_count,
    }
