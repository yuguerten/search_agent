from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.models import CriticDecision
from app.tools.context import ToolContext, canonicalize_papers
from app.tools.scope import evaluate_scope


def evaluate_candidates(
    papers: list[dict[str, Any]],
    target_count: int | None = None,
    min_relevance_score: float | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Apply deterministic critic checks and accumulate approvals by arXiv ID.

    ``target_count`` remains for backwards-compatible tool calls, but the
    configured target is authoritative and cannot be changed by the model.
    """

    settings = get_settings()
    target = settings.max_papers
    threshold = settings.min_relevance_score

    authoritative_papers = papers
    if tool_context is not None and tool_context.state.get("ranked_papers"):
        authoritative_papers = tool_context.state["ranked_papers"]
    canonical_papers = canonicalize_papers(authoritative_papers, tool_context)
    intent = (
        tool_context.state.get("research_intent") if tool_context is not None else None
    )
    decisions: list[CriticDecision] = []
    for paper in canonical_papers:
        reasons: list[str] = []
        if not paper.abstract:
            reasons.append("missing abstract")
        if paper.relevance_score < threshold:
            reasons.append("below relevance threshold")
        _, scope_reasons = evaluate_scope(paper, intent)
        reasons.extend(scope_reasons)
        status = "rejected" if reasons else "approved"
        decisions.append(
            CriticDecision(
                arxiv_id=paper.arxiv_id,
                status=status,
                relevance_score=paper.relevance_score,
                reasons=reasons or ["passed deterministic checks"],
            )
        )

    decisions_by_id = {}
    if tool_context is not None:
        decisions_by_id = {
            decision["arxiv_id"]: decision
            for decision in tool_context.state.get("critic_decisions", [])
            if isinstance(decision, dict) and decision.get("arxiv_id")
        }
    decisions_by_id.update(
        {decision.arxiv_id: decision.model_dump(mode="json") for decision in decisions}
    )

    previous_approved = {}
    if tool_context is not None:
        previous_approved = {
            paper["arxiv_id"]: paper
            for paper in tool_context.state.get("approved_papers", [])
            if isinstance(paper, dict) and paper.get("arxiv_id")
        }
    papers_by_id = {
        paper.arxiv_id: paper.model_dump(mode="json") for paper in canonical_papers
    }
    for arxiv_id, paper in previous_approved.items():
        current = papers_by_id.get(arxiv_id, paper)
        current["critic_status"] = "approved"
        current["critic_reasons"] = paper.get(
            "critic_reasons", ["passed deterministic checks"]
        )
        papers_by_id[arxiv_id] = current
    for paper in papers_by_id.values():
        if paper.get("arxiv_id") in decisions_by_id:
            decision = decisions_by_id[paper["arxiv_id"]]
            if decision["status"] == "approved":
                paper["critic_status"] = "approved"
                paper["critic_reasons"] = decision["reasons"]

    approved = [
        paper
        for paper in papers_by_id.values()
        if paper.get("critic_status") == "approved"
    ][:target]
    decisions_json = list(decisions_by_id.values())
    ready = len(approved) >= target
    if tool_context is not None:
        tool_context.state["critic_decisions"] = decisions_json
        tool_context.state["approved_papers"] = approved
        tool_context.state["target_paper_count"] = target
        tool_context.state["loop_complete"] = ready

    return {
        "decisions": decisions_json,
        "approved_papers": approved,
        "approved_count": len(approved),
        "target_count": target,
        "ready": ready,
    }
