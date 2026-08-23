from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ResearchIntent(BaseModel):
    """Structured intent accumulated during the clarification conversation."""

    original_question: str
    clarified_question: str | None = None
    core_concepts: list[str] = Field(default_factory=list)
    refinement_concepts: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None
    target_paper_count: int = 5


class PaperCandidate(BaseModel):
    """Paper metadata plus deterministic ranking and critic fields."""

    arxiv_id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    arxiv_url: str
    pdf_url: str | None = None
    published_at: date
    updated_at: date | None = None
    doi: str | None = None

    semantic_scholar_id: str | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    citation_source: str | None = None
    citation_retrieved_at: datetime | None = None

    relevance_score: float = 0.0
    citation_score: float = 0.0
    recency_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0

    critic_status: Literal["pending", "approved", "rejected"] = "pending"
    critic_reasons: list[str] = Field(default_factory=list)


class CriticDecision(BaseModel):
    arxiv_id: str
    status: Literal["approved", "rejected"]
    relevance_score: float
    reasons: list[str] = Field(default_factory=list)


class ResearchState(BaseModel):
    """Serializable state shared by the dispatcher, loop, and synthesizer."""

    original_question: str
    clarification_question: str | None = None
    clarification_answers: list[str] = Field(default_factory=list)
    intent: ResearchIntent | None = None
    search_queries: list[str] = Field(default_factory=list)
    candidates: list[PaperCandidate] = Field(default_factory=list)
    ranked_papers: list[PaperCandidate] = Field(default_factory=list)
    critic_decisions: list[CriticDecision] = Field(default_factory=list)
    approved_papers: list[PaperCandidate] = Field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3
    loop_complete: bool = False
    report: str | None = None
