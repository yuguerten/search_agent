from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.models import PaperCandidate


async def embed_papers(papers: list[PaperCandidate]) -> list[list[float]]:
    """Create embeddings through LiteLLM, isolated from agent definitions."""

    from litellm import aembedding

    settings = get_settings()
    inputs = [f"{paper.title}\n{paper.abstract}" for paper in papers]
    response = await aembedding(model=settings.embedding_model, input=inputs)
    return [item["embedding"] for item in response["data"]]


async def embed_papers_tool(papers: list[dict[str, Any]]) -> list[list[float]]:
    """ADK tool wrapper for embedding normalized paper metadata."""

    return await embed_papers(
        [PaperCandidate.model_validate(paper) for paper in papers]
    )
