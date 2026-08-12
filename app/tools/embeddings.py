from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.models import PaperCandidate
from app.tools.context import ToolContext, canonicalize_papers


async def embed_papers(papers: list[PaperCandidate]) -> list[list[float]]:
    """Create embeddings through LiteLLM, isolated from agent definitions."""

    from litellm import aembedding

    settings = get_settings()
    inputs = [f"{paper.title}\n{paper.abstract}" for paper in papers]
    model = settings.embedding_model
    kwargs: dict[str, Any] = {}
    if settings.llm_provider.lower() == "lmstudio":
        if not model.startswith("openai/"):
            model = "openai/" + model
        kwargs = {
            "api_base": settings.litellm_api_base,
            "api_key": settings.litellm_api_key,
        }
    response = await aembedding(model=model, input=inputs, **kwargs)
    return [item["embedding"] for item in response["data"]]


async def embed_papers_tool(
    papers: list[dict[str, Any]], tool_context: ToolContext | None = None
) -> list[list[float]]:
    """ADK tool wrapper for embedding normalized paper metadata."""

    return await embed_papers(canonicalize_papers(papers, tool_context))
