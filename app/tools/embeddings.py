from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.models import PaperCandidate


async def embed_papers(papers: list[PaperCandidate]) -> list[list[float]]:
    """Create embeddings through LiteLLM, isolated from agent definitions."""

    from litellm import aembedding

    settings = get_settings()
    inputs = [f"{paper.title}\n{paper.abstract}" for paper in papers]
    provider = settings.embedding_provider.lower()
    model = settings.embedding_model
    kwargs: dict[str, Any] = {}
    if provider == "lmstudio":
        if not model.startswith("openai/"):
            model = "openai/" + model
        kwargs = {
            "api_base": settings.embedding_api_base,
            "api_key": settings.embedding_api_key,
        }
    elif provider == "openrouter":
        if not model.startswith("openrouter/"):
            model = "openrouter/" + model
        kwargs = {
            "api_base": settings.openrouter_api_base,
            "api_key": settings.openrouter_api_key,
        }
    response = await aembedding(model=model, input=inputs, **kwargs)
    return [item["embedding"] for item in response["data"]]
