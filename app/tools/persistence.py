from __future__ import annotations

from typing import Any

from app.storage.postgres import PostgresPaperStore
from app.tools.context import ToolContext, canonicalize_papers
from app.tools.embeddings import embed_papers


async def persist_papers(
    papers: list[dict[str, Any]] | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Embed and persist papers without exposing vectors to the language model.

    The embedding response stays inside Python: it is passed directly to the
    PostgreSQL adapter and only a small status object is returned to the agent.
    """

    state_ranked = (
        tool_context.state.get("ranked_papers") if tool_context is not None else None
    )
    authoritative_papers = (
        state_ranked if isinstance(state_ranked, list) else papers or []
    )
    candidates = canonicalize_papers(authoritative_papers, tool_context)
    if not candidates:
        return {"status": "skipped", "paper_count": 0}
    embeddings = await embed_papers(candidates)
    store = PostgresPaperStore()
    try:
        await store.create_schema()
        await store.upsert_papers(candidates, embeddings=embeddings)
    finally:
        await store.close()
    return {"status": "stored", "paper_count": len(candidates)}
