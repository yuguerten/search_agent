from __future__ import annotations

from typing import Any

from app.storage.postgres import PostgresPaperStore
from app.tools.context import ToolContext, canonicalize_papers


async def persist_papers(
    papers: list[dict[str, Any]],
    embeddings: list[list[float]] | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Persist ranked papers and embeddings in PostgreSQL with pgvector."""

    candidates = canonicalize_papers(papers, tool_context)
    store = PostgresPaperStore()
    try:
        await store.create_schema()
        await store.upsert_papers(candidates, embeddings=embeddings)
    finally:
        await store.close()
    return {"status": "stored", "paper_count": len(candidates)}
