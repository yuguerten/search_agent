from __future__ import annotations

from typing import Any

from app.models import PaperCandidate
from app.storage.postgres import PostgresPaperStore


async def persist_papers(
    papers: list[dict[str, Any]],
    embeddings: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Persist ranked papers and embeddings in PostgreSQL with pgvector."""

    candidates = [PaperCandidate.model_validate(paper) for paper in papers]
    store = PostgresPaperStore()
    try:
        await store.create_schema()
        await store.upsert_papers(candidates, embeddings=embeddings)
    finally:
        await store.close()
    return {"status": "stored", "paper_count": len(candidates)}
