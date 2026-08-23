from types import SimpleNamespace

import pytest

from app.tools import persistence


def paper() -> dict[str, object]:
    return {
        "arxiv_id": "2401.12345",
        "title": "Scaling laws",
        "abstract": "A study of scaling laws.",
        "authors": ["Author"],
        "arxiv_url": "https://arxiv.org/abs/2401.12345",
        "published_at": "2025-01-01",
    }


@pytest.mark.asyncio
async def test_persistence_embeds_inside_python_without_returning_vectors(monkeypatch):
    vectors = [[0.1, 0.2, 0.3]]
    calls: dict[str, object] = {}

    async def fake_embed(papers):
        calls["embed_papers"] = papers
        return vectors

    class FakeStore:
        async def create_schema(self):
            calls["create_schema"] = True

        async def upsert_papers(self, papers, embeddings=None):
            calls["papers"] = papers
            calls["embeddings"] = embeddings

        async def close(self):
            calls["close"] = True

    monkeypatch.setattr(persistence, "embed_papers", fake_embed)
    monkeypatch.setattr(persistence, "PostgresPaperStore", FakeStore)

    result = await persistence.persist_papers([paper()], SimpleNamespace(state={}))

    assert result == {"status": "stored", "paper_count": 1}
    assert calls["embeddings"] is vectors
    assert calls["close"] is True


@pytest.mark.asyncio
async def test_persistence_skips_authoritative_empty_ranked_state() -> None:
    context = SimpleNamespace(state={"ranked_papers": []})

    result = await persistence.persist_papers(
        papers=["not a paper object"],
        tool_context=context,
    )

    assert result == {"status": "skipped", "paper_count": 0}
