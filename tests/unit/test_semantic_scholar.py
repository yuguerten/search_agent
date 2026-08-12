from datetime import date
from types import SimpleNamespace

import httpx

from app.tools.semantic_scholar import enrich_with_semantic_scholar


def paper(arxiv_id: str) -> dict[str, object]:
    return {
        "arxiv_id": arxiv_id,
        "title": "Scaling laws",
        "abstract": "A study of scaling laws.",
        "authors": ["Author"],
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        "published_at": date(2025, 1, 1).isoformat(),
    }


async def test_rate_limit_returns_papers_without_fabricating_citations(monkeypatch):
    async def fake_request(*args, **kwargs):
        request = httpx.Request("POST", "https://example.test")
        return httpx.Response(429, request=request)

    monkeypatch.setattr("app.tools.semantic_scholar.request_with_retries", fake_request)
    context = SimpleNamespace(state={})

    result = await enrich_with_semantic_scholar([paper("2401.12345v1")], context)

    assert len(result) == 1
    assert result[0]["arxiv_id"] == "2401.12345v1"
    assert result[0]["citation_count"] is None
    assert context.state["semantic_scholar_rate_limited"] is True


async def test_cached_metadata_avoids_second_request(monkeypatch):
    calls = 0

    async def fake_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        request = httpx.Request("POST", "https://example.test")
        return httpx.Response(
            200,
            request=request,
            json=[
                {
                    "paperId": "s2-id",
                    "externalIds": {"DOI": "10.1234/example"},
                    "citationCount": 12,
                    "influentialCitationCount": 3,
                }
            ],
        )

    monkeypatch.setattr("app.tools.semantic_scholar.request_with_retries", fake_request)
    context = SimpleNamespace(state={})

    first = await enrich_with_semantic_scholar([paper("2401.12345v1")], context)
    second = await enrich_with_semantic_scholar([paper("2401.12345v1")], context)

    assert calls == 1
    assert first[0]["citation_count"] == 12
    assert second[0]["semantic_scholar_id"] == "s2-id"


async def test_malformed_model_payload_uses_canonical_session_candidates(monkeypatch):
    async def fake_request(*args, **kwargs):
        request = httpx.Request("POST", "https://example.test")
        return httpx.Response(429, request=request)

    monkeypatch.setattr("app.tools.semantic_scholar.request_with_retries", fake_request)
    context = SimpleNamespace(state={"candidates": [paper("2401.12345v1")]})
    malformed = [
        {
            "title": "Scaling Laws i...",
            "authors": "Nischay Dhankhar, Dos Bah",
        }
    ]

    result = await enrich_with_semantic_scholar(malformed, context)

    assert result[0]["arxiv_id"] == "2401.12345v1"
    assert result[0]["authors"] == ["Author"]
    assert context.state["candidates"][0]["arxiv_id"] == "2401.12345v1"
