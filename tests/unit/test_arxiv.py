from types import SimpleNamespace

import httpx

from app.tools.arxiv import build_arxiv_query, parse_arxiv_feed, search_arxiv
from app.tools.semantic_scholar import _without_arxiv_version


def test_semantic_scholar_ids_drop_arxiv_version_suffix() -> None:
    assert _without_arxiv_version("2607.28611v1") == "2607.28611"


def test_build_arxiv_query_splits_natural_language_terms() -> None:
    assert build_arxiv_query("protein folding algorithms") == (
        "all:protein AND all:folding AND all:algorithms"
    )


def test_parse_arxiv_feed_normalizes_metadata() -> None:
    payload = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.12345v2</id>
        <title>  A   Research Paper  </title>
        <summary> An abstract with   whitespace. </summary>
        <published>2024-01-15T00:00:00Z</published>
        <updated>2024-02-01T00:00:00Z</updated>
        <author><name>Jane Doe</name></author>
        <link title="pdf" href="https://arxiv.org/pdf/2401.12345" />
      </entry>
    </feed>"""

    papers = parse_arxiv_feed(payload)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.12345v2"
    assert papers[0].title == "A Research Paper"
    assert papers[0].abstract == "An abstract with whitespace."
    assert papers[0].authors == ["Jane Doe"]


def test_build_arxiv_query_ignores_year_tokens() -> None:
    query = build_arxiv_query(
        "scaling laws training large language models 2024-08-13 to 2026-08-13"
    )

    assert "2024" not in query
    assert "2026" not in query


def test_build_arxiv_query_uses_topic_anchors_and_optional_terms() -> None:
    query = build_arxiv_query(
        "agentic systems technical architectures monitoring evaluation"
    )

    assert query == (
        "all:agentic AND all:systems AND "
        "(all:technical OR all:architectures OR all:monitoring OR all:evaluation)"
    )


def test_build_arxiv_query_relaxes_five_term_queries() -> None:
    query = build_arxiv_query("agentic technical monitoring real environments")

    assert query == (
        "all:agentic AND all:technical AND "
        "(all:monitoring OR all:real OR all:environments)"
    )


async def test_search_uses_session_plan_instead_of_model_query(monkeypatch) -> None:
    captured = {}

    async def fake_request(client, method, url, **kwargs):
        captured.update(kwargs)
        request = httpx.Request(method, url)
        return httpx.Response(
            200,
            request=request,
            text='<feed xmlns="http://www.w3.org/2005/Atom" />',
        )

    monkeypatch.setattr("app.tools.arxiv.request_with_retries", fake_request)
    context = SimpleNamespace(
        state={
            "search_queries": ["agentic systems monitoring"],
            "arxiv_search_calls": 0,
        }
    )

    await search_arxiv("machine learning healthcare", tool_context=context)

    assert captured["params"]["search_query"] == (
        "all:agentic AND all:systems AND all:monitoring"
    )
    assert context.state["active_search_query"] == "agentic systems monitoring"
