from datetime import date, timedelta
from types import SimpleNamespace

from app.config import get_settings
from app.tools.intent import extract_keyword_candidates, update_intent


def test_extract_keyword_candidates_removes_common_words() -> None:
    keywords = extract_keyword_candidates("What are recent agentic research workflows?")

    assert "what" not in keywords
    assert "agentic" in keywords
    assert "workflows" in keywords


def test_update_intent_is_serializable() -> None:
    intent = update_intent(
        "Find recent papers about agentic research",
        ["Focus on evaluation and multi-agent systems"],
    )

    assert intent["original_question"] == "Find recent papers about agentic research"
    assert "evaluation" in intent["keywords"]
    assert intent["target_paper_count"] == 5
    assert len(intent["search_queries"]) == 3
    assert "want" not in intent["keywords"]
    assert "state" not in intent["keywords"]
    assert all(
        "2024" not in query and "2026" not in query
        for query in intent["search_queries"]
    )
    today = date.today()
    expected_start = today - timedelta(days=get_settings().recent_days)
    assert intent["start_date"] == expected_start.isoformat()
    assert intent["end_date"] == today.isoformat()


def test_update_intent_persists_structured_state() -> None:
    context = SimpleNamespace(state={})

    result = update_intent(
        "Find scaling laws for language model training",
        ["Focus on performance"],
        tool_context=context,
    )

    assert context.state["research_intent"] == result
    assert context.state["search_queries"] == result["search_queries"]


def test_intent_removes_state_of_the_art_boilerplate() -> None:
    keywords = extract_keyword_candidates(
        "I want the state of the art about agentic systems"
    )

    assert keywords == ["agentic", "systems"]


def test_intent_ignores_nonrestrictive_domain_answer() -> None:
    keywords = extract_keyword_candidates(
        "I am open to any domain or industry not important"
    )

    assert keywords == []


def test_update_intent_resets_previous_research_run_state() -> None:
    context = SimpleNamespace(
        state={
            "research_intent": {"original_question": "old topic"},
            "candidates": [{"arxiv_id": "old"}],
            "ranked_papers": [{"arxiv_id": "old"}],
            "approved_papers": [{"arxiv_id": "old"}],
            "critic_decisions": [{"arxiv_id": "old"}],
            "arxiv_search_calls": 4,
            "arxiv_query_cache": {"old": []},
            "loop_complete": True,
            "report": "old report",
        }
    )

    update_intent("new agentic systems topic", [], tool_context=context)

    assert context.state["research_intent"]["original_question"] == (
        "new agentic systems topic"
    )
    assert context.state["candidates"] == []
    assert context.state["approved_papers"] == []
    assert context.state["arxiv_query_cache"] == {}
    assert context.state["loop_complete"] is False
    assert context.state["report"] is None
