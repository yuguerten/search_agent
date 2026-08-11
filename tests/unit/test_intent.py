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
