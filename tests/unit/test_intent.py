from types import SimpleNamespace

from app.tools.intent import (
    build_search_queries,
    extract_concept_candidates,
    extract_keyword_candidates,
    update_intent,
)


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
    assert intent["start_date"] is None
    assert intent["end_date"] is None


def test_update_intent_persists_structured_state() -> None:
    context = SimpleNamespace(state={})

    result = update_intent(
        "Find scaling laws for language model training",
        ["Focus on performance"],
        tool_context=context,
    )

    assert context.state["research_intent"] == result
    assert context.state["search_queries"] == result["search_queries"]


def test_update_intent_prefers_literal_inputs_captured_in_session_state() -> None:
    context = SimpleNamespace(
        state={
            "original_question": (
                "I want to apply distillation on vision language models"
            ),
            "clarification_answers": [
                "any distillation technique",
                "target model size",
                "number of parameters, FLOPs",
            ],
        }
    )

    result = update_intent(
        "For context: dispatcher called tool with parameters",
        ["transcript text"],
        tool_context=context,
    )

    assert result["original_question"] == (
        "I want to apply distillation on vision language models"
    )
    assert result["clarified_question"] == (
        "any distillation technique target model size number of parameters, FLOPs"
    )
    assert "vision" in result["keywords"]
    assert "language" in result["keywords"]
    assert "distillation" in result["keywords"]
    assert "dispatcher" not in result["keywords"]


def test_failed_conversation_builds_phrase_aware_task_queries() -> None:
    result = update_intent(
        (
            "I want to apply distillation on vision language models, "
            "I want state of the art at the moment"
        ),
        [
            "Multimodal approach. All of them please: image captioning and "
            "visual question answering. Any target model size."
        ],
    )

    assert result["core_concepts"] == ["distillation", "vision language models"]
    assert result["refinement_concepts"] == [
        "multimodal",
        "image captioning",
        "visual question answering",
    ]
    assert all(
        '"distillation" AND "vision language models"' in query
        for query in result["search_queries"]
    )
    assert any("image captioning" in query for query in result["search_queries"])
    assert any(
        "visual question answering" in query for query in result["search_queries"]
    )
    assert all("please" not in query for query in result["search_queries"])


def test_conversational_preamble_and_non_choice_do_not_become_queries() -> None:
    context = SimpleNamespace(
        state={
            "original_question": (
                "Hello i am curious to know more about knowledge distillation "
                "applied for vision language models"
            ),
            "clarification_answers": [
                "I would prefer all of them because im not in a position to pick one",
                "deployment on edge",
            ],
        }
    )

    result = update_intent(
        "dispatcher reconstructed the conversation incorrectly",
        ["ignore this transcript"],
        tool_context=context,
    )

    assert result["core_concepts"] == [
        "knowledge distillation",
        "vision language models",
    ]
    assert result["refinement_concepts"] == ["deployment edge"]
    assert result["search_queries"] == [
        '"knowledge distillation" AND "vision language models"',
        '"knowledge distillation" AND "vision language models" AND "deployment edge"',
        '"knowledge distillation"',
    ]
    rejected_noise = {"hello", "curious", "more", "because", "position", "pick"}
    assert rejected_noise.isdisjoint(result["keywords"])


def test_concept_and_query_generation_is_domain_agnostic() -> None:
    biology = update_intent(
        "Find recent methods for protein folding",
        ["Focus on structure prediction and evaluation benchmarks."],
    )
    public_health = update_intent(
        "Study urban heat and cardiovascular mortality",
        ["Include longitudinal studies and vulnerable populations."],
    )

    assert biology["core_concepts"] == ["protein folding"]
    assert biology["search_queries"] == [
        '"protein folding"',
        '"protein folding" AND "structure prediction"',
        '"protein folding" AND "evaluation benchmarks"',
    ]
    assert public_health["core_concepts"] == [
        "urban heat",
        "cardiovascular mortality",
    ]
    assert "include" not in public_health["keywords"]
    assert all(
        '"urban heat" AND "cardiovascular mortality"' in query
        for query in public_health["search_queries"]
    )


def test_uncertain_domain_conversation_keeps_scientific_content_only() -> None:
    result = update_intent(
        "what's the state of the art architecture of time series classification?",
        [
            "I have bacteria trajectories and i want to classify them, tbh i "
            "dunno which domain is this? biomedical as well?",
            "like i have motion and appearance information of my bacteria as a "
            "time serie",
            "yeah for motion i have speed acceleratio.. appearance i have shape info",
        ],
    )

    assert result["core_concepts"] == ["time series classification"]
    noise = {"architecture", "tbh", "dunno", "domain", "my", "yeah", "well"}
    assert noise.isdisjoint(result["keywords"])
    refinements = " ".join(result["refinement_concepts"])
    assert "bacteria trajectories" in refinements
    assert "motion" in refinements
    assert "appearance" in refinements
    assert all(
        '"time series classification"' in query for query in result["search_queries"]
    )
    assert any("bacteria trajectories" in query for query in result["search_queries"])
    assert any("motion appearance" in query for query in result["search_queries"])


def test_query_builder_preserves_phrases_without_domain_rules() -> None:
    concepts = extract_concept_candidates(
        "Compare graph neural networks and molecular property prediction"
    )
    queries = build_search_queries(concepts[:1], concepts[1:])

    assert queries[0] == '"graph neural networks"'
    assert any('"molecular property prediction"' in query for query in queries)


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


def test_intent_discards_orchestration_transcript_words() -> None:
    keywords = extract_keyword_candidates(
        "For context clarifier agent said topic problem you exploring "
        "evolution attention mechanism natural language processing"
    )

    assert keywords == [
        "evolution",
        "attention",
        "mechanism",
        "natural",
        "language",
        "processing",
    ]
