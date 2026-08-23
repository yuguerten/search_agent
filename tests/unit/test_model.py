from app.agents.model import _limit_lmstudio_messages, build_llm


def test_lmstudio_message_limit_preserves_system_and_recent_messages() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old" * 100},
        {"role": "assistant", "content": "recent"},
    ]

    result = _limit_lmstudio_messages(messages, max_chars=80)

    assert result[0]["role"] == "system"
    assert result[-1]["content"] == "recent"
    assert len(result) < len(messages)

def test_openrouter_reasoning_is_disabled_by_default() -> None:
    llm = build_llm()

    assert llm.model == "openrouter/nvidia/nemotron-3.5-lightning:free"
    assert llm._additional_args["reasoning"] == {"enabled": False}
