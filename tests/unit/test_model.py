from app.agents.model import _limit_lmstudio_messages


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
