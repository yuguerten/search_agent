import json
from typing import Any

from google.adk.models.lite_llm import LiteLlm

from app.config import get_settings

_MAX_LMSTUDIO_MESSAGE_CHARS = 120_000


def _normalize_message_content(content: Any) -> str | list[dict[str, Any]]:
    """Convert ADK content values to LM Studio's accepted content shapes."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for block in content:
            if hasattr(block, "model_dump"):
                block = block.model_dump(exclude_none=False)
            elif not isinstance(block, dict):
                block = {"type": "text", "text": str(block)}
            blocks.append(block)
        return blocks
    if hasattr(content, "model_dump"):
        content = content.model_dump(exclude_none=False)
    return json.dumps(content, default=str)


def _normalize_lmstudio_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Normalize ADK messages for LM Studio's OpenAI-compatible API."""

    normalized: list[dict[str, Any]] = []
    for raw_message in messages or []:
        if hasattr(raw_message, "model_dump"):
            message = raw_message.model_dump(exclude_none=False)
        else:
            message = dict(raw_message)
        if message.get("role") == "tool_responses":
            message["role"] = "tool"
        message["content"] = _normalize_message_content(message.get("content"))
        normalized.append(message)
    return normalized


def _limit_lmstudio_messages(
    messages: list[dict[str, Any]],
    max_chars: int = _MAX_LMSTUDIO_MESSAGE_CHARS,
) -> list[dict[str, Any]]:
    """Keep recent context bounded so local models do not exceed their window."""

    if sum(len(json.dumps(message, default=str)) for message in messages) <= max_chars:
        return messages

    system_messages = [
        message for message in messages if message.get("role") == "system"
    ]
    recent: list[dict[str, Any]] = []
    used = sum(len(json.dumps(message, default=str)) for message in system_messages)
    for message in reversed(messages):
        if message.get("role") == "system":
            continue
        size = len(json.dumps(message, default=str))
        if used + size > max_chars and recent:
            break
        recent.append(message)
        used += size
    recent.reverse()
    return [*system_messages, *recent]


def _configure_lmstudio_compatibility() -> None:
    """Patch only the ADK-to-LiteLLM boundary for LM Studio requests."""

    import google.adk.models.lite_llm as adk_lite_llm

    if getattr(adk_lite_llm, "_lmstudio_compatibility_installed", False):
        return

    original_acompletion = adk_lite_llm.acompletion

    async def compatible_acompletion(*args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            normalized = _normalize_lmstudio_messages(kwargs["messages"])
            kwargs["messages"] = _limit_lmstudio_messages(normalized)
        return await original_acompletion(*args, **kwargs)

    adk_lite_llm.acompletion = compatible_acompletion
    adk_lite_llm._lmstudio_compatibility_installed = True


def build_llm() -> LiteLlm:
    """Create the LiteLLM-backed model used by every specialist agent."""

    settings = get_settings()
    if settings.llm_provider.lower() == "lmstudio":
        _configure_lmstudio_compatibility()

    provider = settings.llm_provider.lower()
    model_name = settings.litellm_model
    api_base = settings.litellm_api_base
    api_key = settings.litellm_api_key

    if provider == "lmstudio":
        if not model_name.startswith("openai/"):
            model_name = f"openai/{model_name}"
    elif provider == "openrouter":
        if not model_name.startswith("openrouter/"):
            model_name = f"openrouter/{model_name}"
        api_base = settings.openrouter_api_base
        api_key = settings.openrouter_api_key

    kwargs = {
        "model": model_name,
        "api_base": api_base,
        "api_key": api_key,
    }
    return LiteLlm(**kwargs)
