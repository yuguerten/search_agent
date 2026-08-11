from google.adk.agents import Agent
from google.genai import types

from app.agents.model import build_llm
from app.agents.workflow import research_workflow


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, types.Content):
        return " ".join(part.text or "" for part in (value.parts or []) if part.text)
    return str(value or "")


def _latest_agent_text(callback_context) -> str:
    response_text = _content_text(callback_context.output)
    if response_text.strip():
        return response_text

    for event in reversed(getattr(callback_context.session, "events", [])):
        if event.author != callback_context.agent_name:
            continue
        response_text = _content_text(event.content)
        if response_text.strip():
            return response_text
    return ""


def route_completed_clarification(*, callback_context):
    """Route after a small local model announces completion in plain text.

    Some OpenAI-compatible local models describe a tool call instead of
    returning structured tool-call metadata. ADK can still perform the handoff
    through an event action emitted by this callback.
    """
    response_text = _latest_agent_text(callback_context).lower()
    completion_markers = (
        "clarification is complete",
        "clarification complete",
        "invoking transfer_to_agent",
        "transfer_to_agent",
    )
    if not any(marker in response_text for marker in completion_markers):
        return None

    callback_context.actions.transfer_to_agent = "research_workflow"
    return types.Content(
        role="model",
        parts=[
            types.Part(text="Clarification complete. Starting the research workflow.")
        ],
    )


clarifier_agent = Agent(
    name="clarifier_agent",
    description="Asks one targeted clarification question at a time before research begins.",
    model=build_llm(),
    instruction="""You are the clarification specialist for an academic research assistant.

Ask exactly one concise question per turn. Ask only for information that is still
missing from the user's research intent, such as scope, method, population,
comparison, or desired application. Do not search for papers yet. When the intent
is sufficiently specific, say that clarification is complete and hand off to the
dispatcher. When clarification is complete, invoke the transfer_to_agent tool for research_workflow. Do not describe the transfer in plain text. Never ask multiple questions in one message.""",
    sub_agents=[research_workflow],
    after_agent_callback=route_completed_clarification,
    output_key="clarification_question",
)
