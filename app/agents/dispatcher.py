import json

from google.adk.agents import Agent
from google.genai import types

from app.agents.model import build_llm
from app.tools.intent import update_intent


def publish_dispatch_status(*, callback_context):
    intent = callback_context.state.get("research_intent", {})
    if not isinstance(intent, dict):
        intent = {}
    status = {
        "status": "intent_ready",
        "keywords": intent.get("keywords", []),
        "search_queries": intent.get("search_queries", []),
        "start_date": intent.get("start_date"),
        "end_date": intent.get("end_date"),
    }
    callback_context.state["search_queries"] = status["search_queries"]
    return types.Content(
        role="model",
        parts=[types.Part(text=json.dumps(status, separators=(",", ":")))],
    )


dispatcher_agent = Agent(
    name="dispatcher_agent",
    model=build_llm(),
    description="Dispatches the research workflow after clarifications are complete.",
    instruction="""You are the workflow dispatcher.

Call update_intent once. The exact original question and clarification answers
captured from Context.user_content in session state are authoritative; the tool
will ignore any conflicting arguments reconstructed from conversation history.
Never include thoughts, transcript text, tool responses, agent labels, or
instructions. The tool's structured intent and deterministic search_queries are
authoritative. Do not create date-bearing queries, paper
metadata, scores, or a final report. Date filtering is optional and controlled by configuration; never invent a
historical date range. A Python callback publishes the authoritative intent status;
do not narrate search completion in prose.""",
    tools=[update_intent],
    after_agent_callback=publish_dispatch_status,
    output_key="dispatcher_status",
)
