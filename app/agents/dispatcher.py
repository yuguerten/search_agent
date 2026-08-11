from google.adk.agents import Agent

from app.agents.model import build_llm
from app.tools.intent import update_intent

dispatcher_agent = Agent(
    name="dispatcher_agent",
    model=build_llm(),
    instruction="""You are the workflow dispatcher.

Read the original question and all clarification answers from session state. Keep
the research intent structured and concise. After every user answer, call the
update_intent tool with the complete answer list. Produce search queries that
preserve the user's scope and constraints. Do not write the final report and do
not invent paper metadata. Start the research workflow only when the intent is
specific enough to search.""",
    tools=[update_intent],
    output_key="research_intent",
)
