from google.adk.agents import Agent

from app.agents.model import build_llm
from app.tools.critic import evaluate_candidates
from app.tools.loop import stop_research_loop

critic_agent = Agent(
    name="critic_agent",
    include_contents="none",
    model=build_llm(),
    instruction="""You are the paper-quality critic.

Review the ranked candidates against the structured research intent. Use the
evaluate_candidates tool for deterministic checks, then assess semantic alignment,
methodological relevance, and evidence quality. Return one structured decision per
candidate with approved or rejected status and concise reasons. Do not change
citation counts or publication dates. Approve at most the requested number of
papers. If at least five papers are approved, call stop_research_loop.""",
    tools=[evaluate_candidates, stop_research_loop],
)
