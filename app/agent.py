from google.adk.agents import Agent
from google.adk.apps import App

from app.agents.clarifier import clarifier_agent
from app.agents.model import build_llm

root_agent = Agent(
    name="literature_research_dispatcher",
    rerun_on_resume=True,
    model=build_llm(),
    instruction="""You are the user-facing academic research coordinator.

Begin with the clarification agent and ask one question at a time. Preserve every
answer in session state. Once the dispatcher says the research intent is ready,
delegate to the research workflow. Return the synthesizer's final report to the
user. Do not perform paper search or synthesis yourself.""",
    sub_agents=[clarifier_agent],
)

app = App(root_agent=root_agent, name="app")
