from google.adk.agents import LoopAgent, SequentialAgent

from app.agents.critic import critic_agent
from app.agents.dispatcher import dispatcher_agent
from app.agents.researcher import researcher_agent
from app.agents.synthesizer import synthesizer_agent

research_loop = LoopAgent(
    name="research_critic_loop",
    description="Runs the researcher and critic for up to three bounded iterations.",
    sub_agents=[researcher_agent, critic_agent],
    max_iterations=3,
)


research_workflow = SequentialAgent(
    name="research_workflow",
    description="Dispatches intent, runs bounded research and critique, then synthesizes the report.",
    sub_agents=[dispatcher_agent, research_loop, synthesizer_agent],
)
