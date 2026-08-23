import json

from google.adk.agents import Agent
from google.genai import types

from app.agents.model import build_llm
from app.config import get_settings
from app.tools.critic import evaluate_candidates
from app.tools.loop import stop_research_loop


def run_deterministic_critic(*, callback_context):
    """Evaluate the researcher output without allowing model prose to override it."""

    ranked = callback_context.state.get("ranked_papers", [])
    if isinstance(ranked, list):
        evaluate_candidates(ranked, tool_context=callback_context)
    return publish_critic_status(callback_context=callback_context)


def publish_critic_status(*, callback_context):
    """Replace free-form critic narration with authoritative state-derived status."""

    loop_result = stop_research_loop(callback_context)
    approved = callback_context.state.get("approved_papers", [])
    status = {
        "status": loop_result["status"],
        "approved_count": len(
            {
                paper.get("arxiv_id")
                for paper in approved
                if isinstance(paper, dict) and paper.get("arxiv_id")
            }
        ),
        "target_count": int(loop_result["target_count"]),
        "min_relevance_score": get_settings().min_relevance_score,
        "approved_ids": sorted(
            {
                paper.get("arxiv_id")
                for paper in approved
                if isinstance(paper, dict) and paper.get("arxiv_id")
            }
        ),
    }
    callback_context.state["research_status"] = status
    return types.Content(
        role="model",
        parts=[types.Part(text=json.dumps(status, separators=(",", ":")))],
    )


critic_agent = Agent(
    name="critic_agent",
    include_contents="none",
    model=build_llm(),
    instruction="""You are the paper-quality critic.

Use the ranked candidates and stored dispatcher intent. Call evaluate_candidates
first; its decisions, scope checks, threshold, target count, and approved_papers are
authoritative. Do not create a second approval list, alter scores, or change the
configured target. Do not change citation counts or publication dates. Call
stop_research_loop after evaluation. The post-agent status is generated from
Python state, so do not claim success or failure in prose.""",
    tools=[evaluate_candidates, stop_research_loop],
    before_agent_callback=run_deterministic_critic,
    output_key="critic_status",
)
