import json

from google.adk.agents import Agent
from google.genai import types

from app.agents.model import build_llm
from app.config import get_settings


def _synthesizer_instruction(context) -> str:
    approved = context.state.get("approved_papers", [])
    target = get_settings().max_papers
    evidence = json.dumps(approved, ensure_ascii=False, default=str)
    return f"""You are the final academic synthesizer.

Use only the verified evidence JSON below. Do not create papers, authors, dates,
citation counts, URLs, findings, or references that are not present in it. Every
paper mentioned in the report must use its exact arXiv ID and title from the JSON.
Write a concise report with an introduction, paper-by-paper evidence, synthesis,
comparison, limitations, and research gaps. Cite papers as [n] and copy URLs
exactly from the evidence. The configured target is {target}; do not claim the
research is complete unless that many unique papers are present. Do not claim more
iterations than the configured loop maximum.

Verified evidence JSON:
{evidence}
"""


def guard_without_approved_papers(*, callback_context):
    approved = callback_context.state.get("approved_papers", [])
    target = get_settings().max_papers
    if (
        len({paper.get("arxiv_id") for paper in approved if isinstance(paper, dict)})
        >= target
    ):
        return None

    candidates = callback_context.state.get("candidates", [])
    maximum = get_settings().max_loop_iterations
    return types.Content(
        role="model",
        parts=[
            types.Part(
                text=(
                    "Research did not reach the configured paper target. "
                    f"Approved papers: {len(approved)}/{target}. Validated candidates collected: "
                    f"{len(candidates)}. The research loop was bounded to "
                    f"{maximum} iterations. No paper-specific claims or references "
                    "are included because the critic approved no papers."
                )
            )
        ],
    )


synthesizer_agent = Agent(
    name="synthesizer_agent",
    include_contents="none",
    model=build_llm(),
    instruction=_synthesizer_instruction,
    before_agent_callback=guard_without_approved_papers,
    output_key="final_report",
)
