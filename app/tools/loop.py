from google.adk.tools.tool_context import ToolContext

from app.config import get_settings


def stop_research_loop(tool_context: ToolContext) -> dict[str, str]:
    """Stop only when the cumulative unique approval target is reached."""

    approved = tool_context.state.get("approved_papers", [])
    approved_ids = {
        paper.get("arxiv_id")
        for paper in approved
        if isinstance(paper, dict) and paper.get("arxiv_id")
    }
    target_count = get_settings().max_papers
    if len(approved_ids) >= target_count:
        tool_context.state["loop_complete"] = True
        tool_context.actions.escalate = True
        return {
            "status": "complete",
            "approved_count": str(len(approved_ids)),
            "target_count": str(target_count),
        }
    tool_context.state["loop_complete"] = False
    return {
        "status": "continue",
        "approved_count": str(len(approved_ids)),
        "required_count": str(target_count),
        "target_count": str(target_count),
    }
