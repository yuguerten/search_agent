from google.adk.tools.tool_context import ToolContext


def stop_research_loop(tool_context: ToolContext) -> dict[str, str]:
    """Stop the ADK loop when the critic has approved the target paper count."""

    approved = tool_context.state.get("approved_papers", [])
    target_count = tool_context.state.get("target_paper_count", 5)
    if len(approved) >= target_count:
        tool_context.actions.escalate = True
        return {"status": "complete", "approved_count": str(len(approved))}
    return {
        "status": "continue",
        "approved_count": str(len(approved)),
        "required_count": str(target_count),
    }
