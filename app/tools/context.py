"""Compatibility type for ADK tool context during framework-free unit tests."""

try:
    from google.adk.tools.tool_context import ToolContext
except ImportError:  # pragma: no cover - used only when ADK is not installed

    class ToolContext:  # type: ignore[no-redef]
        state: dict
        actions: object
