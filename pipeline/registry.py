"""
pipeline.registry — ToolRegistry
=================================
Central registry that maps tool names to tool instances and exposes
OpenAI-compatible function schemas for the LLM's `tools=` parameter.

Key responsibilities
--------------------
1. **Registration** — tools are added with `registry.register(tool)`.
2. **Schema exposure** — `registry.schemas_for(skill)` returns the list of
   OpenAI function-schema dicts the LLM should see for a given skill.
   If the skill has an `allowed_tools` whitelist, only those schemas are
   included; otherwise all registered tools are exposed.
3. **Execution** — `registry.execute(name, **kwargs)` looks up the tool by
   name and calls `tool.run(**kwargs)`, returning a plain string result.

This mirrors the LangChain `AgentExecutor` tool-dispatch pattern but is
minimal, dependency-free, and skill-aware.

Usage:
    from pipeline.registry import ToolRegistry
    from pipeline.tools   import BashTool, YouTubeSearchTool

    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(YouTubeSearchTool())

    # Get schemas for a skill that whitelists only bash
    schemas = registry.schemas_for(skill)   # → [bash_schema_dict]

    # Execute a tool call returned by the LLM
    result = registry.execute("bash", command="echo hello")  # → "hello"
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pipeline.skill_loader import SkillSchema


# ---------------------------------------------------------------------------
# Tool protocol — any object with name / description / schema / run qualifies
# ---------------------------------------------------------------------------

@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    schema: dict

    def run(self, **kwargs: Any) -> str:
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Maps tool names → tool instances; mediates schema exposure."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def register(self, tool: Tool) -> None:
        """Add *tool* to the registry under tool.name."""
        if not isinstance(tool, Tool):
            raise TypeError(
                f"{tool!r} does not satisfy the Tool protocol "
                "(needs: name, description, schema, run)"
            )
        self._tools[tool.name] = tool

    # ------------------------------------------------------------------ #
    # Schema exposure                                                      #
    # ------------------------------------------------------------------ #

    def schemas_for(self, skill: SkillSchema) -> list[dict]:
        """
        Return the OpenAI function-schema dicts the LLM should see when
        executing *skill*.

        If `skill.allowed_tools` is non-empty, only those tools are returned
        (unknown names are silently skipped).  If `allowed_tools` is empty,
        all registered tools are returned.
        """
        if skill.allowed_tools:
            return [
                self._tools[name].schema
                for name in skill.allowed_tools
                if name in self._tools
            ]
        return [tool.schema for tool in self._tools.values()]

    def all_schemas(self) -> list[dict]:
        """Return all registered tool schemas (used during skill selection)."""
        return [tool.schema for tool in self._tools.values()]

    # ------------------------------------------------------------------ #
    # Execution                                                            #
    # ------------------------------------------------------------------ #

    def execute(self, tool_name: str, **kwargs: Any) -> str:
        """
        Execute the named tool with the given keyword arguments.

        Parameters
        ----------
        tool_name : str
            Must match a registered tool's name.
        **kwargs
            Arguments forwarded to `tool.run()`.

        Returns
        -------
        str
            Tool output as a plain string.

        Raises
        ------
        KeyError
            If no tool with the given name is registered.
        """
        if tool_name not in self._tools:
            available = ", ".join(self._tools) or "(none)"
            raise KeyError(
                f"Tool '{tool_name}' is not registered. "
                f"Available tools: {available}"
            )
        return self._tools[tool_name].run(**kwargs)

    # ------------------------------------------------------------------ #
    # Introspection                                                         #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        names = list(self._tools)
        return f"<ToolRegistry tools={names}>"

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)
