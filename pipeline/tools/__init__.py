"""
pipeline.tools — concrete tool implementations.

Each tool exposes:
  .name        : str              — identifier used in OpenAI function schema
  .description : str              — shown to the model during tool selection
  .schema      : dict             — OpenAI-compatible function schema dict
  .run(**kwargs) -> str           — executes the tool, always returns str

Import:
    from pipeline.tools.bash    import BashTool
    from pipeline.tools.youtube import YouTubeSearchTool
"""

from pipeline.tools.bash import BashTool
from pipeline.tools.youtube import YouTubeSearchTool

__all__ = ["BashTool", "YouTubeSearchTool"]
