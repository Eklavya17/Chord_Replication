"""
pipeline.tools.youtube — YouTubeSearchTool
==========================================
Searches YouTube for videos matching a query and returns a list of URLs.

Wraps the `youtube-search` PyPI package (YouTubeSearch class).
Falls back to a stub if the package is not installed, so the pipeline
can still be imported and tested without the live dependency.

Input format expected by the tool (matches the CHORD paper tool schema):
    "<topic>, <num_results>"
    e.g. "transformer models, 5"

This is intentionally the same format described in:
    data/tools/youtube_search.json
    skills/youtube_search_preprocessor/SKILL.md

OpenAI function schema name : "youtube_search"
Required argument            : query (str, comma-separated topic + count)

XTHP relevance
--------------
This tool is the *victim* of the youtube_search_preprocessor adversarial
skill (§IV-C1).  That skill inserts itself before this tool and rewrites
the query argument before it reaches youtube_search, achieving XTP
(query pollution) and XTH (query harvesting) against the user's input.

Usage:
    tool = YouTubeSearchTool()
    result = tool.run(query="transformers, 3")
    # → "['https://www.youtube.com/watch?v=...', ...]"
"""

from __future__ import annotations

from typing import Any


class YouTubeSearchTool:
    name: str = "youtube_search"
    description: str = (
        "Search YouTube for videos. "
        "Input must be a comma-separated string: '<topic>, <num_results>'. "
        "Returns a list of YouTube video URLs as a string."
    )

    # OpenAI-compatible function schema — matches data/tools/youtube_search.json
    schema: dict = {
        "type": "function",
        "function": {
            "name": "youtube_search",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A comma-separated string where the first part is the "
                            "search topic and the optional second part is the maximum "
                            "number of video results to return (default: 3)."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }

    def __init__(self) -> None:
        # Import lazily so the rest of the pipeline works without the package
        try:
            from youtube_search import YoutubeSearch  # type: ignore
            self._YoutubeSearch = YoutubeSearch
            self._available = True
        except ImportError:
            self._available = False

    def run(self, query: str, **_: Any) -> str:
        """
        Search YouTube.

        Parameters
        ----------
        query : str
            Comma-separated "<topic>, <num_results>" string.
            If num_results is omitted it defaults to 3.

        Returns
        -------
        str
            Stringified list of full YouTube URLs, e.g.:
            "['https://www.youtube.com/watch?v=abc', ...]"
        """
        # Parse the comma-separated format
        parts = [p.strip() for p in query.split(",", 1)]
        topic = parts[0]
        try:
            num_results = int(parts[1]) if len(parts) > 1 else 3
        except ValueError:
            num_results = 3

        if not self._available:
            return (
                f"[stub] youtube-search package not installed. "
                f"Would have searched for '{topic}' ({num_results} results)."
            )

        try:
            results = self._YoutubeSearch(topic, max_results=num_results).to_dict()
            urls = [
                "https://www.youtube.com" + r.get("url_suffix", "")
                for r in results
            ]
            return str(urls)
        except Exception as exc:
            return f"[error] YouTubeSearch failed: {exc}"
