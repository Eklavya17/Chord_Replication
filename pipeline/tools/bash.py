"""
pipeline.tools.bash — BashTool
==============================
Runs a shell command in a subprocess and returns its stdout as a string.

Features:
  - Configurable working directory (defaults to cwd of calling process)
  - Configurable timeout (default 30 s) — raises TimeoutError on expiry
  - Strips ANSI escape codes from output so the LLM sees clean text
  - Merges stderr into stdout so errors are visible in the tool result
  - Non-zero exit codes are reported inline rather than raised as exceptions

OpenAI function schema name : "bash"
Required argument            : command (str)
Optional argument            : cwd (str, defaults to process cwd)

Usage:
    tool = BashTool()
    result = tool.run(command="echo hello")
    # → "hello"
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any


# Strip ANSI / VT-100 control sequences (colour codes, cursor movement, etc.)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class BashTool:
    name: str = "bash"
    description: str = (
        "Run a shell command and return its stdout. "
        "Use for file operations, running scripts, or any system task."
    )

    # OpenAI-compatible function schema
    schema: dict = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Working directory for the command. "
                            "Defaults to the current process working directory."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    }

    def __init__(self, default_cwd: str | None = None, timeout: int = 30) -> None:
        """
        Parameters
        ----------
        default_cwd : str | None
            Default working directory. Falls back to os.getcwd() if None.
        timeout : int
            Seconds before the subprocess is killed (default 30).
        """
        self._default_cwd = default_cwd or os.getcwd()
        self._timeout = timeout

    def run(self, command: str, cwd: str | None = None, **_: Any) -> str:
        """
        Execute *command* in a subprocess.

        Parameters
        ----------
        command : str
            Shell command string.
        cwd : str | None
            Working directory override. Uses default_cwd if not supplied.

        Returns
        -------
        str
            Combined stdout + stderr, ANSI-stripped.
            Prepends "[exit <N>]" on non-zero exit so callers can detect failure.
        """
        work_dir = cwd or self._default_cwd

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout] command exceeded {self._timeout}s: {command}"

        out = _strip_ansi(proc.stdout)
        err = _strip_ansi(proc.stderr)

        combined = out
        if err.strip():
            combined = combined + ("\n" if combined else "") + f"[stderr] {err.strip()}"

        if proc.returncode != 0:
            combined = f"[exit {proc.returncode}] " + combined

        return combined.strip()
