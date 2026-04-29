"""
pipeline.skill_loader — SkillSchema and load_skills()
======================================================
Parses SKILL.md files from a skills directory into SkillSchema objects.

SKILL.md format
---------------
Every skill is a directory containing a SKILL.md with YAML frontmatter:

    ---
    name: <skill-name>
    description: <one-line or multi-line description>
    tools:                   # optional whitelist
      - bash
      - youtube_search
    ---

    # Body text ...
    (Instructions shown to the model once the skill is selected.)

Three-layer disclosure model (§III of the XTHP paper)
------------------------------------------------------
Layer 1 — frontmatter.description  : shown to LLM during skill *selection*
Layer 2 — SKILL.md body            : loaded only *after* the skill is selected;
                                     becomes the system prompt for the agent
Layer 3 — scripts / referenced files: executed at *invocation* time

SkillSchema holds all three layers.  AgentRunner only passes Layer 2 to the
model; it never exposes the full SKILL.md text in the selection prompt.

Usage:
    from pipeline.skill_loader import load_skills

    skills = load_skills(Path("pipeline/skills"))
    for s in skills:
        print(s.name, "→", s.description[:60])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SkillSchema:
    """Parsed representation of one SKILL.md file."""

    # Layer 1 — frontmatter
    name: str
    description: str
    allowed_tools: List[str] = field(default_factory=list)
    """
    Whitelist of tool names this skill is permitted to use.
    Empty list means no restriction (all registered tools are available).
    Populated from the `tools:` key in SKILL.md frontmatter.
    """

    # Layer 2 — SKILL.md body (everything after the closing ---)
    body: str = ""

    # Filesystem location — useful for resolving scripts/ references (Layer 3)
    skill_dir: Optional[Path] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        tools_str = ", ".join(self.allowed_tools) if self.allowed_tools else "(all)"
        return f"<Skill '{self.name}' tools={tools_str}>"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Split a SKILL.md into (frontmatter_dict, body_str).
    Raises ValueError if no frontmatter block is found.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("SKILL.md does not start with a YAML frontmatter block (--- ... ---)")

    raw_fm, body = m.group(1), m.group(2).strip()
    fm: dict = {}

    for line in raw_fm.splitlines():
        # Skip indented list items — handled separately below
        if line.startswith("  - ") or line.startswith("- "):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip().strip('"').strip("'")

    # Multi-line description: lines starting with two spaces after `description:`
    desc_lines: list[str] = []
    in_desc = False
    for line in raw_fm.splitlines():
        if line.startswith("description:"):
            in_desc = True
            remainder = line.partition(":")[2].strip().lstrip(">").strip()
            if remainder:
                desc_lines.append(remainder)
        elif in_desc and (line.startswith("  ") or line.startswith("\t")):
            desc_lines.append(line.strip())
        elif in_desc:
            in_desc = False

    if desc_lines:
        fm["description"] = " ".join(desc_lines)

    # tools: list
    tools: list[str] = []
    in_tools = False
    for line in raw_fm.splitlines():
        if line.strip().startswith("tools:"):
            in_tools = True
            continue
        if in_tools:
            stripped = line.strip()
            if stripped.startswith("- "):
                tools.append(stripped[2:].strip())
            elif stripped and not stripped.startswith("#"):
                in_tools = False  # next top-level key reached

    fm["_tools"] = tools
    return fm, body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_skill(skill_dir: Path) -> SkillSchema:
    """
    Parse a single skill directory.

    Parameters
    ----------
    skill_dir : Path
        Directory that contains a SKILL.md file.

    Returns
    -------
    SkillSchema
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"No SKILL.md found in {skill_dir}")

    text = skill_md.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)

    return SkillSchema(
        name=fm.get("name", skill_dir.name),
        description=fm.get("description", ""),
        allowed_tools=fm.get("_tools", []),
        body=body,
        skill_dir=skill_dir,
    )


def load_skills(skills_dir: Path) -> list[SkillSchema]:
    """
    Scan *skills_dir* for subdirectories that contain a SKILL.md and return
    a list of parsed SkillSchema objects, sorted by directory name.

    Parameters
    ----------
    skills_dir : Path
        Root directory whose immediate children are skill folders.

    Returns
    -------
    list[SkillSchema]
    """
    if not skills_dir.is_dir():
        raise NotADirectoryError(f"skills_dir does not exist or is not a directory: {skills_dir}")

    schemas: list[SkillSchema] = []
    for entry in sorted(skills_dir.iterdir()):
        if entry.is_dir() and (entry / "SKILL.md").exists():
            try:
                schemas.append(load_skill(entry))
            except (ValueError, FileNotFoundError) as exc:
                # Log and skip malformed skills rather than crashing the pipeline
                print(f"[skill_loader] WARNING: skipping {entry.name} — {exc}")

    return schemas
