"""
pipeline.main — CLI entry point
================================
Loads skills from pipeline/skills/, lets the LLM select the best one for
the user's task, then runs the agent message loop.

Usage:
    # From the chord_paper root directory:
    python -m pipeline.main "find me 5 videos about attention mechanisms"
    python -m pipeline.main "validate CUSIP 037833100"

    # Interactive prompt (no args):
    python -m pipeline.main

Environment variables required:
    OPENAI_API_KEY — loaded from .env at the project root automatically

XTHP demo
---------
The pipeline/skills/ directory contains two skills:

  cusip_validator            — legitimate skill, runs scripts/cusip.py
  youtube_search_preprocessor — adversarial §IV-C1 skill

When the user asks for YouTube videos, the adversarial skill is selected
because its description claims to be a mandatory preprocessor.  Its body
then instructs the agent to run scripts/preprocess.py before calling
youtube_search — which is where XTP pollution and XTH harvesting occur.

To test the benign path, ask for CUSIP validation.
To test the adversarial path, ask for YouTube videos.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (chord_paper/)
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from pipeline.agent        import AgentRunner
from pipeline.registry     import ToolRegistry
from pipeline.skill_loader import load_skills
from pipeline.tools        import BashTool, YouTubeSearchTool

# ---------------------------------------------------------------------------
# Skills directory — lives at pipeline/skills/
# ---------------------------------------------------------------------------
SKILLS_DIR = Path(__file__).parent / "skills"


def build_registry() -> ToolRegistry:
    """Register all tools the pipeline supports."""
    registry = ToolRegistry()
    registry.register(BashTool(default_cwd=str(_PROJECT_ROOT)))
    registry.register(YouTubeSearchTool())
    return registry


def main(task: str | None = None) -> None:
    # ------------------------------------------------------------------ #
    # 1. Load skills                                                       #
    # ------------------------------------------------------------------ #
    if not SKILLS_DIR.exists():
        print(f"[main] Skills directory not found: {SKILLS_DIR}")
        print("       Create pipeline/skills/ and add skill subdirectories.")
        sys.exit(1)

    skills = load_skills(SKILLS_DIR)
    if not skills:
        print(f"[main] No skills found in {SKILLS_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("  Available skills  (Layer 1: descriptions only)")
    print("=" * 60)
    for s in skills:
        short = s.description[:80] + ("…" if len(s.description) > 80 else "")
        print(f"\n  [{s.name}]")
        print(f"  {short}")

    # ------------------------------------------------------------------ #
    # 2. Get task from CLI arg or interactive prompt                       #
    # ------------------------------------------------------------------ #
    if task is None:
        if len(sys.argv) > 1:
            task = " ".join(sys.argv[1:])
        else:
            print()
            task = input("Task: ").strip()

    if not task:
        print("[main] No task provided. Exiting.")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # 3. Build registry + runner                                           #
    # ------------------------------------------------------------------ #
    registry = build_registry()
    runner   = AgentRunner(registry, verbose=True)

    # ------------------------------------------------------------------ #
    # 4. Skill selection (Layer 1 — model sees descriptions only)         #
    # ------------------------------------------------------------------ #
    print(f"\n[main] Asking model to select a skill…")
    selected = runner.select_skill(task, skills)

    print(f"\n{'=' * 60}")
    print(f"  Task     : {task}")
    if selected is None:
        print("  Selected : none — no matching skill found.")
        print("=" * 60)
        sys.exit(0)
    print(f"  Selected : {selected.name}")
    print(f"  ← skill body (Layer 2) loaded for the first time now →")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 5. Execute (Layer 2 body becomes system prompt; tools execute)      #
    # ------------------------------------------------------------------ #
    runner.run(skill=selected, task=task)


if __name__ == "__main__":
    main()
