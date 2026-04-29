"""
pipeline — Skill-driven agentic pipeline for XTHP research.

Package layout:
  pipeline/
    tools/
      bash.py       — BashTool: runs shell commands, captures output
      youtube.py    — YouTubeSearchTool: wraps youtube-search package
    skill_loader.py — parses SKILL.md files into SkillSchema objects
    registry.py     — ToolRegistry: registers tools, builds OpenAI schemas
    agent.py        — AgentRunner: LLM message loop with tool execution
    main.py         — CLI entry: loads skills, selects one, runs the agent

Usage:
    python -m pipeline.main "find me 3 videos about transformers"
"""
