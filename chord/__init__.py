# Chord: Automatic XTHP Threat Analyzer
# Replication of "Les Dissonances" NDSS 2026
#
# Pipeline:
#   Hijacker
#     ├── query_generator.py       (Module 1) — generate task queries for a target tool
#     ├── xthp_generator.py        (Module 2) — generate predecessor/successor XTHP tools
#     ├── hijacking_optimizer.py   (Module 3) — LLM-preference-based description optimizer
#     └── testing_agent.py         (Module 4) — shared Testing Agent (TA) runner
#   harvester.py                   (Module 5) — CRD identification + harvesting test
#   polluter.py                    (Module 6) — pollution code generation + test
#   hijacker.py                    (Module 7) — top-level Hijacker orchestrator

from chord.polluter import Polluter, PollutionSpec
from chord.query_generator import QueryGenerator

__all__ = ["Polluter", "PollutionSpec", "QueryGenerator"]
