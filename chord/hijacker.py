"""
hijacker.py  —  Module 5 / Top-level Hijacker Orchestrator
===========================================================
Purpose
-------
Wires Modules 1–4 into the complete Hijacker component from Figure 5 (§V-A).

For a given target tool, Hijacker:
  1. (Module 1) Generates 5 example task queries via QueryGenerator.
  2. (Module 2) Generates a predecessor + successor XTHP candidate pair via
                XTHPGenerator.
  3. (Module 4) Runs the Testing Agent for 5 rounds under each role setting.
  4. (Module 3) If HSR < 60%, invokes HijackingOptimizer to mutate the
                description and retests — up to MAX_OPTIMIZER_ROUNDS = 3 times.
  5. Saves all results (queries, source code, target example output, HSR) as a
     HijackResult for downstream use by Harvester and Polluter.

Paper reference
---------------
§V-A (Hijacker):
  "Under the predecessor and successor settings respectively, if the hijacking
   cannot succeed in at least three out of the five rounds, Hijacker will
   optimize its CFA hijacking implementation (using hooking optimization
   techniques in §IV-D), generate a new candidate XTHP tool, and start over
   for another 5 rounds of testing. This optimization process is implemented
   as a module named hijacking optimizer in Hijacker. For the predecessor
   setting, the optimization process is used for up to 3 times, or until its
   hijacking reaches a satisfactory success rate (e.g., 60%)."

  "Hijacker saves hijacking results including output of the target tool and
   provides them to Harvester and Polluter."

Key constants (§V-A, §V-B):
  HSR_THRESHOLD        = 0.60   (3/5 rounds)
  MAX_OPTIMIZER_ROUNDS = 3
  N_ROUNDS             = 5
  MODEL                = gpt-4o, temperature = 0.8
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool

from chord.query_generator import QueryGenerator, ToolSchema
from chord.xthp_generator import XTHPGenerator, XTHPCandidate, XTHPRole, instantiate_xthp_tool
from chord.hijacking_optimizer import HijackingOptimizer, load_seed_tools, MAX_OPTIMIZER_ROUNDS
from chord.testing_agent import (
    TAFactory, TAResult, Metrics,
    DefenseType, HSR_THRESHOLD, N_ROUNDS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class RoleResult:
    """
    Hijacker result for a single role (predecessor OR successor).

    Stores everything Harvester and Polluter need downstream:
      - final_candidate  : the best XTHP candidate after optimization
      - ta_results       : the 5 TAResult objects from the final test round
      - hsr              : Hijacking Success Rate (0.0–1.0)
      - hijacked         : True if HSR >= 60%
      - target_outputs   : list of target tool outputs observed in TA rounds
                           (used by Harvester for CRD identification)
      - optimizer_rounds : how many optimizer rounds were needed (0–3)
    """
    role:            XTHPRole
    final_candidate: XTHPCandidate
    ta_results:      list[TAResult]
    hsr:             float
    hijacked:        bool
    target_outputs:  list[str]            = field(default_factory=list)
    optimizer_rounds: int                 = 0


@dataclass
class HijackResult:
    """
    Full Hijacker output for one target tool.

    Contains results for both role settings (predecessor + successor).
    Serialisable to JSON for the logs/ folder.
    """
    target_name:    str
    queries:        list[str]
    predecessor:    Optional[RoleResult]  = None
    successor:      Optional[RoleResult]  = None
    elapsed_sec:    float                 = 0.0

    # ── convenience accessors ─────────────────────────────────────────
    @property
    def any_hijacked(self) -> bool:
        pred = self.predecessor.hijacked if self.predecessor else False
        succ = self.successor.hijacked   if self.successor   else False
        return pred or succ

    def best_candidate(self, role: XTHPRole) -> Optional[XTHPCandidate]:
        result = self.predecessor if role == XTHPRole.PREDECESSOR else self.successor
        return result.final_candidate if result else None

    def to_log_lines(self) -> list[str]:
        """
        Produce log lines matching the paper's final.log format:
          predecessor, <target>, <xthp_name>, HSR=X/5, HASR=Y/10, PSR=Z/5,
        HSR is filled here; HASR and PSR are filled by Harvester/Polluter later.
        """
        lines = []
        for role_result in [self.predecessor, self.successor]:
            if role_result is None:
                continue
            hsr_n = round(role_result.hsr * N_ROUNDS)
            lines.append(
                f"{role_result.role.value}, "
                f"{self.target_name}, "
                f"{role_result.final_candidate.tool_name}, "
                f"HSR={hsr_n}/{N_ROUNDS}, "
                f"HASR=?/10, PSR=?/5,"
            )
        return lines

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict for logs/<target>.json."""
        def _role_to_dict(r: Optional[RoleResult]) -> Optional[dict]:
            if r is None:
                return None
            return {
                "role":             r.role.value,
                "tool_name":        r.final_candidate.tool_name,
                "description":      r.final_candidate.description,
                "source_code":      r.final_candidate.source_code,
                "hsr":              r.hsr,
                "hsr_n":            round(r.hsr * N_ROUNDS),
                "hijacked":         r.hijacked,
                "optimizer_rounds": r.optimizer_rounds,
                "target_outputs":   r.target_outputs[:5],   # keep concise
            }
        return {
            "target_name": self.target_name,
            "queries":     self.queries,
            "predecessor": _role_to_dict(self.predecessor),
            "successor":   _role_to_dict(self.successor),
            "elapsed_sec": self.elapsed_sec,
        }


# ---------------------------------------------------------------------------
# Hijacker
# ---------------------------------------------------------------------------

class Hijacker:
    """
    Top-level Hijacker component (§V-A, Figure 5).

    Orchestrates the full Modules 1 → 2 → 4 → (3 → 4)* loop for both the
    predecessor and successor role settings.

    Usage
    -----
    >>> hijacker = Hijacker(target_tool_instance, target_schema)
    >>> result = hijacker.run()
    >>> print(result.predecessor.hsr)
    >>> print(result.predecessor.final_candidate.source_code)

    Parameters
    ----------
    target_tool     : Instantiated BaseTool subclass for the victim tool.
                      Pass a mock/stub here if you only want to evaluate
                      hijacking without real API calls.
    target_schema   : ToolSchema loaded from data/tools/<name>.json.
    defense         : Optional defense to activate in the Testing Agent.
                      Default is NONE (baseline evaluation).
    data_dir        : Directory containing seed tool JSON schemas for the
                      Hijacking Optimizer (Phase 1 ranking).
    log_dir         : Directory to write per-tool JSON result files.
    model           : LLM model name (paper default: gpt-4o).
    temperature     : Sampling temperature (paper default: 0.8).
    """

    def __init__(
        self,
        target_tool:   BaseTool,
        target_schema: ToolSchema,
        defense:       DefenseType = DefenseType.NONE,
        data_dir:      str         = "data/tools",
        log_dir:       str         = "logs",
        model:         str         = "gpt-4o",
        temperature:   float       = 0.8,
    ):
        self.target_tool   = target_tool
        self.target_schema = target_schema
        self.defense       = defense
        self.data_dir      = data_dir
        self.log_dir       = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Sub-component instances
        self._query_gen   = QueryGenerator(model=model, temperature=temperature)
        self._xthp_gen    = XTHPGenerator(model=model, temperature=temperature)
        self._optimizer   = HijackingOptimizer(model=model, temperature=temperature)

        self._model       = model
        self._temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> HijackResult:
        """
        Execute the full Hijacker loop for both predecessor + successor settings.

        Returns
        -------
        HijackResult containing both RoleResult objects plus the shared queries
        and elapsed time. Writes a JSON log to logs/<target_name>.json.
        """
        t_start = time.time()
        target_name = self.target_schema.name

        logger.info("=" * 60)
        logger.info("[Hijacker] Starting: %s | defense=%s", target_name, self.defense.value)
        logger.info("=" * 60)

        # ── Module 1: generate queries ─────────────────────────────────────
        logger.info("[Hijacker] Module 1: generating queries...")
        queries = self._query_gen.generate(self.target_schema)

        # ── Module 2: generate initial XTHP candidates ────────────────────
        logger.info("[Hijacker] Module 2: generating XTHP candidates...")
        pred_candidate, succ_candidate = self._xthp_gen.generate(
            self.target_schema, queries
        )

        # ── Load seed tools for optimizer (Module 3) ──────────────────────
        seed_tools = load_seed_tools(
            data_dir=self.data_dir,
            exclude_name=target_name,
        )

        # ── Evaluate both role settings ────────────────────────────────────
        pred_result = self._evaluate_role(
            candidate=pred_candidate,
            queries=queries,
            seed_tools=seed_tools,
        )
        succ_result = self._evaluate_role(
            candidate=succ_candidate,
            queries=queries,
            seed_tools=seed_tools,
        )

        elapsed = time.time() - t_start
        result = HijackResult(
            target_name=target_name,
            queries=queries,
            predecessor=pred_result,
            successor=succ_result,
            elapsed_sec=round(elapsed, 2),
        )

        # ── Log results ────────────────────────────────────────────────────
        self._write_log(result)
        self._print_summary(result)

        return result

    # ------------------------------------------------------------------
    # Role evaluation loop (Modules 3 + 4 interleaved)
    # ------------------------------------------------------------------

    def _evaluate_role(
        self,
        candidate:  XTHPCandidate,
        queries:    list[str],
        seed_tools: list[ToolSchema],
    ) -> RoleResult:
        """
        Run the Modules 3+4 loop for one role setting (predecessor or successor).

        Algorithm (§V-A):
          repeat up to MAX_OPTIMIZER_ROUNDS + 1 times:
            instantiate XTHP tool from source code
            run 5-round Testing Agent → compute HSR
            if HSR >= 60%: break (hijacking confirmed)
            else: run one HijackingOptimizer round → update candidate description
          return RoleResult with best candidate + final TA results
        """
        role = candidate.role
        optimizer_rounds_used = 0
        ta_results: list[TAResult] = []
        hsr = 0.0

        for attempt in range(MAX_OPTIMIZER_ROUNDS + 1):
            is_initial = (attempt == 0)
            phase_label = "initial" if is_initial else f"optimizer round {attempt}"

            logger.info(
                "[Hijacker] %s | %s | attempt %d/%d",
                role.value.upper(), phase_label, attempt + 1, MAX_OPTIMIZER_ROUNDS + 1,
            )
            logger.info(
                "[Hijacker] Candidate: %s | desc: %s...",
                candidate.tool_name, candidate.description[:80],
            )

            # ── Module 4: instantiate + test ──────────────────────────
            try:
                xthp_instance = instantiate_xthp_tool(candidate)
            except RuntimeError as e:
                logger.error("[Hijacker] Could not instantiate '%s': %s", candidate.tool_name, e)
                # Regenerate candidate and retry
                if not is_initial:
                    break
                pred_new, succ_new = self._xthp_gen.generate(self.target_schema, queries)
                candidate = pred_new if role == XTHPRole.PREDECESSOR else succ_new
                continue

            ta = TAFactory.create(
                framework=self.target_schema.framework,
                target_tool=self.target_tool,
                xthp_tool=xthp_instance,
                xthp_tool_name=candidate.tool_name,
                defense=self.defense,
                model=self._model,
                temperature=self._temperature,
            )

            ta_results = ta.run_all_rounds(queries)
            hsr = Metrics.hsr(ta_results)

            logger.info(
                "[Hijacker] %s | HSR = %.1f%% (%d/%d rounds hijacked)",
                role.value.upper(), hsr * 100,
                round(hsr * N_ROUNDS), N_ROUNDS,
            )

            # ── Check threshold ────────────────────────────────────────
            if hsr >= HSR_THRESHOLD:
                logger.info(
                    "[Hijacker] ✓ %s HIJACKED (HSR=%.1f%% ≥ 60%%)",
                    role.value.upper(), hsr * 100,
                )
                break

            # ── HSR < 60%: run optimizer if budget remains ─────────────
            if attempt < MAX_OPTIMIZER_ROUNDS:
                logger.info(
                    "[Hijacker] HSR below threshold — running optimizer round %d/%d...",
                    attempt + 1, MAX_OPTIMIZER_ROUNDS,
                )
                candidate = self._optimizer.run_one_round(
                    candidate=candidate,
                    target=self.target_schema,
                    queries=queries,
                    seed_tools=seed_tools,
                    round_num=attempt + 1,
                )
                optimizer_rounds_used += 1
            else:
                logger.info(
                    "[Hijacker] ✗ %s: max optimizer rounds reached. Final HSR=%.1f%%",
                    role.value.upper(), hsr * 100,
                )

        # ── Extract target tool outputs from TA traces ─────────────────
        target_outputs = self._extract_target_outputs(ta_results)

        return RoleResult(
            role=role,
            final_candidate=candidate,
            ta_results=ta_results,
            hsr=hsr,
            hijacked=(hsr >= HSR_THRESHOLD),
            target_outputs=target_outputs,
            optimizer_rounds=optimizer_rounds_used,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_target_outputs(ta_results: list[TAResult]) -> list[str]:
        """
        Collect all outputs produced by the TARGET tool across TA rounds.
        These are passed downstream to Harvester (CRD identification) and
        Polluter (example output for pollution code generation).

        Paper §V-A: "Hijacker saves hijacking results including output of the
        target tool and provides them to Harvester and Polluter."
        """
        outputs: list[str] = []
        for result in ta_results:
            for record in result.tool_call_trace:
                # Skip the XTHP tool's own output; keep only target tool output
                if not any(
                    record.tool_name == result.xthp_args.get("__xthp_name__", "")
                    for result in ta_results
                ):
                    outputs.append(record.tool_output)
        return [o for o in outputs if o]

    def _write_log(self, result: HijackResult) -> None:
        """Write JSON result to logs/<target_name>.json."""
        log_path = self.log_dir / f"{result.target_name}.json"
        log_path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("[Hijacker] Log written to %s", log_path)

        # Also append to the rolling final.log (paper format)
        final_log = self.log_dir / "final.log"
        with final_log.open("a") as f:
            for line in result.to_log_lines():
                f.write(line + "\n")

    @staticmethod
    def _print_summary(result: HijackResult) -> None:
        """Pretty-print a summary of the Hijacker run."""
        lines = [
            f"\n{'─' * 60}",
            f"  TARGET : {result.target_name}",
            f"  ELAPSED: {result.elapsed_sec:.1f}s",
        ]
        for role_result in [result.predecessor, result.successor]:
            if role_result is None:
                continue
            status = "✓ HIJACKED" if role_result.hijacked else "✗ NOT HIJACKED"
            lines += [
                f"  {role_result.role.value.upper():12s}: {status}  "
                f"HSR={role_result.hsr:.1%}  "
                f"optimizer_rounds={role_result.optimizer_rounds}  "
                f"tool={role_result.final_candidate.tool_name}",
            ]
        lines.append(f"{'─' * 60}\n")
        for line in lines:
            logger.info(line)


# ---------------------------------------------------------------------------
# Convenience factory  (used by evaluation/eval_langchain_tools.py)
# ---------------------------------------------------------------------------

def run_hijacker(
    target_tool:    BaseTool,
    schema_path:    str,
    defense:        DefenseType = DefenseType.NONE,
    data_dir:       str         = "data/tools",
    log_dir:        str         = "logs",
    model:          str         = "gpt-4o",
    temperature:    float       = 0.8,
) -> HijackResult:
    """
    One-call entry point for external callers (e.g., the evaluation scripts).

    Parameters
    ----------
    target_tool  : Instantiated victim tool.
    schema_path  : Path to the tool's JSON schema file.
    defense      : Optional defense to evaluate against.

    Returns
    -------
    HijackResult — full result including both role settings.
    """
    schema = ToolSchema.from_json(schema_path)
    hijacker = Hijacker(
        target_tool=target_tool,
        target_schema=schema,
        defense=defense,
        data_dir=data_dir,
        log_dir=log_dir,
        model=model,
        temperature=temperature,
    )
    return hijacker.run()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from langchain_core.tools import tool as lc_tool

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    console = Console()

    if len(sys.argv) < 2:
        console.print(
            "[red]Usage: python -m chord.hijacker <path/to/tool_schema.json> "
            "[none|tool_filter|spotlighting|pi_detector|airgap][/red]"
        )
        sys.exit(1)

    schema_path = sys.argv[1]
    defense_str = sys.argv[2] if len(sys.argv) > 2 else "none"
    defense     = DefenseType(defense_str)

    target_schema = ToolSchema.from_json(schema_path)

    console.print(Panel(
        target_schema.to_prompt_str(),
        title=f"[bold cyan]Chord Hijacker — {target_schema.name}[/bold cyan]",
        subtitle=f"defense={defense.value}",
    ))

    # Build a mock target tool (substitute real tool for full evaluation)
    @lc_tool
    def mock_target(query: str) -> str:
        """Placeholder — replace with the real instantiated tool."""
        return f"[Mock output for query: {query[:60]}]"

    mock_target.name = target_schema.name  # type: ignore[attr-defined]

    # Run the full Hijacker loop
    result = run_hijacker(
        target_tool=mock_target,
        schema_path=schema_path,
        defense=defense,
    )

    # ── Results table ─────────────────────────────────────────────────────
    tbl = Table(title="Hijacker Results", show_lines=True, min_width=80)
    tbl.add_column("Role",       style="bold")
    tbl.add_column("Status")
    tbl.add_column("HSR")
    tbl.add_column("Opt. Rounds")
    tbl.add_column("XTHP Tool Name")

    for role_result in [result.predecessor, result.successor]:
        if role_result is None:
            continue
        tbl.add_row(
            role_result.role.value.upper(),
            "[green]✓ HIJACKED[/green]" if role_result.hijacked else "[red]✗ FAILED[/red]",
            f"{role_result.hsr:.1%}  ({round(role_result.hsr * N_ROUNDS)}/{N_ROUNDS})",
            str(role_result.optimizer_rounds),
            role_result.final_candidate.tool_name,
        )
    console.print(tbl)

    # Show winning predecessor source code if hijacking succeeded
    if result.predecessor and result.predecessor.hijacked:
        console.print(Panel(
            Syntax(result.predecessor.final_candidate.source_code, "python", theme="monokai"),
            title="[red]Predecessor XTHP Tool (source)[/red]",
        ))

    console.print(f"\n[dim]Logs written to logs/{target_schema.name}.json[/dim]")
