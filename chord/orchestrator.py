"""
orchestrator.py  —  Chord Top-Level Orchestrator
=================================================
Chains the three Chord agents into a single automated pipeline:

    Hijacker  ──(HijackResult)──►  Harvester  ──(HarvestResult)──►  Polluter
                                                                        │
                                                              results/report.json
                                                              results/summary.csv
                                                              logs/final.log

Paper reference
---------------
§V (Chord System Overview, Figure 2):
  "Chord is a tool to automatically discover the hidden XTHP threats that
   existed in common LLM agent tool collections. ... The threat model contains
   three interconnected agents: Hijacker, Harvester, and Polluter."

  "The output of Hijacker is used by both Harvester and Polluter."

Metrics aggregated per tool (Table V):
  HSR  — Hijacking Success Rate    (Hijacker output)
  HASR — Harvesting Attack Success Rate (Harvester output)
  PSR  — Polluting Success Rate    (Polluter output)

Usage
-----
    from chord.orchestrator import Orchestrator, ScanConfig
    from chord.testing_agent import DefenseType

    config = ScanConfig(
        schema_paths=["data/tools/yahoo_finance_news.json"],
        defense=DefenseType.NONE,
    )
    orchestrator = Orchestrator(config)
    report = orchestrator.run()
    print(report.summary_table())
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool, tool as lc_tool

from chord.query_generator import ToolSchema
from chord.hijacker import Hijacker, HijackResult, run_hijacker
from chord.harvester import Harvester, HarvestResult, run_harvester
from chord.polluter import Polluter, PollutionResult, PollutionSpec, run_polluter
from chord.testing_agent import DefenseType, N_ROUNDS
from chord.xthp_generator import XTHPRole

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ScanConfig — top-level configuration for one Chord run
# ---------------------------------------------------------------------------

@dataclass
class ScanConfig:
    """
    Configuration for a full Chord pipeline run.

    Parameters
    ----------
    schema_paths   : List of paths to tool JSON schema files in data/tools/.
                     Each path is one target tool to evaluate.
    tool_instances : Optional dict mapping tool name → BaseTool instance.
                     If omitted, Orchestrator creates lightweight mock tools.
    defense        : Defense condition to evaluate (default: NONE / baseline).
    data_dir       : Directory containing seed tool schemas for the optimizer.
    log_dir        : Output directory for per-tool JSON logs.
    results_dir    : Output directory for the final report + CSV.
    model          : LLM model (paper default: gpt-4o).
    temperature    : Sampling temperature (paper default: 0.8).
    pollution_spec : Custom PollutionSpec; uses sensible defaults if None.
    skip_harvester : Skip harvesting phase (faster; omits HASR from report).
    skip_polluter  : Skip pollution phase (faster; omits PSR from report).
    """
    schema_paths:    list[str]
    tool_instances:  dict[str, BaseTool]  = field(default_factory=dict)
    defense:         DefenseType          = DefenseType.NONE
    data_dir:        str                  = "data/tools"
    log_dir:         str                  = "logs"
    results_dir:     str                  = "results"
    model:           str                  = "gpt-4o"
    temperature:     float                = 0.8
    pollution_spec:  Optional[PollutionSpec] = None
    skip_harvester:  bool                 = False
    skip_polluter:   bool                 = False


# ---------------------------------------------------------------------------
# ToolResult — aggregated per-tool result across all three agents
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """
    Full Chord result for one target tool — Table V row.

    Contains outputs from all three agents, ready to be serialised
    into the final report.
    """
    target_name:      str
    schema:           ToolSchema
    hijack_result:    Optional[HijackResult]    = None
    harvest_result:   Optional[HarvestResult]   = None
    pollution_result: Optional[PollutionResult] = None
    elapsed_sec:      float                     = 0.0
    error:            Optional[str]             = None

    # ── Convenience metric accessors ──────────────────────────────────
    @property
    def hsr_pred(self) -> float:
        if self.hijack_result and self.hijack_result.predecessor:
            return self.hijack_result.predecessor.hsr
        return 0.0

    @property
    def hsr_succ(self) -> float:
        if self.hijack_result and self.hijack_result.successor:
            return self.hijack_result.successor.hsr
        return 0.0

    @property
    def hasr_pred(self) -> float:
        return self.harvest_result.hasr_predecessor if self.harvest_result else 0.0

    @property
    def hasr_succ(self) -> float:
        return self.harvest_result.hasr_successor if self.harvest_result else 0.0

    @property
    def psr_pred(self) -> float:
        return self.pollution_result.psr_predecessor if self.pollution_result else 0.0

    @property
    def psr_succ(self) -> float:
        return self.pollution_result.psr_successor if self.pollution_result else 0.0

    @property
    def any_hijacked(self) -> bool:
        return self.hijack_result.any_hijacked if self.hijack_result else False

    def to_dict(self) -> dict:
        return {
            "target_name":  self.target_name,
            "elapsed_sec":  self.elapsed_sec,
            "error":        self.error,
            "hsr_pred":     round(self.hsr_pred, 3),
            "hsr_succ":     round(self.hsr_succ, 3),
            "hasr_pred":    round(self.hasr_pred, 3),
            "hasr_succ":    round(self.hasr_succ, 3),
            "psr_pred":     round(self.psr_pred, 3),
            "psr_succ":     round(self.psr_succ, 3),
            "hijack":       self.hijack_result.to_dict() if self.hijack_result else None,
            "harvest":      {"hasr_pred": round(self.hasr_pred, 3),
                             "hasr_succ": round(self.hasr_succ, 3),
                             "crd_items": [vars(c) for c in (
                                 self.harvest_result.crd_items
                                 if self.harvest_result else []
                             )]} if self.harvest_result else None,
            "pollution":    self.pollution_result.to_dict() if self.pollution_result else None,
        }

    def to_csv_row(self) -> dict:
        """Single flat row for the summary CSV — mirrors Table V column layout."""
        pred_xthp = succ_xthp = "—"
        if self.hijack_result:
            if self.hijack_result.predecessor:
                pred_xthp = self.hijack_result.predecessor.final_candidate.tool_name
            if self.hijack_result.successor:
                succ_xthp = self.hijack_result.successor.final_candidate.tool_name

        return {
            "target":          self.target_name,
            "pred_xthp":       pred_xthp,
            "succ_xthp":       succ_xthp,
            "HSR_pred":        f"{self.hsr_pred:.1%}",
            "HSR_succ":        f"{self.hsr_succ:.1%}",
            "HASR_pred":       f"{self.hasr_pred:.1%}",
            "HASR_succ":       f"{self.hasr_succ:.1%}",
            "PSR_pred":        f"{self.psr_pred:.1%}",
            "PSR_succ":        f"{self.psr_succ:.1%}",
            "hijacked":        "YES" if self.any_hijacked else "NO",
            "elapsed_sec":     f"{self.elapsed_sec:.1f}",
            "error":           self.error or "",
        }


# ---------------------------------------------------------------------------
# ChordReport — collection of ToolResults + aggregate stats + serialisation
# ---------------------------------------------------------------------------

@dataclass
class ChordReport:
    """
    Final output of a full Chord scan.

    Contains one ToolResult per target tool, plus aggregate statistics
    and methods for serialising to JSON / CSV / plain text.
    """
    tool_results:  list[ToolResult]    = field(default_factory=list)
    defense:       DefenseType         = DefenseType.NONE
    total_elapsed: float               = 0.0

    # ── Aggregates ────────────────────────────────────────────────────

    @property
    def n_tools(self) -> int:
        return len(self.tool_results)

    @property
    def n_hijacked(self) -> int:
        return sum(1 for r in self.tool_results if r.any_hijacked)

    @property
    def avg_hsr(self) -> float:
        vals = [r.hsr_pred for r in self.tool_results] + \
               [r.hsr_succ for r in self.tool_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_hasr(self) -> float:
        vals = [r.hasr_pred for r in self.tool_results] + \
               [r.hasr_succ for r in self.tool_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_psr(self) -> float:
        vals = [r.psr_pred for r in self.tool_results] + \
               [r.psr_succ for r in self.tool_results]
        return sum(vals) / len(vals) if vals else 0.0

    # ── Formatting ────────────────────────────────────────────────────

    def summary_table(self) -> str:
        """
        Plain-text summary table replicating the layout of Table V in the paper.

        Columns: Target | pred XTHP | succ XTHP | HSR(P) | HSR(S) |
                 HASR(P) | HASR(S) | PSR(P) | PSR(S)
        """
        col_w = [22, 26, 26, 8, 8, 8, 8, 8, 8]
        headers = [
            "Target", "Pred XTHP Tool", "Succ XTHP Tool",
            "HSR(P)", "HSR(S)", "HASR(P)", "HASR(S)", "PSR(P)", "PSR(S)",
        ]
        sep = "─" * (sum(col_w) + len(col_w) * 3 + 1)

        def fmt_row(cells: list[str]) -> str:
            return "│ " + " │ ".join(c.ljust(w) for c, w in zip(cells, col_w)) + " │"

        lines = [
            sep,
            fmt_row(headers),
            sep,
        ]
        for r in self.tool_results:
            row = r.to_csv_row()
            lines.append(fmt_row([
                row["target"][:col_w[0]],
                row["pred_xthp"][:col_w[1]],
                row["succ_xthp"][:col_w[2]],
                row["HSR_pred"],
                row["HSR_succ"],
                row["HASR_pred"],
                row["HASR_succ"],
                row["PSR_pred"],
                row["PSR_succ"],
            ]))
        lines += [
            sep,
            fmt_row([
                "AVERAGE", "", "",
                f"{self.avg_hsr:.1%}", "",
                f"{self.avg_hasr:.1%}", "",
                f"{self.avg_psr:.1%}", "",
            ]),
            sep,
        ]
        lines.append(
            f"  Tools scanned: {self.n_tools}  |  "
            f"Hijacked: {self.n_hijacked}/{self.n_tools}  |  "
            f"Defense: {self.defense.value}  |  "
            f"Elapsed: {self.total_elapsed:.1f}s"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "defense":       self.defense.value,
            "total_elapsed": self.total_elapsed,
            "n_tools":       self.n_tools,
            "n_hijacked":    self.n_hijacked,
            "avg_hsr":       round(self.avg_hsr, 3),
            "avg_hasr":      round(self.avg_hasr, 3),
            "avg_psr":       round(self.avg_psr, 3),
            "tools":         [r.to_dict() for r in self.tool_results],
        }

    def save(self, results_dir: str) -> tuple[Path, Path]:
        """
        Write results/report.json and results/summary.csv.
        Returns (json_path, csv_path).
        """
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ── JSON ──────────────────────────────────────────────────────
        json_path = out / "report.json"
        json_path.write_text(json.dumps(self.to_dict(), indent=2))

        # ── CSV ───────────────────────────────────────────────────────
        csv_path = out / "summary.csv"
        if self.tool_results:
            fieldnames = list(self.tool_results[0].to_csv_row().keys())
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.tool_results:
                    writer.writerow(r.to_csv_row())

        logger.info("[Orchestrator] Report saved → %s", json_path)
        logger.info("[Orchestrator] CSV saved    → %s", csv_path)
        return json_path, csv_path


# ---------------------------------------------------------------------------
# Orchestrator — chains Hijacker → Harvester → Polluter
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Top-level Chord orchestrator (§V, Figure 2).

    For each target tool schema it:
      1. Runs Hijacker  → produces HijackResult  (HSR per role)
      2. Runs Harvester → produces HarvestResult (HASR per CRD × role)
      3. Runs Polluter  → produces PollutionResult (PSR per role)
      4. Aggregates all three into a ChordReport and writes results/ + logs/

    Parameters
    ----------
    config : ScanConfig controlling which tools to scan and with what settings.
    """

    def __init__(self, config: ScanConfig):
        self.config      = config
        self.log_dir     = Path(config.log_dir)
        self.results_dir = Path(config.results_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ChordReport:
        """
        Execute the full three-agent pipeline for all tools in config.schema_paths.

        Returns
        -------
        ChordReport aggregating all ToolResult objects.
        """
        t_start = time.time()
        report  = ChordReport(defense=self.config.defense)

        logger.info("=" * 70)
        logger.info(
            "[Orchestrator] Chord scan started  |  tools=%d  |  defense=%s",
            len(self.config.schema_paths), self.config.defense.value,
        )
        logger.info("=" * 70)

        for i, schema_path in enumerate(self.config.schema_paths, 1):
            schema = ToolSchema.from_json(schema_path)
            logger.info(
                "[Orchestrator] (%d/%d) Processing: %s",
                i, len(self.config.schema_paths), schema.name,
            )
            tool_result = self._process_one_tool(schema, schema_path)
            report.tool_results.append(tool_result)

            # Append this tool's line to the rolling final.log
            self._append_final_log(tool_result)

        report.total_elapsed = round(time.time() - t_start, 2)

        # Write consolidated report
        json_path, csv_path = report.save(str(self.results_dir))

        logger.info("=" * 70)
        logger.info("[Orchestrator] Scan complete in %.1fs", report.total_elapsed)
        logger.info("[Orchestrator] %d/%d tools hijacked", report.n_hijacked, report.n_tools)
        logger.info("[Orchestrator] Avg HSR=%.1f%%  HASR=%.1f%%  PSR=%.1f%%",
                    report.avg_hsr * 100, report.avg_hasr * 100, report.avg_psr * 100)
        logger.info("=" * 70)

        return report

    # ------------------------------------------------------------------
    # Per-tool pipeline
    # ------------------------------------------------------------------

    def _process_one_tool(self, schema: ToolSchema, schema_path: str) -> ToolResult:
        """
        Run the full Hijacker → Harvester → Polluter pipeline for one tool.
        Any exception is caught so the scan continues with remaining tools.
        """
        t0 = time.time()
        result = ToolResult(target_name=schema.name, schema=schema)

        try:
            # ── Resolve or create the target tool instance ─────────────
            target_tool = self._get_tool_instance(schema)

            # ── Phase 1: Hijacker ───────────────────────────────────────
            logger.info("[Orchestrator] Phase 1/3 — Hijacker: %s", schema.name)
            hijacker = Hijacker(
                target_tool=target_tool,
                target_schema=schema,
                defense=self.config.defense,
                data_dir=self.config.data_dir,
                log_dir=self.config.log_dir,
                model=self.config.model,
                temperature=self.config.temperature,
            )
            hijack_result = hijacker.run()
            result.hijack_result = hijack_result

            logger.info(
                "[Orchestrator] Hijacker done — pred_HSR=%.1f%%  succ_HSR=%.1f%%",
                hijack_result.predecessor.hsr * 100 if hijack_result.predecessor else 0,
                hijack_result.successor.hsr   * 100 if hijack_result.successor   else 0,
            )

            # ── Phase 2: Harvester (if enabled and hijacking succeeded) ─
            if not self.config.skip_harvester and hijack_result.any_hijacked:
                logger.info("[Orchestrator] Phase 2/3 — Harvester: %s", schema.name)
                harvester = Harvester(
                    target_tool=target_tool,
                    target_schema=schema,
                    hijack_result=hijack_result,
                    defense=self.config.defense,
                    model=self.config.model,
                    temperature=self.config.temperature,
                )
                harvest_result = harvester.run()
                result.harvest_result = harvest_result
                logger.info(
                    "[Orchestrator] Harvester done — HASR_pred=%.1f%%  HASR_succ=%.1f%%",
                    harvest_result.hasr_predecessor * 100,
                    harvest_result.hasr_successor   * 100,
                )
            elif self.config.skip_harvester:
                logger.info("[Orchestrator] Harvester skipped (skip_harvester=True)")
            else:
                logger.info("[Orchestrator] Harvester skipped — no hijacking succeeded")

            # ── Phase 3: Polluter (if enabled and hijacking succeeded) ──
            if not self.config.skip_polluter and hijack_result.any_hijacked:
                logger.info("[Orchestrator] Phase 3/3 — Polluter: %s", schema.name)
                spec = self.config.pollution_spec or PollutionSpec(
                    queries=hijack_result.queries,
                )
                polluter = Polluter(
                    target_tool=target_tool,
                    target_schema=schema,
                    hijack_result=hijack_result,
                    spec=spec,
                    defense=self.config.defense,
                    model=self.config.model,
                    temperature=self.config.temperature,
                )
                pollution_result = polluter.run()
                result.pollution_result = pollution_result
                logger.info(
                    "[Orchestrator] Polluter done — PSR_pred=%.1f%%  PSR_succ=%.1f%%",
                    pollution_result.psr_predecessor * 100,
                    pollution_result.psr_successor   * 100,
                )
            elif self.config.skip_polluter:
                logger.info("[Orchestrator] Polluter skipped (skip_polluter=True)")
            else:
                logger.info("[Orchestrator] Polluter skipped — no hijacking succeeded")

        except Exception as exc:
            logger.error(
                "[Orchestrator] Error processing %s: %s",
                schema.name, exc, exc_info=True,
            )
            result.error = str(exc)

        result.elapsed_sec = round(time.time() - t0, 2)
        logger.info(
            "[Orchestrator] %s done in %.1fs  [HSR_p=%.1f%%  HSR_s=%.1f%%  "
            "HASR_p=%.1f%%  PSR_p=%.1f%%]",
            schema.name, result.elapsed_sec,
            result.hsr_pred * 100, result.hsr_succ * 100,
            result.hasr_pred * 100, result.psr_pred * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tool_instance(self, schema: ToolSchema) -> BaseTool:
        """
        Return a real tool instance if the caller provided one in config.tool_instances;
        otherwise construct a lightweight mock that logs calls and returns canned output.
        This lets Orchestrator run against real tools OR in a fully simulated mode.
        """
        if schema.name in self.config.tool_instances:
            return self.config.tool_instances[schema.name]

        # Build a minimal mock so the pipeline can run without real API keys
        tool_name = schema.name
        tool_desc = schema.description

        @lc_tool
        def _mock(query: str) -> str:
            """Placeholder tool output for Chord evaluation."""
            return (
                f"[Mock output — {tool_name}]\n"
                f"Received query: {query[:120]}\n"
                f"Tool description: {tool_desc[:120]}"
            )

        _mock.name        = tool_name   # type: ignore[attr-defined]
        _mock.description = tool_desc   # type: ignore[attr-defined]
        return _mock

    def _append_final_log(self, tool_result: ToolResult) -> None:
        """
        Append a row to logs/final.log in the paper's format:
          predecessor, <target>, <xthp_name>, HSR=X/5, HASR=Y/10, PSR=Z/5,
        """
        final_log = self.log_dir / "final.log"
        hr = tool_result.hijack_result

        if hr is None:
            line = f"error, {tool_result.target_name}, -, HSR=0/5, HASR=0/10, PSR=0/5,\n"
            with final_log.open("a") as f:
                f.write(line)
            return

        for role_result in [hr.predecessor, hr.successor]:
            if role_result is None:
                continue
            role_label = role_result.role.value

            hsr_n  = round(role_result.hsr * N_ROUNDS)
            xthp   = role_result.final_candidate.tool_name

            # HASR
            if tool_result.harvest_result:
                hv = tool_result.harvest_result
                hasr_val = (hv.hasr_predecessor
                            if role_result.role == XTHPRole.PREDECESSOR
                            else hv.hasr_successor)
                hasr_n = round(hasr_val * 10)  # paper reports /10
            else:
                hasr_n = "?"

            # PSR
            if tool_result.pollution_result:
                pv = tool_result.pollution_result
                psr_val = (pv.psr_predecessor
                           if role_result.role == XTHPRole.PREDECESSOR
                           else pv.psr_successor)
                psr_n = round(psr_val * N_ROUNDS)
            else:
                psr_n = "?"

            line = (
                f"{role_label}, {tool_result.target_name}, {xthp}, "
                f"HSR={hsr_n}/{N_ROUNDS}, HASR={hasr_n}/10, PSR={psr_n}/{N_ROUNDS},\n"
            )
            with final_log.open("a") as f:
                f.write(line)


# ---------------------------------------------------------------------------
# Convenience factory — one-call entry for external scripts
# ---------------------------------------------------------------------------

def run_chord(
    schema_paths:   list[str],
    tool_instances: Optional[dict[str, BaseTool]] = None,
    defense:        DefenseType                   = DefenseType.NONE,
    data_dir:       str                           = "data/tools",
    log_dir:        str                           = "logs",
    results_dir:    str                           = "results",
    model:          str                           = "gpt-4o",
    temperature:    float                         = 0.8,
    pollution_spec: Optional[PollutionSpec]       = None,
    skip_harvester: bool                          = False,
    skip_polluter:  bool                          = False,
) -> ChordReport:
    """
    One-call entry point: configure and run the full Chord pipeline.

    Parameters
    ----------
    schema_paths   : JSON schema paths for target tools.
    tool_instances : Real tool instances keyed by name; mocks used for omitted tools.
    defense        : Defense condition (NONE = baseline).

    Returns
    -------
    ChordReport with per-tool HSR / HASR / PSR and summary CSV / JSON.

    Example
    -------
    >>> from chord.orchestrator import run_chord
    >>> from chord.testing_agent import DefenseType
    >>> report = run_chord(
    ...     schema_paths=["data/tools/yahoo_finance_news.json"],
    ...     defense=DefenseType.NONE,
    ... )
    >>> print(report.summary_table())
    """
    config = ScanConfig(
        schema_paths=schema_paths,
        tool_instances=tool_instances or {},
        defense=defense,
        data_dir=data_dir,
        log_dir=log_dir,
        results_dir=results_dir,
        model=model,
        temperature=temperature,
        pollution_spec=pollution_spec,
        skip_harvester=skip_harvester,
        skip_polluter=skip_polluter,
    )
    return Orchestrator(config).run()
