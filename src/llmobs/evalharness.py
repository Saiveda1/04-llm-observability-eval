"""Eval harness: per-model quality scorecards.

Combines DuckDB analytics (cost, latency, error rates) with the injected
ground-truth hallucination rate and a composite quality score to produce a
single per-model scorecard, plus a fleet-level rollup and letter grades.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from . import analytics


def _grade(score: float) -> str:
    for cut, g in ((0.90, "A"), (0.80, "B"), (0.70, "C"), (0.60, "D")):
        if score >= cut:
            return g
    return "F"


def scorecard(con: duckdb.DuckDBPyConnection, data_dir: str) -> pd.DataFrame:
    """Per-model scorecard with a composite 0-1 quality score and letter grade.

    Composite quality = weighted blend of groundedness (1 - halluc_rate),
    reliability (1 - error - timeout), and a latency-health term (p99 vs a
    2s SLO). Weights are explicit and documented for auditability.
    """
    m = analytics.by_model(con, data_dir).copy()

    reliability = 1.0 - (m["error_rate"] + m["timeout_rate"])
    grounded = 1.0 - m["halluc_rate"]
    # Latency health: 1.0 at/under a 2000 ms p99 SLO, decaying above it.
    slo_ms = 2000.0
    latency_health = (slo_ms / m["p99_ms"]).clip(upper=1.0)

    m["quality_score"] = (0.45 * grounded
                          + 0.35 * reliability
                          + 0.20 * latency_health).round(4)
    m["grade"] = m["quality_score"].apply(_grade)
    m["cost_per_1k_traces"] = m["cost_per_1k_traces"].round(4)
    cols = ["model", "traces", "tokens", "cost_usd", "cost_per_1k_traces",
            "p50_ms", "p90_ms", "p99_ms", "error_rate", "timeout_rate",
            "halluc_rate", "quality_score", "grade"]
    return m[cols].sort_values("quality_score", ascending=False).reset_index(drop=True)


def fleet_rollup(scorecard_df: pd.DataFrame) -> dict:
    total_traces = int(scorecard_df["traces"].sum())
    w = scorecard_df["traces"] / total_traces
    return {
        "total_traces": total_traces,
        "total_cost_usd": float(scorecard_df["cost_usd"].sum()),
        "weighted_quality": float((scorecard_df["quality_score"] * w).sum()),
        "weighted_halluc_rate": float((scorecard_df["halluc_rate"] * w).sum()),
        "best_model": scorecard_df.iloc[0]["model"],
        "worst_model": scorecard_df.iloc[-1]["model"],
    }
