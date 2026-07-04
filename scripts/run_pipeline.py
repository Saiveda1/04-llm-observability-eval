#!/usr/bin/env python3
"""Run the full observability pipeline over the Parquet dataset.

DuckDB analytics -> drift detection -> hallucination detector -> eval harness.
Persists every result table into ``benchmarks/`` for the screenshot renderer
and the README, and prints headline numbers + query timings.

Usage:
    python scripts/run_pipeline.py --data data
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from llmobs import analytics, config as C, detector, drift, evalharness  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"


def _timed(label: str, fn, timings: dict):
    t0 = time.time()
    out = fn()
    dt = (time.time() - t0) * 1000
    timings[label] = dt
    print(f"  [{dt:8.1f} ms]  {label}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--eval-sample", type=int, default=200_000)
    ap.add_argument("--drift-per-day", type=int, default=3_000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    data_dir = args.data
    BENCH.mkdir(parents=True, exist_ok=True)
    con = analytics.connect(threads=args.threads)
    timings: dict[str, float] = {}

    print("== DuckDB analytics (out-of-core over Parquet) ==")
    overview = _timed("overview KPIs", lambda: analytics.overview(con, data_dir), timings)
    by_model = _timed("cost/latency by model", lambda: analytics.by_model(con, data_dir), timings)
    by_day = _timed("time-series by day", lambda: analytics.by_day(con, data_dir), timings)
    by_tenant = _timed("cost by tenant", lambda: analytics.by_tenant(con, data_dir), timings)
    by_model_day = _timed("latency percentiles by model/day",
                          lambda: analytics.by_model_day(con, data_dir), timings)
    tput = _timed("throughput (rpm)", lambda: analytics.throughput(con, data_dir), timings)

    print("\n== Drift detection (TF-IDF + SVD -> PSI/KL) ==")
    wr = _timed("sample responses per day",
                lambda: analytics.window_response_sample(con, data_dir, per_day=args.drift_per_day),
                timings)
    drift_rep = _timed("PSI/KL drift timeline", lambda: drift.detect(wr), timings)

    print("\n== Hallucination detector (heuristic + logistic) ==")
    smp = _timed("sample labelled traces",
                 lambda: analytics.sample_for_eval(con, data_dir, n=args.eval_sample), timings)
    det_rep = _timed("train + evaluate detector",
                     lambda: detector.train_and_evaluate(smp), timings)

    print("\n== Eval harness (per-model scorecards) ==")
    scorecard = _timed("model scorecards", lambda: evalharness.scorecard(con, data_dir), timings)
    fleet = evalharness.fleet_rollup(scorecard)

    # ---------------- persist artifacts ----------------
    (BENCH / "overview.json").write_text(json.dumps(overview, indent=2, default=float))
    by_model.to_csv(BENCH / "by_model.csv", index=False)
    by_day.to_csv(BENCH / "by_day.csv", index=False)
    by_tenant.to_csv(BENCH / "by_tenant.csv", index=False)
    by_model_day.to_csv(BENCH / "by_model_day.csv", index=False)
    tput.to_csv(BENCH / "throughput.csv", index=False)
    drift_rep.timeline.to_csv(BENCH / "drift_timeline.csv", index=False)
    pd.DataFrame(det_rep.pr_curve).to_csv(BENCH / "detector_pr.csv", index=False)
    pd.DataFrame({"score": det_rep.test_scores,
                  "label": det_rep.test_labels}).to_csv(BENCH / "detector_scores.csv", index=False)
    (BENCH / "detector_metrics.json").write_text(json.dumps({
        "metrics": det_rep.metrics, "coef": det_rep.coef,
        "prevalence": det_rep.prevalence,
        "n_train": det_rep.n_train, "n_test": det_rep.n_test,
    }, indent=2, default=float))
    scorecard.to_csv(BENCH / "scorecard.csv", index=False)
    (BENCH / "fleet.json").write_text(json.dumps(fleet, indent=2, default=float))
    (BENCH / "query_timings.json").write_text(json.dumps(timings, indent=2))
    (BENCH / "drift_summary.json").write_text(json.dumps({
        "first_alert_day": str(drift_rep.first_alert_day),
        "n_alert_days": len(drift_rep.alert_days),
        "threshold": drift_rep.threshold,
        "baseline_days": [str(d) for d in drift_rep.baseline_days],
        "cutover_day_index": C.CUTOVER_DAY,
    }, indent=2, default=str))

    # ---------------- headline numbers ----------------
    dm = det_rep.summary_row()
    print("\n================= HEADLINE =================")
    print(f"  traces            : {overview['total_traces']:,}")
    print(f"  total tokens      : {overview['total_tokens']:,}")
    print(f"  total spend       : ${overview['total_cost']:,.2f}")
    print(f"  p50 / p99 latency : {overview['p50_latency']:.0f} / {overview['p99_latency']:.0f} ms")
    print(f"  error / timeout   : {overview['error_rate']*100:.2f}% / {overview['timeout_rate']*100:.2f}%")
    print(f"  halluc rate (gt)  : {overview['halluc_rate']*100:.2f}%")
    print(f"  peak throughput   : {tput['peak_rpm'].iloc[0]:,.0f} req/min")
    print(f"  drift 1st alert   : {drift_rep.first_alert_day}  "
          f"(cutover index {C.CUTOVER_DAY}, {len(drift_rep.alert_days)} alert days)")
    print(f"  detector AUC/AP   : {dm['clf_auc']:.3f} / {dm['clf_ap']:.3f}  "
          f"(random AP {dm['random_ap']:.3f})")
    print(f"  detector P/R/F1   : {dm['clf_precision']:.3f} / {dm['clf_recall']:.3f} / {dm['clf_f1']:.3f}")
    print(f"  fleet quality     : {fleet['weighted_quality']:.3f}  "
          f"(best {fleet['best_model']}, worst {fleet['worst_model']})")
    print(f"  total query time  : {sum(timings.values()):.0f} ms across {len(timings)} ops")
    print("  artifacts written -> benchmarks/")


if __name__ == "__main__":
    main()
