#!/usr/bin/env python3
"""Scaling benchmark: generation throughput + out-of-core query latency.

Generates the dataset at several sizes, times generation and a representative
set of DuckDB analytics queries, and writes benchmarks/scaling.csv + .md.
Each size is generated into a temp dir and removed afterwards, so peak disk is
bounded by the largest single run.

Usage:
    python scripts/benchmark_scaling.py --sizes 100000 500000 1000000 2000000
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from llmobs import analytics, drift, generate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"


def _bench_one(n: int) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="obs_bench_"))
    try:
        gen = generate.generate(n, tmp, verbose=False)
        con = analytics.connect()
        # warm the parquet metadata cache
        analytics.overview(con, tmp)

        def _t(fn):
            t0 = time.time()
            fn()
            return (time.time() - t0) * 1000

        q_overview = _t(lambda: analytics.overview(con, tmp))
        q_by_model = _t(lambda: analytics.by_model(con, tmp))
        q_by_day = _t(lambda: analytics.by_day(con, tmp))
        q_tenant = _t(lambda: analytics.by_tenant(con, tmp))
        wr = analytics.window_response_sample(con, tmp, per_day=1500)
        q_drift = _t(lambda: drift.detect(wr))

        return {
            "rows": n,
            "parquet_gb": round(gen["gb"], 3),
            "gen_s": round(gen["elapsed_s"], 2),
            "gen_rows_per_s": round(gen["rows_per_s"]),
            "q_overview_ms": round(q_overview, 1),
            "q_by_model_ms": round(q_by_model, 1),
            "q_by_day_ms": round(q_by_day, 1),
            "q_by_tenant_ms": round(q_tenant, 1),
            "drift_detect_ms": round(q_drift, 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100_000, 500_000, 1_000_000, 2_000_000])
    args = ap.parse_args()

    BENCH.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.sizes:
        print(f"benchmarking {n:,} rows ...", flush=True)
        r = _bench_one(n)
        rows.append(r)
        print(f"  gen {r['gen_s']}s ({r['gen_rows_per_s']:,} rows/s), "
              f"{r['parquet_gb']} GB, overview {r['q_overview_ms']} ms", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(BENCH / "scaling.csv", index=False)
    with open(BENCH / "scaling.md", "w") as f:
        f.write("# Scaling benchmark\n\n")
        f.write("Generation throughput and out-of-core DuckDB query latency by "
                "dataset size (single 4-core box, zstd Parquet).\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"\nwrote {BENCH/'scaling.csv'} and {BENCH/'scaling.md'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
