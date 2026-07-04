#!/usr/bin/env python3
"""Generate the synthetic LLM trace dataset -> partitioned Parquet.

Usage:
    python scripts/generate_data.py --traces 5000000 --out data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmobs import config as C  # noqa: E402
from llmobs import generate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=int, default=5_000_000,
                    help="total number of traces to generate")
    ap.add_argument("--out", type=str, default="data", help="output data dir")
    ap.add_argument("--seed", type=int, default=C.GLOBAL_SEED)
    ap.add_argument("--days", type=int, default=C.N_DAYS)
    args = ap.parse_args()

    print(f"Generating {args.traces:,} traces over {args.days} days "
          f"(seed={args.seed}) -> {args.out}/{C.PARQUET_DIRNAME}")
    summary = generate.generate(args.traces, args.out, seed=args.seed,
                                n_days=args.days, verbose=True)

    out = Path(args.out)
    bench = out.parent / "benchmarks"
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "scale.json").write_text(json.dumps(summary, indent=2))

    print("\n=== generation complete ===")
    print(f"  traces      : {summary['total_traces']:,}")
    print(f"  partitions  : {summary['n_partitions']} (daily)")
    print(f"  parquet size: {summary['gb']:.2f} GB")
    print(f"  elapsed     : {summary['elapsed_s']:.1f} s "
          f"({summary['rows_per_s']:,.0f} rows/s)")
    print(f"  wrote scale summary -> {bench / 'scale.json'}")


if __name__ == "__main__":
    main()
