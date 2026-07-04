"""Analytics correctness: percentile math, cost aggregation, partition pruning."""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmobs import analytics, config as C  # noqa: E402


def _scan(data_dir):
    return (f"read_parquet('{data_dir}/{C.PARQUET_DIRNAME}/**/*.parquet', "
            f"hive_partitioning=1)")


def test_percentile_math_matches_numpy(dataset, con):
    """DuckDB quantile_cont must match numpy's linear-interp percentile."""
    lat = con.execute(
        f"SELECT latency_ms FROM {_scan(dataset)} WHERE model='sparrow-4b-lite'"
    ).df()["latency_ms"].to_numpy()
    for q in (0.5, 0.9, 0.99):
        duck = con.execute(
            f"SELECT quantile_cont(latency_ms, {q}) FROM {_scan(dataset)} "
            f"WHERE model='sparrow-4b-lite'").fetchone()[0]
        npq = np.percentile(lat, q * 100)
        assert abs(duck - npq) <= 1e-4 * max(1.0, abs(npq))


def test_percentile_monotonic(dataset, con):
    ov = analytics.overview(con, dataset)
    assert ov["p50_latency"] <= ov["p99_latency"]
    bm = analytics.by_model(con, dataset)
    assert (bm["p50_ms"] <= bm["p90_ms"]).all()
    assert (bm["p90_ms"] <= bm["p99_ms"]).all()


def test_cost_aggregation_consistency(dataset, con):
    """Overview total cost == sum of per-model cost == sum of per-day cost."""
    ov = analytics.overview(con, dataset)
    by_model = analytics.by_model(con, dataset)
    by_day = analytics.by_day(con, dataset)
    total = ov["total_cost"]
    assert abs(by_model["cost_usd"].sum() - total) < 1e-6 * total
    assert abs(by_day["cost_usd"].sum() - total) < 1e-6 * total
    # tokens too
    assert int(by_model["tokens"].sum()) == int(ov["total_tokens"])


def test_error_and_timeout_rates_bounded(dataset, con):
    ov = analytics.overview(con, dataset)
    assert 0.0 <= ov["error_rate"] < 0.1
    assert 0.0 <= ov["timeout_rate"] < 0.05
    # errors climb after cutover -> global rate above the baseline injection rate
    assert ov["error_rate"] > C.BASE_ERROR_RATE * 0.8


def test_tenant_cost_sums_to_total(dataset, con):
    ov = analytics.overview(con, dataset)
    bt = analytics.by_tenant(con, dataset)
    assert set(bt["tenant"]) == set(C.TENANTS)
    assert abs(bt["cost_usd"].sum() - ov["total_cost"]) < 1e-6 * ov["total_cost"]


def test_throughput_positive(dataset, con):
    tp = analytics.throughput(con, dataset)
    assert tp["peak_rpm"].iloc[0] >= tp["median_rpm"].iloc[0] > 0
