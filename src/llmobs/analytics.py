"""DuckDB analytics layer over the partitioned Parquet trace dataset.

Every metric is expressed as real SQL that DuckDB executes out-of-core: only
the columns a query touches are scanned, and aggregations stream, so these run
over datasets far larger than RAM. Nothing here materialises the full table.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from . import config as C


def connect(threads: int = 4, memory_limit: str = "4GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    return con


def _glob(data_dir: str | Path) -> str:
    root = Path(data_dir) / C.PARQUET_DIRNAME
    return str(root / "**" / "*.parquet")


def _scan(data_dir: str | Path) -> str:
    """A read_parquet(...) table expression with Hive partition columns."""
    return (f"read_parquet('{_glob(data_dir)}', hive_partitioning=1, "
            f"union_by_name=1)")


def overview(con: duckdb.DuckDBPyConnection, data_dir: str | Path) -> dict:
    """Fleet-wide KPI tiles in a single scan."""
    row = con.execute(f"""
        SELECT
            count(*)                                   AS total_traces,
            sum(total_tokens)                          AS total_tokens,
            sum(cost_usd)                              AS total_cost,
            avg(latency_ms)                            AS avg_latency,
            quantile_cont(latency_ms, 0.50)            AS p50_latency,
            quantile_cont(latency_ms, 0.99)            AS p99_latency,
            avg(is_error::INT)                         AS error_rate,
            avg(is_timeout::INT)                       AS timeout_rate,
            avg(is_hallucination::INT)                 AS halluc_rate,
            count(DISTINCT tenant)                     AS tenants,
            count(DISTINCT model)                      AS models
        FROM {_scan(data_dir)}
    """).fetchone()
    cols = ["total_traces", "total_tokens", "total_cost", "avg_latency",
            "p50_latency", "p99_latency", "error_rate", "timeout_rate",
            "halluc_rate", "tenants", "models"]
    return dict(zip(cols, row))


def by_model(con: duckdb.DuckDBPyConnection, data_dir: str | Path) -> pd.DataFrame:
    """Cost / token / latency-percentile / error scorecard per model."""
    return con.execute(f"""
        SELECT
            model,
            count(*)                                   AS traces,
            sum(total_tokens)                          AS tokens,
            sum(cost_usd)                              AS cost_usd,
            avg(cost_usd) * 1000                       AS cost_per_1k_traces,
            quantile_cont(latency_ms, 0.50)            AS p50_ms,
            quantile_cont(latency_ms, 0.90)            AS p90_ms,
            quantile_cont(latency_ms, 0.99)            AS p99_ms,
            avg(is_error::INT)                         AS error_rate,
            avg(is_timeout::INT)                       AS timeout_rate,
            avg(is_hallucination::INT)                 AS halluc_rate
        FROM {_scan(data_dir)}
        GROUP BY model
        ORDER BY cost_usd DESC
    """).df()


def by_day(con: duckdb.DuckDBPyConnection, data_dir: str | Path) -> pd.DataFrame:
    """Daily time-series: volume, spend, latency percentiles, error rate."""
    return con.execute(f"""
        SELECT
            date,
            count(*)                                   AS traces,
            sum(cost_usd)                              AS cost_usd,
            sum(total_tokens)                          AS tokens,
            quantile_cont(latency_ms, 0.50)            AS p50_ms,
            quantile_cont(latency_ms, 0.90)            AS p90_ms,
            quantile_cont(latency_ms, 0.99)            AS p99_ms,
            avg(is_error::INT)                         AS error_rate,
            avg(is_timeout::INT)                       AS timeout_rate,
            avg(is_hallucination::INT)                 AS halluc_rate
        FROM {_scan(data_dir)}
        GROUP BY date
        ORDER BY date
    """).df()


def by_tenant(con: duckdb.DuckDBPyConnection, data_dir: str | Path) -> pd.DataFrame:
    """Cost attribution and quality per tenant (chargeback view)."""
    return con.execute(f"""
        SELECT
            tenant,
            count(*)                                   AS traces,
            sum(cost_usd)                              AS cost_usd,
            sum(total_tokens)                          AS tokens,
            quantile_cont(latency_ms, 0.99)            AS p99_ms,
            avg(is_hallucination::INT)                 AS halluc_rate
        FROM {_scan(data_dir)}
        GROUP BY tenant
        ORDER BY cost_usd DESC
    """).df()


def by_model_day(con: duckdb.DuckDBPyConnection, data_dir: str | Path) -> pd.DataFrame:
    """Per-model daily latency percentiles (for the percentile-band chart)."""
    return con.execute(f"""
        SELECT
            date, model,
            count(*)                                   AS traces,
            quantile_cont(latency_ms, 0.50)            AS p50_ms,
            quantile_cont(latency_ms, 0.90)            AS p90_ms,
            quantile_cont(latency_ms, 0.99)            AS p99_ms
        FROM {_scan(data_dir)}
        GROUP BY date, model
        ORDER BY date, model
    """).df()


def throughput(con: duckdb.DuckDBPyConnection, data_dir: str | Path) -> pd.DataFrame:
    """Requests-per-minute distribution (peak / sustained throughput)."""
    return con.execute(f"""
        WITH per_min AS (
            SELECT date_trunc('minute', ts) AS minute, count(*) AS rpm
            FROM {_scan(data_dir)}
            GROUP BY 1
        )
        SELECT
            quantile_cont(rpm, 0.50) AS median_rpm,
            quantile_cont(rpm, 0.95) AS p95_rpm,
            max(rpm)                 AS peak_rpm
        FROM per_min
    """).df()


def sample_for_eval(con: duckdb.DuckDBPyConnection, data_dir: str | Path,
                    n: int = 200_000, seed: int = C.GLOBAL_SEED) -> pd.DataFrame:
    """Deterministic reservoir-style sample of successful traces for the
    hallucination detector (mirrors real eval sampling on a firehose)."""
    con.execute(f"SELECT setseed({(seed % 1000) / 1000.0})")
    return con.execute(f"""
        SELECT trace_id, date, model, topic, prompt, response,
               prompt_tokens, completion_tokens, n_tool_calls,
               latency_ms, is_hallucination
        FROM {_scan(data_dir)}
        WHERE status = 'ok'
        USING SAMPLE {n} ROWS (reservoir, {seed})
    """).df()


def window_response_sample(con: duckdb.DuckDBPyConnection, data_dir: str | Path,
                           per_day: int = 3_000, seed: int = C.GLOBAL_SEED) -> pd.DataFrame:
    """Balanced per-day sample of responses for drift analysis."""
    return con.execute(f"""
        SELECT date, response
        FROM (
            SELECT date, response,
                   row_number() OVER (PARTITION BY date
                                      ORDER BY hash(trace_id + {seed})) AS rn
            FROM {_scan(data_dir)}
        )
        WHERE rn <= {per_day}
        ORDER BY date
    """).df()
