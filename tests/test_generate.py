"""Generator correctness: schema, determinism, partitioning, injected regimes."""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmobs import config as C, generate  # noqa: E402


def _scan(data_dir):
    return (f"read_parquet('{data_dir}/{C.PARQUET_DIRNAME}/**/*.parquet', "
            f"hive_partitioning=1)")


def test_row_count_and_partitions(dataset):
    con = duckdb.connect()
    n = con.execute(f"SELECT count(*) FROM {_scan(dataset)}").fetchone()[0]
    assert n == 120_000
    parts = list((Path(dataset) / C.PARQUET_DIRNAME).glob("date=*/data.parquet"))
    assert len(parts) == C.N_DAYS


def test_schema_and_no_nulls(dataset):
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM {_scan(dataset)} LIMIT 5").df()
    required = {"trace_id", "ts", "model", "tenant", "prompt", "response",
                "prompt_tokens", "completion_tokens", "latency_ms", "cost_usd",
                "n_tool_calls", "status", "is_hallucination", "regime"}
    assert required <= set(df.columns)
    nulls = con.execute(
        f"SELECT sum((prompt IS NULL)::INT) + sum((cost_usd IS NULL)::INT) "
        f"FROM {_scan(dataset)}").fetchone()[0]
    assert nulls == 0


def test_trace_ids_unique(dataset):
    con = duckdb.connect()
    total, distinct = con.execute(
        f"SELECT count(*), count(DISTINCT trace_id) FROM {_scan(dataset)}").fetchone()
    assert total == distinct


def test_determinism():
    """Same seed -> identical bytes for a given day partition."""
    import numpy as np
    rng1 = np.random.default_rng(C.GLOBAL_SEED + 1000 + 5)
    rng2 = np.random.default_rng(C.GLOBAL_SEED + 1000 + 5)
    t1 = generate._generate_day(5, 2000, rng1, 0, 0)
    t2 = generate._generate_day(5, 2000, rng2, 0, 0)
    assert t1.to_pandas().equals(t2.to_pandas())


def test_cost_formula_matches_pricing(dataset):
    """Recompute cost from tokens+pricing and match stored cost_usd."""
    con = duckdb.connect()
    df = con.execute(
        f"SELECT model, prompt_tokens, completion_tokens, cost_usd "
        f"FROM {_scan(dataset)} USING SAMPLE 5000 ROWS").df()
    price_in = {m.name: m.price_in for m in C.MODELS}
    price_out = {m.name: m.price_out for m in C.MODELS}
    expected = (df["prompt_tokens"] / 1e6 * df["model"].map(price_in)
                + df["completion_tokens"] / 1e6 * df["model"].map(price_out))
    assert np.allclose(expected.to_numpy(), df["cost_usd"].to_numpy(), rtol=1e-6)


def test_injected_regime_shift(dataset):
    """Post-cutover regime must show a higher hallucination rate than baseline."""
    con = duckdb.connect()
    df = con.execute(
        f"SELECT regime, avg(is_hallucination::INT) hr, avg(is_error::INT) er "
        f"FROM {_scan(dataset)} GROUP BY regime").df().set_index("regime")
    assert df.loc["drift", "hr"] > df.loc["baseline", "hr"] + 0.01
    assert df.loc["drift", "er"] > df.loc["baseline", "er"]


def test_regressed_model_tail_latency(dataset):
    """The regressed model's post-cutover p99 latency must inflate."""
    con = duckdb.connect()
    q = f"""
      SELECT regime, quantile_cont(latency_ms, 0.99) p99
      FROM {_scan(dataset)}
      WHERE model = '{C.REGRESSED_MODEL}' AND status='ok'
      GROUP BY regime
    """
    df = con.execute(q).df().set_index("regime")
    assert df.loc["drift", "p99"] > 1.4 * df.loc["baseline", "p99"]
