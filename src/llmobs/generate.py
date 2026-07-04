"""Synthetic LLM trace generator.

Streams traces day-by-day into a Hive-partitioned Parquet dataset
(``data/traces/date=YYYY-MM-DD/data.parquet``). Memory is bounded by one day
of traffic regardless of the total ``--traces``, so the same code path scales
from 10k to billions of rows.

Each trace carries request/response text, model, prompt/completion tokens,
latency, cost, timestamp, tool calls, tenant/user, and *injected* quality
issues (errors, timeouts, hallucinations) plus a post-cutover drift regime.

Determinism: a per-day seed is derived from ``GLOBAL_SEED`` so the dataset is
byte-reproducible and each day can be regenerated independently.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C

# Fixed synthetic text lengths (words). Token *accounting* below is independent
# and realistic; these only shape the TF-IDF / groundedness signal.
_PROMPT_WORDS = 10
_RESP_WORDS = 16

# Materialise topic vocab as a padded 2D object array for vectorised gather.
_TOPIC_VOCAB = np.array([C.TOPICS[t] for t in C.TOPIC_NAMES], dtype=object)  # (T, 8)
_VOCAB_W = _TOPIC_VOCAB.shape[1]
_TOPIC_NAME_ARR = np.array(C.TOPIC_NAMES, dtype=object)
_HEDGE_ARR = np.array(C.HEDGE_TOKENS, dtype=object)
_MODEL_PRICE_IN = np.array([m.price_in for m in C.MODELS])
_MODEL_PRICE_OUT = np.array([m.price_out for m in C.MODELS])
_MODEL_BASE_LAT = np.array([m.base_latency_ms for m in C.MODELS])
_MODEL_WEIGHTS = np.array([m.weight for m in C.MODELS])
_REGRESSED_IDX = C.MODEL_NAMES.index(C.REGRESSED_MODEL)


def _arrow_schema() -> pa.schema:
    return pa.schema([
        ("trace_id", pa.int64()),
        ("ts", pa.timestamp("ms")),
        ("model", pa.string()),
        ("tenant", pa.string()),
        ("user_id", pa.int32()),
        ("topic", pa.string()),
        ("prompt", pa.string()),
        ("response", pa.string()),
        ("prompt_tokens", pa.int32()),
        ("completion_tokens", pa.int32()),
        ("total_tokens", pa.int32()),
        ("latency_ms", pa.float32()),
        ("cost_usd", pa.float64()),
        ("n_tool_calls", pa.int16()),
        ("status", pa.string()),
        ("is_error", pa.bool_()),
        ("is_timeout", pa.bool_()),
        ("is_hallucination", pa.bool_()),
        ("regime", pa.string()),
    ])


def _day_counts(total: int, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Split ``total`` traces across days with mild weekly seasonality + growth."""
    days = np.arange(n_days)
    weekly = 1.0 + 0.18 * np.sin(2 * np.pi * days / 7.0)        # weekday rhythm
    growth = 1.0 + 0.6 * days / max(n_days - 1, 1)              # traffic grows over time
    shape = weekly * growth
    shape = shape / shape.sum()
    counts = np.floor(shape * total).astype(np.int64)
    counts[-1] += total - counts.sum()  # exact total
    return counts


def _vjoin(mat: np.ndarray) -> np.ndarray:
    """Vectorised space-join of a (n, k) string/object matrix into (n,) strings."""
    out = mat[:, 0].astype(str)
    for j in range(1, mat.shape[1]):
        out = np.char.add(np.char.add(out, " "), mat[:, j].astype(str))
    return out


def _build_text(topic_idx: np.ndarray, is_halluc: np.ndarray,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Fully-vectorised construction of prompt/response text.

    Grounded answers reuse the prompt's topic vocabulary (high token overlap);
    hallucinated answers pull from a *foreign* topic and append "hedge" filler
    (low overlap). Label noise blurs the boundary so detection is non-trivial.
    Returns (prompts, responses) as numpy string arrays.
    """
    n = topic_idx.shape[0]
    T = len(C.TOPIC_NAMES)

    # ----- prompt: topic name + words sampled from that topic's vocab -----
    p_ids = rng.integers(0, _VOCAB_W, size=(n, _PROMPT_WORDS))
    p_words = _TOPIC_VOCAB[topic_idx[:, None], p_ids]                 # (n, Wp)
    prompt_mat = np.concatenate([_TOPIC_NAME_ARR[topic_idx][:, None], p_words], axis=1)

    # ----- response topic. Overlapping distributions keep detection non-trivial:
    #   * ~35% of hallucinations answer *on-topic* (plausible-but-wrong) -> look
    #     grounded, the hard positives.
    #   * ~10% of grounded answers drift to a foreign topic -> hard negatives.
    shift = rng.integers(1, T, size=n)
    foreign_topic = (topic_idx + shift) % T
    ontopic_halluc = is_halluc & (rng.random(n) < 0.35)
    foreign_halluc = is_halluc & ~ontopic_halluc
    drift_grounded = (~is_halluc) & (rng.random(n) < 0.10)
    resp_topic = np.where(foreign_halluc | drift_grounded, foreign_topic, topic_idx)

    r_ids = rng.integers(0, _VOCAB_W, size=(n, _RESP_WORDS))
    r_words = _TOPIC_VOCAB[resp_topic[:, None], r_ids]               # (n, Wr)

    # "Hedge" filler tokens: present in ~55% of hallucinations, but also a ~6%
    # false-positive rate on grounded answers, so hedge count is a noisy signal.
    hedge_present = (is_halluc & (rng.random(n) < 0.55)) \
        | ((~is_halluc) & (rng.random(n) < 0.06))
    h_ids = rng.integers(0, len(_HEDGE_ARR), size=(n, 2))
    hedge_words = _HEDGE_ARR[h_ids]
    r_words[:, -2:] = np.where(hedge_present[:, None], hedge_words, r_words[:, -2:])

    resp_mat = np.concatenate([_TOPIC_NAME_ARR[resp_topic][:, None], r_words], axis=1)
    return _vjoin(prompt_mat), _vjoin(resp_mat)


def _generate_day(day: int, n: int, rng: np.random.Generator,
                  start_epoch_ms: int, id_offset: int) -> pa.Table:
    is_drift = day >= C.CUTOVER_DAY
    topic_w = np.array(C.POST_TOPIC_WEIGHTS if is_drift else C.PRE_TOPIC_WEIGHTS)
    topic_w = topic_w / topic_w.sum()

    # ----- categorical draws -----
    model_idx = rng.choice(len(C.MODELS), size=n, p=_MODEL_WEIGHTS / _MODEL_WEIGHTS.sum())
    tenant_idx = rng.choice(len(C.TENANTS), size=n,
                            p=np.array(C.TENANT_WEIGHTS) / sum(C.TENANT_WEIGHTS))
    user_id = rng.integers(0, C.USERS_PER_TENANT, size=n).astype(np.int32) \
        + tenant_idx.astype(np.int32) * C.USERS_PER_TENANT
    topic_idx = rng.choice(len(C.TOPIC_NAMES), size=n, p=topic_w)

    # ----- token accounting (independent of the text length above) -----
    prompt_tokens = np.clip(rng.lognormal(mean=4.9, sigma=0.55, size=n), 8, 8000).astype(np.int32)
    comp_mu = 4.9 + (0.35 if is_drift else 0.0)  # completions get longer post-cutover
    # agents/reasoning-style topics (last four) produce longer completions
    long_topic = topic_idx >= (len(C.TOPIC_NAMES) - 4)
    completion_tokens = np.clip(
        rng.lognormal(mean=comp_mu + 0.25 * long_topic, sigma=0.6, size=n), 1, 16000
    ).astype(np.int32)
    total_tokens = prompt_tokens + completion_tokens

    # ----- cost -----
    cost = (prompt_tokens / 1e6) * _MODEL_PRICE_IN[model_idx] \
        + (completion_tokens / 1e6) * _MODEL_PRICE_OUT[model_idx]

    # ----- status: timeouts + errors (errors climb after cutover) -----
    err_rate = C.BASE_ERROR_RATE * (C.DRIFT_ERROR_MULT if is_drift else 1.0)
    is_timeout = rng.random(n) < C.BASE_TIMEOUT_RATE
    is_error = (~is_timeout) & (rng.random(n) < err_rate)
    status = np.where(is_timeout, "timeout", np.where(is_error, "error", "ok"))

    # ----- latency -----
    length_factor = 0.3 + completion_tokens / 200.0
    noise = rng.lognormal(mean=0.0, sigma=0.35, size=n)
    latency = _MODEL_BASE_LAT[model_idx] * length_factor * noise
    if is_drift:  # the regressed model's tail latency inflates
        rmask = model_idx == _REGRESSED_IDX
        latency[rmask] *= 1.8 * rng.lognormal(mean=0.0, sigma=0.45, size=int(rmask.sum()))
    latency = np.where(is_timeout, C.TIMEOUT_LATENCY_MS, latency).astype(np.float32)

    # ----- tool calls (agents topics + reasoning model call more tools) -----
    tool_lam = 0.3 + 1.4 * long_topic + 0.8 * (model_idx == C.MODEL_NAMES.index("orion-reason-xl"))
    n_tool_calls = rng.poisson(tool_lam).astype(np.int16)

    # ----- hallucination label (ground truth) -----
    hall_rate = np.full(n, C.BASE_HALLUCINATION_RATE)
    if is_drift:
        hall_rate[model_idx == _REGRESSED_IDX] = C.DRIFT_HALLUCINATION_RATE
        hall_rate[long_topic] = np.maximum(hall_rate[long_topic], 0.16)
    # errors/timeouts have no usable answer -> not scored as hallucination
    is_halluc = (rng.random(n) < hall_rate) & (status == "ok")

    prompts, responses = _build_text(topic_idx, is_halluc, rng)

    # ----- timestamps: spread across the day with a diurnal pattern -----
    day_ms = 86_400_000
    frac = rng.beta(2.0, 2.0, size=n)  # busier mid-day
    ts = (start_epoch_ms + day * day_ms + (frac * day_ms)).astype(np.int64)
    order = np.argsort(ts)

    trace_id = (id_offset + np.arange(n)).astype(np.int64)
    regime = np.full(n, "drift" if is_drift else "baseline", dtype=object)

    tbl = pa.table({
        "trace_id": trace_id[order],
        "ts": pa.array(ts[order], type=pa.timestamp("ms")),
        "model": np.array(C.MODEL_NAMES, dtype=object)[model_idx][order],
        "tenant": np.array(C.TENANTS, dtype=object)[tenant_idx][order],
        "user_id": user_id[order],
        "topic": _TOPIC_NAME_ARR[topic_idx][order],
        "prompt": prompts[order],
        "response": responses[order],
        "prompt_tokens": prompt_tokens[order],
        "completion_tokens": completion_tokens[order],
        "total_tokens": total_tokens[order],
        "latency_ms": latency[order],
        "cost_usd": cost[order],
        "n_tool_calls": n_tool_calls[order],
        "status": status[order],
        "is_error": is_error[order],
        "is_timeout": is_timeout[order],
        "is_hallucination": is_halluc[order],
        "regime": regime[order],
    }, schema=_arrow_schema())
    return tbl


def generate(total: int, out_dir: str | Path, *, seed: int = C.GLOBAL_SEED,
             n_days: int = C.N_DAYS, verbose: bool = True) -> dict:
    """Generate ``total`` traces into a partitioned Parquet dataset.

    Returns a summary dict with real counts, bytes, and elapsed seconds.
    """
    out = Path(out_dir)
    root = out / C.PARQUET_DIRNAME
    root.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timezone
    start_epoch_ms = int(datetime.fromisoformat(C.START_DATE)
                         .replace(tzinfo=timezone.utc).timestamp() * 1000)
    counts = _day_counts(total, n_days, np.random.default_rng(seed))

    t0 = time.time()
    total_bytes = 0
    id_offset = 0
    for day in range(n_days):
        n = int(counts[day])
        if n == 0:
            continue
        rng = np.random.default_rng(seed + 1000 + day)  # independent per-day stream
        tbl = _generate_day(day, n, rng, start_epoch_ms, id_offset)
        from datetime import timedelta
        date_str = (datetime.fromisoformat(C.START_DATE) + timedelta(days=day)).strftime("%Y-%m-%d")
        part_dir = root / f"{C.PARTITION_COL}={date_str}"
        part_dir.mkdir(parents=True, exist_ok=True)
        fp = part_dir / "data.parquet"
        pq.write_table(tbl, fp, compression="zstd", compression_level=3)
        total_bytes += fp.stat().st_size
        id_offset += n
        if verbose and (day % 10 == 0 or day == n_days - 1):
            print(f"  day {day:>3}/{n_days}  {date_str}  rows={n:>8,}  "
                  f"cum_bytes={total_bytes/1e6:8.1f} MB  "
                  f"elapsed={time.time()-t0:6.1f}s", flush=True)

    elapsed = time.time() - t0
    summary = {
        "total_traces": int(total),
        "n_days": n_days,
        "n_partitions": int((counts > 0).sum()),
        "bytes": int(total_bytes),
        "gb": total_bytes / 1e9,
        "elapsed_s": elapsed,
        "rows_per_s": total / elapsed if elapsed else 0.0,
        "path": str(root),
    }
    return summary
