# Architecture

## Problem

Production LLM platforms emit a firehose of traces — one per request — each
carrying text, token counts, latency, cost, tool calls, and a tenant. At fleet
scale that is **billions of rows/month**. Teams need three things from that
firehose, continuously and cheaply:

1. **Operational analytics** — spend, tokens, latency percentiles, error/timeout
   rates, throughput, sliced by model / day / tenant.
2. **Drift detection** — catch when the input or output distribution shifts
   (a model cutover, a prompt-template change, an abuse wave) *before* it shows
   up as a customer-visible quality regression.
3. **Quality / hallucination monitoring** — score groundedness at sampling rate
   and track a per-model quality scorecard over time.

This repo is a self-contained, offline reference implementation of all three.

## Component map

```
            ┌─────────────────────────────────────────────────────────┐
            │  generate.py  — synthetic trace firehose (streamed)       │
            │  vectorised NumPy, day-by-day, bounded memory             │
            └───────────────┬─────────────────────────────────────────┘
                            │  Hive-partitioned Parquet (zstd)
                            │  data/traces/date=YYYY-MM-DD/data.parquet
                            ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  DuckDB analytics (analytics.py)  — out-of-core SQL over Parquet    │
   │   overview · by_model · by_day · by_tenant · by_model_day · rpm     │
   └───────┬───────────────────────┬────────────────────────┬───────────┘
           │ per-day response       │ labelled sample         │ group-bys
           │ sample                 │ (reservoir)             │
           ▼                        ▼                         ▼
   ┌───────────────┐      ┌──────────────────┐      ┌────────────────────┐
   │ drift.py      │      │ detector.py      │      │ evalharness.py     │
   │ TF-IDF+SVD    │      │ heuristic +      │      │ per-model quality  │
   │ → PSI / KL    │      │ logistic clf     │      │ scorecards + grade │
   └───────────────┘      └──────────────────┘      └────────────────────┘
```

## Key design decisions & trade-offs

### Storage: Hive-partitioned Parquet, partitioned by day
- **Why Parquet + partitions:** columnar scans mean a cost query touches only
  `cost_usd`, not the (large) text columns; day partitions let DuckDB prune
  entire files for time-ranged queries. zstd keeps the text columns small
  because the synthetic vocabulary dictionary-encodes well.
- **Why day granularity:** matches the natural query grain (daily dashboards,
  daily drift windows) and keeps per-file row counts in the ~100k range — big
  enough to amortise metadata, small enough to prune finely.
- **Trade-off:** many small tenants share a file. A production build would add a
  second partition level (`tenant_bucket`) or Z-order/cluster on tenant for
  tenant-scoped queries; here a single scan is fast enough and simpler.

### Generation: streamed, vectorised, deterministic
- The generator emits **one day at a time**; peak memory is one day of traffic,
  not the whole dataset. That is what lets the *same code* scale from 100k to
  billions of rows — the `--traces` knob only changes how many day-shards get
  written, never the working-set size.
- Text is built with **fully vectorised `np.char`** joins (no Python per-row
  loop), so 5M rows generate in tens of seconds on one box.
- Every day-shard is seeded independently (`GLOBAL_SEED + day`), so the dataset
  is byte-reproducible and any single day can be regenerated in isolation.

### Analytics: DuckDB out-of-core
- All metrics are plain SQL executed by DuckDB directly against the Parquet
  glob (`read_parquet(..., hive_partitioning=1)`). Aggregations stream; the
  engine never materialises the full table, so queries run over datasets much
  larger than RAM. Percentiles use `quantile_cont` (linear interpolation),
  validated against NumPy in the test suite.

### Drift: TF-IDF + SVD → PSI / KL
- Embeddings are **deterministic and offline**: a `HashingVectorizer` (fixed
  feature space, no vocabulary to fit) + `TfidfTransformer` + `TruncatedSVD`
  fit on the baseline window. No model download, no API.
- Each response is reduced to a scalar drift score (projection on the leading
  baseline component). We bin the baseline into deciles and compute **PSI** and
  **KL divergence** of each day against the baseline. PSI ≥ 0.25 raises an
  alert (industry-standard threshold). Drift runs on a *per-day sample*, not the
  full firehose — the standard, cheap way to monitor distributional drift.

### Detector: heuristic + logistic, evaluated honestly
- A cheap **groundedness heuristic** (prompt/response Jaccard overlap + hedge-
  phrase count) runs inline and is reported as a standalone baseline.
- A **logistic classifier** over groundedness + hedge + length/token/tool
  features is trained on a labelled sample and evaluated on a held-out split.
- We report precision/recall/F1/ROC-AUC/AP for both **and** the random baseline
  (AP == prevalence), so the lift is unambiguous. The injected labels have
  overlapping feature distributions (on-topic hallucinations, drifting grounded
  answers, noisy hedge signal) so the task is deliberately non-trivial.

## Scaling to 1B+ rows

| Concern | This repo (measured) | Path to 1B/month |
|---|---|---|
| **Generation** | streamed day-shards, bounded memory, ~2–3M rows/s core loop | embarrassingly parallel by day/shard across workers; write to object store |
| **Storage** | day-partitioned zstd Parquet | add `tenant_bucket` partition + Z-order; land in S3/GCS; lifecycle to cold tiers |
| **Analytics** | DuckDB out-of-core, single box | DuckDB scales vertically; for horizontal, the identical SQL runs on a lakehouse engine (Athena/Trino/BigQuery/Spark) over the same Parquet |
| **Drift** | per-day sampled embeddings | sample rate is size-independent — cost is O(windows), not O(rows); precompute embeddings in the ingest path |
| **Detector** | fixed-size labelled sample (200k) | sample size, not corpus size, drives training cost; retrain on a schedule; serve the linear model inline at ingest |
| **Roll-ups** | recomputed on read | pre-aggregate to hourly/daily cubes on ingest so dashboards read cubes, not raw traces |

The invariant: **every stage's cost is a function of windows or samples, not of
the raw row count** — which is exactly why the design extrapolates to billions
without changing shape. Only the generator and the group-by scans touch every
row, and both are streaming/columnar and horizontally partitionable.

## Reproducibility
- `GLOBAL_SEED` threads through generation, sampling, SVD, and the classifier.
- `MPLBACKEND=Agg`, fixed palette (`viztheme.py`), fixed figure DPI.
- `make all` reproduces data → metrics → screenshots from scratch.
