# LLM Observability & Eval Project Document

**Prepared For:** Sai Veda  
**GitHub Publishing Account:** Nikeshk834  
**Repository Slug:** `04-llm-observability-eval`  
**Verified Test Count From Portfolio Index:** 21  

## Background

Trace analytics + quality monitoring for LLM fleets at scale — a self-contained,
**fully offline** reference implementation. It generates a realistic multi-tenant
LLM trace firehose, lands it as partitioned Parquet, and runs four production
concerns over it: **cost/latency analytics (DuckDB, out-of-core)**,
**embedding-drift detection (PSI/KL)**, a **hallucination detector
(heuristic + logistic)**, and a **per-model eval scorecard** — with generated,
product-grade dashboards.

> **Real scale achieved:** **5,000,000 traces** over 60 days across 5 models and
> 8 tenants → **1.79 billion tokens**, **$7,529 tracked spend**, **0.34 GB**
> zstd Parquet (60 daily partitions), generated in **233 s**. Full analytics
> layer answers fleet-wide KPI, cost-by-model, and latency-percentile queries in
> **0.6–2.0 s each, out-of-core** on a single 4-core box. Architected for
> **1B+ rows/month** — see [ARCHITECTURE.md](ARCHITECTURE.md).

## Project Purpose

This repository is part of the AI engineering portfolio and focuses on the following problem space:

- Trace analytics + drift + hallucination detection
- Headline result from the portfolio index: **5M traces / 1.79B tokens**; PSI flags injected drift

## What This Project Solves

This project provides a production-style implementation with benchmark evidence and operational checks committed into the repository.

## Technical Approach

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
  `cost_usd`, not

## Benchmark And Validation Evidence

The portfolio root documents **21 passing tests** for this project, and the repo quickstart uses `make test` as the standard validation path. The benchmark outputs committed in `benchmarks/` and the generated visuals in `assets/` are the evidence package for this delivery.

### scaling.md

# Scaling benchmark

Generation throughput and out-of-core DuckDB query latency by dataset size (single 4-core box, zstd Parquet).

## Visual Artifacts Reviewed

- `assets/dashboard.png`: Fleet dashboard — KPI tiles, latency-over-time, spend/tokens by model.
- `assets/drift_timeline.png`: Drift detection — PSI crosses the alert threshold exactly at the model cutover.
- `assets/latency_bands.png`: Latency percentile bands — the injected tail-latency regression is unmistakable.
- `assets/detector.png`: Hallucination detector — PR curve + score distribution vs a random baseline.

## Engineering Notes

The primary design and scale decisions are documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md). The benchmark markdown in [`benchmarks/`](./benchmarks) and the generated figures in [`assets/`](./assets) should be read together: the markdown gives the measured numbers, and the screenshots make those results easier to inspect quickly during review.

## Files Included In This Repo

- [`README.md`](./README.md) for project overview, quickstart, and headline results
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) for system design and scaling choices
- [`benchmarks/`](./benchmarks) for measured results from the committed runs
- [`assets/`](./assets) for generated screenshots and dashboards
- [`tests/`](./tests) for the automated validation suite

## Delivery Summary

This project document was prepared for **Sai Veda** so the repository reads like a real project handoff: what the system is for, what problem it solves, what evidence supports it, and where the benchmark and test artifacts live inside the repo.
