"""LLM Observability & Evaluation Platform.

A repo-ready reference implementation of trace analytics and quality
monitoring for large-scale LLM deployments:

- ``generate``  : synthetic LLM trace generator (streams to partitioned Parquet)
- ``analytics`` : DuckDB out-of-core cost/token/latency/error analytics
- ``drift``     : TF-IDF + SVD embedding drift detection via PSI / KL divergence
- ``detector``  : heuristic + logistic hallucination/quality proxy scorer
- ``evalharness``: per-model quality scorecards

Everything is deterministic (seeded) and runs fully offline.
"""
from __future__ import annotations

from . import analytics, config, detector, drift, evalharness, generate

__all__ = ["analytics", "config", "detector", "drift", "evalharness", "generate"]
__version__ = "1.0.0"
