"""Central configuration: models, pricing, tenants, drift regime, seeds.

All names are synthetic. Pricing is in USD per 1M tokens and is representative
of the market, not any specific vendor.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
GLOBAL_SEED = 1234

# ---------------------------------------------------------------------------
# Time span of the synthetic fleet
# ---------------------------------------------------------------------------
START_DATE = "2026-01-01"
N_DAYS = 60
# The "cutover" is a model/infra rollout that introduces a drift regime:
# response distribution shifts, hallucination rate spikes, one model regresses
# on tail latency. Detectors must catch this.
CUTOVER_DAY = 40  # 0-indexed day within the span


@dataclass(frozen=True)
class Model:
    name: str
    price_in: float  # USD per 1M prompt tokens
    price_out: float  # USD per 1M completion tokens
    base_latency_ms: float  # median first-token+gen latency at a nominal length
    weight: float  # share of traffic


# Synthetic model fleet (fictional names).
MODELS: tuple[Model, ...] = (
    Model("atlas-large-v2", price_in=3.00, price_out=15.00, base_latency_ms=900.0, weight=0.22),
    Model("atlas-mini-v2", price_in=0.25, price_out=1.25, base_latency_ms=280.0, weight=0.38),
    Model("nova-8b-instruct", price_in=0.15, price_out=0.60, base_latency_ms=210.0, weight=0.25),
    Model("orion-reason-xl", price_in=5.00, price_out=25.00, base_latency_ms=1400.0, weight=0.10),
    Model("sparrow-4b-lite", price_in=0.05, price_out=0.20, base_latency_ms=140.0, weight=0.05),
)
MODEL_NAMES: tuple[str, ...] = tuple(m.name for m in MODELS)

# The model that regresses (tail-latency + hallucination spike) after cutover.
REGRESSED_MODEL = "atlas-large-v2"

# ---------------------------------------------------------------------------
# Tenants (multi-tenant SaaS control plane)
# ---------------------------------------------------------------------------
TENANTS: tuple[str, ...] = (
    "acme-search", "globex-support", "initech-copilot", "umbra-analytics",
    "wayne-devtools", "stark-agents", "hooli-chatops", "piedpiper-rag",
)
TENANT_WEIGHTS: tuple[float, ...] = (0.20, 0.18, 0.15, 0.12, 0.11, 0.10, 0.08, 0.06)
USERS_PER_TENANT = 500

# ---------------------------------------------------------------------------
# Quality-issue injection rates
# ---------------------------------------------------------------------------
BASE_ERROR_RATE = 0.018
BASE_TIMEOUT_RATE = 0.006
BASE_HALLUCINATION_RATE = 0.070
# Post-cutover regime: hallucination on the regressed model spikes.
DRIFT_HALLUCINATION_RATE = 0.260
DRIFT_ERROR_MULT = 1.9  # errors climb after the cutover across the fleet
TIMEOUT_LATENCY_MS = 30_000.0

# ---------------------------------------------------------------------------
# Text / topic vocabulary for synthetic request+response bodies.
# Topics carry the drift signal: the post-cutover regime shifts topic mix and
# response length, which the TF-IDF+SVD/PSI pipeline must detect.
# ---------------------------------------------------------------------------
TOPICS: dict[str, list[str]] = {
    "billing": ["invoice", "refund", "charge", "subscription", "payment", "receipt", "proration", "credit"],
    "auth": ["login", "password", "token", "oauth", "session", "mfa", "sso", "credential"],
    "database": ["query", "index", "shard", "replica", "transaction", "schema", "partition", "vacuum"],
    "networking": ["latency", "packet", "gateway", "dns", "throughput", "firewall", "route", "handshake"],
    "ml_ops": ["model", "training", "checkpoint", "gpu", "gradient", "epoch", "inference", "quantize"],
    "kubernetes": ["pod", "cluster", "namespace", "ingress", "deployment", "sidecar", "autoscaler", "helm"],
    # Emergent post-cutover topics (rare before the cutover, common after):
    "agents": ["planner", "toolchain", "handoff", "reflection", "trajectory", "scratchpad", "critic", "subgoal"],
    "compliance": ["audit", "retention", "gdpr", "consent", "redaction", "policy", "residency", "lineage"],
    "streaming": ["kafka", "offset", "consumer", "backpressure", "windowing", "watermark", "topic", "flush"],
    "security": ["exploit", "cve", "patch", "sandbox", "escalation", "payload", "hardening", "zeroday"],
}
TOPIC_NAMES: tuple[str, ...] = tuple(TOPICS.keys())
# Pre-cutover topic distribution (first 6 dominate; last 4 rare).
PRE_TOPIC_WEIGHTS = [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.03, 0.03, 0.02, 0.02]
# Post-cutover: the emergent topics surge.
POST_TOPIC_WEIGHTS = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.16, 0.13, 0.11, 0.15]

# "Hedge" filler tokens that appear disproportionately in hallucinated answers
# (unsupported, low-groundedness completions). This is the injected signal the
# detector learns; groundedness (prompt/response overlap) is the heuristic.
HEDGE_TOKENS = [
    "presumably", "allegedly", "as-is-widely-known", "obviously-therefore",
    "it-is-guaranteed", "without-any-doubt", "the-official-figure", "per-the-manual",
]

# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------
PARQUET_DIRNAME = "traces"  # under data/
PARTITION_COL = "date"
