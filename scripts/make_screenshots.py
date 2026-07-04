#!/usr/bin/env python3
"""Render product-grade observability 'screenshots' from pipeline artifacts.

Reads the CSV/JSON tables written by run_pipeline.py into ``benchmarks/`` and
produces four PNGs into ``assets/``:

  1. dashboard.png          KPI tiles + latency-over-time + cost-by-model
  2. drift_timeline.png     PSI/KL over time crossing the alert threshold
  3. latency_bands.png      p50/p90/p99 latency bands over time
  4. detector.png           PR curve + score-distribution for the detector

Everything is styled with the shared viztheme (dark product palette).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llmobs import viztheme as V  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
ASSETS = ROOT / "assets"

# Fixed categorical color assignment: model -> hue (never cycled by rank).
MODEL_ORDER = ["atlas-large-v2", "atlas-mini-v2", "nova-8b-instruct",
               "orion-reason-xl", "sparrow-4b-lite"]
MODEL_COLOR = {m: V.PALETTE[i] for i, m in enumerate(MODEL_ORDER)}


def _load():
    d = {}
    d["overview"] = json.loads((BENCH / "overview.json").read_text())
    d["by_model"] = pd.read_csv(BENCH / "by_model.csv")
    d["by_day"] = pd.read_csv(BENCH / "by_day.csv", parse_dates=["date"])
    d["by_model_day"] = pd.read_csv(BENCH / "by_model_day.csv", parse_dates=["date"])
    d["by_tenant"] = pd.read_csv(BENCH / "by_tenant.csv")
    d["drift"] = pd.read_csv(BENCH / "drift_timeline.csv", parse_dates=["date"])
    d["pr"] = pd.read_csv(BENCH / "detector_pr.csv")
    d["scores"] = pd.read_csv(BENCH / "detector_scores.csv")
    d["scorecard"] = pd.read_csv(BENCH / "scorecard.csv")
    d["detector_metrics"] = json.loads((BENCH / "detector_metrics.json").read_text())
    d["fleet"] = json.loads((BENCH / "fleet.json").read_text())
    d["drift_summary"] = json.loads((BENCH / "drift_summary.json").read_text())
    d["scale"] = json.loads((BENCH / "scale.json").read_text())
    return d


def _human(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n/div:.2f}{unit}"
    return f"{n:.0f}"


def _cutover_date(by_day: pd.DataFrame, cutover_idx: int):
    return by_day["date"].iloc[cutover_idx]


# ---------------------------------------------------------------------------
# 1. Multi-panel dashboard
# ---------------------------------------------------------------------------
def dashboard(d) -> None:
    ov = d["overview"]
    by_day = d["by_day"]
    by_model = d["by_model"].set_index("model").reindex(MODEL_ORDER)
    scale = d["scale"]

    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.8, 1.5, 1.4],
                  hspace=0.42, wspace=0.28, left=0.06, right=0.97, top=0.9, bottom=0.08)

    # ---- KPI tiles ----
    tiles = [
        ("Total Traces", _human(ov["total_traces"]), f"{scale['n_partitions']} daily partitions", V.ACCENT),
        ("Total Tokens", _human(ov["total_tokens"]), "prompt + completion", V.PALETTE[6]),
        ("Total Spend", f"${_human(ov['total_cost'])}", f"{scale['gb']:.1f} GB parquet", V.GOOD),
        ("p99 Latency", f"{ov['p99_latency']/1000:.1f}s", f"p50 {ov['p50_latency']:.0f} ms", V.WARN),
    ]
    for i, (label, value, sub, color) in enumerate(tiles):
        ax = fig.add_subplot(gs[0, i])
        V.kpi(ax, label, value, sub, color=color)

    # ---- latency over time (fleet p50 / p90 / p99) ----
    ax_lat = fig.add_subplot(gs[1, :3])
    cutover = _cutover_date(by_day, d["drift_summary"]["cutover_day_index"])
    for pct, col, lw in (("p50_ms", V.PALETTE[1], 1.8),
                         ("p90_ms", V.PALETTE[3], 1.8),
                         ("p99_ms", V.BAD, 2.2)):
        ax_lat.plot(by_day["date"], by_day[pct] / 1000.0, color=col, lw=lw,
                    label=pct.replace("_ms", "").upper())
    ax_lat.axvline(cutover, color=V.MUTED, ls="--", lw=1.2)
    ax_lat.text(cutover, ax_lat.get_ylim()[1] * 0.94, "  cutover", color=V.MUTED,
                fontsize=8, va="top")
    ax_lat.set_title("Fleet latency over time", loc="left")
    ax_lat.set_ylabel("latency (s)")
    ax_lat.legend(loc="upper left", ncol=3, fontsize=8)
    ax_lat.margins(x=0.01)

    # ---- error / halluc mini tile column ----
    ax_rate = fig.add_subplot(gs[1, 3])
    ax_rate.axis("off")
    lines = [
        ("Error rate", f"{ov['error_rate']*100:.2f}%", V.WARN),
        ("Timeout rate", f"{ov['timeout_rate']*100:.2f}%", V.BAD),
        ("Halluc rate", f"{ov['halluc_rate']*100:.2f}%", V.PALETTE[6]),
        ("Tenants", f"{ov['tenants']}", V.TEXT),
        ("Models", f"{ov['models']}", V.TEXT),
    ]
    y = 0.92
    ax_rate.text(0.0, 1.02, "FLEET HEALTH", color=V.MUTED, fontsize=9, fontweight="bold")
    for label, val, col in lines:
        ax_rate.text(0.0, y, label, color=V.MUTED, fontsize=9, va="center")
        ax_rate.text(1.0, y, val, color=col, fontsize=12, fontweight="bold",
                     va="center", ha="right")
        y -= 0.19

    # ---- cost by model (horizontal bars, fixed color per model) ----
    ax_cost = fig.add_subplot(gs[2, :2])
    order = by_model["cost_usd"].sort_values()
    colors = [MODEL_COLOR[m] for m in order.index]
    ax_cost.barh(order.index, order.values, color=colors, height=0.62)
    for m, v in order.items():
        ax_cost.text(v, m, f" ${_human(v)}", va="center", ha="left",
                     fontsize=8, color=V.TEXT)
    ax_cost.set_title("Spend by model", loc="left")
    ax_cost.set_xlabel("USD")
    ax_cost.margins(x=0.14)

    # ---- tokens by model (horizontal bars) ----
    ax_tok = fig.add_subplot(gs[2, 2:])
    order2 = by_model["tokens"].sort_values()
    colors2 = [MODEL_COLOR[m] for m in order2.index]
    ax_tok.barh(order2.index, order2.values, color=colors2, height=0.62)
    for m, v in order2.items():
        ax_tok.text(v, m, f" {_human(v)}", va="center", ha="left",
                    fontsize=8, color=V.TEXT)
    ax_tok.set_title("Tokens by model", loc="left")
    ax_tok.set_xlabel("tokens")
    ax_tok.margins(x=0.14)

    V.save_panel(fig, str(ASSETS / "dashboard.png"),
                 suptitle="LLM Observability — Fleet Dashboard")


# ---------------------------------------------------------------------------
# 2. Drift timeline
# ---------------------------------------------------------------------------
def drift_timeline(d) -> None:
    dr = d["drift"]
    ds = d["drift_summary"]
    by_day = d["by_day"]
    cutover = _cutover_date(by_day, ds["cutover_day_index"])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[2.2, 1.0],
                                  sharex=True)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.9, bottom=0.09, hspace=0.18)

    alert = ds["threshold"]
    watch = 0.10
    # shade regimes
    ax.axhspan(alert, max(dr["psi"].max() * 1.1, alert * 1.5), color=V.BAD, alpha=0.06)
    ax.axhspan(watch, alert, color=V.WARN, alpha=0.06)

    ax.plot(dr["date"], dr["psi"], color=V.ACCENT, lw=2.2, label="PSI (response embedding)")
    # highlight alerting points
    al = dr[dr["psi"] >= alert]
    ax.scatter(al["date"], al["psi"], color=V.BAD, s=22, zorder=5, label="alert")

    ax.axhline(alert, color=V.BAD, ls="--", lw=1.3)
    ax.axhline(watch, color=V.WARN, ls="--", lw=1.1)
    ax.text(dr["date"].iloc[0], alert, "  alert  0.25", color=V.BAD, fontsize=8, va="bottom")
    ax.text(dr["date"].iloc[0], watch, "  watch  0.10", color=V.WARN, fontsize=8, va="bottom")
    ax.axvline(cutover, color=V.MUTED, ls="--", lw=1.2)
    ax.text(cutover, ax.get_ylim()[1] * 0.96, "  model cutover", color=V.MUTED,
            fontsize=9, va="top")

    if ds["first_alert_day"] and ds["first_alert_day"] != "None":
        fa = pd.to_datetime(ds["first_alert_day"])
        yv = dr.loc[dr["date"] == fa, "psi"]
        if len(yv):
            ax.annotate("first alert", xy=(fa, float(yv.iloc[0])),
                        xytext=(15, -28), textcoords="offset points", color=V.BAD,
                        fontsize=9, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=V.BAD, lw=1.2))
    ax.set_ylabel("Population Stability Index")
    ax.set_title("Embedding drift detection — PSI crosses alert threshold at cutover",
                 loc="left")
    ax.legend(loc="center left", fontsize=9)

    # KL as a small multiple (separate axis, not dual-scale)
    ax2.plot(dr["date"], dr["kl"], color=V.PALETTE[4], lw=1.8)
    ax2.axvline(cutover, color=V.MUTED, ls="--", lw=1.0)
    ax2.set_ylabel("KL div (nats)")
    ax2.set_xlabel("date")

    V.save_panel(fig, str(ASSETS / "drift_timeline.png"))


# ---------------------------------------------------------------------------
# 3. Latency percentile bands
# ---------------------------------------------------------------------------
def latency_bands(d) -> None:
    by_day = d["by_day"].copy()
    bmd = d["by_model_day"]
    ds = d["drift_summary"]
    cutover = _cutover_date(by_day, ds["cutover_day_index"])

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.9, bottom=0.1)

    x = by_day["date"]
    p50 = by_day["p50_ms"] / 1000
    p90 = by_day["p90_ms"] / 1000
    p99 = by_day["p99_ms"] / 1000

    ax.fill_between(x, p50, p90, color=V.ACCENT, alpha=0.16, label="p50–p90")
    ax.fill_between(x, p90, p99, color=V.BAD, alpha=0.13, label="p90–p99")
    ax.plot(x, p50, color=V.GOOD, lw=2.0, label="p50")
    ax.plot(x, p90, color=V.WARN, lw=2.0, label="p90")
    ax.plot(x, p99, color=V.BAD, lw=2.4, label="p99")

    # overlay the regressed model's p99 to show the tail regression
    reg = bmd[bmd["model"] == "atlas-large-v2"].sort_values("date")
    ax.plot(reg["date"], reg["p99_ms"] / 1000, color=MODEL_COLOR["atlas-large-v2"],
            lw=1.6, ls=":", label="atlas-large-v2 p99")

    ax.axvline(cutover, color=V.MUTED, ls="--", lw=1.2)
    ax.text(cutover, ax.get_ylim()[1] * 0.95, "  cutover (tail-latency regression)",
            color=V.MUTED, fontsize=9, va="top")
    ax.set_title("Fleet latency percentile bands over time", loc="left")
    ax.set_ylabel("latency (s)")
    ax.set_xlabel("date")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.margins(x=0.01)

    V.save_panel(fig, str(ASSETS / "latency_bands.png"))


# ---------------------------------------------------------------------------
# 4. Detector PR curve + score distribution
# ---------------------------------------------------------------------------
def detector_chart(d) -> None:
    pr = d["pr"]
    scores = d["scores"]
    m = d["detector_metrics"]["metrics"]
    prevalence = d["detector_metrics"]["prevalence"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.24)

    # --- PR curve ---
    ax1.plot(pr["recall"], pr["precision"], color=V.ACCENT, lw=2.4,
             label=f"classifier  AP={m['clf']['avg_precision']:.3f}")
    ax1.axhline(prevalence, color=V.MUTED, ls="--", lw=1.2,
                label=f"random  AP={m['random']['avg_precision']:.3f}")
    # operating point
    ax1.scatter([m["clf"]["recall"]], [m["clf"]["precision"]], color=V.BAD, s=45,
                zorder=5, label=(f"operating pt  "
                                 f"P={m['clf']['precision']:.2f} R={m['clf']['recall']:.2f}"))
    ax1.set_title("Hallucination detector — precision/recall", loc="left")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right", fontsize=8)

    # --- score distribution by class ---
    pos = scores.loc[scores["label"] == 1, "score"]
    neg = scores.loc[scores["label"] == 0, "score"]
    bins = np.linspace(0, 1, 40)
    ax2.hist(neg, bins=bins, color=V.GOOD, alpha=0.55, label="grounded (truth=0)",
             density=True)
    ax2.hist(pos, bins=bins, color=V.BAD, alpha=0.6, label="hallucination (truth=1)",
             density=True)
    ax2.axvline(m["clf"]["threshold"], color=V.TEXT, ls="--", lw=1.3)
    ax2.text(m["clf"]["threshold"], ax2.get_ylim()[1] * 0.96,
             " decision thr", color=V.TEXT, fontsize=8, va="top")
    ax2.set_title(f"Score distribution   (ROC-AUC={m['clf']['roc_auc']:.3f}, "
                  f"F1={m['clf']['f1']:.3f})", loc="left")
    ax2.set_xlabel("predicted P(hallucination)")
    ax2.set_ylabel("density")
    ax2.legend(loc="upper center", fontsize=8)

    V.save_panel(fig, str(ASSETS / "detector.png"),
                 suptitle="Quality / Hallucination Detector Evaluation")


def main() -> None:
    V.apply_theme()
    ASSETS.mkdir(parents=True, exist_ok=True)
    d = _load()
    dashboard(d)
    drift_timeline(d)
    latency_bands(d)
    detector_chart(d)
    print(f"wrote 4 screenshots -> {ASSETS}")
    for p in ("dashboard", "drift_timeline", "latency_bands", "detector"):
        fp = ASSETS / f"{p}.png"
        print(f"  {fp.name:20s} {fp.stat().st_size/1024:6.1f} KB")


if __name__ == "__main__":
    main()
