"""Drift detection: PSI math + PSI must spike at the injected cutover boundary."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmobs import analytics, config as C, drift  # noqa: E402


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    edges = np.quantile(x, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    psi = drift.population_stability_index(x, x, edges)
    assert psi < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 20000)
    edges = np.quantile(base, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    psis = [drift.population_stability_index(base, rng.normal(s, 1, 20000), edges)
            for s in (0.0, 0.5, 1.0, 2.0)]
    assert psis[0] < psis[1] < psis[2] < psis[3]
    assert psis[-1] > drift.PSI_ALERT


def test_kl_nonnegative():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 10000)
    edges = np.quantile(base, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    kl = drift.kl_divergence(base, rng.normal(1.5, 1, 10000), edges)
    assert kl >= 0


def test_drift_detected_at_cutover(dataset, con):
    """The end-to-end pipeline must (a) stay calm pre-cutover and (b) alert at
    or after the injected cutover day."""
    wr = analytics.window_response_sample(con, dataset, per_day=2500)
    rep = drift.detect(wr)
    tl = rep.timeline.reset_index(drop=True)

    cutover = C.CUTOVER_DAY
    pre = tl.iloc[:cutover - 2]        # comfortably before the boundary
    post = tl.iloc[cutover:]           # from the cutover onward

    # pre-cutover windows are stable
    assert pre["psi"].max() < drift.PSI_ALERT
    # post-cutover windows breach the alert threshold
    assert post["psi"].max() >= drift.PSI_ALERT
    # mean PSI jumps by a wide margin across the boundary
    assert post["psi"].mean() > pre["psi"].mean() + 0.1
    # first alert lands at/after the cutover, not before
    assert rep.first_alert_day is not None
    first_idx = tl.index[tl["date"] == rep.first_alert_day][0]
    assert first_idx >= cutover - 1
