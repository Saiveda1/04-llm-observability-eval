"""Embedding-drift detection via PSI / KL divergence.

Pipeline (fully offline & deterministic):

1. Embed response text with a hashing TF-IDF vectoriser (fixed feature space,
   no vocabulary to fit -> reproducible and memory-bounded) followed by a
   TruncatedSVD projection fit on the *baseline* window.
2. Reduce each response to a scalar drift score (its projection on the leading
   baseline component).
3. Bin the baseline scores into deciles and, for every daily window, compute
   the Population Stability Index (PSI) and KL divergence of that window's
   score distribution against the baseline reference.
4. Flag windows whose PSI crosses the alert threshold.

PSI convention (industry standard):
    PSI < 0.10  -> stable
    0.10-0.25   -> moderate shift (watch)
    PSI >= 0.25 -> significant shift (alert)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from . import config as C

PSI_WATCH = 0.10
PSI_ALERT = 0.25
_N_BINS = 10
_EPS = 1e-6


def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                               bin_edges: np.ndarray) -> float:
    """PSI between an expected (baseline) and actual sample over fixed bins."""
    e_hist, _ = np.histogram(expected, bins=bin_edges)
    a_hist, _ = np.histogram(actual, bins=bin_edges)
    e = e_hist / max(e_hist.sum(), 1)
    a = a_hist / max(a_hist.sum(), 1)
    e = np.clip(e, _EPS, None)
    a = np.clip(a, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def kl_divergence(expected: np.ndarray, actual: np.ndarray,
                  bin_edges: np.ndarray) -> float:
    """KL(actual || expected) over fixed bins (nats)."""
    e_hist, _ = np.histogram(expected, bins=bin_edges)
    a_hist, _ = np.histogram(actual, bins=bin_edges)
    e = np.clip(e_hist / max(e_hist.sum(), 1), _EPS, None)
    a = np.clip(a_hist / max(a_hist.sum(), 1), _EPS, None)
    return float(np.sum(a * np.log(a / e)))


@dataclass
class DriftReport:
    timeline: pd.DataFrame          # per-day psi, kl, score_mean, alert
    baseline_days: list[str]
    alert_days: list[str] = field(default_factory=list)
    threshold: float = PSI_ALERT

    @property
    def first_alert_day(self) -> str | None:
        return self.alert_days[0] if self.alert_days else None


def _embed(responses: list[str], n_features: int = 2 ** 18,
           n_components: int = 8, seed: int = C.GLOBAL_SEED,
           svd: TruncatedSVD | None = None,
           tfidf: TfidfTransformer | None = None):
    """Hashing TF-IDF -> SVD. If svd/tfidf are provided they are reused
    (transform only); otherwise they are fit here and returned."""
    hv = HashingVectorizer(n_features=n_features, alternate_sign=False,
                           norm=None, ngram_range=(1, 2))
    counts = hv.transform(responses)
    if tfidf is None:
        tfidf = TfidfTransformer()
        X = tfidf.fit_transform(counts)
    else:
        X = tfidf.transform(counts)
    if svd is None:
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        emb = svd.fit_transform(X)
    else:
        emb = svd.transform(X)
    return emb, svd, tfidf


def detect(window_df: pd.DataFrame, *, baseline_frac: float = 0.4,
           seed: int = C.GLOBAL_SEED) -> DriftReport:
    """Run drift detection over a per-day response sample.

    ``window_df`` needs columns ``date`` and ``response``. The earliest
    ``baseline_frac`` of days form the reference distribution; every day is
    then scored against it.
    """
    days = sorted(window_df["date"].unique())
    n_base = max(2, int(len(days) * baseline_frac))
    baseline_days = days[:n_base]

    base_mask = window_df["date"].isin(baseline_days)
    base_resp = window_df.loc[base_mask, "response"].tolist()

    # Fit the embedding on baseline responses, then score all responses.
    base_emb, svd, tfidf = _embed(base_resp, seed=seed)
    all_emb, _, _ = _embed(window_df["response"].tolist(), svd=svd, tfidf=tfidf, seed=seed)

    # Scalar drift score = projection on the leading baseline component.
    base_scores = base_emb[:, 0]
    window_df = window_df.copy()
    window_df["score"] = all_emb[:, 0]

    # Fixed baseline decile bin edges (open-ended tails).
    qs = np.linspace(0, 1, _N_BINS + 1)
    edges = np.quantile(base_scores, qs)
    edges[0], edges[-1] = -np.inf, np.inf

    rows = []
    for day in days:
        s = window_df.loc[window_df["date"] == day, "score"].to_numpy()
        psi = population_stability_index(base_scores, s, edges)
        kl = kl_divergence(base_scores, s, edges)
        rows.append({"date": day, "n": len(s), "score_mean": float(s.mean()),
                     "psi": psi, "kl": kl,
                     "alert": psi >= PSI_ALERT,
                     "watch": PSI_WATCH <= psi < PSI_ALERT})
    timeline = pd.DataFrame(rows)
    alert_days = timeline.loc[timeline["alert"], "date"].tolist()
    return DriftReport(timeline=timeline, baseline_days=list(baseline_days),
                       alert_days=alert_days)
