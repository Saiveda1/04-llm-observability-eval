"""Hallucination detector: must beat random and separate the classes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmobs import analytics, detector  # noqa: E402


def test_groundedness_bounds():
    assert detector.groundedness("a b c", "a b c") == 1.0
    assert detector.groundedness("a b c", "x y z") == 0.0
    g = detector.groundedness("a b c d", "a b x y")
    assert 0.0 < g < 1.0


def test_detector_beats_random(dataset, con):
    smp = analytics.sample_for_eval(con, dataset, n=60_000)
    rep = detector.train_and_evaluate(smp)
    m = rep.metrics

    # ROC-AUC well above chance
    assert m["clf"]["roc_auc"] > 0.75
    # average precision far above the random (prevalence) baseline
    assert m["clf"]["avg_precision"] > 3 * m["random"]["avg_precision"]
    assert m["clf"]["avg_precision"] > m["random"]["avg_precision"] + 0.15
    # a usable operating point
    assert m["clf"]["precision"] > 0.3
    assert m["clf"]["recall"] > 0.3
    assert m["clf"]["f1"] > 0.35


def test_heuristic_also_beats_random(dataset, con):
    smp = analytics.sample_for_eval(con, dataset, n=60_000)
    rep = detector.train_and_evaluate(smp)
    # groundedness heuristic alone is a real signal
    assert rep.metrics["heuristic"]["roc_auc"] > 0.65


def test_groundedness_coef_sign(dataset, con):
    """Lower groundedness should raise hallucination probability (neg coef)."""
    smp = analytics.sample_for_eval(con, dataset, n=60_000)
    rep = detector.train_and_evaluate(smp)
    assert rep.coef["groundedness"] < 0
    assert rep.coef["hedge_count"] > 0
