"""Hallucination / quality proxy scorer.

Two layers, both trained/evaluated on the injected ground-truth labels:

1. **Heuristic groundedness** — token overlap (Jaccard) between prompt and
   response plus a "hedge phrase" count. Cheap, explainable, runs inline on the
   firehose. Reported as a standalone baseline detector.
2. **Logistic classifier** — logistic regression over groundedness + hedge
   count + length / token / tool-call features. Trained on a labelled sample,
   evaluated on a held-out split.

We report precision / recall / F1 / ROC-AUC / average-precision for both, and
compare against a random baseline (AP == positive prevalence).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config as C

_HEDGE_SET = set(C.HEDGE_TOKENS)


def groundedness(prompt: str, response: str) -> float:
    """Jaccard token overlap between prompt and response tokens in [0, 1].

    Low groundedness => the answer shares little vocabulary with the question,
    a classic unsupported-claim signal.
    """
    p = set(prompt.split())
    r = set(response.split())
    if not p and not r:
        return 0.0
    return len(p & r) / len(p | r)


def _hedge_count(response: str) -> int:
    return sum(1 for tok in response.split() if tok in _HEDGE_SET)


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised feature extraction from a sampled trace frame."""
    prompts = df["prompt"].to_numpy()
    responses = df["response"].to_numpy()
    ground = np.empty(len(df))
    hedge = np.empty(len(df))
    resp_len = np.empty(len(df))
    for i in range(len(df)):
        p = prompts[i]
        r = responses[i]
        ps = set(p.split())
        rs = r.split()
        rset = set(rs)
        union = len(ps | rset)
        ground[i] = (len(ps & rset) / union) if union else 0.0
        hedge[i] = sum(1 for t in rs if t in _HEDGE_SET)
        resp_len[i] = len(rs)
    feats = pd.DataFrame({
        "groundedness": ground,
        "hedge_count": hedge,
        "resp_len": resp_len,
        "log_completion_tokens": np.log1p(df["completion_tokens"].to_numpy()),
        "log_prompt_tokens": np.log1p(df["prompt_tokens"].to_numpy()),
        "n_tool_calls": df["n_tool_calls"].to_numpy(),
    })
    return feats


@dataclass
class DetectorReport:
    metrics: dict            # model + heuristic + random metrics
    pr_curve: dict           # precision/recall arrays for the classifier
    coef: dict               # feature -> logistic coefficient
    prevalence: float
    n_train: int
    n_test: int
    test_scores: np.ndarray  # held-out predicted probabilities (subsample)
    test_labels: np.ndarray  # held-out ground-truth labels (subsample)

    def summary_row(self) -> dict:
        m = self.metrics
        return {
            "clf_precision": m["clf"]["precision"],
            "clf_recall": m["clf"]["recall"],
            "clf_f1": m["clf"]["f1"],
            "clf_auc": m["clf"]["roc_auc"],
            "clf_ap": m["clf"]["avg_precision"],
            "heur_auc": m["heuristic"]["roc_auc"],
            "random_ap": m["random"]["avg_precision"],
        }


def _binary_metrics(y_true: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = (score >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    return {
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "avg_precision": float(average_precision_score(y_true, score)),
        "threshold": float(thr),
    }


def train_and_evaluate(df: pd.DataFrame, *, seed: int = C.GLOBAL_SEED,
                       test_size: float = 0.3) -> DetectorReport:
    """Train the logistic detector and evaluate it + the heuristic baseline."""
    X = featurize(df)
    y = df["is_hallucination"].astype(int).to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(scaler.transform(X_tr), y_tr)
    proba = clf.predict_proba(scaler.transform(X_te))[:, 1]

    # Pick the decision threshold that maximises F1 on the PR curve.
    prec, rec, thr = precision_recall_curve(y_te, proba)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best = int(np.nanargmax(f1s[:-1])) if len(thr) else 0
    best_thr = float(thr[best]) if len(thr) else 0.5

    clf_metrics = _binary_metrics(y_te, proba, best_thr)

    # Heuristic baseline: low groundedness => hallucination. Score = 1 - ground
    # (+ a nudge for hedge phrases). Threshold chosen the same way.
    heur_score = (1.0 - X_te["groundedness"].to_numpy()) \
        + 0.15 * np.clip(X_te["hedge_count"].to_numpy(), 0, 4)
    hp, hr, ht = precision_recall_curve(y_te, heur_score)
    hf1 = 2 * hp * hr / np.clip(hp + hr, 1e-9, None)
    hbest = int(np.nanargmax(hf1[:-1])) if len(ht) else 0
    heur_metrics = _binary_metrics(y_te, heur_score,
                                   float(ht[hbest]) if len(ht) else 0.5)

    prevalence = float(y_te.mean())
    rng = np.random.default_rng(seed)
    rand_score = rng.random(len(y_te))
    random_metrics = {
        "roc_auc": float(roc_auc_score(y_te, rand_score)),
        "avg_precision": float(average_precision_score(y_te, rand_score)),
        "prevalence": prevalence,
    }

    sub = rng.choice(len(y_te), size=min(20_000, len(y_te)), replace=False)
    return DetectorReport(
        metrics={"clf": clf_metrics, "heuristic": heur_metrics,
                 "random": random_metrics},
        pr_curve={"precision": prec, "recall": rec},
        coef=dict(zip(X.columns, clf.coef_[0].tolist())),
        prevalence=prevalence, n_train=len(y_tr), n_test=len(y_te),
        test_scores=proba[sub], test_labels=y_te[sub],
    )
