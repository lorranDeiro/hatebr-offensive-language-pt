"""Metrics with bootstrap confidence intervals, so small gaps are not over-read."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def scores(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ci(y_true, y_pred, metric=f1_score, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    vals = [metric(y_true[i], y_pred[i], zero_division=0) for i in idx]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def report(name: str, y_true, y_pred) -> dict:
    s = scores(y_true, y_pred)
    s["f1_ci95"] = bootstrap_ci(y_true, y_pred)
    s["name"] = name
    lo, hi = s["f1_ci95"]
    print(f"{name:<42} acc={s['accuracy']:.4f}  P={s['precision']:.4f}  R={s['recall']:.4f}  "
          f"F1={s['f1']:.4f} [{lo:.3f}, {hi:.3f}]")
    return s
