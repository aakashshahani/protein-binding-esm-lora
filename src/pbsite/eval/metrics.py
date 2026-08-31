"""Per-residue evaluation metrics.

Primary metric is AUPRC (average precision) because binding residues are rare
and threshold-free ranking quality matters most. We also report AUROC, and
threshold-dependent F1 / precision / recall / MCC at the F1-optimal threshold
chosen on validation (never on test).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray, n_steps: int = 200) -> float:
    """Threshold in (0,1) maximizing F1 on the given (validation) arrays."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def residue_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute the full metric suite from flat arrays of true labels / probs."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # AUROC/AUPRC are undefined with a single class present; guard for tests.
    both_classes = len(np.unique(y_true)) == 2
    return {
        "auprc": float(average_precision_score(y_true, y_prob)) if both_classes else float("nan"),
        "auroc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if both_classes else 0.0,
        "threshold": float(threshold),
        "n_pos": int(y_true.sum()),
        "n": int(y_true.size),
    }
