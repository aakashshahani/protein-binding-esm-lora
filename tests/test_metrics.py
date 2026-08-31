import numpy as np

from pbsite.eval.metrics import best_f1_threshold, residue_metrics


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    m = residue_metrics(y, p, threshold=0.5)
    assert m["auprc"] == 1.0
    assert m["auroc"] == 1.0
    assert m["f1"] == 1.0
    assert m["mcc"] == 1.0


def test_single_class_guarded():
    y = np.zeros(5, dtype=int)
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    m = residue_metrics(y, p, threshold=0.5)
    assert np.isnan(m["auprc"])
    assert m["mcc"] == 0.0


def test_best_threshold_in_range():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.2, 0.6, 0.3, 0.7, 0.9])
    t = best_f1_threshold(y, p)
    assert 0.0 < t < 1.0
