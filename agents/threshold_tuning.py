"""Grid search over classification thresholds for business trade-offs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def tune_thresholds(y_true, pred_prob, thresholds: np.ndarray | None = None) -> list[dict]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred_prob, dtype=float)
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.01)
    results: list[dict] = []
    for threshold in thresholds:
        pred_class = (p >= threshold).astype(int)
        cm = confusion_matrix(y, pred_class, labels=[0, 1])
        if cm.size == 1:
            tn = fp = fn = tp = 0
            if cm.shape == (1, 1):
                if y.sum() == 0 and pred_class.sum() == 0:
                    tn = int(cm[0, 0])
                elif y.sum() == len(y) and pred_class.sum() == len(y):
                    tp = int(cm[0, 0])
        else:
            tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results.append(
            {
                "threshold": round(float(threshold), 2),
                "accuracy": float(accuracy_score(y, pred_class)),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision_score(y, pred_class, zero_division=0)),
            }
        )
    results.sort(
        key=lambda x: (x["sensitivity"], x["specificity"], x["accuracy"]),
        reverse=True,
    )
    return results
