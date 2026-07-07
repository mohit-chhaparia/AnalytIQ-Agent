"""
Threshold tuning for logistic regression.
Sweeps thresholds 0.10–0.90 and returns a ranked table of sensitivity, specificity, precision, and accuracy per threshold.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)


def tune_thresholds(y_true, pred_prob: list) -> dict:
    """
    Sweep classification thresholds and return a ranked results table.

    Parameters
    ----------
    y_true   : array-like of 0/1 true labels
    pred_prob: array-like of predicted probabilities (from logistic model)

    Returns
    -------
    dict with keys:
        "threshold_table" : list of dicts, one per threshold
        "best_youden"     : threshold maximising Youden's J (sensitivity + specificity - 1)
        "best_f1"         : threshold maximising F1 score
        "auc"             : overall AUC
    """
    y_true = np.array(y_true)
    pred_prob = np.array(pred_prob)

    results = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        t = round(float(threshold), 2)
        pred_class = (pred_prob >= t).astype(int)

        cm = confusion_matrix(y_true, pred_class)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden_j = sensitivity + specificity - 1

        results.append({
            "threshold":   t,
            "accuracy":    round(float(accuracy_score(y_true, pred_class)), 4),
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "precision":   round(float(precision_score(y_true, pred_class, zero_division=0)), 4),
            "f1":          round(float(f1_score(y_true, pred_class, zero_division=0)), 4),
            "youden_j":    round(float(youden_j), 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

    best_youden = max(results, key=lambda x: x["youden_j"])
    best_f1     = max(results, key=lambda x: x["f1"])

    try:
        auc = round(float(roc_auc_score(y_true, pred_prob)), 4)
    except Exception:
        auc = None

    return {
        "threshold_table": results,
        "best_youden":     best_youden,
        "best_f1":         best_f1,
        "auc":             auc,
    }


def interpret_threshold_results(tuning_result: dict) -> list:
    """Return human-readable diagnostic notes from threshold tuning."""
    notes = []
    by = tuning_result.get("best_youden", {})
    bf = tuning_result.get("best_f1", {})
    auc = tuning_result.get("auc")

    if auc is not None:
        if auc >= 0.9:
            notes.append(f"AUC = {auc:.3f} — Excellent discriminative ability.")
        elif auc >= 0.8:
            notes.append(f"AUC = {auc:.3f} — Good discriminative ability.")
        elif auc >= 0.7:
            notes.append(f"AUC = {auc:.3f} — Acceptable discriminative ability.")
        else:
            notes.append(f"AUC = {auc:.3f} — Poor discriminative ability. Consider alternative models.")

    if by:
        notes.append(
            f"Best threshold by Youden's J: {by['threshold']} "
            f"(Sensitivity={by['sensitivity']:.3f}, Specificity={by['specificity']:.3f})."
        )
    if bf:
        notes.append(
            f"Best threshold by F1: {bf['threshold']} "
            f"(F1={bf['f1']:.3f}, Precision={bf['precision']:.3f}, Recall={bf['sensitivity']:.3f})."
        )
    if by and bf and by["threshold"] != bf["threshold"]:
        notes.append(
            "Youden and F1 optimal thresholds differ. "
            "Use Youden's threshold when false negatives and false positives are equally costly. "
            "Use F1 threshold when positive class precision matters more."
        )
    return notes
