import numpy as np
import pytest
from agents.threshold_tuning import tune_thresholds, interpret_threshold_results


def test_returns_required_keys():
    np.random.seed(0)
    y = np.random.randint(0, 2, 100)
    p = np.clip(y + np.random.normal(0, 0.2, 100), 0, 1)
    r = tune_thresholds(y, p)
    assert "best_youden" in r
    assert "best_f1"     in r
    assert "auc"         in r
    assert len(r["threshold_table"]) == 81   # 0.10 to 0.90 step 0.01


def test_perfect_predictor():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    r = tune_thresholds(y, p)
    assert r["auc"] == 1.0


def test_interpret_returns_strings():
    np.random.seed(1)
    y = np.random.randint(0, 2, 200)
    p = np.clip(y + np.random.normal(0, 0.3, 200), 0, 1)
    r = tune_thresholds(y, p)
    notes = interpret_threshold_results(r)
    assert isinstance(notes, list)
    assert all(isinstance(n, str) for n in notes)
    assert len(notes) >= 2


