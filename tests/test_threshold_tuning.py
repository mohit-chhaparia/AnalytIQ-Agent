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


