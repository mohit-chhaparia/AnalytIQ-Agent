import pandas as pd

from agents.ml_agent import run_tabular_ml
from agents.time_series_agent import run_time_series_summary


def test_ml_binary_smoke():
    df = pd.DataFrame({"y": [0, 1] * 20, "a": range(40), "b": ["x", "y"] * 20})
    out = run_tabular_ml(df, "y", ["a", "b"], task="classify", cv_folds=3)
    assert "cv_mean" in out
    assert out["task"] == "classify"



