from agents.data_profiler import profile_dataframe
from agents.model_selector import recommend_models
import pandas as pd


def test_recommend_logistic():
    df = pd.DataFrame({"y": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
    prof = profile_dataframe(df)
    rec = recommend_models(prof, "y", "predict churn")
    assert "error" not in rec
    assert rec["outcome_type"] == "binary"
    assert any("Logistic" in r["model"] for r in rec["recommendations"])
