from agents.data_profiler import profile_dataframe
from agents.intent_agent import infer_analysis_modes
import pandas as pd


def test_intent_time_series_keywords():
    df = pd.DataFrame({"y": [1, 2, 3], "t": pd.date_range("2020-01-01", periods=3)})
    prof = profile_dataframe(df)
    r = infer_analysis_modes("ARIMA forecast for demand", prof, "y")
    assert r["flags"]["time_series"] is True


def test_intent_ml_keywords():
    df = pd.DataFrame({"y": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
    prof = profile_dataframe(df)
    r = infer_analysis_modes("Use random forest with cross validation", prof, "y")
    assert r["flags"]["tabular_ml"] is True
