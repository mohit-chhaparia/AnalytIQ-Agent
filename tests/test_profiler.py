import pandas as pd

from agents.data_profiler import infer_variable_type, profile_dataframe


def test_profile_basic():
    df = pd.DataFrame(
        {
            "a": list(range(25)),
            "b": (["x", "y", "z"] * 8) + ["x"],
            "c": [0, 1, 0, 1, 0] * 5,
        }
    )
    p = profile_dataframe(df)
    assert p["shape"]["rows"] == 25
    assert len(p["columns"]) == 3
    names = {c["name"]: c["inferred_type"] for c in p["columns"]}
    assert names["a"] == "continuous_numeric"
    assert names["b"] == "categorical"
    assert names["c"] == "binary"


