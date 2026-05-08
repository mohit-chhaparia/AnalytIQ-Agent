"""Tabular machine learning: preprocessing, cross-validation, feature importance."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_xy(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    sub = df[[outcome] + predictors].dropna()
    X = sub[predictors]
    y = sub[outcome].values
    return X, y


def _make_pipeline(X: pd.DataFrame, task: str) -> Pipeline:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No numeric or categorical columns detected in predictors.")
    pre = ColumnTransformer(transformers, remainder="drop")
    if task == "classify":
        est = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced_subsample",
        )
    else:
        est = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    return Pipeline([("prep", pre), ("model", est)])


def _encode_binary_y(y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    s = pd.Series(y)
    if pd.api.types.is_numeric_dtype(s) and set(pd.unique(s.dropna())).issubset({0, 1}):
        return s.astype(float).values, {"encoding": "numeric_0_1"}
    uniq = pd.unique(s.astype(str))
    if len(uniq) != 2:
        raise ValueError("Classification requires a binary outcome with two distinct values.")
    low, high = sorted(uniq.tolist(), key=str)
    y_enc = (s.astype(str) == high).astype(int).values
    return y_enc, {"encoding": "two_class", "positive_class": str(high), "negative_class": str(low)}


