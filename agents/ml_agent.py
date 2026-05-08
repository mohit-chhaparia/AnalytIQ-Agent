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


def run_tabular_ml(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    task: str,
    cv_folds: int = 5,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """
    task: 'classify' | 'regress'
    Returns metrics, CV scores, and feature importances (aggregated post one-hot).
    """
    if not predictors:
        raise ValueError("Select at least one predictor for ML.")
    task = task.lower()
    if task not in ("classify", "regress"):
        raise ValueError("task must be 'classify' or 'regress'")

    X, y = _build_xy(df, outcome, predictors)
    if len(X) < 10:
        raise ValueError("Need at least 10 complete rows for a stable ML baseline.")

    if task == "classify":
        y_use, enc_meta = _encode_binary_y(y)
        n_splits = min(cv_folds, max(2, min(5, len(np.unique(y_use)))))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        y_use = pd.to_numeric(pd.Series(y), errors="coerce").values
        if np.isnan(y_use).any():
            raise ValueError("Regression requires a numeric outcome.")
        enc_meta = {"encoding": "numeric"}
        n_splits = min(cv_folds, max(2, len(y_use) // 3))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    pipe = _make_pipeline(X, task)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "classify":
            cv_scores = cross_val_score(pipe, X, y_use, cv=splitter, scoring="roc_auc")
        else:
            cv_scores = cross_val_score(pipe, X, y_use, cv=splitter, scoring="r2")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_use,
        test_size=test_size,
        random_state=42,
        stratify=y_use if task == "classify" else None,
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    model = pipe.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)

    feat_names: list[str] | None = None
    try:
        feat_names = list(pipe.named_steps["prep"].get_feature_names_out())
    except Exception:  # noqa: BLE001
        feat_names = None

    metrics: dict[str, float] = {}
    if task == "classify":
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics["accuracy"] = float(accuracy_score(y_test, preds))
        try:
            metrics["roc_auc_holdout"] = float(roc_auc_score(y_test, proba))
        except ValueError:
            metrics["roc_auc_holdout"] = float("nan")
    else:
        metrics["rmse_holdout"] = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics["r2_holdout"] = float(r2_score(y_test, preds))

    imp_list: list[dict[str, float]] = []
    if importances is not None and feat_names is not None and len(feat_names) == len(importances):
        order = np.argsort(importances)[::-1][:25]
        imp_list = [{"feature": feat_names[i], "importance": float(importances[i])} for i in order]

    return {
        "model_type": "RandomForest " + ("Classifier" if task == "classify" else "Regressor"),
        "task": task,
        "outcome": outcome,
        "predictors": predictors,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_metric": "roc_auc" if task == "classify" else "r2",
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "metrics": metrics,
        "target_encoding": enc_meta,
        "top_feature_importances": imp_list,
        "_pipeline": pipe,
    }
