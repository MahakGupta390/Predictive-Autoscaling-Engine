"""
Prediction engine: Linear Regression baseline and XGBoost behind one
Predictor interface, config-selectable, plus the chronological
train/test/compare harness used to justify picking one over the other.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


FEATURE_COLUMNS = [
    "request_rate",
    "cpu_pct",
    "mem_pct",
    "rr_roll_mean_short",
    "rr_roll_mean_long",
    "rr_ewma",
    "rr_lag_1",
    "rr_lag_5",
]
TARGET_COLUMN = "request_rate"


def build_supervised_dataset(
    features_df: pd.DataFrame, horizon: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Shift the target forward by `horizon` ticks: row i's target is the
    request_rate `horizon` ticks in the future. Drops the final `horizon`
    rows, which have no future value to predict."""
    df = features_df.copy()
    df["target"] = df[TARGET_COLUMN].shift(-horizon)
    df = df.dropna(subset=["target"])

    X = df[FEATURE_COLUMNS]
    y = df["target"]
    return X, y


def chronological_split(
    X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-ordered split — NEVER shuffle. The last test_frac of rows (in
    time order) become the test set, so no future information leaks into
    training."""
    split_idx = int(len(X) * (1 - test_frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # guard against division by ~0 request rates
    denom = np.clip(np.abs(y_true), 1e-3, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def naive_baseline_mape(y_true: pd.Series, X_test: pd.DataFrame) -> float:
    """'Tomorrow = today' baseline — predict the target equals the most
    recent known request_rate. Any trained model must beat this to be
    worth deploying."""
    naive_pred = X_test["request_rate"].to_numpy()
    return mape(y_true.to_numpy(), naive_pred)


@dataclass
class PredictionResult:
    predicted_load: np.ndarray
    model_used: str


class Predictor(ABC):
    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


class LinearRegressionPredictor(Predictor):
    """Ridge (L2-regularized) rather than plain OLS: the feature set is
    highly multicollinear by construction (cpu_pct/mem_pct are near-linear
    functions of request_rate, and the lag/rolling features track it too),
    which makes unregularized OLS coefficients unstable. Ridge is the
    standard fix and is still "linear regression" for baseline purposes.

    Trains on log1p(target) rather than the raw value. Training minimizes
    MSE, but evaluation uses MAPE (a relative-error metric) — on raw
    values those disagree: MSE is dominated by the large absolute errors
    during bursts, so an MSE-trained model sacrifices accuracy on the many
    small-value quiet-period rows, which is exactly what MAPE penalizes
    hardest. Log-space MSE approximates relative error, aligning the two.
    """

    name = "linear_regression"

    def __init__(self, alpha: float = 1.0):
        self._model = Ridge(alpha=alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, np.log1p(y))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.expm1(self._model.predict(X))


class XGBoostPredictor(Predictor):
    """Same log1p/expm1 treatment as the linear model, for the same
    MSE-vs-MAPE reason — and it keeps the two models' comparison fair,
    since both are now optimizing the same effective objective."""

    name = "xgboost"

    def __init__(self, n_estimators: int = 200, max_depth: int = 4, learning_rate: float = 0.05):
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:squarederror",
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, np.log1p(y))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.expm1(self._model.predict(X))


_REGISTRY = {
    "linear_regression": LinearRegressionPredictor,
    "xgboost": XGBoostPredictor,
}


class PredictionService:
    """Config-driven wrapper: which model is active is a string, not a
    code change. Everything downstream (fusion, accuracy tracker) only
    ever talks to this class."""

    def __init__(self, model_name: str = "linear_regression", horizon: int = 5):
        if model_name not in _REGISTRY:
            raise ValueError(f"unknown model '{model_name}', choices: {list(_REGISTRY)}")
        self.model_name = model_name
        self.horizon = horizon
        self._predictor: Predictor = _REGISTRY[model_name]()
        self._is_trained = False

    def train(self, features_df: pd.DataFrame) -> None:
        X, y = build_supervised_dataset(features_df, self.horizon)
        self._predictor.fit(X, y)
        self._is_trained = True

    def predict_next(self, recent_features_df: pd.DataFrame) -> PredictionResult:
        if not self._is_trained:
            raise RuntimeError("call train() before predict_next()")
        X = recent_features_df[FEATURE_COLUMNS]
        preds = self._predictor.predict(X)
        return PredictionResult(predicted_load=preds, model_used=self.model_name)


def compare_models(
    features_df: pd.DataFrame, horizon: int = 5, test_frac: float = 0.2
) -> dict:
    """Train every registered model on the identical chronological split
    and return {model_name: mape}, plus the naive baseline for reference.
    This is the table that goes in the resume writeup."""
    X, y = build_supervised_dataset(features_df, horizon)
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_frac)

    results = {"naive_baseline": naive_baseline_mape(y_test, X_test)}
    for name, cls in _REGISTRY.items():
        model = cls()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = mape(y_test.to_numpy(), preds)

    return results