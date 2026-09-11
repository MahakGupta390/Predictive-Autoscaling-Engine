import numpy as np
import pandas as pd

from metrics_source import SyntheticMetricSource, SyntheticConfig
from preprocessing import process_batch, PreprocessConfig
from prediction_service import (
    build_supervised_dataset,
    chronological_split,
    compare_models,
    PredictionService,
    FEATURE_COLUMNS,
)


def _make_features_df(duration_minutes=480, seed=5):
    cfg = SyntheticConfig(duration_minutes=duration_minutes, seed=seed, burst_probability=0.015)
    raw = SyntheticMetricSource(cfg).generate_dataset()
    return process_batch(raw, PreprocessConfig())


def test_target_is_shifted_correctly():
    df = pd.DataFrame({
        "request_rate": [10, 20, 30, 40, 50],
        "cpu_pct": [1, 1, 1, 1, 1],
        "mem_pct": [1, 1, 1, 1, 1],
        "rr_roll_mean_short": [1, 1, 1, 1, 1],
        "rr_roll_mean_long": [1, 1, 1, 1, 1],
        "rr_ewma": [1, 1, 1, 1, 1],
        "rr_lag_1": [1, 1, 1, 1, 1],
        "rr_lag_5": [1, 1, 1, 1, 1],
    })
    X, y = build_supervised_dataset(df, horizon=2)
    # row 0's target should be request_rate at row 2 -> 30
    assert y.iloc[0] == 30
    # last 2 rows dropped (no future value available)
    assert len(X) == 3


def test_chronological_split_has_no_leakage():
    df = _make_features_df(duration_minutes=120)
    X, y = build_supervised_dataset(df, horizon=5)
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_frac=0.2)

    # index order preserved -> every train row index precedes every test row index
    assert X_train.index.max() < X_test.index.min(), "test rows must all come after train rows in time"
    assert len(X_train) + len(X_test) == len(X)


def test_linear_regression_beats_naive_baseline():
    df = _make_features_df(duration_minutes=480, seed=7)
    results = compare_models(df, horizon=5, test_frac=0.2)
    assert results["linear_regression"] < results["naive_baseline"], (
        f"LR ({results['linear_regression']:.2f}%) should beat naive "
        f"({results['naive_baseline']:.2f}%) or it's not adding value"
    )


def test_xgboost_produces_finite_comparable_predictions():
    df = _make_features_df(duration_minutes=480, seed=7)
    results = compare_models(df, horizon=5, test_frac=0.2)
    assert np.isfinite(results["xgboost"])
    assert results["xgboost"] < results["naive_baseline"] * 2, "xgboost output looks unreasonable"


def test_registry_swap_produces_consistent_output_shape():
    df = _make_features_df(duration_minutes=240, seed=9)
    recent_window = df[FEATURE_COLUMNS].tail(10)

    for model_name in ["linear_regression", "xgboost"]:
        svc = PredictionService(model_name=model_name, horizon=5)
        svc.train(df)
        result = svc.predict_next(recent_window)
        assert result.model_used == model_name
        assert len(result.predicted_load) == len(recent_window)
        assert np.all(np.isfinite(result.predicted_load))


if __name__ == "__main__":
    test_target_is_shifted_correctly()
    test_chronological_split_has_no_leakage()
    test_linear_regression_beats_naive_baseline()
    test_xgboost_produces_finite_comparable_predictions()
    test_registry_swap_produces_consistent_output_shape()
    print("all prediction engine tests passed")