import matplotlib.pyplot as plt
import pandas as pd

from metrics_source import SyntheticMetricSource, SyntheticConfig
from preprocessing import process_batch, PreprocessConfig
from prediction_service import (
    build_supervised_dataset,
    chronological_split,
    LinearRegressionPredictor,
    XGBoostPredictor,
    mape,
)


# -----------------------------
# 1. Generate + preprocess data
# -----------------------------
cfg = SyntheticConfig(
    duration_minutes=480,
    seed=7,
    burst_probability=0.015,
)

raw = SyntheticMetricSource(cfg).generate_dataset()
features = process_batch(raw, PreprocessConfig())


# -----------------------------
# 2. Build future-prediction dataset
# -----------------------------
horizon = 5

X, y = build_supervised_dataset(features, horizon)

X_train, X_test, y_train, y_test = chronological_split(
    X, y, test_frac=0.2
)


# -----------------------------
# 3. Train both models
# -----------------------------
lr = LinearRegressionPredictor()
lr.fit(X_train, y_train)

xgb = XGBoostPredictor()
xgb.fit(X_train, y_train)


# -----------------------------
# 4. Generate predictions
# -----------------------------
naive_pred = X_test["request_rate"].to_numpy()

lr_pred = lr.predict(X_test)
xgb_pred = xgb.predict(X_test)


# -----------------------------
# 5. Calculate MAPE
# -----------------------------
naive_mape = mape(y_test.to_numpy(), naive_pred)
lr_mape = mape(y_test.to_numpy(), lr_pred)
xgb_mape = mape(y_test.to_numpy(), xgb_pred)


print(f"Naive MAPE:           {naive_mape:.2f}%")
print(f"Linear Regression:    {lr_mape:.2f}%")
print(f"XGBoost:              {xgb_mape:.2f}%")


# -----------------------------
# 6. Plot
# -----------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    y_test.index,
    y_test.to_numpy(),
    label="actual"
)

plt.plot(
    y_test.index,
    naive_pred,
    label="naive (persistence)"
)

plt.plot(
    y_test.index,
    lr_pred,
    label="linear regression"
)

plt.plot(
    y_test.index,
    xgb_pred,
    label="xgboost"
)

plt.title(
    f"Hold-out test set: actual vs naive vs LR vs XGBoost "
    f"({horizon}-tick horizon)"
)

plt.xlabel("Time / test-set index")
plt.ylabel("Request rate")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot_predictions_comparison.png", dpi=130)
plt.show()