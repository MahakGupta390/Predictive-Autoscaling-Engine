"""
Stage 2: Preprocessing / feature engineering.

Cleans raw MetricSamples, maintains a bounded history, and derives rolling
features. process_batch() and Preprocessor.process() share the same feature
logic so training-time (batch) and live-tick (streaming) computation never
diverge.
"""

from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque, List, Optional

import numpy as np
import pandas as pd

from metrics_source import MetricSample


@dataclass
class FeatureVector:
    timestamp: object
    pod_name: str
    request_rate: float
    cpu_pct: float
    mem_pct: float
    rr_roll_mean_short: float   # short window rolling mean of request_rate
    rr_roll_mean_long: float    # long window rolling mean
    rr_ewma: float              # exponentially weighted moving average
    rr_lag_1: float             # request_rate 1 tick ago
    rr_lag_5: float             # request_rate 5 ticks ago
    hour_sin: float             # cyclic encoding of hour-of-day (avoids 23->0 discontinuity)
    hour_cos: float
    is_weekend: int             # 1 for Sat/Sun, else 0


@dataclass
class PreprocessConfig:
    short_window: int = 5
    long_window: int = 20
    ewma_span: int = 10
    lag_steps: tuple = (1, 5)
    outlier_z_thresh: float = 4.0     # clip points beyond this many std devs
    history_max_len: int = 500        # bounded buffer size


class HistoryStore:
    """Bounded buffer of cleaned MetricSamples. Backs both the streaming
    Preprocessor (needs the recent window) and can be dumped to a DataFrame
    for batch training."""

    def __init__(self, max_len: int = 500):
        self._buf: Deque[MetricSample] = deque(maxlen=max_len)

    def append(self, sample: MetricSample) -> None:
        self._buf.append(sample)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in self._buf])

    def __len__(self) -> int:
        return len(self._buf)


def _clean(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Fill missing values, clip statistical outliers on request_rate."""
    df = df.copy()
    numeric_cols = ["request_rate", "cpu_pct", "mem_pct"]

    # raw samples may arrive as ints (e.g. hand-crafted test data); force
    # float64 so later outlier-clipping assignments never hit a dtype error
    df[numeric_cols] = df[numeric_cols].astype(float)

    # forward-fill then back-fill any gaps (live telemetry can drop samples)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # rolling z-score outlier clipping on request_rate — use a window so a
    # legitimate sustained burst doesn't get treated as an outlier
    # shift(1) first: a point must be compared against the window BEFORE it,
    # not a window that includes itself — otherwise a spike drags its own
    # mean/std up and "clipping" ends up targeting a still-huge value
    prior = df["request_rate"].shift(1)
    roll_mean = prior.rolling(cfg.long_window, min_periods=1).mean()
    roll_std = prior.rolling(cfg.long_window, min_periods=1).std().fillna(0)
    # a perfectly flat recent history gives std == 0, which would divide by
    # zero (or NaN out) and silently let the very next spike through — use a
    # small epsilon floor instead so a flat-then-spike pattern still trips
    roll_std_safe = roll_std.clip(lower=1e-6)
    z = (df["request_rate"] - roll_mean) / roll_std_safe
    is_outlier = z.abs() > cfg.outlier_z_thresh
    clip_value = roll_mean + np.sign(df["request_rate"] - roll_mean) * cfg.outlier_z_thresh * roll_std
    df.loc[is_outlier.fillna(False), "request_rate"] = clip_value[is_outlier.fillna(False)]

    return df


def process_batch(raw_df: pd.DataFrame, cfg: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    """Batch feature computation — used for training data prep, and reused
    tick-by-tick by Preprocessor.process() so serving matches training."""
    df = _clean(raw_df, cfg)

    df["rr_roll_mean_short"] = df["request_rate"].rolling(
        cfg.short_window, min_periods=1
    ).mean()
    df["rr_roll_mean_long"] = df["request_rate"].rolling(
        cfg.long_window, min_periods=1
    ).mean()
    df["rr_ewma"] = df["request_rate"].ewm(span=cfg.ewma_span, adjust=False).mean()

    for lag in cfg.lag_steps:
        # cold start: no data `lag` steps back yet -> fall back to current value
        df[f"rr_lag_{lag}"] = df["request_rate"].shift(lag).fillna(df["request_rate"])

    # calendar features, derived purely from the timestamp -- this is why
    # they live here (stage 2) and not on MetricSample itself: any source
    # (synthetic or real Prometheus) already carries a timestamp, so this
    # computation is shared rather than duplicated per-source
    ts = pd.to_datetime(df["timestamp"])
    hour_fraction = (ts.dt.hour + ts.dt.minute / 60.0) / 24.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_fraction)
    df["hour_cos"] = np.cos(2 * np.pi * hour_fraction)
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)

    return df


class Preprocessor:
    """Streaming wrapper: feed one MetricSample at a time, get one
    FeatureVector back, computed via the same process_batch() logic."""

    def __init__(self, config: PreprocessConfig = PreprocessConfig()):
        self.config = config
        self.history = HistoryStore(max_len=config.history_max_len)

    def process(self, sample: MetricSample) -> FeatureVector:
        self.history.append(sample)
        raw_df = self.history.as_dataframe()
        featured = process_batch(raw_df, self.config)
        last = featured.iloc[-1]

        return FeatureVector(
            timestamp=last["timestamp"],
            pod_name=last["pod_name"],
            request_rate=float(last["request_rate"]),
            cpu_pct=float(last["cpu_pct"]),
            mem_pct=float(last["mem_pct"]),
            rr_roll_mean_short=float(last["rr_roll_mean_short"]),
            rr_roll_mean_long=float(last["rr_roll_mean_long"]),
            rr_ewma=float(last["rr_ewma"]),
            rr_lag_1=float(last[f"rr_lag_1"]),
            rr_lag_5=float(last[f"rr_lag_5"]),
            hour_sin=float(last["hour_sin"]),
            hour_cos=float(last["hour_cos"]),
            is_weekend=int(last["is_weekend"]),
        )