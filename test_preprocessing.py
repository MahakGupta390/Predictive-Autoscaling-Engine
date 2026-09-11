from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from metrics_source import MetricSample, SyntheticMetricSource, SyntheticConfig
from preprocessing import Preprocessor, PreprocessConfig, process_batch


def _make_samples(values):
    t0 = datetime.now()
    return [
        MetricSample(
            timestamp=t0 + timedelta(seconds=15 * i),
            pod_name="test-pod",
            cpu_pct=v * 0.5,
            mem_pct=v * 0.3,
            request_rate=v,
        )
        for i, v in enumerate(values)
    ]


def test_rolling_mean_matches_hand_calc():
    # values chosen so a 3-tick rolling mean is easy to check by hand
    values = [10, 20, 30, 40, 50]
    samples = _make_samples(values)
    cfg = PreprocessConfig(short_window=3)
    pre = Preprocessor(cfg)

    outputs = [pre.process(s) for s in samples]

    # last tick's short rolling mean should be mean(30,40,50) = 40
    assert abs(outputs[-1].rr_roll_mean_short - 40.0) < 1e-6


def test_lag_features_correct():
    values = [10, 20, 30, 40, 50, 60]
    samples = _make_samples(values)
    pre = Preprocessor(PreprocessConfig())
    outputs = [pre.process(s) for s in samples]

    # lag_1 at tick i should equal request_rate at tick i-1
    assert abs(outputs[3].rr_lag_1 - values[2]) < 1e-6
    # cold start: tick 0 has no history, lag falls back to current value
    assert abs(outputs[0].rr_lag_1 - values[0]) < 1e-6


def test_outlier_gets_clipped():
    values = [50] * 30 + [5000] + [50] * 10  # one absurd spike
    samples = _make_samples(values)
    pre = Preprocessor(PreprocessConfig(outlier_z_thresh=4.0))
    outputs = [pre.process(s) for s in samples]

    spike_output = outputs[30]
    assert spike_output.request_rate < 500, (
        f"outlier should have been clipped, got {spike_output.request_rate}"
    )


def test_streaming_matches_batch():
    cfg = SyntheticConfig(duration_minutes=60, seed=3)
    raw_df = SyntheticMetricSource(cfg).generate_dataset()
    pcfg = PreprocessConfig()

    batch_result = process_batch(raw_df, pcfg)

    pre = Preprocessor(pcfg)
    stream_results = []
    for _, row in raw_df.iterrows():
        sample = MetricSample(
            timestamp=row["timestamp"],
            pod_name=row["pod_name"],
            cpu_pct=row["cpu_pct"],
            mem_pct=row["mem_pct"],
            request_rate=row["request_rate"],
        )
        stream_results.append(pre.process(sample))

    stream_short_means = np.array([o.rr_roll_mean_short for o in stream_results])
    batch_short_means = batch_result["rr_roll_mean_short"].to_numpy()

    assert np.allclose(stream_short_means, batch_short_means, atol=1e-6), (
        "streaming and batch feature computation diverged — train/serve skew risk"
    )


if __name__ == "__main__":
    test_rolling_mean_matches_hand_calc()
    test_lag_features_correct()
    test_outlier_gets_clipped()
    test_streaming_matches_batch()
    print("all stage 2 tests passed")
