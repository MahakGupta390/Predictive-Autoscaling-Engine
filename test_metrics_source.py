import numpy as np
from metrics_source import SyntheticMetricSource, SyntheticConfig


def test_schema_and_ranges():
    src = SyntheticMetricSource(SyntheticConfig(duration_minutes=30))
    df = src.generate_dataset()

    assert set(df.columns) == {
        "timestamp", "pod_name", "cpu_pct", "mem_pct", "request_rate"
    }
    assert not df.isnull().values.any(), "no NaNs allowed downstream"
    assert df.cpu_pct.between(0, 100).all()
    assert df.mem_pct.between(0, 100).all()
    assert (df.request_rate >= 0).all()


def test_burst_is_visible():
    # burst_probability raised so a 3hr run reliably contains at least one
    cfg = SyntheticConfig(duration_minutes=180, burst_probability=0.02)
    df = SyntheticMetricSource(cfg).generate_dataset()

    baseline = df.request_rate.quantile(0.5)
    peak = df.request_rate.max()
    assert peak > 2 * baseline, "synthetic dataset has no meaningful burst"


def test_reproducible_with_seed():
    cfg = SyntheticConfig(seed=7, duration_minutes=20)
    a = SyntheticMetricSource(cfg).generate_dataset()
    b = SyntheticMetricSource(cfg).generate_dataset()
    assert np.allclose(a.request_rate, b.request_rate)


def test_cpu_correlates_with_request_rate():
    # cpu_pct should track request_rate, not be independent noise —
    # otherwise there's nothing for the predictor to learn later
    cfg = SyntheticConfig(duration_minutes=120, resource_noise_std=1.0)
    df = SyntheticMetricSource(cfg).generate_dataset()
    corr = df.request_rate.corr(df.cpu_pct)
    assert corr > 0.8, f"expected strong correlation, got {corr:.2f}"


if __name__ == "__main__":
    test_schema_and_ranges()
    test_burst_is_visible()
    test_reproducible_with_seed()
    test_cpu_correlates_with_request_rate()
    print("all stage 1 tests passed")
