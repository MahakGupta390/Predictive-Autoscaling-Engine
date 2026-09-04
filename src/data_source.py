import numpy as np
import pandas as pd

def fabricate_data(hours=48, interval_seconds=15, seed=42):
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / interval_seconds)
    timestamps = pd.date_range("2026-01-01", periods=n, freq=f"{interval_seconds}s")

    t = np.arange(n)
    steps_per_day = int(24 * 3600 / interval_seconds)
    daily = 30 + 20 * np.sin(2 * np.pi * t / steps_per_day)

    bursts = np.zeros(n)
    burst_starts = rng.choice(n, size=max(1, hours // 6), replace=False)
    for s in burst_starts:
        length = rng.integers(20, 80)
        end = min(s + length, n)
        bursts[s:end] += rng.uniform(30, 60)

    noise = rng.normal(0, 3, n)
    cpu_pct = np.clip(daily + bursts + noise, 0, 100)
    mem_pct = np.clip(cpu_pct * 0.7 + rng.normal(0, 5, n) + 20, 0, 100)
    request_rate = np.clip(cpu_pct * 4 + rng.normal(0, 15, n), 0, None)

    return pd.DataFrame({
        "timestamp": timestamps,
        "pod_name": "app-pod-1",
        "cpu_pct": cpu_pct,
        "mem_pct": mem_pct,
        "request_rate": request_rate,
    })

if __name__ == "__main__":
    df = fabricate_data()
    df.to_csv("data/fabricated_workload.csv", index=False)
    print(f"Wrote {len(df)} rows to data/fabricated_workload.csv")