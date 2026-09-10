"""
Stage 1: Metrics collection.

Defines the common MetricSample schema and MetricSource interface that both
the synthetic (bootstrap) and live (Prometheus/Metrics Server) sources will
implement, plus the synthetic generator itself.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MetricSample:
    """One reading, for one pod, at one point in time. The shared contract
    every downstream stage (preprocessing, HPA calc, accuracy tracker)
    consumes."""
    timestamp: datetime
    pod_name: str
    cpu_pct: float
    mem_pct: float
    request_rate: float


class MetricSource(ABC):
    """Common interface so synthetic and live sources are interchangeable.
    The reconciler only ever talks to this interface."""

    @abstractmethod
    def stream(self) -> Iterator[MetricSample]:
        """Yield MetricSample objects, one per polling interval."""
        raise NotImplementedError


@dataclass
class SyntheticConfig:
    pod_name: str = "app-pod-1"
    duration_minutes: int = 180
    interval_seconds: int = 15

    # baseline + predictable (diurnal) component
    baseline_request_rate: float = 50.0
    diurnal_amplitude: float = 20.0
    diurnal_period_minutes: float = 60.0

    # unpredictable noise
    noise_std: float = 3.0

    # burst state machine
    burst_probability: float = 0.01          # per-tick chance a new burst starts
    burst_magnitude_range: Tuple[float, float] = (2.0, 4.0)   # multiplier on baseline
    burst_duration_ticks_range: Tuple[int, int] = (10, 40)

    # how request_rate drives cpu/mem
    cpu_per_request: float = 0.8
    mem_per_request: float = 0.5
    resource_noise_std: float = 1.5

    seed: Optional[int] = 42


class SyntheticMetricSource(MetricSource):
    """Generates a fabricated but structured workload: baseline + diurnal
    wave + Gaussian noise + injected multi-tick bursts, with cpu/mem derived
    from request_rate so the columns are actually correlated."""

    def __init__(self, config: SyntheticConfig = SyntheticConfig()):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def generate_dataset(self) -> pd.DataFrame:
        cfg = self.config
        n_ticks = int(cfg.duration_minutes * 60 / cfg.interval_seconds)
        start = datetime.now()

        samples = []
        burst_remaining = 0
        burst_level = 0.0

        for i in range(n_ticks):
            t = start + timedelta(seconds=i * cfg.interval_seconds)
            minutes = i * cfg.interval_seconds / 60.0

            diurnal = cfg.diurnal_amplitude * np.sin(
                2 * np.pi * minutes / cfg.diurnal_period_minutes
            )
            noise = self._rng.normal(0, cfg.noise_std)

            # burst state machine: start a new burst, or continue an existing one
            if burst_remaining <= 0 and self._rng.random() < cfg.burst_probability:
                burst_remaining = int(self._rng.integers(*cfg.burst_duration_ticks_range))
                burst_level = cfg.baseline_request_rate * self._rng.uniform(
                    *cfg.burst_magnitude_range
                )

            if burst_remaining > 0:
                burst_contrib = burst_level
                burst_remaining -= 1
            else:
                burst_contrib = 0.0

            request_rate = max(
                0.0, cfg.baseline_request_rate + diurnal + noise + burst_contrib
            )

            cpu_pct = cfg.cpu_per_request * request_rate + self._rng.normal(
                0, cfg.resource_noise_std
            )
            mem_pct = cfg.mem_per_request * request_rate + self._rng.normal(
                0, cfg.resource_noise_std
            )
            cpu_pct = float(np.clip(cpu_pct, 0.0, 100.0))
            mem_pct = float(np.clip(mem_pct, 0.0, 100.0))

            samples.append(
                MetricSample(t, cfg.pod_name, cpu_pct, mem_pct, request_rate)
            )

        return pd.DataFrame([asdict(s) for s in samples])

    def stream(self) -> Iterator[MetricSample]:
        """Replay the generated dataset row by row, matching the live
        source's interface (one sample per call)."""
        df = self.generate_dataset()
        for _, row in df.iterrows():
            yield MetricSample(
                timestamp=row["timestamp"],
                pod_name=row["pod_name"],
                cpu_pct=row["cpu_pct"],
                mem_pct=row["mem_pct"],
                request_rate=row["request_rate"],
            )

    def save_csv(self, path: str) -> None:
        self.generate_dataset().to_csv(path, index=False)
