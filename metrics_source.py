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
import time

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
    burst_duration_ticks_range: Tuple[int, int] = (10, 40)    # hold time at peak
    burst_ramp_ticks_range: Tuple[int, int] = (5, 15)         # ramp up/down time

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
        burst_profile: list = []   # precomputed per-tick contribution for an active burst

        for i in range(n_ticks):
            t = start + timedelta(seconds=i * cfg.interval_seconds)
            minutes = i * cfg.interval_seconds / 60.0

            diurnal = cfg.diurnal_amplitude * np.sin(
                2 * np.pi * minutes / cfg.diurnal_period_minutes
            )
            noise = self._rng.normal(0, cfg.noise_std)

            # burst state machine: ramp up to peak, hold, ramp back down —
            # NOT an instant jump. A step-function burst has zero lead-time
            # signal (it's memoryless), so no model could ever beat naive
            # persistence at predicting it; a ramp gives a real precursor
            # to learn from, same as an actual traffic surge building up.
            if not burst_profile and self._rng.random() < cfg.burst_probability:
                hold_ticks = int(self._rng.integers(*cfg.burst_duration_ticks_range))
                ramp_ticks = int(self._rng.integers(*cfg.burst_ramp_ticks_range))
                peak = cfg.baseline_request_rate * self._rng.uniform(*cfg.burst_magnitude_range)
                ramp_up = np.linspace(0, peak, ramp_ticks, endpoint=False)
                hold = np.full(hold_ticks, peak)
                ramp_down = np.linspace(peak, 0, ramp_ticks, endpoint=False)
                burst_profile = list(ramp_up) + list(hold) + list(ramp_down)

            burst_contrib = burst_profile.pop(0) if burst_profile else 0.0

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


class ClockBasedDemoSource(MetricSource):
    """Demo-only source: request_rate tracks the REAL current wall-clock
    time (a daily sine wave anchored to actual hour-of-day, plus a weekend
    discount and noise) rather than an internal tick counter. Useful for a
    live demo with no cluster available -- numbers move with whatever time
    you happen to be presenting at.

    Still synthetic. This is NOT stage 12's real live source (that one
    reads actual Prometheus/Metrics Server data) -- don't conflate the two
    in a writeup.
    """

    def __init__(
        self,
        pod_name: str = "app-pod-1",
        poll_interval_seconds: int = 15,
        config: Optional[SyntheticConfig] = None,
        seed: Optional[int] = None,
    ):
        self.pod_name = pod_name
        self.poll_interval_seconds = poll_interval_seconds
        self.cfg = config or SyntheticConfig()
        # intentionally non-deterministic by default (seed=None) -- this
        # is for demo variety, not reproducible testing like the seeded
        # SyntheticMetricSource
        self._rng = np.random.default_rng(seed)

    def get_current_sample(self, now: Optional[datetime] = None) -> MetricSample:
        """now is injectable so this is still unit-testable without
        depending on the real clock."""
        now = now or datetime.now()
        cfg = self.cfg

        day_fraction = (now.hour * 3600 + now.minute * 60 + now.second) / 86400.0
        # trough near midnight, peak near midday
        diurnal = cfg.diurnal_amplitude * -np.cos(2 * np.pi * day_fraction)

        request_rate = cfg.baseline_request_rate + diurnal
        if now.weekday() >= 5:   # Saturday=5, Sunday=6
            request_rate *= 0.7

        request_rate = max(0.0, request_rate + self._rng.normal(0, cfg.noise_std))

        cpu_pct = float(np.clip(
            cfg.cpu_per_request * request_rate + self._rng.normal(0, cfg.resource_noise_std),
            0.0, 100.0,
        ))
        mem_pct = float(np.clip(
            cfg.mem_per_request * request_rate + self._rng.normal(0, cfg.resource_noise_std),
            0.0, 100.0,
        ))

        return MetricSample(now, self.pod_name, cpu_pct, mem_pct, request_rate)

    def stream(self) -> Iterator[MetricSample]:
        while True:
            yield self.get_current_sample()
            time.sleep(self.poll_interval_seconds)