"""
Safety + decision module.

Mirrors real Kubernetes HPA's asymmetric stabilization strategy: scale-up
applies immediately (under-provisioning is the emergency to avoid),
scale-down only applies the highest recommendation seen over a trailing
stabilization window (dropped load has to STAY dropped before we shrink).
Then rate-limits the resulting delta and clamps to min/max.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Optional, Tuple

from hpa_calculator import ScaleAction


@dataclass
class SafetyConfig:
    min_replicas: int = 1
    max_replicas: int = 10
    stabilization_window_seconds: int = 300   # matches k8s HPA scale-down default
    max_scale_up_step: Optional[int] = None    # None = no limit, react fully
    max_scale_down_step: Optional[int] = 1     # conservative default: shrink 1 at a time


@dataclass
class FinalDecision:
    replicas: int
    action: ScaleAction


class SafetyState:
    """Holds the rolling window of recent fusion recommendations and the
    last replica count actually applied. Must persist across ticks —
    this is the one module in the pipeline with real memory."""

    def __init__(self, initial_replicas: int = 1):
        self.last_applied_replicas: int = initial_replicas
        self._history: Deque[Tuple[datetime, int]] = deque()

    def record(self, now: datetime, recommended_replicas: int, window_seconds: int) -> None:
        self._history.append((now, recommended_replicas))
        cutoff = now - timedelta(seconds=window_seconds)
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def max_in_window(self) -> int:
        return max((r for _, r in self._history), default=self.last_applied_replicas)


def apply_safety(
    recommended_replicas: int,
    now: datetime,
    state: SafetyState,
    cfg: SafetyConfig = SafetyConfig(),
) -> FinalDecision:
    state.record(now, recommended_replicas, cfg.stabilization_window_seconds)

    if recommended_replicas >= state.last_applied_replicas:
        # scale-up (or hold): react immediately, no stabilization delay
        candidate = recommended_replicas
    else:
        # scale-down: only shrink to the HIGHEST recommendation seen across
        # the trailing window, not this tick's value alone
        candidate = state.max_in_window()

    # rate limiting, applied per direction
    delta = candidate - state.last_applied_replicas
    if delta > 0 and cfg.max_scale_up_step is not None:
        delta = min(delta, cfg.max_scale_up_step)
    elif delta < 0 and cfg.max_scale_down_step is not None:
        delta = max(delta, -cfg.max_scale_down_step)
    candidate = state.last_applied_replicas + delta

    # final clamp
    candidate = max(cfg.min_replicas, min(cfg.max_replicas, candidate))

    if candidate > state.last_applied_replicas:
        action = ScaleAction.SCALE_UP
    elif candidate < state.last_applied_replicas:
        action = ScaleAction.SCALE_DOWN
    else:
        action = ScaleAction.MAINTAIN

    state.last_applied_replicas = candidate
    return FinalDecision(replicas=candidate, action=action)