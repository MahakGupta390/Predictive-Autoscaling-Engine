"""
Accuracy tracker.

Predictions are claims about the future, so they can't be scored the
moment they're made -- they're parked under the TICK they're targeting
(current_tick + horizon) and only resolved once that tick's real metric
actually arrives. Tick index rather than timestamp, since it's exact and
avoids any clock-drift/float-precision matching issues.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class PredictionLogEntry:
    timestamp: object
    model_name: str
    predicted: float
    actual: float
    error_pct: float
    resolved_tick: int = -1


def _point_error_pct(predicted: float, actual: float) -> float:
    denom = max(abs(actual), 1e-3)
    return abs(predicted - actual) / denom * 100


class AccuracyTracker:
    def __init__(self, max_log_len: int = 5000):
        # target_tick -> [(model_name, predicted_value), ...]
        self._pending: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        self._log: Deque[PredictionLogEntry] = deque(maxlen=max_log_len)

    def log_prediction(
        self, current_tick: int, horizon: int, model_name: str, predicted_value: float
    ) -> None:
        target_tick = current_tick + horizon
        self._pending[target_tick].append((model_name, predicted_value))

    def record_actual(
        self, current_tick: int, actual_value: float, timestamp: Optional[object] = None
    ) -> List[PredictionLogEntry]:
        """Resolve any predictions that were targeting this tick. Returns
        the entries just resolved (empty if nothing was pending for this
        tick -- most ticks won't have a matching prediction unless horizon
        divides evenly into the logging cadence)."""
        entries = self._pending.pop(current_tick, [])
        resolved = []
        for model_name, predicted in entries:
            err = _point_error_pct(predicted, actual_value)
            row = PredictionLogEntry(
                timestamp=timestamp, model_name=model_name,
                predicted=predicted, actual=actual_value, error_pct=err,
                resolved_tick=current_tick,
            )
            self._log.append(row)
            resolved.append(row)
        return resolved

    def count_since(self, model_name: str, since_tick: int) -> int:
        """How many predictions for this model have resolved at or after
        since_tick -- used to gate the degradation check so it doesn't
        fire off a handful of noisy samples."""
        return sum(
            1 for e in self._log
            if e.model_name == model_name and e.resolved_tick >= since_tick
        )

    def rolling_mape(self, model_name: str, window: Optional[int] = None) -> Optional[float]:
        errors = [e.error_pct for e in self._log if e.model_name == model_name]
        if window is not None:
            errors = errors[-window:]
        if not errors:
            return None
        return sum(errors) / len(errors)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self._log])

    def pending_count(self) -> int:
        return sum(len(v) for v in self._pending.values())