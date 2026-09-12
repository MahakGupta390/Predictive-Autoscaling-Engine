"""
Retraining trigger.

Two independent conditions can fire a retrain: a scheduled interval
(keep the model current as traffic patterns drift, even if accuracy
looks fine), or MAPE degradation relative to the baseline recorded right
after the last training. Degradation is gated by a minimum fresh-sample
count so it can't fire off a handful of noisy predictions.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from accuracy_tracker import AccuracyTracker


@dataclass
class RetrainPolicy:
    scheduled_interval_ticks: int = 2000
    degradation_threshold_pct: float = 25.0   # trigger if MAPE is >25% worse than baseline
    mape_window: int = 50                     # how many recent predictions to average
    min_predictions_before_check: int = 20    # sample gate for the degradation check


@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: str


class RetrainingTrigger:
    def __init__(self, tracker: AccuracyTracker, policy: RetrainPolicy = RetrainPolicy()):
        self.tracker = tracker
        self.policy = policy
        self._baseline_mape: Dict[str, float] = {}
        self._last_retrain_tick: Dict[str, int] = {}

    def set_baseline(self, model_name: str, mape_value: float, current_tick: int) -> None:
        """Call this right after (re)training a model, using the freshly
        measured MAPE (e.g. from compare_models()'s held-out split)."""
        self._baseline_mape[model_name] = mape_value
        self._last_retrain_tick[model_name] = current_tick

    def should_retrain(self, model_name: str, current_tick: int) -> RetrainDecision:
        if model_name not in self._baseline_mape:
            return RetrainDecision(False, "no baseline established yet -- call set_baseline() after initial training")

        last_tick = self._last_retrain_tick[model_name]

        # scheduled check -- fires regardless of measured accuracy
        ticks_since_retrain = current_tick - last_tick
        if ticks_since_retrain >= self.policy.scheduled_interval_ticks:
            return RetrainDecision(
                True, f"scheduled interval elapsed ({ticks_since_retrain} ticks since last retrain)"
            )

        # degradation check -- gated by a minimum sample count
        fresh_samples = self.tracker.count_since(model_name, since_tick=last_tick)
        if fresh_samples < self.policy.min_predictions_before_check:
            return RetrainDecision(
                False, f"only {fresh_samples} fresh predictions since last retrain, too few to judge"
            )

        current_mape = self.tracker.rolling_mape(model_name, window=self.policy.mape_window)
        baseline = self._baseline_mape[model_name]
        allowed = baseline * (1 + self.policy.degradation_threshold_pct / 100)

        if current_mape is not None and current_mape > allowed:
            return RetrainDecision(
                True,
                f"MAPE degraded to {current_mape:.2f}% vs baseline {baseline:.2f}% "
                f"(threshold {self.policy.degradation_threshold_pct}%)",
            )

        return RetrainDecision(False, f"MAPE {current_mape:.2f}% still within tolerance of baseline {baseline:.2f}%")