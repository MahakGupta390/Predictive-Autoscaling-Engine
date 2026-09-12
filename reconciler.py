"""
Reconciler.

Orchestrates every stage into one tick: metrics -> features -> reactive
HPA calc (in parallel with) prediction -> fusion + SLA -> safety -> apply
-> accuracy tracking / retraining check. Nothing here is new logic -- it's
wiring together modules already built and independently tested.

Bootstrap: runs in a naive fallback (predicted = current value) until
enough history exists to train the prediction service, then switches over.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd

from metrics_source import MetricSample
from preprocessing import Preprocessor, PreprocessConfig, process_batch
from prediction_service import PredictionService, FEATURE_COLUMNS, compare_models, default_model_path
from hpa_calculator import calculate_desired_replicas, HPAConfig, HPADecision
from capacity_model import CapacityConfig
from sla_evaluator import SLAConfig, check_sla_violation, SLAViolationReport
from fusion import combine, FusedDecision
from safety import SafetyState, SafetyConfig, apply_safety, FinalDecision
from resource_manager import ResourceManager, ScaleResult
from accuracy_tracker import AccuracyTracker
from retraining_trigger import RetrainingTrigger, RetrainPolicy, RetrainDecision


@dataclass
class ReconcilerConfig:
    model_name: str = "xgboost"
    horizon: int = 5
    warmup_ticks: int = 60             # ticks of history before the first training

    preprocess_cfg: PreprocessConfig = field(default_factory=PreprocessConfig)
    hpa_cfg: HPAConfig = field(default_factory=HPAConfig)
    capacity_cfg: CapacityConfig = field(default_factory=CapacityConfig)
    sla_cfg: SLAConfig = field(default_factory=SLAConfig)
    safety_cfg: SafetyConfig = field(default_factory=SafetyConfig)
    retrain_policy: RetrainPolicy = field(default_factory=RetrainPolicy)

    initial_replicas: int = 2
    model_save_dir: Optional[str] = None   # if set, saves after every (re)train


@dataclass
class TickResult:
    tick: int
    timestamp: object
    current_replicas: int
    request_rate: float
    cpu_pct: float
    predicted_request_rate: Optional[float]
    model_used: Optional[str]
    hpa_decision: HPADecision
    fused_decision: FusedDecision
    final_decision: FinalDecision
    sla_violation: SLAViolationReport
    scale_result: Optional[ScaleResult]
    retrain_decision: Optional[RetrainDecision]


def _validate_shared_limits(cfg: ReconcilerConfig) -> None:
    """HPAConfig, SafetyConfig, and CapacityConfig can each independently
    define min/max replicas -- if they ever disagree, the pipeline ends up
    with clamps fighting each other in ways that are hard to debug later.
    Fail fast at construction time instead."""
    mins = {cfg.hpa_cfg.min_replicas, cfg.safety_cfg.min_replicas, cfg.capacity_cfg.min_replicas}
    if len(mins) > 1:
        raise ValueError(f"min_replicas disagrees across configs: {mins} -- must all match")
    maxes = {cfg.hpa_cfg.max_replicas, cfg.safety_cfg.max_replicas}
    if len(maxes) > 1:
        raise ValueError(f"max_replicas disagrees across configs: {maxes} -- must all match")


class Reconciler:
    def __init__(
        self,
        cfg: ReconcilerConfig = ReconcilerConfig(),
        resource_manager: Optional[ResourceManager] = None,
    ):
        _validate_shared_limits(cfg)
        self.cfg = cfg
        self.resource_manager = resource_manager

        self.preprocessor = Preprocessor(cfg.preprocess_cfg)
        self.prediction_service = PredictionService(model_name=cfg.model_name, horizon=cfg.horizon)
        self.safety_state = SafetyState(initial_replicas=cfg.initial_replicas)
        self.accuracy_tracker = AccuracyTracker()
        self.retraining_trigger = RetrainingTrigger(self.accuracy_tracker, cfg.retrain_policy)

        self._tick = 0
        self._is_trained = False

    def _get_current_replicas(self) -> int:
        if self.resource_manager is not None:
            return self.resource_manager.get_current_replicas()
        # no cluster attached -- last-applied count IS the ground truth here
        return self.safety_state.last_applied_replicas

    def _train(self, current_tick: int) -> None:
        history_df = self.preprocessor.history.as_dataframe()
        featured = process_batch(history_df, self.cfg.preprocess_cfg)

        self.prediction_service.train(featured)
        self._is_trained = True

        offline = compare_models(featured, horizon=self.cfg.horizon, test_frac=0.2)
        model_mape = offline[self.cfg.model_name]
        self.retraining_trigger.set_baseline(self.cfg.model_name, model_mape, current_tick)

        if self.cfg.model_save_dir:
            path = default_model_path(self.cfg.model_save_dir, self.cfg.model_name)
            self.prediction_service.save(path)

    def run_tick(self, sample: MetricSample) -> TickResult:
        tick = self._tick
        self._tick += 1

        feature_vector = self.preprocessor.process(sample)

        # resolve any prediction that was targeting this tick BEFORE logging
        # a new one for this tick (order matters: don't resolve your own
        # brand-new prediction against itself)
        self.accuracy_tracker.record_actual(tick, sample.request_rate, sample.timestamp)

        current_replicas = self._get_current_replicas()

        # the synthetic generator models cpu_pct as a function of TOTAL
        # request_rate with no notion of replica count (see metrics_source.py) --
        # real per-pod Kubernetes metrics naturally reflect load spreading
        # across pods, so approximate that here. Without this, the reactive
        # HPA path never sees any relief from scaling up and gets stuck at
        # max_replicas forever, since cpu_pct never drops regardless of
        # how many replicas are running.
        approx_per_pod_cpu_pct = feature_vector.cpu_pct / max(1, current_replicas)

        hpa_decision = calculate_desired_replicas(
            current_replicas=current_replicas,
            current_utilization_pct=approx_per_pod_cpu_pct,
            cfg=self.cfg.hpa_cfg,
        )

        # bootstrap: not enough history yet to train
        if not self._is_trained and len(self.preprocessor.history) >= self.cfg.warmup_ticks:
            self._train(current_tick=tick)

        if self._is_trained:
            recent = pd.DataFrame([feature_vector.__dict__])[FEATURE_COLUMNS]
            result = self.prediction_service.predict_next(recent)
            predicted_request_rate = float(result.predicted_load[0])
            model_used = result.model_used
            self.accuracy_tracker.log_prediction(
                current_tick=tick, horizon=self.cfg.horizon,
                model_name=model_used, predicted_value=predicted_request_rate,
            )
        else:
            # naive fallback during warmup: predicted = current
            predicted_request_rate = feature_vector.request_rate
            model_used = None

        fused_decision = combine(
            hpa_decision, predicted_request_rate, self.cfg.sla_cfg, self.cfg.capacity_cfg
        )
        final_decision = apply_safety(
            fused_decision.target_replicas, now=sample.timestamp,
            state=self.safety_state, cfg=self.cfg.safety_cfg,
        )

        sla_violation = check_sla_violation(
            current_replicas=current_replicas, current_request_rate=feature_vector.request_rate,
            current_cpu_pct=feature_vector.cpu_pct, sla_cfg=self.cfg.sla_cfg, capacity_cfg=self.cfg.capacity_cfg,
        )

        scale_result = self.resource_manager.apply(final_decision) if self.resource_manager else None

        retrain_decision = None
        if self._is_trained:
            retrain_decision = self.retraining_trigger.should_retrain(self.cfg.model_name, current_tick=tick)
            if retrain_decision.should_retrain:
                self._train(current_tick=tick)

        return TickResult(
            tick=tick, timestamp=sample.timestamp, current_replicas=current_replicas,
            request_rate=feature_vector.request_rate, cpu_pct=feature_vector.cpu_pct,
            predicted_request_rate=predicted_request_rate, model_used=model_used,
            hpa_decision=hpa_decision, fused_decision=fused_decision, final_decision=final_decision,
            sla_violation=sla_violation, scale_result=scale_result, retrain_decision=retrain_decision,
        )

    def run(self, samples: Iterable[MetricSample], max_ticks: Optional[int] = None) -> list:
        results = []
        for i, sample in enumerate(samples):
            if max_ticks is not None and i >= max_ticks:
                break
            results.append(self.run_tick(sample))
        return results