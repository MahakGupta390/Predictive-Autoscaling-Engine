"""
Stateless "what-if" preview.

Backs Tab 2 (Live Control Panel) and Tab 3 (What-If Scenarios). Unlike a
real reconciler tick, this does NOT touch SafetyState -- there's no
"previous tick" for a hypothetical, so stabilization/rate-limiting simply
don't apply here. This shows what the system WOULD recommend right now,
not what it would actually do after safety smoothing.

Our prediction features are trailing-window (rolling mean, EWMA, lags) --
not a single point in time -- so a bare slider value has nothing to
compute those from. The fix: splice the override onto the tail of an
already-run simulation's history, recompute features for just that last
row, and predict from there.
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from prediction_service import PredictionService, FEATURE_COLUMNS
from preprocessing import process_batch, PreprocessConfig
from hpa_calculator import calculate_desired_replicas, HPAConfig, HPADecision
from capacity_model import CapacityConfig
from sla_evaluator import SLAConfig, check_sla_violation, required_replicas_for_sla,SLAViolationReport
from fusion import combine, FusedDecision


@dataclass
class PreviewResult:
    predicted_request_rate: float
    predicted_delta: float              # vs override_request_rate
    hpa_decision: HPADecision
    predicted_sla_replicas: int
    fused_decision: FusedDecision
    sla_violation: SLAViolationReport
    replica_delta: int                  # fused target vs override_replicas
    out_of_range_warning: Optional[str] # None if the override was within the training range


def _training_range(history_df: pd.DataFrame) -> tuple:
    return float(history_df["request_rate"].min()), float(history_df["request_rate"].max())


def preview_decision(
    history_df: pd.DataFrame,
    override_request_rate: float,
    override_cpu_pct: float,
    override_replicas: int,
    prediction_service: PredictionService,
    hpa_cfg: HPAConfig,
    sla_cfg: SLAConfig,
    capacity_cfg: CapacityConfig,
    preprocess_cfg: PreprocessConfig = PreprocessConfig(),
) -> PreviewResult:
    """history_df: raw MetricSample-shaped rows (timestamp, pod_name,
    cpu_pct, mem_pct, request_rate) from an already-run simulation --
    NOT feature-engineered yet, this function does that itself so the
    override is reflected in the rolling features, not bolted on after."""
    spliced = history_df.copy()
    last_idx = spliced.index[-1]
    spliced.loc[last_idx, "request_rate"] = override_request_rate
    spliced.loc[last_idx, "cpu_pct"] = override_cpu_pct

    featured = process_batch(spliced, preprocess_cfg)
    last_row = featured[FEATURE_COLUMNS].tail(1)

    result = prediction_service.predict_next(last_row)
    predicted = float(result.predicted_load[0])

    # extrapolation warning: XGBoost can't extrapolate past training-range
    # leaf values at all -- it'll silently return the nearest leaf
    # regardless of how far out the input goes, which looks authoritative
    # but isn't. Flag it rather than let the number stand unqualified.
    lo, hi = _training_range(history_df)
    warning = None
    if override_request_rate < lo or override_request_rate > hi:
        warning = (
            f"override request_rate ({override_request_rate:.0f}) is outside the loaded "
            f"simulation's training range ({lo:.0f}-{hi:.0f}) -- the model cannot reliably "
            f"extrapolate here, treat this prediction with caution"
        )

    hpa_decision = calculate_desired_replicas(
    current_replicas=override_replicas,
    current_utilization_pct=override_cpu_pct / max(1, override_replicas),
    cfg=hpa_cfg,
)

    predicted_sla_replicas = required_replicas_for_sla(
        predicted,
        sla_cfg,
        capacity_cfg,
)

    fused_decision = combine(
        hpa_decision,
        predicted,
        sla_cfg,
        capacity_cfg,
)
    sla_violation = check_sla_violation(
        current_replicas=override_replicas, current_request_rate=override_request_rate,
        current_cpu_pct=override_cpu_pct, sla_cfg=sla_cfg, capacity_cfg=capacity_cfg,
    )

    return PreviewResult(
        predicted_request_rate=predicted,
        predicted_delta=predicted - override_request_rate,
        hpa_decision=hpa_decision,
        predicted_sla_replicas=predicted_sla_replicas,
        fused_decision=fused_decision,
        sla_violation=sla_violation,
        replica_delta=fused_decision.target_replicas - override_replicas,
        out_of_range_warning=warning,
)