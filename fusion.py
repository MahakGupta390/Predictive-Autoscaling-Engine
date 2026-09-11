"""
Fusion combiner.

Deliberately narrow: takes already-computed replica counts from the
reactive HPA calculator and the SLA evaluator, and blends them. Does not
re-run the predictor or re-derive SLA logic itself — that separation is
what makes this module testable with plain mocked inputs.
"""

from dataclasses import dataclass

from hpa_calculator import HPADecision
from sla_evaluator import SLAConfig, required_replicas_for_sla
from capacity_model import CapacityConfig


@dataclass
class FusedDecision:
    target_replicas: int
    rationale: str


def combine(
    hpa_decision: HPADecision,
    predicted_request_rate: float,
    sla_cfg: SLAConfig,
    capacity_cfg: CapacityConfig,
) -> FusedDecision:
    """Fusion rule: take the max of what reactive HPA wants and what the
    SLA-derated capacity model says the predicted load will require. A
    prediction is never allowed to scale BELOW what reactive alone would
    give — it can only push replicas higher, ahead of demand arriving."""
    sla_replicas = required_replicas_for_sla(predicted_request_rate, sla_cfg, capacity_cfg)

    if sla_replicas > hpa_decision.desired_replicas:
        target = sla_replicas
        rationale = (
            f"predicted load requires {sla_replicas} replicas to hold SLA "
            f"(reactive HPA alone wanted {hpa_decision.desired_replicas})"
        )
    elif hpa_decision.desired_replicas > sla_replicas:
        target = hpa_decision.desired_replicas
        rationale = (
            f"reactive HPA wants {hpa_decision.desired_replicas} replicas based on "
            f"current utilization (SLA-driven prediction only required {sla_replicas})"
        )
    else:
        target = sla_replicas
        rationale = f"reactive and predicted SLA requirement agree at {target} replicas"

    return FusedDecision(target_replicas=target, rationale=rationale)