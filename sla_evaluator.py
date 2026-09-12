"""
SLA evaluator.

Converts user-defined SLA thresholds into a required replica count. The
key idea: a tighter latency target means each container safely handles
LESS traffic before latency degrades, so the capacity model's
requests-per-container number gets derated based on how the target
compares to the reference point it was measured at.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from capacity_model import CapacityConfig, required_replicas_for_capacity


@dataclass
class SLAConfig:
    target_latency_ms: float = 150.0
    target_error_rate_pct: float = 1.0
    target_availability_pct: float = 99.9
    cpu_utilization_threshold_pct: float = 80.0   # secondary, diagnostic-only signal

    # the load test that produced CapacityConfig.requests_per_container was
    # run at this latency — this is the reference point derating is relative to
    reference_latency_ms: float = 200.0
    latency_derate_exponent: float = 1.5   # >1: capacity falls off faster than linear as target tightens


def _latency_derating_factor(cfg: SLAConfig) -> float:
    """1.0 = full measured capacity (target as loose or looser than the
    reference load test). Below 1.0 as the target tightens. Never exceeds
    1.0 — a looser-than-reference target doesn't get to claim more
    capacity than was actually measured."""
    ratio = cfg.target_latency_ms / cfg.reference_latency_ms
    return min(1.0, ratio ** cfg.latency_derate_exponent)


def effective_capacity_for_sla(
    sla_cfg: SLAConfig, capacity_cfg: CapacityConfig
) -> CapacityConfig:
    """Returns a derated CapacityConfig reflecting the SLA's latency
    target. error_rate/availability thresholds are tracked in SLAConfig
    for logging/future use but don't currently derate capacity — noted as
    a known simplification, not an oversight."""
    factor = _latency_derating_factor(sla_cfg)
    derated_capacity = capacity_cfg.requests_per_container * factor
    return CapacityConfig(
        requests_per_container=derated_capacity,
        safety_margin=capacity_cfg.safety_margin,
        min_replicas=capacity_cfg.min_replicas,
    )


def required_replicas_for_sla(
    request_rate: float, sla_cfg: SLAConfig, capacity_cfg: CapacityConfig
) -> int:
    derated = effective_capacity_for_sla(sla_cfg, capacity_cfg)
    return required_replicas_for_capacity(request_rate, derated)


@dataclass
class SLAViolationReport:
    violated: bool
    reasons: List[str]


def check_sla_violation(
    current_replicas: int,
    current_request_rate: float,
    current_cpu_pct: Optional[float],
    sla_cfg: SLAConfig,
    capacity_cfg: CapacityConfig,
) -> SLAViolationReport:
    """Diagnostic-only: reports whether the SLA is being violated RIGHT
    NOW and why, independent of what the scaling decision should be.
    Deliberately does not feed back into required_replicas_for_sla() or
    combine() -- this exists for dashboards/alerting, not for the replica
    math, so it doesn't ripple into fusion.py or safety.py."""
    reasons: List[str] = []

    required = required_replicas_for_sla(current_request_rate, sla_cfg, capacity_cfg)
    if current_replicas < required:
        reasons.append(
            f"currently {current_replicas} replicas but {required} are required to hold "
            f"the {sla_cfg.target_latency_ms:.0f}ms latency target at "
            f"{current_request_rate:.0f} req/s"
        )

    if current_cpu_pct is not None and current_cpu_pct > sla_cfg.cpu_utilization_threshold_pct:
        reasons.append(
            f"CPU utilization {current_cpu_pct:.1f}% exceeds "
            f"{sla_cfg.cpu_utilization_threshold_pct:.0f}% threshold"
        )

    return SLAViolationReport(violated=len(reasons) > 0, reasons=reasons)