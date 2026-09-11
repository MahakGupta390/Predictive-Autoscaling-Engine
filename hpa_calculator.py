"""
Stage 3: Reactive HPA calculator.

Pure function, no I/O, no history — mirrors the standard Kubernetes HPA
ratio formula, including its tolerance band, so this is a fair baseline to
compare predictive scaling against (not a strawman version of HPA).
"""

import math
from dataclasses import dataclass
from enum import Enum


class ScaleAction(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class HPAConfig:
    target_utilization_pct: float = 50.0
    min_replicas: int = 1
    max_replicas: int = 10
    tolerance: float = 0.10   # ±10%, matches real Kubernetes HPA default


@dataclass
class HPADecision:
    desired_replicas: int
    raw_ratio: float          # current_utilization / target, before clamping
    action: ScaleAction


def calculate_desired_replicas(
    current_replicas: int,
    current_utilization_pct: float,
    cfg: HPAConfig = HPAConfig(),
) -> HPADecision:
    ratio = current_utilization_pct / cfg.target_utilization_pct

    # tolerance band: don't react to small fluctuations around target
    if abs(ratio - 1.0) <= cfg.tolerance:
        desired = current_replicas
    else:
        # ceil, never floor — under-provisioning is the worse failure mode
        desired = math.ceil(current_replicas * ratio)

    desired = max(cfg.min_replicas, min(cfg.max_replicas, desired))

    if desired > current_replicas:
        action = ScaleAction.SCALE_UP
    elif desired < current_replicas:
        action = ScaleAction.SCALE_DOWN
    else:
        action = ScaleAction.MAINTAIN

    return HPADecision(desired_replicas=desired, raw_ratio=ratio, action=action)
