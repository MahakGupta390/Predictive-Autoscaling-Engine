"""
Stage 4: Capacity model.

Converts a request rate into "replicas needed to stay under per-container
capacity." The requests_per_container number is a placeholder until real
load-test data is available — the config load/save is built now so that
swap is a YAML edit later, not a code change.
"""

import math
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


@dataclass
class CapacityConfig:
    requests_per_container: float = 40.0   # placeholder — replace after load testing
    safety_margin: float = 0.15            # reserve 15% headroom per container
    min_replicas: int = 1

    @property
    def effective_capacity(self) -> float:
        """Capacity after reserving headroom — never let a container run
        right up to its theoretical limit."""
        return self.requests_per_container * (1.0 - self.safety_margin)


def required_replicas_for_capacity(
    request_rate: float, cfg: CapacityConfig = CapacityConfig()
) -> int:
    if request_rate <= 0:
        return cfg.min_replicas

    replicas = math.ceil(request_rate / cfg.effective_capacity)
    return max(cfg.min_replicas, replicas)


def save_capacity_config(cfg: CapacityConfig, path: str) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(cfg)))


def load_capacity_config(path: str) -> CapacityConfig:
    data = yaml.safe_load(Path(path).read_text())
    return CapacityConfig(**data)