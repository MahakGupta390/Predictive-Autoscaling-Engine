import tempfile
import os

from capacity_model import (
    CapacityConfig,
    required_replicas_for_capacity,
    save_capacity_config,
    load_capacity_config,
)


def test_exact_boundary():
    # effective_capacity = 40 * (1 - 0.15) = 34 per container.
    # 3 containers' worth exactly at capacity -> should need exactly 3
    cfg = CapacityConfig(requests_per_container=40.0, safety_margin=0.15)
    replicas = required_replicas_for_capacity(3 * cfg.effective_capacity, cfg)
    assert replicas == 3


def test_just_over_boundary_rounds_up():
    cfg = CapacityConfig(requests_per_container=40.0, safety_margin=0.15)
    replicas = required_replicas_for_capacity(3 * cfg.effective_capacity + 1, cfg)
    assert replicas == 4, "even 1 request over capacity must round up, not down"


def test_safety_margin_increases_required_replicas():
    request_rate = 100.0
    no_margin = required_replicas_for_capacity(
        request_rate, CapacityConfig(requests_per_container=40.0, safety_margin=0.0)
    )
    with_margin = required_replicas_for_capacity(
        request_rate, CapacityConfig(requests_per_container=40.0, safety_margin=0.15)
    )
    assert with_margin >= no_margin, "adding headroom should never reduce required replicas"


def test_zero_traffic_clamps_to_min():
    cfg = CapacityConfig(min_replicas=2)
    assert required_replicas_for_capacity(0.0, cfg) == 2


def test_config_roundtrip():
    cfg = CapacityConfig(requests_per_container=55.5, safety_margin=0.2, min_replicas=3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "capacity.yaml")
        save_capacity_config(cfg, path)
        loaded = load_capacity_config(path)
        assert loaded == cfg


if __name__ == "__main__":
    test_exact_boundary()
    test_just_over_boundary_rounds_up()
    test_safety_margin_increases_required_replicas()
    test_zero_traffic_clamps_to_min()
    test_config_roundtrip()
    print("all stage 4 tests passed")