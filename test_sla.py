from capacity_model import CapacityConfig
from sla_evaluator import SLAConfig, required_replicas_for_sla, effective_capacity_for_sla


def test_tighter_sla_never_reduces_required_replicas():
    request_rate = 200.0
    cap_cfg = CapacityConfig(requests_per_container=40.0, safety_margin=0.15)

    loose = required_replicas_for_sla(
        request_rate,
        SLAConfig(target_latency_ms=300.0, reference_latency_ms=200.0),
        cap_cfg,
    )
    tight = required_replicas_for_sla(
        request_rate,
        SLAConfig(target_latency_ms=100.0, reference_latency_ms=200.0),
        cap_cfg,
    )
    assert tight >= loose, "tightening the SLA must never lower the required replica count"


def test_looser_than_reference_capped_at_measured_capacity():
    cap_cfg = CapacityConfig(requests_per_container=40.0, safety_margin=0.0)
    # target much looser than the reference -> factor should cap at 1.0,
    # not claim more capacity than was actually load-tested
    derated = effective_capacity_for_sla(
        SLAConfig(target_latency_ms=1000.0, reference_latency_ms=200.0), cap_cfg
    )
    assert derated.requests_per_container == cap_cfg.requests_per_container


def test_at_reference_latency_matches_undecorated_capacity():
    cap_cfg = CapacityConfig(requests_per_container=40.0, safety_margin=0.1)
    sla_cfg = SLAConfig(target_latency_ms=200.0, reference_latency_ms=200.0)
    derated = effective_capacity_for_sla(sla_cfg, cap_cfg)
    assert abs(derated.requests_per_container - cap_cfg.requests_per_container) < 1e-9


def test_much_tighter_sla_meaningfully_increases_replicas():
    request_rate = 300.0
    cap_cfg = CapacityConfig(requests_per_container=40.0, safety_margin=0.15)
    normal = required_replicas_for_sla(
        request_rate, SLAConfig(target_latency_ms=200.0, reference_latency_ms=200.0), cap_cfg
    )
    strict = required_replicas_for_sla(
        request_rate, SLAConfig(target_latency_ms=50.0, reference_latency_ms=200.0), cap_cfg
    )
    assert strict > normal


if __name__ == "__main__":
    test_tighter_sla_never_reduces_required_replicas()
    test_looser_than_reference_capped_at_measured_capacity()
    test_at_reference_latency_matches_undecorated_capacity()
    test_much_tighter_sla_meaningfully_increases_replicas()
    print("all SLA evaluator tests passed")