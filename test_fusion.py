from hpa_calculator import HPADecision, ScaleAction
from sla_evaluator import SLAConfig
from capacity_model import CapacityConfig
from fusion import combine


# Fixed, generous capacity config so we can reason about SLA-required
# replicas by hand across these tests.
CAP_CFG = CapacityConfig(requests_per_container=40.0, safety_margin=0.0)
SLA_CFG = SLAConfig(target_latency_ms=200.0, reference_latency_ms=200.0)  # no derating


def test_prediction_wins_when_it_requires_more():
    # request_rate 400 / 40 per container = 10 replicas needed for SLA
    hpa = HPADecision(desired_replicas=3, raw_ratio=1.0, action=ScaleAction.MAINTAIN)
    result = combine(hpa, predicted_request_rate=400.0, sla_cfg=SLA_CFG, capacity_cfg=CAP_CFG)
    assert result.target_replicas == 10
    assert "predicted load requires" in result.rationale


def test_reactive_wins_when_it_requires_more():
    # request_rate 40 -> only 1 replica needed for SLA, but reactive already at 6
    hpa = HPADecision(desired_replicas=6, raw_ratio=2.0, action=ScaleAction.SCALE_UP)
    result = combine(hpa, predicted_request_rate=40.0, sla_cfg=SLA_CFG, capacity_cfg=CAP_CFG)
    assert result.target_replicas == 6
    assert "reactive HPA wants" in result.rationale


def test_tie_break_reports_agreement():
    # request_rate 120 -> exactly 3 replicas for SLA, reactive also says 3
    hpa = HPADecision(desired_replicas=3, raw_ratio=1.0, action=ScaleAction.MAINTAIN)
    result = combine(hpa, predicted_request_rate=120.0, sla_cfg=SLA_CFG, capacity_cfg=CAP_CFG)
    assert result.target_replicas == 3
    assert "agree" in result.rationale


def test_never_scales_below_reactive_floor():
    # even with near-zero predicted load, fusion must not undercut reactive
    hpa = HPADecision(desired_replicas=8, raw_ratio=3.0, action=ScaleAction.SCALE_UP)
    result = combine(hpa, predicted_request_rate=1.0, sla_cfg=SLA_CFG, capacity_cfg=CAP_CFG)
    assert result.target_replicas >= hpa.desired_replicas


if __name__ == "__main__":
    test_prediction_wins_when_it_requires_more()
    test_reactive_wins_when_it_requires_more()
    test_tie_break_reports_agreement()
    test_never_scales_below_reactive_floor()
    print("all fusion tests passed")