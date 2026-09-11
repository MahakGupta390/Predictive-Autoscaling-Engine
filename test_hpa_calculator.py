from hpa_calculator import calculate_desired_replicas, HPAConfig, ScaleAction


def test_exactly_at_target_maintains():
    cfg = HPAConfig(target_utilization_pct=50.0)
    d = calculate_desired_replicas(current_replicas=4, current_utilization_pct=50.0, cfg=cfg)
    assert d.desired_replicas == 4
    assert d.action == ScaleAction.MAINTAIN


def test_within_tolerance_band_does_not_flap():
    # 54% vs 50% target is within the default ±10% tolerance -> no change
    cfg = HPAConfig(target_utilization_pct=50.0, tolerance=0.10)
    d = calculate_desired_replicas(current_replicas=4, current_utilization_pct=54.0, cfg=cfg)
    assert d.action == ScaleAction.MAINTAIN
    assert d.desired_replicas == 4


def test_double_target_roughly_doubles_replicas():
    cfg = HPAConfig(target_utilization_pct=50.0)
    d = calculate_desired_replicas(current_replicas=4, current_utilization_pct=100.0, cfg=cfg)
    assert d.desired_replicas == 8
    assert d.action == ScaleAction.SCALE_UP


def test_low_utilization_scales_down():
    cfg = HPAConfig(target_utilization_pct=50.0)
    d = calculate_desired_replicas(current_replicas=8, current_utilization_pct=10.0, cfg=cfg)
    assert d.desired_replicas < 8
    assert d.action == ScaleAction.SCALE_DOWN


def test_zero_utilization_clamps_to_min_not_zero():
    cfg = HPAConfig(target_utilization_pct=50.0, min_replicas=2)
    d = calculate_desired_replicas(current_replicas=5, current_utilization_pct=0.0, cfg=cfg)
    assert d.desired_replicas == 2


def test_extreme_utilization_clamps_to_max():
    cfg = HPAConfig(target_utilization_pct=50.0, max_replicas=10)
    d = calculate_desired_replicas(current_replicas=5, current_utilization_pct=1000.0, cfg=cfg)
    assert d.desired_replicas == 10


def test_rounds_up_never_down():
    # current=3, ratio = 70/50 = 1.4 -> 3*1.4 = 4.2 -> must round to 5, not 4
    cfg = HPAConfig(target_utilization_pct=50.0)
    d = calculate_desired_replicas(current_replicas=3, current_utilization_pct=70.0, cfg=cfg)
    assert d.desired_replicas == 5


if __name__ == "__main__":
    test_exactly_at_target_maintains()
    test_within_tolerance_band_does_not_flap()
    test_double_target_roughly_doubles_replicas()
    test_low_utilization_scales_down()
    test_zero_utilization_clamps_to_min_not_zero()
    test_extreme_utilization_clamps_to_max()
    test_rounds_up_never_down()
    print("all stage 3 tests passed")
