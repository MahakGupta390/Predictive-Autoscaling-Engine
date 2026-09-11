from datetime import datetime, timedelta

from hpa_calculator import ScaleAction
from safety import SafetyConfig, SafetyState, apply_safety


def test_scale_up_applies_immediately():
    cfg = SafetyConfig(stabilization_window_seconds=300, max_scale_up_step=None)
    state = SafetyState(initial_replicas=3)
    now = datetime.now()

    result = apply_safety(recommended_replicas=8, now=now, state=state, cfg=cfg)
    assert result.replicas == 8
    assert result.action == ScaleAction.SCALE_UP


def test_scale_down_suppressed_within_window():
    cfg = SafetyConfig(stabilization_window_seconds=300, max_scale_down_step=None)
    state = SafetyState(initial_replicas=2)
    t0 = datetime.now()

    # a burst pushes us up to 10
    apply_safety(10, t0, state, cfg)
    # 30 seconds later, load has already dropped -> recommendation says 2
    result = apply_safety(2, t0 + timedelta(seconds=30), state, cfg)

    # window (300s) hasn't elapsed since the 10-replica recommendation,
    # so we must NOT drop straight to 2 -- should hold near the recent max
    assert result.replicas == 10
    assert result.action == ScaleAction.MAINTAIN


def test_scale_down_permitted_once_window_elapses():
    cfg = SafetyConfig(stabilization_window_seconds=300, max_scale_down_step=None)
    state = SafetyState(initial_replicas=2)
    t0 = datetime.now()

    apply_safety(10, t0, state, cfg)
    # 310 seconds later -- the old high recommendation has aged out of the window
    result = apply_safety(2, t0 + timedelta(seconds=310), state, cfg)

    assert result.replicas == 2
    assert result.action == ScaleAction.SCALE_DOWN


def test_rate_limit_caps_scale_up_delta():
    cfg = SafetyConfig(max_scale_up_step=2)
    state = SafetyState(initial_replicas=3)
    now = datetime.now()

    result = apply_safety(recommended_replicas=9, now=now, state=state, cfg=cfg)
    assert result.replicas == 5, "should only step up by max_scale_up_step, not jump straight to 9"


def test_rate_limit_caps_scale_down_delta():
    cfg = SafetyConfig(stabilization_window_seconds=0, max_scale_down_step=1)
    state = SafetyState(initial_replicas=9)
    now = datetime.now()

    result = apply_safety(recommended_replicas=2, now=now, state=state, cfg=cfg)
    assert result.replicas == 8, "should only step down by max_scale_down_step, not jump straight to 2"


def test_clamped_to_min_and_max():
    cfg = SafetyConfig(min_replicas=2, max_replicas=6, max_scale_up_step=None, max_scale_down_step=None)
    state = SafetyState(initial_replicas=4)
    now = datetime.now()

    over = apply_safety(recommended_replicas=20, now=now, state=state, cfg=cfg)
    assert over.replicas == 6

    state2 = SafetyState(initial_replicas=4)
    under = apply_safety(recommended_replicas=0, now=now, state=state2, cfg=cfg)
    assert under.replicas == 2


def test_flap_simulation_produces_no_scale_down_events():
    # spike -> drop -> spike, all within the stabilization window --
    # a system without proper stabilization would flap down and back up;
    # this must produce zero scale-down actions across the sequence
    cfg = SafetyConfig(stabilization_window_seconds=300, max_scale_down_step=None)
    state = SafetyState(initial_replicas=2)
    t0 = datetime.now()

    sequence = [10, 2, 9, 3, 10]   # rapid noisy recommendations
    actions = []
    for i, rec in enumerate(sequence):
        result = apply_safety(rec, t0 + timedelta(seconds=20 * i), state, cfg)
        actions.append(result.action)

    assert ScaleAction.SCALE_DOWN not in actions, f"unexpected scale-down in flap sequence: {actions}"


if __name__ == "__main__":
    test_scale_up_applies_immediately()
    test_scale_down_suppressed_within_window()
    test_scale_down_permitted_once_window_elapses()
    test_rate_limit_caps_scale_up_delta()
    test_rate_limit_caps_scale_down_delta()
    test_clamped_to_min_and_max()
    test_flap_simulation_produces_no_scale_down_events()
    print("all safety module tests passed")