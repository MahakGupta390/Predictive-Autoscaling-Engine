from accuracy_tracker import AccuracyTracker


def test_prediction_resolves_at_correct_target_tick():
    tracker = AccuracyTracker()
    tracker.log_prediction(current_tick=10, horizon=5, model_name="linear_regression", predicted_value=100.0)

    # not yet due at tick 14
    resolved_early = tracker.record_actual(current_tick=14, actual_value=999.0)
    assert resolved_early == []
    assert tracker.pending_count() == 1

    # due exactly at tick 15 (10 + 5)
    resolved = tracker.record_actual(current_tick=15, actual_value=120.0)
    assert len(resolved) == 1
    assert resolved[0].model_name == "linear_regression"
    assert resolved[0].predicted == 100.0
    assert resolved[0].actual == 120.0
    assert tracker.pending_count() == 0


def test_error_matches_hand_calculation():
    tracker = AccuracyTracker()
    tracker.log_prediction(current_tick=0, horizon=1, model_name="xgboost", predicted_value=80.0)
    resolved = tracker.record_actual(current_tick=1, actual_value=100.0)

    # |80-100|/100 * 100 = 20%
    assert abs(resolved[0].error_pct - 20.0) < 1e-9


def test_multiple_models_same_target_tick_resolve_independently():
    tracker = AccuracyTracker()
    tracker.log_prediction(current_tick=5, horizon=5, model_name="linear_regression", predicted_value=90.0)
    tracker.log_prediction(current_tick=5, horizon=5, model_name="xgboost", predicted_value=105.0)

    resolved = tracker.record_actual(current_tick=10, actual_value=100.0)
    resolved_by_model = {r.model_name: r for r in resolved}

    assert len(resolved) == 2
    assert abs(resolved_by_model["linear_regression"].error_pct - 10.0) < 1e-9
    assert abs(resolved_by_model["xgboost"].error_pct - 5.0) < 1e-9


def test_rolling_mape_aggregates_correctly_per_model():
    tracker = AccuracyTracker()
    # two linear_regression predictions with known errors: 10% and 20%
    tracker.log_prediction(0, 1, "linear_regression", 90.0)
    tracker.record_actual(1, 100.0)
    tracker.log_prediction(1, 1, "linear_regression", 80.0)
    tracker.record_actual(2, 100.0)

    assert abs(tracker.rolling_mape("linear_regression") - 15.0) < 1e-9
    assert tracker.rolling_mape("xgboost") is None  # no data logged for this model


def test_unresolved_prediction_stays_pending_indefinitely_until_its_tick():
    tracker = AccuracyTracker()
    tracker.log_prediction(current_tick=100, horizon=5, model_name="linear_regression", predicted_value=50.0)
    for tick in range(101, 105):
        tracker.record_actual(current_tick=tick, actual_value=999.0)
    assert tracker.pending_count() == 1  # still waiting for tick 105


if __name__ == "__main__":
    test_prediction_resolves_at_correct_target_tick()
    test_error_matches_hand_calculation()
    test_multiple_models_same_target_tick_resolve_independently()
    test_rolling_mape_aggregates_correctly_per_model()
    test_unresolved_prediction_stays_pending_indefinitely_until_its_tick()
    print("all accuracy tracker unit tests passed")