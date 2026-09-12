from accuracy_tracker import AccuracyTracker
from retraining_trigger import RetrainingTrigger, RetrainPolicy


def _log_n_predictions(tracker, model_name, n, error_pct_each, start_tick=0):
    """Helper: log+resolve n predictions with a controlled error, one per tick."""
    for i in range(n):
        tick = start_tick + i
        # predicted vs actual chosen to produce error_pct_each exactly
        actual = 100.0
        predicted = actual * (1 + error_pct_each / 100)
        tracker.log_prediction(current_tick=tick, horizon=1, model_name=model_name, predicted_value=predicted)
        tracker.record_actual(current_tick=tick + 1, actual_value=actual)


def test_no_trigger_without_baseline():
    tracker = AccuracyTracker()
    trigger = RetrainingTrigger(tracker, RetrainPolicy())
    decision = trigger.should_retrain("xgboost", current_tick=100)
    assert not decision.should_retrain
    assert "no baseline" in decision.reason


def test_scheduled_interval_fires_regardless_of_accuracy():
    tracker = AccuracyTracker()
    policy = RetrainPolicy(scheduled_interval_ticks=500, min_predictions_before_check=1000)
    trigger = RetrainingTrigger(tracker, policy)
    trigger.set_baseline("xgboost", mape_value=15.0, current_tick=0)

    # good accuracy, but scheduled interval has elapsed
    _log_n_predictions(tracker, "xgboost", n=10, error_pct_each=2.0)
    decision = trigger.should_retrain("xgboost", current_tick=600)

    assert decision.should_retrain
    assert "scheduled interval" in decision.reason


def test_insufficient_fresh_samples_suppresses_degradation_check():
    tracker = AccuracyTracker()
    policy = RetrainPolicy(scheduled_interval_ticks=100000, min_predictions_before_check=50)
    trigger = RetrainingTrigger(tracker, policy)
    trigger.set_baseline("xgboost", mape_value=15.0, current_tick=0)

    # only 5 fresh predictions logged, all with terrible error -- should
    # NOT trigger, since the sample gate isn't satisfied yet
    _log_n_predictions(tracker, "xgboost", n=5, error_pct_each=90.0)
    decision = trigger.should_retrain("xgboost", current_tick=10)

    assert not decision.should_retrain
    assert "too few" in decision.reason


def test_degradation_beyond_threshold_triggers():
    tracker = AccuracyTracker()
    policy = RetrainPolicy(
        scheduled_interval_ticks=100000,
        min_predictions_before_check=20,
        degradation_threshold_pct=25.0,
        mape_window=50,
    )
    trigger = RetrainingTrigger(tracker, policy)
    trigger.set_baseline("xgboost", mape_value=15.0, current_tick=0)

    # baseline is 15% -- allowed up to 15*1.25 = 18.75%; log 30 predictions at 40% error
    _log_n_predictions(tracker, "xgboost", n=30, error_pct_each=40.0)
    decision = trigger.should_retrain("xgboost", current_tick=40)

    assert decision.should_retrain
    assert "degraded" in decision.reason


def test_within_tolerance_does_not_trigger():
    tracker = AccuracyTracker()
    policy = RetrainPolicy(scheduled_interval_ticks=100000, min_predictions_before_check=20)
    trigger = RetrainingTrigger(tracker, policy)
    trigger.set_baseline("xgboost", mape_value=15.0, current_tick=0)

    # 16% is within the default 25% tolerance band of a 15% baseline
    _log_n_predictions(tracker, "xgboost", n=30, error_pct_each=16.0)
    decision = trigger.should_retrain("xgboost", current_tick=40)

    assert not decision.should_retrain
    assert "within tolerance" in decision.reason


def test_set_baseline_resets_both_counters():
    tracker = AccuracyTracker()
    policy = RetrainPolicy(scheduled_interval_ticks=500, min_predictions_before_check=20)
    trigger = RetrainingTrigger(tracker, policy)
    trigger.set_baseline("xgboost", mape_value=15.0, current_tick=0)

    _log_n_predictions(tracker, "xgboost", n=30, error_pct_each=40.0)
    assert trigger.should_retrain("xgboost", current_tick=40).should_retrain

    # simulate a retrain happening now, at tick 40, with a fresh baseline
    trigger.set_baseline("xgboost", mape_value=12.0, current_tick=40)

    # scheduled clock and degradation window both reset -- no immediate re-trigger
    decision = trigger.should_retrain("xgboost", current_tick=41)
    assert not decision.should_retrain


if __name__ == "__main__":
    test_no_trigger_without_baseline()
    test_scheduled_interval_fires_regardless_of_accuracy()
    test_insufficient_fresh_samples_suppresses_degradation_check()
    test_degradation_beyond_threshold_triggers()
    test_within_tolerance_does_not_trigger()
    test_set_baseline_resets_both_counters()
    print("all retraining trigger tests passed")