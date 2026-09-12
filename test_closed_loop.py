"""
Closed-loop integration test.

Not a unit test of one module -- replays synthetic ticks through the real
preprocessor (feeding each post-scaling sample into the history buffer,
exactly as the reconciler will), a trained prediction service, and the
accuracy tracker together, confirming the tracker's rolling MAPE lands in
the same ballpark as compare_models()'s offline evaluation.
"""

from metrics_source import SyntheticMetricSource, SyntheticConfig
from preprocessing import Preprocessor, PreprocessConfig, process_batch
from prediction_service import PredictionService, FEATURE_COLUMNS, compare_models
from accuracy_tracker import AccuracyTracker


def test_closed_loop_tracks_accuracy_consistently_with_offline_eval():
    horizon = 5
    train_cfg = SyntheticConfig(duration_minutes=480, seed=7, burst_probability=0.015)
    raw_train = SyntheticMetricSource(train_cfg).generate_dataset()
    train_features = process_batch(raw_train, PreprocessConfig())

    # train once on a batch, same as stage 5 would before going live
    svc = PredictionService(model_name="xgboost", horizon=horizon)
    svc.train(train_features)

    # sanity-check offline number to compare the live loop against
    offline = compare_models(train_features, horizon=horizon, test_frac=0.2)

    # now replay a FRESH synthetic run tick-by-tick, exactly like the
    # reconciler will: preprocess -> predict -> log -> next tick's actual
    # resolves the prediction from `horizon` ticks ago
    live_cfg = SyntheticConfig(duration_minutes=180, seed=99, burst_probability=0.015)
    raw_live = SyntheticMetricSource(live_cfg).generate_dataset()

    pre = Preprocessor(PreprocessConfig())
    tracker = AccuracyTracker()

    for tick, (_, row) in enumerate(raw_live.iterrows()):
        from metrics_source import MetricSample
        sample = MetricSample(
            timestamp=row["timestamp"], pod_name=row["pod_name"],
            cpu_pct=row["cpu_pct"], mem_pct=row["mem_pct"], request_rate=row["request_rate"],
        )

        # this IS "feeding actual post-scaling metrics into the history
        # buffer" -- the append happens inside process()
        feature_vector = pre.process(sample)

        # resolve any prediction whose target tick is NOW
        tracker.record_actual(current_tick=tick, actual_value=sample.request_rate, timestamp=sample.timestamp)

        # only predict once enough history exists for stable features
        if tick >= 30:
            import pandas as pd
            recent = pd.DataFrame([feature_vector.__dict__])[FEATURE_COLUMNS]
            result = svc.predict_next(recent)
            tracker.log_prediction(
                current_tick=tick, horizon=horizon,
                model_name=result.model_used, predicted_value=float(result.predicted_load[0]),
            )

    live_mape = tracker.rolling_mape("xgboost")

    assert live_mape is not None, "tracker should have resolved a meaningful number of predictions"
    assert tracker.as_dataframe().shape[0] > 50, "closed loop should have logged many resolved predictions"
    # not expecting an exact match to the offline split (different data,
    # different noise draw) -- just that it's in a sane, comparable range
    assert live_mape < offline["naive_baseline"] * 1.5, (
        f"live-loop MAPE ({live_mape:.2f}%) is wildly worse than the offline "
        f"naive baseline ({offline['naive_baseline']:.2f}%) -- something diverged "
        f"between training-time and live-loop feature computation"
    )


if __name__ == "__main__":
    test_closed_loop_tracks_accuracy_consistently_with_offline_eval()
    print("closed-loop integration test passed")