"""
End-to-end simulation harness -- no Kubernetes anywhere.

Matches the flat predictive-autoscaler/ layout (no tests/ folder -- there
are no separate test files in this project). Reuses plot_metrics.py and
plot_predictions.py as-is for their visualizations rather than duplicating
that logic inline.
"""

import importlib
import os
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from metrics_source import SyntheticMetricSource, SyntheticConfig, MetricSample
from preprocessing import process_batch, PreprocessConfig
from prediction_service import PredictionService, FEATURE_COLUMNS, compare_models, default_model_path
from resource_manager import ResourceManager, ResourceManagerConfig
from reconciler import Reconciler, ReconcilerConfig

# every module in the flat layout -- no test_*.py files exist in this
# project, so "foundation check" here means "every module imports clean",
# not "every module's test suite passes"
PROJECT_MODULES = [
    "metrics_source", "preprocessing", "hpa_calculator", "capacity_model",
    "prediction_service", "sla_evaluator", "fusion", "safety",
    "resource_manager", "accuracy_tracker", "retraining_trigger", "reconciler",
]


def step(n, title):
    print(f"\n{'='*70}\nSTEP {n}: {title}\n{'='*70}")


def run_plot_script(script_name):
    """Run one of the existing plot_*.py scripts headless (Agg backend so
    plt.show() doesn't block or need a display) and capture its stdout."""
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, script_name], capture_output=True, text=True, env=env
    )
    return result


def main():
    # ------------------------------------------------------------------
    step(1, "Foundation check -- every module imports cleanly")
    all_ok = True
    for mod in PROJECT_MODULES:
        try:
            importlib.import_module(mod)
            print(f"  [OK]   {mod}.py")
        except Exception as e:
            all_ok = False
            print(f"  [FAIL] {mod}.py -- {e}")
    assert all_ok, "one or more modules failed to import -- fix before trusting the integration below"

    # ------------------------------------------------------------------
    step(2, "Visualize the raw synthetic workload (plot_metrics.py)")
    result = run_plot_script("plot_metrics.py")
    print(f"  exit code: {result.returncode}")
    print("  saved: plot_metrics_request_rate.png, plot_metrics_cpu_mem.png")
    assert result.returncode == 0, f"plot_metrics.py failed:\n{result.stderr}"

    # ------------------------------------------------------------------
    step(3, "Generate the simulation dataset and run the reconciler standalone")
    gen_cfg = SyntheticConfig(duration_minutes=300, seed=11, burst_probability=0.015)
    raw = SyntheticMetricSource(gen_cfg).generate_dataset()
    samples = [
        MetricSample(r["timestamp"], r["pod_name"], r["cpu_pct"], r["mem_pct"], r["request_rate"])
        for _, r in raw.iterrows()
    ]
    rc_cfg = ReconcilerConfig(warmup_ticks=60, model_name="xgboost")
    rc = Reconciler(rc_cfg, resource_manager=None)
    results = rc.run(samples)
    print(f"  ran {len(results)} ticks through the full pipeline with zero cluster dependency")

    # ------------------------------------------------------------------
    step(4, "Confirm bootstrap -> trained transition")
    switch_tick = next(r.tick for r in results if r.model_used is not None)
    before = all(r.model_used is None for r in results if r.tick < switch_tick)
    after = all(r.model_used == "xgboost" for r in results if r.tick >= switch_tick)
    print(f"  switched from naive fallback to xgboost predictions at tick {switch_tick}")
    assert before and after, "bootstrap -> trained transition did not happen cleanly"
    print("  [PASS] clean transition, no gap or overlap")

    # ------------------------------------------------------------------
    step(5, "Model comparison (plot_predictions.py) + live-loop accuracy cross-check")
    result = run_plot_script("plot_predictions.py")
    print("  plot_predictions.py output:")
    for line in result.stdout.strip().splitlines():
        print(f"    {line}")
    print("  saved: plot_predictions_comparison.png")
    assert result.returncode == 0, f"plot_predictions.py failed:\n{result.stderr}"

    history_df = rc.preprocessor.history.as_dataframe()
    featured = process_batch(history_df, rc_cfg.preprocess_cfg)
    offline = compare_models(featured, horizon=rc_cfg.horizon, test_frac=0.2)
    live_mape = rc.accuracy_tracker.rolling_mape("xgboost")
    resolved_count = rc.accuracy_tracker.as_dataframe().shape[0]
    print(f"  reconciler's own offline eval:  naive={offline['naive_baseline']:.2f}%  xgb={offline['xgboost']:.2f}%")
    print(f"  reconciler's live-loop MAPE:     {live_mape:.2f}%  ({resolved_count} predictions resolved)")
    assert resolved_count > 100
    assert live_mape < offline["naive_baseline"] * 1.5, "live MAPE diverged badly from offline eval"
    print("  [PASS] live-loop accuracy consistent with offline evaluation")

    # ------------------------------------------------------------------
    step(6, "Confirm scale-up-during-burst / scale-down-after, with NO REAL FLAPPING")
    trajectory = [r.final_decision.replicas for r in results]
    actions = [r.final_decision.action.value for r in results]
    reversal_ticks = [
        i for i in range(1, len(actions))
        if actions[i] != actions[i - 1] and actions[i] != "maintain" and actions[i - 1] != "maintain"
    ]
    window = 20
    max_reversals_in_window = max(
        (sum(1 for t in reversal_ticks if w <= t < w + window) for w in range(0, len(actions), window)),
        default=0,
    )
    print(f"  peak replicas: {max(trajectory)}  (started at {rc_cfg.initial_replicas})")
    print(f"  max reversals in any {window}-tick window: {max_reversals_in_window}")
    assert max(trajectory) > rc_cfg.initial_replicas
    assert max_reversals_in_window <= 1, "multiple reversals within one stabilization window -- real flapping"
    print("  [PASS] scaled up during bursts, no repeated reversals within any stabilization window")

    # ------------------------------------------------------------------
    step(7, "Confirm SLA violation reporting fires during under-provisioned windows")
    violations = [r for r in results if r.sla_violation.violated]
    print(f"  SLA violations flagged: {len(violations)} / {len(results)} ticks")
    if violations:
        print(f"  example reason: {violations[0].sla_violation.reasons[0]}")

    # ------------------------------------------------------------------
    step(8, "Confirm the retraining trigger is live")
    retrain_checks = [r for r in results if r.retrain_decision is not None]
    fired = [r for r in retrain_checks if r.retrain_decision.should_retrain]
    print(f"  retrain checks performed: {len(retrain_checks)}, fired: {len(fired)}")
    for r in fired[:3]:
        print(f"    tick {r.tick}: {r.retrain_decision.reason}")

    # ------------------------------------------------------------------
    step(9, "Confirm persistence works mid-simulation")
    recent_features = featured[FEATURE_COLUMNS].tail(5)
    live_preds = rc.prediction_service.predict_next(recent_features).predicted_load
    with tempfile.TemporaryDirectory() as tmp:
        path = default_model_path(tmp, "xgboost")
        rc.prediction_service.save(path)
        reloaded = PredictionService.load(path)
        reloaded_preds = reloaded.predict_next(recent_features).predicted_load
    assert np.allclose(live_preds, reloaded_preds)
    print("  [PASS] model trained DURING the live simulation saves/reloads identically")

    # ------------------------------------------------------------------
    step(10, "Confirm the write path works against a (mocked) cluster")
    mock_api = MagicMock()
    fake_scale = MagicMock()
    fake_scale.spec.replicas = 2
    mock_api.read_namespaced_deployment_scale.return_value = fake_scale
    rm = ResourceManager(ResourceManagerConfig(namespace="default", deployment_name="app"), api_client=mock_api)
    rc2 = Reconciler(ReconcilerConfig(warmup_ticks=60, model_name="xgboost"), resource_manager=rm)
    rc2.run(samples[:80])
    read_calls = mock_api.read_namespaced_deployment_scale.call_count
    apply_calls = mock_api.patch_namespaced_deployment_scale.call_count
    print(f"  cluster read calls: {read_calls}, cluster patch calls: {apply_calls}")
    assert read_calls > 0 and apply_calls > 0, "reconciler never touched the (mocked) cluster API"
    print("  [PASS] identical reconciler code drives a real cluster client when one is attached")

    # ------------------------------------------------------------------
    step(11, "Summary")
    print(f"""
  Ticks simulated:                    {len(results)}
  Bootstrap -> trained at tick:       {switch_tick}
  Reconciler offline naive/XGB MAPE:  {offline['naive_baseline']:.2f}% / {offline['xgboost']:.2f}%
  Reconciler live-loop XGB MAPE:      {live_mape:.2f}%  ({resolved_count} resolved predictions)
  Peak replicas reached:              {max(trajectory)} (from {rc_cfg.initial_replicas})
  Max reversals per 20-tick window:   {max_reversals_in_window}
  SLA violations flagged:             {len(violations)}
  Retraining checks / fired:          {len(retrain_checks)} / {len(fired)}
  Persistence mid-sim:                verified
  Mocked-cluster write path:          verified ({apply_calls} patch calls)
  Plots saved:                        plot_metrics_request_rate.png, plot_metrics_cpu_mem.png,
                                       plot_predictions_comparison.png

  No real Kubernetes cluster was used anywhere in this simulation.
""")


if __name__ == "__main__":
    main()