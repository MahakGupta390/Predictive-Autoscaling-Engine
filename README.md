# Predictive Autoscaling Engine

An autoscaler for Kubernetes-style workloads that fuses a **predictive** signal (Ridge regression + XGBoost, trained on rolling-window traffic features) with a **reactive** HPA baseline, so it can scale ahead of a traffic ramp instead of only reacting after latency has already degraded.

**Live demo:** https://predictive-autoscaling-engine-8xjxrsehmmuwgbdmxg92q8.streamlit.app/


## Why this exists

Standard reactive autoscaling (Kubernetes HPA) can only respond *after* load has already risen — during a fast ramp, that lag means either real latency/error-rate degradation while replicas catch up, or permanently over-provisioned capacity to buffer against that lag. This project fuses a trained forecast into the scaling decision so replicas can be added *ahead of* a ramp completing, while keeping the reactive path as a safety net so a bad prediction never scales the system below what current load already justifies.

Full narrative, the tradeoffs made, and the bugs found along the way: see [`case_study.md`](./case_study.md).

## Running it

```bash
pip install -r requirements.txt
python -m streamlit run dashboard.py
```

Click **Train / Retrain Model** in the sidebar to load a simulation, then explore the five tabs. No Kubernetes cluster required — everything runs against a synthetic workload generator.

To run the non-interactive end-to-end proof instead:
```bash
python simulate_end_to_end.py
```

## Module map

| File | Provides |
|---|---|
| `metrics_source.py` | `MetricSource`, `SyntheticMetricSource`, `ClockBasedDemoSource` |
| `preprocessing.py` | `Preprocessor`, `process_batch` — rolling/EWMA/lag features, calendar features |
| `hpa_calculator.py` | `calculate_desired_replicas` — reactive baseline |
| `capacity_model.py` | `required_replicas_for_capacity` |
| `prediction_service.py` | `PredictionService` (Ridge + XGBoost), `compare_models` |
| `sla_evaluator.py` | `required_replicas_for_sla`, `check_sla_violation` |
| `fusion.py` | `combine` — max-rule blend of reactive + predicted |
| `safety.py` | `apply_safety` — asymmetric stabilization, rate limiting, clamps |
| `resource_manager.py` | `ResourceManager` — Kubernetes API read/patch |
| `accuracy_tracker.py` | `AccuracyTracker` — predicted vs actual, rolling MAPE |
| `retraining_trigger.py` | `RetrainingTrigger` — scheduled + degradation-based |
| `reconciler.py` | `Reconciler` — ties every module into one tick loop |
| `preview.py` | `preview_decision` — stateless what-if, backs the dashboard |
| `dashboard.py` | Streamlit UI |

## Status

- Core pipeline: built and tested end to end against a synthetic workload
- Dashboard: deployed, interactive
- Real-cluster deployment (minikube, live Prometheus source): not done — see `case_study.md` for what that would take
