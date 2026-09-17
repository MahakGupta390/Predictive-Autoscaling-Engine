Predictive Autoscaler

A predictive autoscaling engine that combines workload forecasting with reactive HPA-style scaling, capacity limits, SLA-aware scaling, and safety controls. The included Streamlit dashboard runs a complete simulation using synthetic workload metrics, so the dashboard can be explored without a Kubernetes cluster.

Installation

Clone or download the repository, then install the Python dependencies:

pip install -r requirements.txt

Run

Start the interactive dashboard with:

streamlit run dashboard.py

The dashboard opens in your browser and provides simulation controls, workload forecasts, scaling decisions, SLA checks, what-if scenarios, and an architecture view.

Project Structure

File

Purpose

dashboard.py

Streamlit application containing the interactive simulation dashboard and replay logic.

metrics_source.py

Defines the metric schema/source interface and generates synthetic workload, CPU, and memory metrics.

preprocessing.py

Cleans metrics, maintains history, and creates rolling/calendar features for prediction.

prediction_service.py

Provides the prediction layer with Ridge-based linear regression and XGBoost models, plus model comparison and persistence helpers.

hpa_calculator.py

Implements the reactive HPA-style replica calculation and scaling action logic.

capacity_model.py

Converts request demand and per-container capacity into a required replica count.

sla_evaluator.py

Converts latency/CPU SLA constraints into replica requirements and reports SLA violations.

fusion.py

Combines the reactive HPA recommendation with the predictive/SLA capacity recommendation.

safety.py

Applies replica limits, rate limiting, and asymmetric scale-up/scale-down stabilization.

preview.py

Provides stateless decision previews used by the dashboard's live-control and what-if views.

reconciler.py

Orchestrates the end-to-end autoscaling loop for a real controller: metrics → prediction/HPA → fusion → safety → resource update → tracking/retraining.

resource_manager.py

Handles Kubernetes deployment replica reads and scale operations.

accuracy_tracker.py

Tracks predictions against later observations and calculates prediction accuracy metrics such as MAPE.

retraining_trigger.py

Determines when model retraining should occur based on scheduled intervals or accuracy degradation.

simulation.py

End-to-end simulation harness for exercising the autoscaling pipeline without Kubernetes.

plot_metrics.py

Standalone plotting utility for generated workload metrics.

plot_predictions.py

Standalone utility for preprocessing data, training predictions, and visualizing prediction results.

.streamlit/config.toml

Streamlit theme and dashboard configuration.

Architecture

The main dashboard flow is:

Synthetic Metrics
       ↓
Preprocessing / Feature Engineering
       ↓
Prediction Model ─────────┐
                          │
Reactive HPA ─────────────┤
                          ↓
                  Fusion + SLA
                          ↓
                       Safety
                          ↓
                 Replica Decision

The dashboard separates the expensive workload-generation/training/prediction stage from the lightweight decision replay stage. This allows SLA, capacity, and initial-replica controls to update the scaling decisions without retraining the prediction model on every slider change.

Kubernetes

The Streamlit dashboard is designed for simulation and does not require a Kubernetes cluster.

resource_manager.py contains the Kubernetes integration used by the controller-side reconciliation path. It reads the deployment's current replica count and can apply scaling decisions through the Kubernetes API when the project is run in a Kubernetes environment.

Notes

Workload data used by the dashboard is synthetic.

The prediction layer supports xgboost and linear_regression modes.

The dashboard's simulation controls can change workload generation, model configuration, SLA/capacity settings, and starting replica count.