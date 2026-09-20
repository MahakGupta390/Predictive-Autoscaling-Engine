
"""
Predictive Autoscaler -- interactive dashboard. No Kubernetes needed.

Architecture: split into an expensive CACHED core (data generation,
preprocessing, training, per-tick prediction) and a cheap REPLAY (HPA
calc + fusion + safety + SLA check per tick, pure arithmetic, no ML).
SLA/capacity/initial-replica sliders only trigger replay -- that's what
makes "every slider instantly updates all tabs" literally true instead of
re-running the whole simulation on each tweak.
"""

from dataclasses import asdict

import pandas as pd
import streamlit as st
import altair as alt

from metrics_source import SyntheticMetricSource, SyntheticConfig, MetricSample
from preprocessing import Preprocessor, process_batch, PreprocessConfig
from prediction_service import PredictionService, FEATURE_COLUMNS, compare_models
from hpa_calculator import calculate_desired_replicas, HPAConfig
from capacity_model import CapacityConfig
from sla_evaluator import SLAConfig, check_sla_violation
from fusion import combine
from safety import SafetyState, SafetyConfig, apply_safety
from preview import preview_decision

st.set_page_config(page_title="Predictive Autoscaler", layout="wide")


@st.cache_data(show_spinner="Generating workload, training model, predicting each tick...")
def run_core_pipeline(duration_minutes, seed, burst_probability, model_name, horizon, warmup_ticks):
    gen_cfg = SyntheticConfig(duration_minutes=duration_minutes, seed=seed, burst_probability=burst_probability)
    raw = SyntheticMetricSource(gen_cfg).generate_dataset()

    pre = Preprocessor(PreprocessConfig())
    svc = PredictionService(model_name=model_name, horizon=horizon)
    is_trained = False
    predicted, model_used = [], []

    for _, row in raw.iterrows():
        sample = MetricSample(row["timestamp"], row["pod_name"], row["cpu_pct"], row["mem_pct"], row["request_rate"])
        fv = pre.process(sample)

        if not is_trained and len(pre.history) >= warmup_ticks:
            featured_so_far = process_batch(pre.history.as_dataframe(), PreprocessConfig())
            svc.train(featured_so_far)
            is_trained = True

        if is_trained:
            recent = pd.DataFrame([fv.__dict__])[FEATURE_COLUMNS]
            predicted.append(float(svc.predict_next(recent).predicted_load[0]))
            model_used.append(model_name)
        else:
            predicted.append(row["request_rate"])
            model_used.append(None)

    core_df = raw.copy()
    core_df["predicted_request_rate"] = predicted
    core_df["model_used"] = model_used

    featured_full = process_batch(raw, PreprocessConfig())
    offline = compare_models(featured_full, horizon=horizon, test_frac=0.2) if is_trained else {}

    return core_df, svc, offline


def replay_decisions(core_df, initial_replicas, hpa_cfg, sla_cfg, capacity_cfg, safety_cfg):
    state = SafetyState(initial_replicas=initial_replicas)
    rows = []
    for tick, row in core_df.reset_index(drop=True).iterrows():
        current_replicas = state.last_applied_replicas
        approx_per_pod_cpu = row["cpu_pct"] / max(1, current_replicas)
        hpa_decision = calculate_desired_replicas(current_replicas, approx_per_pod_cpu, hpa_cfg)
        fused = combine(hpa_decision, row["predicted_request_rate"], sla_cfg, capacity_cfg)
        final = apply_safety(fused.target_replicas, row["timestamp"], state, safety_cfg)
        violation = check_sla_violation(
            current_replicas, row["request_rate"], row["cpu_pct"], sla_cfg, capacity_cfg
        )
        rows.append({
            "tick": tick, "timestamp": row["timestamp"],
            "request_rate": row["request_rate"], "cpu_pct": row["cpu_pct"],
            "predicted_request_rate": row["predicted_request_rate"], "model_used": row["model_used"],
            "hpa_desired": hpa_decision.desired_replicas,
            "fused_target": fused.target_replicas, "fused_rationale": fused.rationale,
            "final_replicas": final.replicas, "final_action": final.action.value,
            "current_replicas": current_replicas,
            "sla_violated": violation.violated, "sla_reasons": "; ".join(violation.reasons),
        })
    return pd.DataFrame(rows)


def sidebar_controls():
    with st.sidebar:
        st.header("Data / Model")
        duration_minutes = st.slider("Duration (minutes)", 60, 600, 300, step=30)
        seed = st.number_input("Random seed", value=11, step=1)
        burst_probability = st.slider("Burst probability (per tick)", 0.0, 0.05, 0.015, step=0.005)
        model_name = st.radio("Prediction model", ["xgboost", "linear_regression"])
        warmup_ticks = st.slider("Warmup ticks before first training", 20, 120, 60, step=10)
        load_clicked = st.button("Train / Retrain Model", type="primary")

        st.header("SLA & Capacity  (instant)")
        target_latency_ms = st.slider("Target Latency (ms)", 50, 500, 150, step=10)
        cpu_threshold_pct = st.slider(
            "Max CPU Utilization (%)", 50, 100, 80,
            help="Diagnostic only -- flags SLA violations, does not itself change the replica count. "
                 "Target Latency above is the real scaling driver.",
        )
        requests_per_container = st.slider("Max Requests per Container", 25, 300, 100)
        initial_replicas = st.slider("Initial Containers", 1, 10, 3)

    return {
        "duration_minutes": duration_minutes, "seed": int(seed), "burst_probability": burst_probability,
        "model_name": model_name, "warmup_ticks": warmup_ticks, "load_clicked": load_clicked,
        "target_latency_ms": target_latency_ms, "cpu_threshold_pct": cpu_threshold_pct,
        "requests_per_container": requests_per_container, "initial_replicas": initial_replicas,
    }


def build_configs(ctrl):
    hpa_cfg = HPAConfig(min_replicas=1, max_replicas=15)
    safety_cfg = SafetyConfig(min_replicas=1, max_replicas=15)
    capacity_cfg = CapacityConfig(requests_per_container=ctrl["requests_per_container"], min_replicas=1)
    sla_cfg = SLAConfig(
        target_latency_ms=ctrl["target_latency_ms"],
        cpu_utilization_threshold_pct=ctrl["cpu_threshold_pct"],
    )
    return hpa_cfg, sla_cfg, capacity_cfg, safety_cfg


def main():
    st.title("Predictive Autoscaler")
    st.caption("Runs entirely against the synthetic source -- zero Kubernetes dependency.")

    ctrl = sidebar_controls()
    hpa_cfg, sla_cfg, capacity_cfg, safety_cfg = build_configs(ctrl)

    if ctrl["load_clicked"] or "core_df" not in st.session_state:
        if not ctrl["load_clicked"] and "core_df" not in st.session_state:
            st.info("Click **Train / Retrain Model** in the sidebar to load a simulation.")
            return
        core_df, svc, offline = run_core_pipeline(
            ctrl["duration_minutes"], ctrl["seed"], ctrl["burst_probability"],
            ctrl["model_name"], 5, ctrl["warmup_ticks"],
        )
        st.session_state["core_df"] = core_df
        st.session_state["svc"] = svc
        st.session_state["offline"] = offline

    core_df = st.session_state["core_df"]
    svc = st.session_state["svc"]
    offline = st.session_state["offline"]

    results_df = replay_decisions(core_df, ctrl["initial_replicas"], hpa_cfg, sla_cfg, capacity_cfg, safety_cfg)
    results_df["sla_cpu_threshold_line"] = ctrl["cpu_threshold_pct"]

    tab1, tab2, tab3, tab4= st.tabs(
        ["Full Simulation", "Live Control Panel", "What-If Scenarios", "Step-by-Step"]
    )

    # ---------------- TAB 1: Full Simulation ----------------
    with tab1:
        c1, c2, c3, c4 = st.columns(4,gap="medium")
        c1.metric("Peak actual load", f"{results_df.request_rate.max():.0f} req/s",border=True)
        c2.metric("Peak predicted load", f"{results_df.predicted_request_rate.max():.0f} req/s",border=True)
        c3.metric("Max containers", int(results_df.final_replicas.max()),border=True)
        c4.metric("Scale events", int((results_df.final_action != "maintain").sum()),border=True)

        st.subheader("Workload: actual vs predicted")
        st.line_chart(
            results_df.set_index("tick")[["request_rate", "predicted_request_rate"]],
            x_label="Simulation tick",
            y_label="Requests / second",
    )

        st.subheader("Allocated containers")
        st.line_chart(
            results_df.set_index("tick")[["final_replicas"]],
            x_label="Simulation tick",
            y_label="Containers",
)

        st.subheader("CPU utilization vs SLA threshold")
        st.line_chart(
            results_df.set_index("tick")[["cpu_pct", "sla_cpu_threshold_line"]],
            x_label="Simulation tick",
            y_label="CPU utilization (%)",
    )

        st.subheader("Scaling decision log")
        st.dataframe(
            results_df[["tick", "timestamp", "final_action", "final_replicas", "fused_rationale", "sla_reasons"]],
            width='stretch', height=300,
        )

        if offline:
                st.header("Offline model comparison")
                offline_df = pd.DataFrame({
                "Model": ["Linear Regression", "Naive Baseline", "XGBoost"],
                "MAPE": list(offline.values())
                 })
                chart = alt.Chart(offline_df).mark_bar().encode(
                x=alt.X("Model:N", axis=alt.Axis(labelAngle=0,labelFontSize=14,labelFontWeight="bold", labelLimit=200,
                  labelLineHeight=18, title="Model", titleFontSize=14)),
                y=alt.Y("MAPE:Q", title="MAPE (%)"),
                tooltip=["Model", alt.Tooltip("MAPE:Q", format=".2f")]
                ).properties(height=360)
                st.altair_chart(chart, width="stretch")
    # ---------------- TAB 2: Live Control Panel ----------------
    with tab2:
        st.write("Splice an override onto the loaded simulation's history at a chosen point in time, "
                 "and see the instant prediction + scaling decision. Does not affect the real run above.")
        max_tick = len(core_df) - 1
        t_col, r_col, c_col, n_col = st.columns(4)
        timestamp_idx = t_col.slider("Timestamp (tick)", 1, max_tick, min(200, max_tick))
        override_rr = r_col.slider("Current Requests/sec", 0, 800, 150)
        override_cpu = c_col.slider("Current CPU Usage (%)", 0, 100, 50)
        override_replicas = n_col.slider("Current Running Containers", 1, 15, 3)

        history_slice = core_df.iloc[: timestamp_idx + 1]
        preview = preview_decision(
            history_slice, override_rr, override_cpu, override_replicas,
            svc, hpa_cfg, sla_cfg, capacity_cfg,
        )

        if preview.out_of_range_warning:
            st.warning(preview.out_of_range_warning)

        with st.container(border=True):
            p1, p2, p3, p4 = st.columns(4,gap="small")

            p1.metric(
                "Predicted next-step load",
                f"{preview.predicted_request_rate:.0f}",
                f"{preview.predicted_delta:+.0f}",
            )

            p2.metric(
                "Required containers",
                preview.fused_decision.target_replicas,
                f"{preview.replica_delta:+d}",
            )

            p3.metric(
                "Current capacity",
                f"{override_replicas * ctrl['requests_per_container']:.0f} req/s",
            )

            if preview.replica_delta > 0:
                action_label = "SCALE UP"
            elif preview.replica_delta < 0:
                action_label = "SCALE DOWN"
            else:
                action_label = "NO CHANGE"

            p4.markdown(
                f"""
                <div style="margin-top: 0.25rem;">
                    <div style="font-size: 1rem; color: #d6d6d6;">
                        Scaling action
                    </div>
                    <div style="font-size: 1.6rem; font-weight: 400; margin-top: 0.35rem;">
                        {action_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
)

            st.write(
                "**Decision breakdown:**",
                preview.fused_decision.rationale,
            )

            if preview.sla_violation.violated:
                st.error(
                    "SLA violated: " + "; ".join(preview.sla_violation.reasons)
                )
            else:
                st.success("SLA satisfied at this point")

        chart_df = pd.DataFrame({
            "metric": ["Current Capacity", "Predicted Demand", "New Capacity"],
            "value": [
                override_replicas * ctrl["requests_per_container"],
                preview.predicted_request_rate,
                preview.fused_decision.target_replicas * ctrl["requests_per_container"],
            ],
        })

        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X(
                 "metric:N",
                  axis=alt.Axis(
                  labelAngle=0,
                  labelFontSize=14,
                  labelFontWeight="bold",
                  labelLimit=200,
                  labelLineHeight=18,
                  title="Metric",
                  titleFontSize=14,
                ),
            ),
            y=alt.Y(
                "value:Q",
                title="Requests / second",
            ),
            tooltip=[
                "metric",
                alt.Tooltip("value:Q", format=".0f"),
            ],
        ).properties(height=360)

        st.altair_chart(chart, width="stretch")

    # ---------------- TAB 3: What-If Scenarios ----------------
    with tab3:
        st.write("Three independent scenarios, compared side by side. Each uses the same loaded "
                 "simulation's most recent history as context for the rolling features.")
        defaults = {"A (Low Traffic)": (80, 25, 3), "B (Medium Traffic)": (250, 60, 5), "C (Peak Traffic)": (450, 92, 8)}
        cols = st.columns(3)
        scenario_results = []
        for col, (name, (d_rr, d_cpu, d_containers)) in zip(cols, defaults.items()):
            with col:
                st.markdown(f"**{name}**")
                s_rr = st.number_input("Requests/sec", 0, 800, d_rr, key=f"{name}_rr")
                s_cpu = st.number_input("CPU Usage (%)", 0, 100, d_cpu, key=f"{name}_cpu")
                s_containers = st.number_input("Current Containers", 1, 15, d_containers, key=f"{name}_c")

                pr = preview_decision(core_df, s_rr, s_cpu, s_containers, svc, hpa_cfg, sla_cfg, capacity_cfg)
                if pr.out_of_range_warning:
                    st.caption(f"Warning: {pr.out_of_range_warning}")
                st.metric("Predicted load", f"{pr.predicted_request_rate:.0f}")
                st.metric("Required containers", pr.fused_decision.target_replicas)
                scenario_results.append({"scenario": name, "predicted_load": pr.predicted_request_rate,
                                          "required_containers": pr.fused_decision.target_replicas,
                                          "sla_violated": pr.sla_violation.violated})

        st.subheader("Comparison")

        comp_df = pd.DataFrame(scenario_results).set_index("scenario")

        st.dataframe(comp_df, width='stretch')

        chart_df = comp_df.reset_index().melt(
            id_vars="scenario",
            value_vars=["predicted_load", "required_containers"],
            var_name="Metric",
            value_name="Value",
        )

        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X(
                "scenario:N",
                axis=alt.Axis(
                    labelAngle=0,
                    labelFontSize=14,
                    labelFontWeight="bold",
                    labelLimit=200,
                    labelLineHeight=18,
                    title="Scenario",
                    titleFontSize=14,
                ),
            ),
            xOffset=alt.XOffset("Metric:N"),
            y=alt.Y(
                "Value:Q",
                title="Value",
            ),
            tooltip=[
                "scenario",
                "Metric",
                alt.Tooltip("Value:Q", format=".0f"),
            ],
        ).properties(height=360)

        st.altair_chart(chart, width="stretch")

    # ---------------- TAB 4: Step-by-Step ----------------
    with tab4:
        step_idx = st.slider("Time step", 0, len(results_df) - 1, 7)
        row = results_df.iloc[step_idx]

        s1, s2, s3 = st.columns(3)
        s1.metric("Requests/sec", f"{row.request_rate:.0f}")
        s2.metric("CPU %", f"{row.cpu_pct:.1f}")
        s3.metric("Containers", int(row.final_replicas))

        # st.write(f"**Predicted:** {row.predicted_request_rate:.0f} req/s  (model: {row.model_used or 'naive fallback'})")
        if pd.isna(row.model_used):
            model_display = "warmup — actual load used"
        else:
             model_display = row.model_used

        st.write( f"**Predicted:** {row.predicted_request_rate:.0f} req/s "f"(model: {model_display})"
)
        st.write(f"**Action:** {row.final_action}  —  {row.fused_rationale}")
        if row.sla_violated:
            st.error("SLA violation: " + row.sla_reasons)
        else:
            st.success("SLA satisfied at this step")

        st.subheader("History up to this step")
        history_so_far = results_df.iloc[: step_idx + 1].set_index("tick")
        st.line_chart(history_so_far[["request_rate", "predicted_request_rate"]])
        st.line_chart(history_so_far[["final_replicas"]])
if __name__ == "__main__":
    main()