# Case Study: Predictive Autoscaling Engine

## The problem

Kubernetes' standard autoscaler (HPA) is purely reactive: it looks at current CPU/memory utilization and adjusts replica count from there. That has a structural weakness — **it can only respond to load that has already arrived.** During a fast traffic ramp, there's an unavoidable window between "load rose" and "replicas caught up," and during that window the system is either serving degraded latency or dropping requests. The usual workaround is over-provisioning — running enough spare capacity that the ramp never actually stresses the system — which is a permanent cost paid every quiet hour of every day to cover a problem that only exists during bursts.

A **predictive** signal — a model trained to forecast load a few ticks ahead — can start adding replicas *before* a ramp completes, closing that lag without paying for standing capacity around the clock. That's the premise this project tests: does a trained forecast measurably beat naive "assume tomorrow looks like today" reactive scaling, and can it be fused with a reactive baseline safely enough that a bad prediction never makes things worse than reactive-only would have?

## Architecture, in one line

`metrics → preprocessing (rolling/EWMA/lag features) → dual-model prediction (Ridge + XGBoost) → [reactive HPA calc] + [SLA-derated capacity model] → fusion (max-rule) → safety (asymmetric stabilization) → resource manager`, closed by an accuracy tracker and retraining trigger. Thirteen independently-tested modules, tied together by one reconciler tick loop, fronted by a Streamlit dashboard. Full module-by-module detail in the README.

## The tradeoffs that actually happened, not a list written in hindsight

Every item below is a real decision point that surfaced *during* implementation, usually because something broke first.

**Ridge over plain OLS for the linear baseline.** The feature set is deliberately multicollinear (`cpu_pct`/`mem_pct` are near-linear functions of `request_rate`, and the lag/rolling features track it too) — that's realistic, not a bug, but it makes unregularized OLS numerically unstable. Ridge was the direct fix, and it's still "linear regression" for baseline-comparison purposes.

**Bursts had to be ramped, not instantaneous.** The first version of the synthetic burst generator jumped straight to peak magnitude — a memoryless step function. Naive persistence is *provably* unbeatable against a memoryless jump (there's no precursor to learn from), which meant the first honest test of "does ML beat naive" failed for a mathematically inevitable reason, not a modeling flaw. Fixed by giving bursts a ramp-up/hold/ramp-down profile, mirroring how real traffic surges actually build.

**Training on `log1p(target)`, not the raw value.** Even after the burst fix, the model still lost to naive. Diagnosis: training minimizes MSE, evaluation reports MAPE (a relative-error metric) — on raw values those disagree, since MSE is dominated by the large absolute errors during bursts at the expense of the many small-value quiet-period rows MAPE weights heavily. Training in log-space aligns the two objectives. After this fix: naive 22.19% MAPE, Ridge 21.19%, XGBoost 18.72% — the first result that actually validated the project's premise.

**Chronological split, never random.** A random train/test split on time-series data lets the model train on rows that come *after* some test rows — the model could effectively see the tail of the event it's being asked to "predict." Every evaluation in this project uses a strict time-ordered split for this reason.

**A real bug found integrating the reconciler: the reactive path had no notion of replica count.** The synthetic generator models `cpu_pct` as a function of *total* request load, with no per-pod division — so once the reactive HPA path scaled up, `cpu_pct` never showed any relief, and the system got stuck at `max_replicas` permanently. Fixed by approximating per-pod utilization (`cpu_pct / current_replicas`) before it reaches the HPA calculator — a one-line fix, but the kind of bug that only surfaces once components are integrated, not visible in any single module's unit tests.

**Asymmetric stabilization, matching real Kubernetes HPA rather than inventing a custom scheme.** Scale-up applies immediately; scale-down only applies the *highest* recommendation seen across a trailing window, so a load drop has to stay dropped before the system shrinks. Verified in the final end-to-end run: with production settings, zero repeated oscillations across a full multi-burst simulation — clean staircases up, clean staircases down.

**CPU kept diagnostic-only, latency made the real scaling driver.** It would have been easy to let a CPU threshold directly drive replica count (simpler, matches some reference implementations) — but the actual capacity math in this project is latency-derated (`effective_capacity = base_capacity × min(1, (target_latency/reference_latency)^exponent)`), and folding CPU into that same decision would have meant two drivers silently fighting each other. CPU stays a separate, honest violation *signal*, not a hidden second lever.

**XGBoost's extrapolation failure, demonstrated rather than assumed.** Pushing a what-if input (750 req/s) far outside a loaded simulation's observed range returned a prediction that didn't move meaningfully with the input at all — a tree model can't extrapolate past the leaf values it saw in training, it just returns the nearest one. This wasn't a defect to hide; the dashboard surfaces a visible warning whenever an override falls outside the training range, because a demo tool that lets a broken number look authoritative is worse than one that flags its own blind spot.

## Quantified results

- Naive baseline 22.19% MAPE → Ridge 21.19% → XGBoost 18.72%, on an identical held-out chronological split
- A full multi-burst simulation run: zero flapping (zero repeated direction reversals within any stabilization window) under production safety settings
- SLA violations, when they occur, cluster precisely at the *start* of each burst (the unavoidable replicas-catching-up lag) and disappear once replicas reach target — not scattered randomly, which is the expected, correct pattern for a system that's actually working
- Live-loop prediction accuracy (measured tick-by-tick during a running simulation) stays consistent with the offline, controlled evaluation — no train/serve skew between the batch-trained model and the streaming inference path

## Honest limitations

- **Not deployed against a real cluster.** The resource manager talks to the real Kubernetes API and is tested against a mocked client, but there's no minikube integration test proving it against a live cluster — a deliberate scope cut, not an oversight.
- **The synthetic generator's diurnal cycle isn't calendar-anchored.** Calendar features (hour-of-day, weekend) are computed correctly but excluded from the model, because this generator's cycle repeats every simulated hour, unrelated to the real clock — they're pure noise against this dataset, though they would matter against real production traffic.
- **CPU-as-scaling-driver was deliberately not built**, in favor of keeping the tested latency/capacity math as the single source of truth (see tradeoffs above).
- **Free-tier hosting sleeps after inactivity** — the public demo link may need a manual wake click if it hasn't been visited recently.

## What "done" looks like from here

The core engineering claim — a fused predictive/reactive autoscaler measurably beats naive persistence and doesn't flap — is proven, tested, and reproducible via `simulate_end_to_end.py`. What's left is entirely optional extension, not a gap in the core result: a real cluster deployment, a true live metrics source, and calendar-aware training against real (not synthetic) traffic.