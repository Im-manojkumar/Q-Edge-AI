"""Q-Edge AI dashboard.  Run with:  streamlit run dashboard/app.py"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from qedge.orchestrator.engine import Orchestrator  # noqa: E402
from qedge.orchestrator.policies import EXTRA_POLICIES, POLICIES, QEdgeAdaptivePolicy, QEdgeCostPolicy, Weights  # noqa: E402
from qedge.quantum.backends import provider_status  # noqa: E402
from qedge.resources import ResourceState, load_catalogue, scale  # noqa: E402
from qedge.simulation.simulator import EXTRA_SCENARIOS, SCENARIOS, BatchConfig, Simulator, get_scenario, summarise  # noqa: E402
from qedge.orchestrator import split  # noqa: E402
from qedge.workloads import KINDS, load_calibration, make_qsim  # noqa: E402

st.set_page_config(page_title="Q-Edge AI", layout="wide")
load_calibration()
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TARGET_COLORS = {"classical": "#b4b2aa", "quantum simulator": "#1baf7a", "QPU": "#2a78d6"}

st.title("Q-Edge AI")
st.caption("Adaptive hybrid quantum-classical orchestration: for each AI workload, decide between edge CPU/GPU, "
           "cloud CPU/GPU, a quantum simulator, or offloading the circuit part to a QPU. Quantum assists classical "
           "AI here; it does not replace it.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Decision weights")
    W = Weights(
        latency=st.slider("latency", 0.0, 5.0, 1.0, 0.1),
        edge_energy=st.slider("edge battery energy", 0.0, 5.0, 0.3, 0.1),
        total_energy=st.slider("total energy (incl. QPU cryostat)", 0.0, 1.0, 0.05, 0.01),
        cost=st.slider("money", 0.0, 5.0, 0.3, 0.1),
        quality=st.slider("result quality", 0.0, 30.0, 10.0, 0.5),
    )
    st.header("Quantum services")
    sim_stack = st.selectbox("Quantum simulator stack", ["qiskit", "pennylane", "cirq", "braket-local"])
    qpu_stack = st.selectbox("QPU route", ["ibm-auto", "ibm", "braket", "ibm-emulator"],
                             help="ibm-auto picks the IBM device and qubits from calibration data. Real hardware needs "
                                  "credentials in environment variables; otherwise it uses calibration snapshots and a "
                                  "local emulator and says so.")

tabs = st.tabs(["Orchestrator", "Split model", "Simulation", "What-if map", "Experiment results", "Providers",
                "Calibration"])


def load_split_profiles() -> str | None:
    """Reload results/split_profile.json (E8) so a new measurement shows without a restart."""
    try:
        split.load_profiles(keep=True)
    except ValueError as e:
        split.load_profiles("", keep=True)
        return f"results/split_profile.json is malformed, so the built-in profiles are used: {e}"
    return None


SPLIT_ERROR = load_split_profiles()

# ---------------------------------------------------------------- orchestrator
with tabs[0]:
    st.subheader("Plan and run one workload")
    c1, c2 = st.columns([1, 2])
    with c1:
        kind = st.selectbox("Workload", KINDS, index=2)
        params = {}
        if kind == "inference":
            params["batch"] = st.slider("batch size", 1, 64, 8)
            params["deadline_s"] = st.number_input("deadline (s)", 0.05, 10.0, 0.5)
            params["privacy"] = st.checkbox("data must stay on the edge")
        elif kind == "training":
            params["samples"] = st.slider("samples", 200, 20000, 800, 100)
            params["epochs"] = st.slider("epochs", 1, 10, 1)
            params["deadline_s"] = st.number_input("deadline (s)", 1.0, 3600.0, 120.0)
            params["privacy"] = st.checkbox("data must stay on the edge")
        elif kind == "optimization":
            params["n"] = st.slider("graph nodes (qubits)", 4, 120, 8)
            params["p"] = st.slider("QAOA layers p", 1, 4, 1)
            params["deadline_s"] = st.number_input("deadline (s)", 1.0, 3600.0, 300.0)
        elif kind == "kernel_classification":
            params["features"] = st.slider("features (qubits)", 2, 10, 3)
            params["n_train"] = st.slider("training samples", 10, 120, 30)
            params["deadline_s"] = st.number_input("deadline (s)", 1.0, 3600.0, 300.0)
            params["privacy"] = st.checkbox("data must stay on the edge")
        elif kind == "split_inference":
            params["profile"] = st.selectbox("model profile", list(split.PROFILES),
                                             help="digits is the real model (it runs); camera is an assumption profile "
                                                  "(planning only)")
            params["k"] = st.slider("embedding size k (= qubits of the quantum head)", 2, 8, split.DEFAULT_K)
            params["batch"] = st.slider("images in the batch", 1, 64, 8)
            params["deadline_s"] = st.number_input("deadline (s)", 0.01, 3600.0, 5.0)
            params["privacy"] = st.checkbox("private: only tensors measured as private may leave the edge")
        else:
            params["n"] = st.slider("spin-chain qubits", 4, 120, 8)
            params["steps"] = st.slider("Trotter steps", 1, 40, 4)
            params["deadline_s"] = st.number_input("deadline (s)", 1.0, 7200.0, 600.0)
        adaptive = st.checkbox("use feedback (adaptive)", value=True)

    @st.cache_resource
    def orchestrator(sim, qpu, adaptive_flag):
        res = load_catalogue()
        pol = (QEdgeAdaptivePolicy if adaptive_flag else QEdgeCostPolicy)(res)
        return Orchestrator(pol, sim_stack=sim, qpu_stack=qpu)

    orch = orchestrator(sim_stack, qpu_stack, adaptive)
    orch.policy.w = W
    w, d = orch.plan(kind, **params)
    with c2:
        st.markdown(f"**Decision: `{d.placement.label}`**")
        if kind == "split_inference":
            st.caption("Split: " + split.describe_placement(w, d.placement, orch.resources))
        rows = pd.DataFrame([e.row() for e in d.estimates])
        rows["chosen"] = rows.placement == d.placement.label
        show = rows[["placement", "chosen", "feasible", "latency_s", "edge_energy_j", "energy_j", "cost_usd",
                     "quality", "fidelity", "score", "reasons"]]
        st.dataframe(show.style.format({"latency_s": "{:.3g}", "edge_energy_j": "{:.3g}", "energy_j": "{:.3g}",
                                        "cost_usd": "${:.3g}", "quality": "{:.3f}", "fidelity": "{:.3f}",
                                        "score": "{:.3g}"}, na_rep="-"), hide_index=True)
        fe = rows[rows.feasible & np.isfinite(rows.score.astype(float))]
        if len(fe):
            fig = px.bar(fe.sort_values("score"), x="score", y="placement", orientation="h",
                         color="chosen", color_discrete_map={True: PALETTE[0], False: "#b4b2aa"},
                         title="Score of each feasible placement (lower is better)")
            fig.update_layout(height=280, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig)
        for label, checks in d.quantum_checks.items():
            verdict = all(ok for _, ok, _ in checks)
            with st.expander(f"Quantum gate for {label}: {'open' if verdict else 'closed'}", expanded=not verdict):
                for name, ok, why in checks:
                    st.write(("✅ " if ok else "❌ ") + f"**{name}**: {why}")

    st.divider()
    force = st.selectbox("Placement to execute", ["(the decision)"] + [e.placement.label for e in d.estimates])
    big = (kind == "quantum_simulation" and params["n"] > 20) or (kind == "optimization" and params["n"] > 16 and
                                                                   "+" in (force if force != "(the decision)" else d.placement.label))
    no_model = kind == "split_inference" and params["profile"] != "digits"
    rc1, rc2 = st.columns(2)
    race = rc1.checkbox("Race mode", help="If this runs on the QPU, run the best classical placement at the same time "
                                          "and keep the QPU result only if it arrives before the deadline.")
    race_deadline = rc2.number_input("race deadline (s)", 1.0, 3600.0, 10.0, disabled=not race)
    if kind == "split_inference" and not no_model:
        from qedge.executor import Executor
        from qedge.ml import zoo
        if not os.path.exists(zoo.model_path(params["k"], cache_dir=Executor.model_cache)):
            st.caption(f"The k={params['k']} model is not trained yet: the first run trains it and saves it to "
                       f"{Executor.model_cache} (measured: 28 s at k=4, 146 s at k=8, mostly the quantum head).")
    if st.button("Run it for real", disabled=big or no_model,
                 help="Too large to execute in the prototype" if big else
                 "The camera profile is an assumption profile with no trained model" if no_model else None):
        with st.spinner("executing..."):
            # The race deadline replaces the workload's own; both reach submit as deadline_s.
            rest = {k: v for k, v in params.items() if k != "deadline_s"}
            rep = orch.submit(kind, force=None if force == "(the decision)" else force, race=race,
                              deadline_s=race_deadline if race else params.get("deadline_s"), **rest)
        if rep.race is not None:
            (st.success if rep.winner == "quantum" else st.info)(rep.race.summary())
        if rep.error:
            st.error(rep.error)
        else:
            r = rep.result
            where = r.backend or "local classical"
            if r.fallback_from:
                st.warning(f"{r.fallback_from} is not configured, so the QPU path ran on {r.backend}.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(r.quality_metric, f"{r.quality:.3f}", f"{r.quality - rep.decision.chosen.quality:+.3f} vs predicted")
            m2.metric("latency (s)", f"{rep.measured_latency_s:.3g}")
            m3.metric("quantum time (s)", f"{r.quantum_s:.3g}")
            m4.metric("ran on", where)
            if r.kind == "split_inference":
                from qedge.orchestrator.engine import split_summary
                st.info(split_summary(r.details).replace("\n  ", "\n\n"))
            st.json(r.details)
    if big:
        st.caption("Execution is disabled for this size: the prototype only runs circuits a laptop can simulate "
                   "and verify. The decision above still applies.")
    if no_model:
        st.caption("The camera profile is an assumption (MobileNetV2 published figures), so there is no model to run; "
                   "the decision above still applies.")

# ---------------------------------------------------------------- split model
with tabs[1]:
    st.subheader("Split model: where to cut the network, and is a quantum head worth it?")
    st.caption("A CNN backbone runs on the edge up to a cut point; the rest runs in the cloud, or the k-number embedding "
               "goes to a k-qubit quantum head. The router picks the cut from the profile below. The quantum head "
               "assists a classical model: measured so far it is level with or below the classical heads.")
    if SPLIT_ERROR:
        st.error(SPLIT_ERROR)
    sp_name = st.selectbox("Profile", list(split.PROFILES), key="sp_profile")
    prof = split.get_profile(sp_name)
    measured = prof["source"].startswith("measured")
    (st.success if measured else st.warning)(f"Source: {prof['source']}")
    by_k, cut_rows = split.profile_tables(prof)
    acc = pd.DataFrame(by_k)
    a1, a2 = st.columns([3, 2])
    with a1:
        st.markdown("**Test accuracy per head and embedding size k**")
        st.dataframe(acc[["k", "linear", "mlp", "quantum", "quantum_l0", "source"]], hide_index=True)
    with a2:
        placeholder = acc["source"] == split.PLACEHOLDER_NOTE
        long = acc[~placeholder | placeholder.all()].melt(id_vars="k", value_vars=["linear", "mlp", "quantum", "quantum_l0"], var_name="head",
                        value_name="accuracy")
        long["accuracy"] = pd.to_numeric(long["accuracy"], errors="coerce")
        figa = px.line(long, x="k", y="accuracy", color="head", markers=True, color_discrete_sequence=PALETTE,
                       height=260)
        figa.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title="test accuracy")
        st.plotly_chart(figa, width="stretch")
    if placeholder.any() and not placeholder.all():
        st.caption(f"k = {', '.join(str(k) for k in acc.k[placeholder])} not measured: those rows keep the built-in "
                   "placeholders and are left out of the chart.")
    st.caption("quantum_l0 is the ablation with no trainable circuit (encoding + linear readout only). "
               + ("Accuracies are inherited from digits, not measured on this model." if prof["name"] == "camera"
                  else "" if measured else "These are placeholders, not measurements, until E8 writes "
                                           "results/split_profile.json."))
    st.markdown("**What an attacker recovers from each transmitted tensor**")
    leak = pd.DataFrame(cut_rows + [{"cut": f"embedding (k={r['k']})", "values": r["k"], "bytes": r["bytes"],
                                      "front_flop": int(split.embedding(prof, r["k"])["front_flop"]),
                                      "reid": r["reid"], "reid_beyond_label": r["reid_beyond_label"],
                                      "r2_beyond_label": r["r2_beyond_label"], "verdict": r["verdict"]}
                                     for r in by_k])
    st.dataframe(leak, hide_index=True)
    priv = prof.get("privacy") or {}
    st.caption("reid: share of test images whose reconstruction is nearest to their own original (chance = 1 in the "
               "number of test images); beyond_label: what the tensor reveals that the predicted label does not. "
               "A verdict holds against these attacks only"
               + (f" ({', '.join(priv['attacks'])} decoders; rule: {priv['rule']})" if priv.get("attacks") else "")
               + "; a stronger attacker could do better, so 'private' is evidence, never a guarantee. "
                 "'-' means not measured, which counts as not private.")

    st.divider()
    st.markdown("**Router: every candidate placement for one batch**")
    r1, r2, r3, r4 = st.columns(4)
    sp_k = r1.slider("k", 2, 8, split.DEFAULT_K, key="sp_k")
    sp_batch = r1.slider("batch (images)", 1, 64, 16, key="sp_batch")
    sp_up = r2.select_slider("uplink (Mbps)", [0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000], value=20,
                             key="sp_up", help="uplink of every cloud resource; the catalogue default is 20 Mbps")
    sp_edge = r2.selectbox("edge device", ["edge_gpu", "edge_cpu", "both"], key="sp_edge")
    sp_priv = r3.checkbox("private batch", key="sp_priv")
    sp_dead = r3.number_input("deadline (s)", 0.01, 3600.0, 5.0, key="sp_dead")
    sp_head = r4.selectbox("quantum head", list(split.QHEADS), key="sp_qhead")
    cat = load_catalogue()
    for r in cat.values():
        if not r.is_edge:
            r.uplink_mbps = sp_up
        elif not r.is_quantum and sp_edge != "both" and r.name != sp_edge:
            r.available = False
    rpol = QEdgeCostPolicy(cat)
    rpol.w = W
    from qedge.workloads import make_split_inference
    sw = make_split_inference(0, np.random.default_rng(0), profile=sp_name, k=sp_k, batch=sp_batch, privacy=sp_priv,
                              deadline_s=sp_dead, qhead=sp_head)
    sd = rpol.decide(sw, orch.monitor.snapshot(sw))
    rows = []
    for e in sd.estimates:
        sent = split.transmitted(e.placement, cat)
        size = sum(prof["input_bytes"] if t == "input" else split.cut_info(prof, t, sp_k)["bytes"] for t in sent)
        rows.append({"placement": e.placement.label, "chosen": e.placement == sd.placement, "feasible": e.feasible,
                     "leaves edge": ", ".join(sent) or "nothing", "bytes / image": size,
                     "latency_s": e.latency_s, "edge_energy_j": e.edge_energy_j, "cost_usd": e.cost_usd,
                     "quality": e.quality, "score": f"{e.score:.3g}" if e.feasible and e.score is not None else "-",
                     "why not": "; ".join(e.reasons)})
    cand = pd.DataFrame(rows)
    st.markdown(f"Decision: **`{sd.placement.label}`**: {split.describe_placement(sw, sd.placement, cat)}")
    st.dataframe(cand.style.format({"latency_s": "{:.3g}", "edge_energy_j": "{:.3g}", "cost_usd": "${:.3g}",
                                    "quality": "{:.3f}"}, na_rep="-"), hide_index=True,
                 height=35 * (len(cand) + 1) + 3)
    figc = px.bar(cand, x="latency_s", y="placement", orientation="h", color="chosen", log_x=True,
                  color_discrete_map={True: PALETTE[0], False: "#b4b2aa"}, height=40 + 22 * len(cand),
                  hover_data=["leaves edge", "bytes / image", "feasible", "why not"],
                  title="Predicted latency per candidate (log scale)")
    figc.add_vline(x=sp_dead, line_dash="dash", line_color=PALETTE[1])
    # On a log axis an annotation's x is given in log10 units (a shape's is not).
    figc.add_annotation(x=float(np.log10(sp_dead)), y=1.0, yref="paper", text="deadline", showarrow=False,
                        xanchor="left", yanchor="bottom", font=dict(color=PALETTE[1]))
    figc.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0), yaxis=dict(autorange="reversed"))
    st.plotly_chart(figc, width="stretch")
    for label, checks in sd.quantum_checks.items():
        verdict = all(ok for _, ok, _ in checks)
        with st.expander(f"Quantum gate for {label}: {'open' if verdict else 'closed'}"):
            for name, ok, why in checks:
                st.write(("✅ " if ok else "❌ ") + f"**{name}**: {why}")
    st.caption("Estimates from the cost model: catalogue resources with the uplink above, the QPU's figures from "
               "calibration snapshots (live calibration with IBM credentials). The network is modelled, not measured. "
               "Run a batch for real in the Orchestrator tab (workload split_inference).")

    st.divider()
    import glob
    e8 = sorted(glob.glob(os.path.join("figures", "e8_*.png")))
    st.markdown("**Experiment E8 figures**")
    if e8:
        for f in e8:
            st.markdown(f"`{os.path.basename(f)}`")
            st.image(f)
    else:
        st.caption("No E8 figures yet: run `python -m qedge experiments --only e8`.")

# ---------------------------------------------------------------- simulation
with tabs[2]:
    st.subheader("Discrete-event simulation")
    c1, c2, c3 = st.columns(3)
    sc = c1.selectbox("Scenario", list(SCENARIOS) + list(EXTRA_SCENARIOS))
    n_tasks = c2.slider("tasks", 100, 3000, 600, 100)
    seed = c3.number_input("seed", 0, 100, 0)
    st.caption(get_scenario(sc).description)
    all_pols = {**POLICIES, **EXTRA_POLICIES}
    pols = st.multiselect("Policies", list(all_pols), default=list(POLICIES),
                          help="qedge_hedged runs a classical fallback next to risky QPU jobs (race mode)")
    b1, b2 = st.columns(2)
    window = b1.select_slider("QPU batching window (s)", [0, 5, 30, 120], value=0,
                              help="0 = every QPU task is its own job. Otherwise tasks arriving within the window "
                                   "share one job (one queue wait, one task fee, shared shots when they fit side by side).")
    max_batch = b2.slider("most tasks per QPU job", 2, 16, 8)
    if st.button("Run simulation"):
        sim = Simulator(sc, seed=int(seed), batching=BatchConfig(window, max_batch) if window else None)
        wl = sim.workloads(n_tasks)
        frames = []
        for p in pols:
            pol = all_pols[p](copy.deepcopy(sim.nominal), seed=int(seed))
            if hasattr(pol, "w"):
                pol.w = W
            frames.append(sim.run(pol, wl))
        df = pd.concat(frames)
        summ = summarise(df).drop(columns=["scenario"])
        st.dataframe(summ.round(4), hide_index=True)
        if "batch_id" in df.columns or "hedged" in df.columns:
            q = df[df.quantum_resource == "qpu"]
            if len(q):
                extra = q.groupby("policy").agg(qpu_tasks=("id", "count"), qpu_cost_per_task=("cost_usd", "mean"),
                                                qpu_deadline_hit_rate=("deadline_met", "mean"))
                if "batch_size" in q:
                    extra["mean_batch_size"] = q.groupby("policy").batch_size.mean()
                if "hedged" in q:
                    extra["hedged_share"] = q.groupby("policy").hedged.mean()
                st.markdown("QPU-routed tasks")
                st.dataframe(extra.reset_index().round(4), hide_index=True)
        m = st.selectbox("Metric", ["deadline_hit_rate", "mean_latency_s", "p95_latency_s", "edge_energy_j",
                                    "total_energy_kj", "cost_usd_per_task", "mean_quality", "quantum_offload_rate"])
        colors = {p: (PALETTE[0] if p.startswith("qedge") else "#b4b2aa") for p in summ.policy}
        fig = px.bar(summ, x=m, y="policy", orientation="h", color="policy", color_discrete_map=colors)
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig)
        pl = df.groupby(["policy", "placement"]).size().rename("tasks").reset_index()
        st.plotly_chart(px.bar(pl, x="tasks", y="policy", color="placement", orientation="h",
                               color_discrete_sequence=PALETTE, title="Where each policy placed work"),
                        width="stretch")
        st.session_state["last_sim"] = df

# ---------------------------------------------------------------- what-if
with tabs[3]:
    st.subheader("When does quantum simulation go to the QPU?")
    st.caption("Change the QPU's hardware figures and see which spin-chain sizes Q-Edge would offload. "
               "Today's figures: 2-qubit error 3e-3, queue 120 s.")
    c1, c2, c3 = st.columns(3)
    err = c1.select_slider("gate error vs today", [1.0, 0.3, 0.1, 0.03, 0.01, 0.003], value=1.0)
    queue = c2.select_slider("queue (s)", [1, 10, 30, 120, 600, 3600], value=120)
    price = c3.select_slider("price vs today", [0.0, 0.01, 0.1, 1.0], value=1.0)
    cat = scale(load_catalogue(), qpu__p1=err, qpu__p2=err, qpu__readout_err=err,
                qpu__cost_per_task=price, qpu__cost_per_shot=price)
    cat["qpu"].queue_mean_s = queue
    pol = QEdgeCostPolicy(cat)
    pol.w = W
    rows = []
    for n in range(8, 129, 8):
        for steps in range(2, 41, 2):
            wk = make_qsim(0, np.random.default_rng(0), n=n, steps=steps, deadline_s=3600)
            dd = pol.decide(wk, ResourceState())
            tgt = "QPU" if dd.placement.quantum == "qpu" else ("quantum simulator" if dd.placement.quantum else "classical")
            rows.append({"qubits": n, "steps": steps, "target": tgt, "placement": dd.placement.label})
    mp = pd.DataFrame(rows)
    fig = px.scatter(mp, x="qubits", y="steps", color="target", symbol="target", color_discrete_map=TARGET_COLORS,
                     hover_data=["placement"], height=460)
    fig.update_traces(marker=dict(size=13, line=dict(width=1, color="white")))
    st.plotly_chart(fig)
    st.write(f"QPU chosen for **{(mp.target == 'QPU').mean():.0%}** of this grid.")

# ---------------------------------------------------------------- results
with tabs[4]:
    st.subheader("Experiment results (from `python -m qedge experiments`)")
    figs = [("e1_stack_speed.png", "E1: same circuits on four quantum stacks"),
            ("e2_maxcut.png", "E2: QAOA vs classical MaxCut"), ("e2_kernel.png", "E2: quantum vs classical kernels"),
            ("e3_policies_baseline.png", "E3: policy comparison (baseline)"),
            ("e3_policies_network_congested.png", "E3: policy comparison (congested network)"),
            ("e3_scenarios.png", "E3: deadline hit rate across scenarios"),
            ("e4_qsim_map.png", "E4: QPU decision map, quantum simulation"),
            ("e4_qaoa_advantage.png", "E4: QAOA advantage needed before offloading"),
            ("e6_calibration_choice.png", "E6: output error, fixed device vs calibration-aware routing"),
            ("e6_calibration_prediction.png", "E6: predicted vs measured, static figures vs calibration"),
            ("e7_packing_fidelity.png", "E7: fidelity cost of packing several programs into one job"),
            ("e7_packing_cost.png", "E7: QPU time and price per program when packed"),
            ("e7_qpu_cost.png", "E7: QPU cost per task with batching (simulation)"),
            ("e7_deadline_hit.png", "E7: deadline hit rate with batching and race mode (simulation)"),
            ("e7_race.png", "E7: live race mode, delivered time vs deadline")]
    for f, cap in figs:
        p = os.path.join("figures", f)
        if os.path.exists(p):
            st.markdown(f"**{cap}**")
            st.image(p)
    for csv in ["e3_summary.csv", "e2_qaoa_stacks.csv", "e5_live.csv"]:
        p = os.path.join("results", csv)
        if os.path.exists(p):
            st.markdown(f"`{csv}`")
            st.dataframe(pd.read_csv(p), hide_index=True)
    if os.path.exists("results/calibration.json"):
        st.markdown("Calibration used by the cost model")
        st.json(json.load(open("results/calibration.json")))

# ---------------------------------------------------------------- providers
with tabs[5]:
    st.subheader("Quantum stacks and real hardware")
    st.dataframe(pd.DataFrame(provider_status()), hide_index=True)
    st.markdown("""
**To use real hardware** set these before starting the dashboard (never paste keys into the app):

| Provider | Free option | Variables |
|---|---|---|
| IBM Quantum | Open Plan, 10 QPU-minutes / 28 days | `QISKIT_IBM_TOKEN`, optional `QISKIT_IBM_INSTANCE`, `IBM_QUANTUM_BACKEND` |
| Amazon Braket | AWS Free Tier: 1 hour/month of managed simulators (QPUs are billed) | AWS credentials, `AWS_DEFAULT_REGION`, `BRAKET_DEVICE_ARN` |

Runs on real hardware are capped by `QEDGE_HW_MAX_CIRCUITS` (default 60) and `QEDGE_HW_MAX_SHOTS` (default 2000).
""")
    if os.path.exists("results/live_log.jsonl"):
        log = [json.loads(line) for line in open("results/live_log.jsonl")][-50:]
        st.markdown("Recent live executions")
        st.dataframe(pd.DataFrame([{"kind": r["workload"]["kind"], "placement": r["placement"],
                                    "predicted_latency_s": r["predicted"]["latency_s"],
                                    "measured_latency_s": r["measured_latency_s"],
                                    "quality": (r["result"] or {}).get("quality"),
                                    "backend": (r["result"] or {}).get("backend"), "error": r["error"]} for r in log]),
                     hide_index=True)

# ---------------------------------------------------------------- calibration
with tabs[6]:
    from qedge.executor import representative_circuit
    from qedge.quantum import calibration as calmod
    from qedge.quantum.backends.qiskit_backend import to_qiskit
    from qedge.workloads import BUILDERS

    st.subheader("Calibration-aware routing")

    @st.cache_resource
    def devices(prefer_live: bool):
        return calmod.load_devices(prefer_live)

    devs = devices(True)
    live = devs[0].source == "live"
    if live:
        st.success(f"Live calibration from {len(devs)} IBM devices on your account (reading it uses no QPU time).")
    else:
        st.info("No IBM credentials: using calibration snapshots of real IBM devices, emulated locally. "
                "Set QISKIT_IBM_TOKEN to route on today's live data.")
    summ = pd.DataFrame([d.summary() for d in devs]).sort_values("median_2q_error")
    st.dataframe(summ, hide_index=True)

    c1, c2 = st.columns(2)
    kind = c1.selectbox("Workload to place", ["quantum_simulation", "optimization", "kernel_classification"])
    if kind == "quantum_simulation":
        params = {"n": c2.slider("qubits", 2, 20, 8), "steps": c2.slider("Trotter steps", 1, 6, 3)}
    elif kind == "optimization":
        params = {"n": c2.slider("graph nodes (qubits)", 3, 16, 8), "p": 1}
    else:
        params = {"features": c2.slider("features (qubits)", 2, 12, 6), "n_train": 20, "n_test": 10}

    @st.cache_data(show_spinner="Scoring every device and qubit layout ...")
    def ranking(kind, params_json):
        w = BUILDERS[kind](0, np.random.default_rng(0), **json.loads(params_json))
        best, allc = calmod.select_device(to_qiskit(representative_circuit(w)), devices(True))
        return best.device, pd.DataFrame([c.row() for c in allc]).sort_values("esp", ascending=False)

    best_dev, rank = ranking(kind, json.dumps(params))
    st.markdown(f"**Chosen: {best_dev}.** Estimated success probability (ESP) per device, with the qubits it "
                "would use. `default_esp` is what Qiskit's own placement gets on that device.")
    st.dataframe(rank[["device", "esp", "default_esp", "two_q_count", "depth", "queue_s", "layout"]], hide_index=True)
    fig = px.bar(rank, x="device", y="esp", color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(height=280, margin=dict(t=10, b=10), yaxis_title="estimated success probability")
    st.plotly_chart(fig, use_container_width=True)

    show = st.selectbox("Qubit map for device", list(summ.device), index=list(summ.device).index(best_dev))

    @st.cache_data
    def qubit_positions(device):
        import rustworkx as rx
        cal = next(d for d in devices(True) if d.name == device)
        g = rx.PyGraph()
        g.add_nodes_from(range(cal.num_qubits))
        for a, b in cal.two_q:
            if a < b:
                g.add_edge(a, b, None)
        pos = rx.spring_layout(g, seed=3, num_iter=2000)
        return {q: tuple(pos[q]) for q in range(cal.num_qubits)}

    import plotly.graph_objects as go
    cal = next(d for d in devs if d.name == show)
    pos = qubit_positions(show)
    chosen = set(rank.loc[rank.device == show, "layout"].iloc[0]) if show in set(rank.device) else set()
    figm = go.Figure()
    for label, sel, color, width in [("coupler", lambda e: e < 0.01, "#d9d7d0", 2),
                                     ("coupler, error >= 1%", lambda e: 0.01 <= e < calmod.BROKEN, "#eb6834", 3),
                                     ("broken coupler", lambda e: e >= calmod.BROKEN, "#e34948", 5)]:
        xs, ys = [], []
        for (a, b), e in cal.two_q.items():
            if a < b and sel(e):
                xs += [pos[a][0], pos[b][0], None]
                ys += [pos[a][1], pos[b][1], None]
        figm.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width), name=label,
                                  hoverinfo="skip"))
    qs = list(range(cal.num_qubits))
    figm.add_trace(go.Scatter(
        x=[pos[q][0] for q in qs], y=[pos[q][1] for q in qs], mode="markers", name="qubit (colour = readout error)",
        marker=dict(size=[13 if q in chosen else 8 for q in qs], color=[cal.readout[q] for q in qs],
                    colorscale="YlOrRd", cmin=0, cmax=0.05, colorbar=dict(title="readout error", thickness=10),
                    line=dict(color=["#2a78d6" if q in chosen else "rgba(0,0,0,0)" for q in qs], width=3)),
        text=[f"qubit {q}<br>readout error {cal.readout[q]:.4f}<br>T1 {cal.t1_us.get(q, float('nan')):.0f} us"
              + ("<br><b>used by this workload</b>" if q in chosen else "") for q in qs],
        hoverinfo="text"))
    figm.update_layout(height=560, margin=dict(t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False),
                       legend=dict(orientation="h", y=-0.02))
    st.caption("Qubits outlined in blue are the ones this workload would use. Positions are a drawing of the "
               "coupling graph, not the chip's physical layout.")
    st.plotly_chart(figm, use_container_width=True)

    drift = calmod.drift_from_trace()
    if drift:
        st.markdown("**Drift between the first and last logged calibration** (`python -m qedge calibration log`)")
        st.dataframe(pd.DataFrame(drift), hide_index=True)
    else:
        st.caption("Log live calibration and queues over time with `python -m qedge calibration log --count 144` "
                   "to see drift here and to run the simulator on the real queue trace (`--scenario ibm_trace`).")
