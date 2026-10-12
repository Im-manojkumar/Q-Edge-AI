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
from qedge.orchestrator.policies import POLICIES, QEdgeAdaptivePolicy, QEdgeCostPolicy, Weights  # noqa: E402
from qedge.quantum.backends import provider_status  # noqa: E402
from qedge.resources import ResourceState, load_catalogue, scale  # noqa: E402
from qedge.simulation.simulator import SCENARIOS, Simulator, summarise  # noqa: E402
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
    qpu_stack = st.selectbox("QPU route", ["ibm", "braket", "ibm-emulator"],
                             help="Real hardware needs credentials in environment variables; otherwise it falls back "
                                  "to a local emulator and says so.")

tabs = st.tabs(["Orchestrator", "Simulation", "What-if map", "Experiment results", "Providers"])

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
    if st.button("Run it for real", disabled=big, help="Too large to execute in the prototype" if big else None):
        with st.spinner("executing..."):
            rep = orch.submit(kind, force=None if force == "(the decision)" else force, **params)
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
            st.json(r.details)
    if big:
        st.caption("Execution is disabled for this size: the prototype only runs circuits a laptop can simulate "
                   "and verify. The decision above still applies.")

# ---------------------------------------------------------------- simulation
with tabs[1]:
    st.subheader("Discrete-event simulation")
    c1, c2, c3 = st.columns(3)
    sc = c1.selectbox("Scenario", list(SCENARIOS))
    n_tasks = c2.slider("tasks", 100, 3000, 600, 100)
    seed = c3.number_input("seed", 0, 100, 0)
    st.caption(SCENARIOS[sc].description)
    pols = st.multiselect("Policies", list(POLICIES), default=list(POLICIES))
    if st.button("Run simulation"):
        sim = Simulator(sc, seed=int(seed))
        wl = sim.workloads(n_tasks)
        frames = []
        for p in pols:
            pol = POLICIES[p](copy.deepcopy(sim.nominal), seed=int(seed))
            if hasattr(pol, "w"):
                pol.w = W
            frames.append(sim.run(pol, wl))
        df = pd.concat(frames)
        summ = summarise(df).drop(columns=["scenario"])
        st.dataframe(summ.round(4), hide_index=True)
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
with tabs[2]:
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
with tabs[3]:
    st.subheader("Experiment results (from `python -m qedge experiments`)")
    figs = [("e1_stack_speed.png", "E1: same circuits on four quantum stacks"),
            ("e2_maxcut.png", "E2: QAOA vs classical MaxCut"), ("e2_kernel.png", "E2: quantum vs classical kernels"),
            ("e3_policies_baseline.png", "E3: policy comparison (baseline)"),
            ("e3_policies_network_congested.png", "E3: policy comparison (congested network)"),
            ("e3_scenarios.png", "E3: deadline hit rate across scenarios"),
            ("e4_qsim_map.png", "E4: QPU decision map, quantum simulation"),
            ("e4_qaoa_advantage.png", "E4: QAOA advantage needed before offloading")]
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
with tabs[4]:
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
