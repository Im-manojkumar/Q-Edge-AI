"""Builds REPORT.md from results/*.csv. Every number in the report comes from those files."""
from __future__ import annotations

import json
import os

import pandas as pd


def _r(path):
    p = os.path.join("results", path)
    return pd.read_csv(p) if os.path.exists(p) else None


INT_COLS = {"n", "n_qubits", "features", "steps", "tasks"}


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and c in INT_COLS and not pd.isna(v) and float(v).is_integer():
                cells.append(str(int(v)))
            elif isinstance(v, float):
                cells.append("-" if pd.isna(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def build(path: str = "REPORT.md", narrative: str = "docs/narrative.md"):
    meta = json.load(open("results/run_meta.json")) if os.path.exists("results/run_meta.json") else {}
    cal = json.load(open("results/calibration.json")) if os.path.exists("results/calibration.json") else {}
    parts = []
    if os.path.exists(narrative):
        parts.append(open(narrative).read().rstrip())
    parts.append("\n---\n\n# Appendix: generated result tables\n")
    parts.append(f"Generated from `results/` by `python -m qedge report`. Run settings: "
                 f"`{json.dumps({k: meta.get(k) for k in ('quick', 'python', 'machine', 'versions', 'timings_s')})}`\n")

    e1 = _r("e1_stacks.csv")
    if e1 is not None:
        nl = e1[e1.experiment == "noiseless"]
        small = nl[nl.n_qubits <= 8]
        t = (small.groupby(["stack"]).agg(mean_tv_to_exact_n_le_8=("tv_to_exact", "mean"),
                                          max_tv_to_exact_n_le_8=("tv_to_exact", "max")).reset_index())
        if "mean_z_error" in nl:
            t = t.merge(nl.groupby("stack").agg(max_mean_z_error_all_n=("mean_z_error", "max")).reset_index())
        big = nl[nl.n_qubits == nl.n_qubits.max()].pivot_table(index="family", columns="stack", values="time_s").reset_index()
        parts.append("## A1. Stack agreement and speed (E1)\n")
        parts.append("Agreement with the exact state vector (4000 shots). Total-variation distance is only meaningful "
                     "while 4000 shots can cover the outcomes (up to 8 qubits); the average-<Z> error is checked at "
                     "every size up to 20 qubits:\n")
        parts.append(_md(t))
        parts.append(f"\nWall time (s) per run at {int(nl.n_qubits.max())} qubits:\n")
        parts.append(_md(big))
        noisy = e1[e1.experiment == "noisy"]
        if len(noisy):
            parts.append("\nSame depolarising noise model in all four SDKs, TV distance to Qiskit's noisy result:\n")
            parts.append(_md(noisy.pivot_table(index="n_qubits", columns="stack", values="tv_to_qiskit_noisy").reset_index()))
        parts.append("\n![](figures/e1_stack_speed.png)\n")

    st = _r("e2_qaoa_stacks.csv")
    mc = _r("e2_maxcut.csv")
    if mc is not None:
        parts.append("## A2. MaxCut: QAOA vs classical (E2)\n")
        t = mc.pivot_table(index="n", columns="method", values="ratio").reset_index()
        parts.append("Mean best-cut / optimum:\n")
        parts.append(_md(t))
        ex = mc[mc.method.str.startswith("qaoa")].pivot_table(index="n", columns="method", values="expected_ratio")
        rnd = mc[mc.method == "exact"].assign(r=lambda d: (d.edges / 2) / d.optimum).groupby("n").r.mean()
        ex.insert(0, "random_sample", rnd)
        parts.append("\nMean cut of a single sample / optimum (what QAOA changes; best-of-shots hides it):\n")
        parts.append(_md(ex.reset_index()))
        tt = mc.pivot_table(index="n", columns="method", values="time_s").reset_index()
        parts.append("\nMean wall time (s):\n")
        parts.append(_md(tt, ".4f"))
        if st is not None:
            parts.append("\nThe same QAOA run on every stack:\n")
            parts.append(_md(st))
        parts.append("\n![](figures/e2_maxcut.png)\n")
    kn = _r("e2_kernel.csv")
    if kn is not None:
        parts.append("## A3. Kernel classification (E2)\n")
        parts.append(_md(kn.pivot_table(index="features", columns="method", values="accuracy").reset_index()))
        parts.append("\n![](figures/e2_kernel.png)\n")
    tf = _r("e2_tfim.csv")
    if tf is not None:
        parts.append("## A4. Spin-chain simulation error (E2)\n")
        parts.append("Absolute error in average magnetisation vs exact:\n")
        parts.append(_md(tf.pivot_table(index=["n", "steps"], columns="method", values="abs_error").reset_index()))
    if cal:
        parts.append(f"\nCalibration written for the cost model: `{json.dumps(cal)}`\n")

    s = _r("e3_summary.csv")
    if s is not None:
        parts.append("## A5. Policy comparison (E3)\n")
        cols = ["scenario", "policy", "deadline_hit_rate", "mean_latency_s", "p95_latency_s", "edge_energy_j",
                "total_energy_kj", "cost_usd_per_task", "mean_quality", "quantum_offload_rate"]
        agg = s.groupby(["scenario", "policy"]).mean(numeric_only=True).reset_index()[cols]
        parts.append(_md(agg))
        parts.append("\n![](figures/e3_policies_baseline.png)\n\n![](figures/e3_policies_network_congested.png)\n\n"
                     "![](figures/e3_scenarios.png)\n")
    m = _r("e4_qsim_map.csv")
    if m is not None:
        parts.append("## A6. Decision boundaries (E4)\n")
        parts.append(_md(m.groupby("regime").agg(qpu_share=("uses_qpu", "mean"), any_quantum=("uses_quantum", "mean"))
                         .reset_index()))
        parts.append("\n![](figures/e4_qsim_map.png)\n\n![](figures/e4_qaoa_advantage.png)\n")
    live = _r("e5_live.csv")
    if live is not None:
        parts.append("## A7. Live prototype runs (E5)\n")
        cols = ["tag", "kind", "params", "placement", "method", "backend", "fallback_from", "predicted_quality",
                "quality", "predicted_latency_s", "measured_latency_s", "predicted_cost_usd", "error"]
        parts.append(_md(live[[c for c in cols if c in live.columns]], ".3g"))
    e6 = _r("e6_summary.csv")
    if e6 is not None:
        parts.append("## A8. Calibration-aware routing (E6)\n")
        src = _r("e6_calibration.csv")
        kind = "live IBM calibration" if src is not None and (src.source == "live").all() else \
            "calibration snapshots of real IBM devices (local emulation of each device)"
        parts.append(f"Data: {kind}. Output error = 1 - Hellinger fidelity to the ideal distribution, 4000 shots.\n")
        cols = ["case", "n_qubits", "v1_error", "calibrated_error", "calibrated_device", "oracle_device",
                "picked_oracle", "static_pred", "calibrated_esp"]
        parts.append(_md(e6[[c for c in cols if c in e6.columns]]))
        pr = _r("e6_prediction.csv")
        if pr is not None:
            parts.append("\nHow well each predictor tracks measured fidelity (Spearman rank correlation). 'All' mostly "
                         "reflects that bigger circuits fail more, which any size-based predictor captures; 'within "
                         "circuit' asks whether it ranks the devices correctly, which is what routing needs:\n")
            parts.append(_md(pr))
        parts.append("\n![](figures/e6_calibration_choice.png)\n\n![](figures/e6_calibration_prediction.png)\n")
    pk = _r("e7_packing_summary.csv")
    if pk is not None:
        parts.append("## A9. Job packing (E7)\n")
        parts.append("Several small circuits share one QPU job on disjoint qubits (emulated from calibration snapshots, "
                     "4000 shots). `loss` = mean solo Hellinger fidelity minus mean packed fidelity; `loss_z` = loss "
                     "over its shot-noise standard error (|z| >= 2 counts as resolved). Crosstalk factor c is an "
                     "assumption, applied only where programs touch (buffer 0).\n")
        cols = ["device", "mix", "buffer", "crosstalk_factor", "k", "capacity", "mean_fidelity_packed",
                "mean_fidelity_solo", "fidelity_loss", "loss_z", "programs_exposed", "qpu_s_per_program",
                "price_per_program", "programs_per_free_10min"]
        parts.append(_md(pk[[c for c in cols if c in pk.columns]], ".4g"))
        parts.append("\n![](figures/e7_packing_fidelity.png)\n\n![](figures/e7_packing_cost.png)\n")
    bt = _r("e7_batching_summary.csv")
    if bt is not None:
        parts.append("## A10. Batching and hedged race mode in simulation (E7)\n")
        parts.append("800 tasks x 3 seeds per row; QPU figures are the hypothetical future QPU of E3. `-allfit` is the "
                     "strict rule (all members side by side or all back to back); plain batch rows pack the subset "
                     "that fits and run the rest back to back. Packing is assumed to cost no fidelity here.\n")
        cols = ["scenario", "variant", "qpu_tasks_per_run", "qpu_jobs_per_run", "mean_batch_size",
                "qpu_cost_per_qpu_task", "d_qpu_cost_per_qpu_task_pct", "deadline_hit_rate", "d_deadline_hit_pp",
                "qpu_deadline_hit_rate", "qpu_mean_latency_s", "hedge_rate", "fallback_win_rate",
                "wasted_cost_usd_per_task", "wasted_energy_j_per_task"]
        parts.append(_md(bt[bt.scenario != "baseline"][[c for c in cols if c in bt.columns]], ".4g"))
        parts.append("\nOn `baseline` every variant gives the same result (no task reaches the QPU).\n")
        parts.append("\n![](figures/e7_qpu_cost.png)\n\n![](figures/e7_deadline_hit.png)\n")
    rc = _r("race_demo.csv")
    if rc is not None:
        parts.append("## A11. Live race mode (E7)\n")
        parts.append("Real orchestrator, QPU path forced onto `ibm-auto` (local emulator of the best snapshot device), "
                     "classical fallback run at the same time. Times are wall-clock seconds from submission.\n")
        cols = ["scenario", "run", "deadline_s", "fallback", "winner", "quantum_status", "delivered_s",
                "overshoot_s", "classical_wall_s", "quantum_wall_s", "qpu_ran_before_cancel_s", "delivered_quality"]
        parts.append(_md(rc[[c for c in cols if c in rc.columns]], ".4g"))
        lag = _r("race_gil_lag.csv")
        if lag is not None:
            parts.append("\nHow late a timed 50 ms wait wakes up (ms), idle vs while the emulator runs in another "
                         "thread on the same machine:\n")
            parts.append(_md(lag, ".4g"))
        parts.append("\n![](figures/e7_race.png)\n")
    vq = _r("e8_vqe.csv")
    if vq is not None:
        parts.append("## A12. VQE ground-state energy (E8)\n")
        parts.append("Energy quality = 1 - (E - E0)/(Emax - E0): 1 at the exact ground state, about 0.5 for a fully "
                     "mixed (pure noise) state. Mean over seeds:\n")
        t = vq.pivot_table(index=["model", "n"], columns="method", values="quality").reset_index()
        parts.append(_md(t, ".4f"))
        h2 = vq[(vq.model == "h2") & (vq.kind == "hybrid")]
        if len(h2):
            parts.append("\nH2 error in mHa (chemical accuracy is 1.6 mHa):\n")
            parts.append(_md(h2.assign(error_mHa=1000 * h2.error)
                             .groupby("method").error_mHa.agg(["mean", "min", "max"]).reset_index(), ".2f"))
        st = _r("e8_vqe_stacks.csv")
        if st is not None:
            parts.append("\nH2 with VQE on every stack:\n")
            parts.append(_md(st, ".4f"))
        parts.append("\n![](figures/e8_vqe.png)\n")
    sc = _r("e8_scheduling.csv")
    if sc is not None:
        parts.append("## A13. Edge/cloud task scheduling: QAOA vs classical (E8)\n")
        parts.append("Mean optimum / cost found (1 = optimal assignment):\n")
        parts.append(_md(sc.pivot_table(index="n", columns="method", values="ratio").reset_index()))
        parts.append("\n`random_sampling` draws as many uniform samples as the noiseless QAOA run measured in total "
                     "(about 38 circuits x 1000 shots), because QAOA keeps the best sample it ever saw; "
                     "`random_sampling_1000` draws 1000.\n")
        ex = sc[sc.method.isin(["qaoa_noiseless", "qaoa_noisy", "random_sampling"])]
        parts.append("\nMean cost of a single sample (optimum / mean cost); best-of-shots hides this:\n")
        parts.append(_md(ex.pivot_table(index="n", columns="method", values="expected_ratio").reset_index()))
        parts.append("\n![](figures/e8_scheduling.png)\n")
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")
    return path
