"""Builds REPORT.md from results/*.csv. Every number in the report comes from those files."""
from __future__ import annotations

import json
import os

import pandas as pd


def _r(path):
    p = os.path.join("results", path)
    return pd.read_csv(p) if os.path.exists(p) else None


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
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
        t = (nl.groupby(["stack"]).agg(max_tv_to_exact=("tv_to_exact", "max"), mean_tv=("tv_to_exact", "mean"))
             .reset_index())
        big = nl[nl.n_qubits == nl.n_qubits.max()].pivot_table(index="family", columns="stack", values="time_s").reset_index()
        parts.append("## A1. Stack agreement and speed (E1)\n")
        parts.append("Total-variation distance between each stack's sampled distribution and the exact one "
                     "(4000 shots; shot noise alone gives roughly 0.01 to 0.1 depending on how spread the distribution is):\n")
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
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")
    return path
