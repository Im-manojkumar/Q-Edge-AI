"""Figures for the report. Q-Edge policies are highlighted; baselines stay gray."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, MUTED, GRID = "#b4b2aa", "#0b0b0b", "#52514e", "#e6e4df"
STACK_COLORS = {"qiskit": BLUE, "pennylane": ORANGE, "cirq": AQUA, "braket-local": YELLOW}
POLICY_ORDER = ["edge_only", "cloud_only", "quantum_first", "random", "latency_greedy", "qedge_static", "qedge_adaptive"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False,
    "figure.dpi": 130, "lines.linewidth": 2,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def stacks(df: pd.DataFrame, out: str):
    d = df[df.experiment == "noiseless"]
    fams = list(d.family.unique())
    fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.2), sharey=True)
    for ax, fam in zip(axes, fams):
        for stack, color in STACK_COLORS.items():
            s = d[(d.family == fam) & (d["stack"] == stack)]
            ax.plot(s.n_qubits, s.time_s, color=color, marker="o", markersize=4, label=stack)
        ax.set_yscale("log")
        ax.set_title(fam.replace("_", " "))
        ax.set_xlabel("qubits")
    axes[0].set_ylabel("wall time per run (s, log)")
    axes[-1].legend(loc="upper left", fontsize=8)
    _save(fig, out)


def maxcut(df: pd.DataFrame, out: str):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    order = [("exact", "exact (classical)", INK), ("simulated_annealing", "simulated annealing", AQUA),
             ("greedy", "greedy", GRAY), ("qaoa_noiseless", "QAOA p=1, noiseless", BLUE),
             ("qaoa_noisy", "QAOA p=1, NISQ noise", ORANGE), ("random_sampling", "random sampling", MAGENTA)]
    g = df.groupby(["method", "n"]).ratio.mean().reset_index()
    for m, label, color in order:
        s = g[g.method == m]
        ax.plot(s.n, s.ratio, marker="o", markersize=4, color=color, label=label,
                linestyle="--" if m == "random_sampling" else "-")
    emu = df[df.method == "qaoa_ibm_emulator"]
    if len(emu):
        ax.scatter(emu.n, emu.ratio, marker="D", s=40, color=YELLOW, edgecolor="white", zorder=5,
                   label="QAOA, IBM-device emulator")
    ax.set_ylim(0.85, 1.01)
    ax.set_xlabel("graph nodes (= qubits)")
    ax.set_ylabel("best cut / optimum")
    ax.text(14, 1.004, "exact, annealing and QAOA overlap at 1.0", ha="right", fontsize=8, color=MUTED)
    ax.set_title("Best of 1000 shots: everything finds the optimum")
    ax.legend(fontsize=7.5, loc="lower left")
    # Mean cut of a single sample: the part QAOA actually changes.
    ex = df[df.method == "exact"].assign(ratio=lambda d: (d.edges / 2) / d.optimum).groupby("n").ratio.mean()
    ax2.plot(ex.index, ex.values, color=MAGENTA, linestyle="--", marker="o", markersize=4, label="random sample")
    for m, label, color in [("qaoa_noiseless", "QAOA p=1, noiseless", BLUE), ("qaoa_noisy", "QAOA p=1, NISQ noise", ORANGE),
                            ("qaoa_ibm_emulator", "QAOA, IBM-device emulator", YELLOW)]:
        s = df[df.method == m].groupby("n").expected_ratio.mean()
        if len(s):
            ax2.plot(s.index, s.values, color=color, marker="o", markersize=4, label=label)
    ax2.axhline(1.0, color=INK, linewidth=1)
    ax2.text(ex.index.max(), 0.985, "exact / annealing", ha="right", va="top", fontsize=8, color=MUTED)
    ax2.set_xlabel("graph nodes (= qubits)")
    ax2.set_ylabel("mean cut of one sample / optimum")
    ax2.set_title("Mean sample: QAOA's real gain over random")
    ax2.legend(fontsize=7.5, loc="lower right")
    _save(fig, out)


def kernel(df: pd.DataFrame, out: str):
    order = [("rbf_svm", "RBF SVM", INK), ("logistic", "logistic regression", GRAY),
             ("quantum_kernel_noiseless", "quantum kernel", BLUE), ("quantum_kernel_noisy", "quantum kernel, noisy", ORANGE)]
    g = df.groupby(["features", "method"]).accuracy.mean().unstack()
    x = np.arange(len(g.index))
    wdt = 0.2
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (m, label, color) in enumerate(order):
        ax.bar(x + (i - 1.5) * wdt, g[m], wdt * 0.9, color=color, label=label)
    ax.set_xticks(x, [f"{d} features" for d in g.index])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("test accuracy")
    ax.set_title("Breast-cancer classification: quantum vs classical kernels")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, out)


def policies(summ: pd.DataFrame, out: str, scenario: str = "baseline"):
    s = summ[summ.scenario == scenario].groupby("policy").mean(numeric_only=True).reindex(POLICY_ORDER)
    metrics = [("deadline_hit_rate", "deadline hit rate", False), ("mean_latency_s", "mean latency (s, log)", True),
               ("edge_energy_j", "edge energy per task (J)", False), ("cost_usd_per_task", "cloud+QPU cost per task ($, log)", True)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    colors = [BLUE if p.startswith("qedge") else GRAY for p in s.index]
    for ax, (m, title, log) in zip(axes, metrics):
        vals = s[m].clip(lower=1e-6) if log else s[m]
        ax.barh(range(len(s)), vals, color=colors, height=0.7)
        ax.set_yticks(range(len(s)), s.index if ax is axes[0] else [""] * len(s))
        ax.invert_yaxis()
        if log:
            ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Policy comparison, scenario: {scenario}", color=INK, x=0.01, ha="left")
    _save(fig, out)


def scenarios(summ: pd.DataFrame, out: str):
    piv = summ.groupby(["scenario", "policy"]).deadline_hit_rate.mean().unstack().reindex(columns=POLICY_ORDER)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else INK)
    fig.colorbar(im, ax=ax, label="deadline hit rate")
    ax.set_title("Deadline hit rate by scenario and policy")
    _save(fig, out)


def qsim_map(df: pd.DataFrame, out: str):
    regimes = list(df.regime.unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.2 * len(regimes), 3.6), sharey=True)
    for ax, reg in zip(axes, regimes):
        d = df[df.regime == reg]
        d = d.assign(code=d.uses_quantum.astype(int) + d.uses_qpu.astype(int))
        piv = d.pivot(index="steps", columns="n_qubits", values="code").astype(float)
        ax.imshow(piv.values, origin="lower", cmap=matplotlib.colors.ListedColormap(["#f0efec", AQUA, BLUE]),
                  vmin=0, vmax=2, aspect="auto",
                  extent=[piv.columns.min() - 4, piv.columns.max() + 4, piv.index.min() - 1, piv.index.max() + 1])
        ax.grid(False)
        share = d.uses_qpu.mean()
        ax.set_title(f"{reg.replace('_', ' ')}: QPU chosen in {share:.0%}")
        ax.set_xlabel("spin-chain qubits")
    axes[0].set_ylabel("Trotter steps (circuit depth / 3)")
    handles = [matplotlib.patches.Patch(color=c, label=l) for l, c in
               [("classical (state vector / MPS)", "#d9d7d0"), ("quantum simulator", AQUA), ("QPU", BLUE)]]
    axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle("Spin-chain simulation: where Q-Edge runs it as QPU hardware improves", color=INK, x=0.01, ha="left")
    _save(fig, out)


def qaoa_adv(df: pd.DataFrame, out: str):
    ns = sorted(df.n.unique())
    codes = {"classical": 0, "quantum simulator": 1, "QPU": 2}
    cmap = matplotlib.colors.ListedColormap(["#f0efec", AQUA, BLUE])
    fig, axes = plt.subplots(1, len(ns), figsize=(4 * len(ns), 3.4), sharey=True)
    for ax, n in zip(axes, ns):
        d = df[df.n == n].assign(code=lambda x: x.target.map(codes))
        piv = d.pivot(index="quality_advantage", columns="price_multiplier", values="code")
        ax.imshow(piv.values, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="auto")
        cost = d.groupby("price_multiplier").qpu_cost_usd.first()
        ax.set_xticks(range(len(piv.columns)), [("free" if cost[c] < 0.01 else f"${cost[c]:.2f}") for c in piv.columns])
        ax.set_yticks(range(0, len(piv.index), 2), [f"+{v:.2f}" for v in piv.index[::2]])
        ax.grid(False)
        fid = d.qpu_fidelity.iloc[0]
        ax.set_title(f"MaxCut, {n} nodes (QPU fidelity {fid:.2f})")
        ax.set_xlabel("QPU price per task")
    axes[0].set_ylabel("hypothetical QAOA quality edge")
    handles = [matplotlib.patches.Patch(color=c, label=l) for l, c in
               [("classical", "#d9d7d0"), ("quantum simulator", AQUA), ("QPU", BLUE)]]
    axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle("Hard MaxCut instances (classical heuristic at 0.70): where would Q-Edge run QAOA?",
                 color=INK, x=0.01, ha="left")
    _save(fig, out)


def make_all(res: str = "results", fig: str = "figures"):
    os.makedirs(fig, exist_ok=True)
    jobs = [("e1_stacks.csv", stacks, "e1_stack_speed.png"), ("e2_maxcut.csv", maxcut, "e2_maxcut.png"),
            ("e2_kernel.csv", kernel, "e2_kernel.png"), ("e3_summary.csv", policies, "e3_policies_baseline.png"),
            ("e3_summary.csv", scenarios, "e3_scenarios.png"), ("e4_qsim_map.csv", qsim_map, "e4_qsim_map.png"),
            ("e4_qaoa_advantage.csv", qaoa_adv, "e4_qaoa_advantage.png")]
    for csv, fn, png in jobs:
        p = os.path.join(res, csv)
        if os.path.exists(p):
            fn(pd.read_csv(p), os.path.join(fig, png))
    p = os.path.join(res, "e3_summary.csv")
    if os.path.exists(p):
        policies(pd.read_csv(p), os.path.join(fig, "e3_policies_network_congested.png"), "network_congested")
