"""E8: the two added algorithms, VQE and QAOA task scheduling, against their classical baselines.

Same protocol as E2: run the hybrid algorithm on a noiseless simulator, a uniformly
noisy one and the IBM-device emulator, run the classical method on the same
instances, and fit the quality curves the cost model uses. The fits are merged
into results/calibration.json (the E2 entries are kept).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from ..quantum.backends import SIMULATOR_STACKS, get_backend
from ..quantum.scheduling import (run_scheduling_qaoa, scheduling_annealing, scheduling_greedy,
                                  scheduling_instance)
from ..quantum.vqe import (CHEMICAL_ACCURACY, energy_quality, hamiltonian_for, measurement_groups, run_vqe,
                           spectrum_bounds)
from .e2_quantum_vs_classical import NOISE


def vqe(quick: bool) -> pd.DataFrame:
    systems = [("h2", 2)] + [("tfim", n) for n in ([4, 6] if quick else [4, 6, 8, 10])]
    seeds = range(2 if quick else 3)
    clean, noisy, emu = get_backend("qiskit"), get_backend("qiskit", NOISE), get_backend("ibm-emulator")
    rows = []
    for model, n in systems:
        H, n = hamiltonian_for(model, n)
        t0 = time.perf_counter()
        e0, emax = spectrum_bounds(H, n)
        t_exact = time.perf_counter() - t0
        base = {"model": model, "n": n, "exact_energy": e0}
        rows.append({**base, "seed": 0, "method": "exact_diagonalisation", "energy": e0, "error": 0.0,
                     "quality": 1.0, "time_s": t_exact, "kind": "classical"})
        for seed in seeds:
            runs = [("vqe_noiseless", clean, "rotosolve"), ("vqe_cobyla_noiseless", clean, "cobyla"),
                    ("vqe_noisy", noisy, "rotosolve")]
            if n <= 4 and seed == 0:
                runs.append(("vqe_ibm_emulator", emu, "rotosolve"))
            for name, be, opt in runs:
                t0 = time.perf_counter()
                # COBYLA gets the same number of energy evaluations Rotosolve uses (one per job).
                kw = {"sweeps": 5} if opt == "rotosolve" else {"maxiter": 2 * 5 * n * 2}
                r = run_vqe(be, H, n, 1, shots=2000, seed=seed, optimizer=opt, **kw)
                rows.append({**base, "seed": seed, "method": name, "energy": r.energy, "error": r.energy - e0,
                             "quality": energy_quality(r.energy, e0, emax), "time_s": time.perf_counter() - t0,
                             "circuit_evals": r.n_circuit_evals, "jobs": r.n_jobs, "kind": "hybrid",
                             "chemical_accuracy": abs(r.energy - e0) <= CHEMICAL_ACCURACY if model == "h2" else None})
    return pd.DataFrame(rows)


def vqe_all_stacks() -> pd.DataFrame:
    H, n = hamiltonian_for("h2")
    e0, emax = spectrum_bounds(H, n)
    rows = []
    for s in SIMULATOR_STACKS:
        t0 = time.perf_counter()
        r = run_vqe(get_backend(s), H, n, 1, shots=4000, sweeps=5, seed=0)
        rows.append({"stack": s, "model": "h2", "energy": r.energy, "error_mHa": 1000 * (r.energy - e0),
                     "quality": energy_quality(r.energy, e0, emax), "circuit_evals": r.n_circuit_evals,
                     "time_s": time.perf_counter() - t0})
    return pd.DataFrame(rows)


def scheduling(quick: bool) -> pd.DataFrame:
    sizes = [4, 6, 8, 10] if quick else [4, 6, 8, 10, 12, 14]
    instances = 2 if quick else 3
    clean, noisy, emu = get_backend("qiskit"), get_backend("qiskit", NOISE), get_backend("ibm-emulator")
    rows = []
    for n in sizes:
        for g in range(instances):
            inst = scheduling_instance(n, 0.3, seed=1000 * n + g)
            t0 = time.perf_counter()
            costs = inst.all_costs()
            opt = float(costs.min())
            t_exact = time.perf_counter() - t0
            base = {"n": n, "instance": g, "optimum": opt, "pairs_communicating": len(inst.comm)}
            (_, gc), t_g = _timed(scheduling_greedy, inst)
            (_, sc), t_sa = _timed(scheduling_annealing, inst, 1000, g)
            rows += [{**base, "method": "exact", "ratio": 1.0, "time_s": t_exact, "kind": "classical"},
                     {**base, "method": "greedy", "ratio": opt / gc, "time_s": t_g, "kind": "classical"},
                     {**base, "method": "simulated_annealing", "ratio": opt / sc, "time_s": t_sa, "kind": "classical"}]
            runs = [("qaoa_noiseless", clean), ("qaoa_noisy", noisy)]
            if n <= 8 and g == 0:
                runs.append(("qaoa_ibm_emulator", emu))
            budget = 0
            for name, be in runs:
                t0 = time.perf_counter()
                r = run_scheduling_qaoa(be, inst, p=1, shots=1000, maxiter=25 if name != "qaoa_ibm_emulator" else 15,
                                        seed=g)
                rows.append({**base, "method": name, "ratio": opt / r.best_cost, "expected_ratio": opt / r.expected_cost,
                             "time_s": time.perf_counter() - t0, "circuit_evals": r.n_circuit_evals, "kind": "hybrid"})
                if name == "qaoa_noiseless":
                    budget = r.n_circuit_evals * r.shots
            # QAOA keeps the best of every sample it draws while optimising (~38 circuits x 1000 shots),
            # so the fair random baseline draws the same number of uniform samples.
            rng = np.random.default_rng(g)
            for name, size in [("random_sampling", budget), ("random_sampling_1000", 1000)]:
                rnd = costs[rng.integers(0, len(costs), size)]
                rows.append({**base, "method": name, "ratio": opt / rnd.min(), "expected_ratio": opt / rnd.mean(),
                             "time_s": 0.0, "samples": size, "kind": "baseline"})
    return pd.DataFrame(rows)


def _timed(fn, *a):
    t0 = time.perf_counter()
    out = fn(*a)
    return out, time.perf_counter() - t0


def _linear(series: pd.Series, at: int, floor: float) -> dict:
    x, y = series.index.values.astype(float), series.values.astype(float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0
    value = float(np.interp(at, x, y))
    return {"slope": min(slope, 0.0), "at": at, "value": min(1.0, value), "floor": floor}


def fit_calibration(vq: pd.DataFrame, sc: pd.DataFrame) -> dict:
    v = vq[vq.method == "vqe_noiseless"].groupby("n").quality.mean()
    # QAOA's measured edge over uniform random sampling with the same number of samples. Random
    # sampling's own curve is fitted on 1000 samples, the only budget at which it falls below 1
    # within the sizes we can verify exactly; it is the floor that noise pulls QAOA down to.
    q = sc[sc.method == "qaoa_noiseless"].groupby("n").ratio.mean()
    same = sc[sc.method == "random_sampling"].groupby("n").ratio.mean()
    r = sc[sc.method == "random_sampling_1000"].groupby("n").ratio.mean()
    sa = sc[sc.method == "simulated_annealing"].groupby("n").ratio.mean()
    return {"vqe_quality": _linear(v, 2, 0.85),
            "sched_qaoa_advantage": float(max(0.0, (q - same).mean())),
            "sched_random": _linear(r, 10, 0.1),
            "sched_sa": {"slope": 0.0, "at": 10, "value": float(min(1.0, sa.mean())), "floor": float(min(0.9, sa.min()))}}


def run(quick: bool = False, out_dir: str = "results") -> dict[str, pd.DataFrame]:
    vq = vqe(quick)
    st = vqe_all_stacks()
    sc = scheduling(quick)
    path = os.path.join(out_dir, "calibration.json")
    cal = json.load(open(path)) if os.path.exists(path) else {}
    cal.update(fit_calibration(vq, sc))
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)
    return {"e8_vqe": vq, "e8_vqe_stacks": st, "e8_scheduling": sc}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def make_figures(res: str = "results", fig: str = "figures"):
    from .figures import AQUA, BLUE, GRAY, INK, MAGENTA, ORANGE, YELLOW, _save, plt

    p = os.path.join(res, "e8_vqe.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df["system"] = np.where(df.model == "h2", "H2 (2q)", "TFIM " + df.n.astype(str) + "q")
        order = list(dict.fromkeys(df.sort_values("n").system))
        methods = [("exact_diagonalisation", "exact (classical)", INK), ("vqe_noiseless", "VQE, Rotosolve", BLUE),
                   ("vqe_cobyla_noiseless", "VQE, COBYLA", AQUA), ("vqe_noisy", "VQE, NISQ noise", ORANGE),
                   ("vqe_ibm_emulator", "VQE, IBM emulator", YELLOW)]
        g = df.groupby(["system", "method"]).quality.mean().unstack().reindex(order)
        fig_, ax = plt.subplots(figsize=(7.5, 3.6))
        x = np.arange(len(order))
        present = [m for m in methods if m[0] in g.columns]
        width = 0.8 / len(present)
        for k, (m, label, color) in enumerate(present):
            ax.bar(x + (k - (len(present) - 1) / 2) * width, g[m].values, width, color=color, label=label)
        ax.set_xticks(x, order)
        ax.set_ylim(0.85, 1.005)
        ax.set_ylabel("energy quality (1 = ground state)")
        ax.set_title("VQE vs exact diagonalisation")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
        _save(fig_, os.path.join(fig, "e8_vqe.png"))
    p = os.path.join(res, "e8_scheduling.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        fig_, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
        order = [("exact", "exact (classical)", INK), ("simulated_annealing", "simulated annealing", AQUA),
                 ("greedy", "greedy", GRAY), ("qaoa_noiseless", "QAOA p=1, noiseless", BLUE),
                 ("qaoa_noisy", "QAOA p=1, NISQ noise", ORANGE),
                 ("random_sampling", "random, same samples as QAOA", MAGENTA),
                 ("random_sampling_1000", "random, 1000 samples", MAGENTA)]
        g = df.groupby(["method", "n"])[["ratio", "expected_ratio"]].mean().reset_index()
        for m, label, color in order:
            s_ = g[g.method == m]
            ax.plot(s_.n, s_.ratio, marker="o", markersize=4, color=color, label=label,
                    linestyle=":" if m == "random_sampling_1000" else "--" if m == "random_sampling" else "-")
        emu = df[df.method == "qaoa_ibm_emulator"]
        if len(emu):
            ax.scatter(emu.n, emu.ratio, color=YELLOW, zorder=5, label="QAOA, IBM emulator")
        ax.set_xlabel("tasks (qubits)")
        ax.set_ylabel("optimum / best cost found")
        ax.set_title("Best assignment found")
        for m, label, color in order[3:6]:
            s_ = g[g.method == m]
            ax2.plot(s_.n, s_.expected_ratio, marker="o", markersize=4, color=color, label=label,
                     linestyle="--" if m == "random_sampling" else "-")
        ax2.set_xlabel("tasks (qubits)")
        ax2.set_ylabel("optimum / mean cost of one sample")
        ax2.set_title("Quality of a single sample")
        handles, labels = ax.get_legend_handles_labels()
        fig_.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
        _save(fig_, os.path.join(fig, "e8_scheduling.png"))
