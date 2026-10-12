"""E2: does the quantum path actually help? Measured, not assumed.

For each quantum-capable workload we run the hybrid algorithm on simulators
(noiseless, uniformly noisy, and the IBM-device emulator) and the classical
baseline on the same instances, then fit the quality curves the cost model
uses (written to results/calibration.json).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..classical.solvers import maxcut_annealing, maxcut_greedy, timed
from ..executor import KERNEL_BANDWIDTH, kernel_dataset
from ..quantum.algorithms import (cut_value, magnetisation, maxcut_optimum, quantum_kernel, random_graph,
                                  run_qaoa, tfim_exact_magnetisation, tfim_trotter)
from ..quantum.backends import SIMULATOR_STACKS, NoiseSpec, get_backend
from ..quantum.circuit import Circuit

NOISE = NoiseSpec(p1=3e-4, p2=3e-3)  # roughly today's best superconducting 1q/2q error rates


def random_sampling_best(backend, n, edges, shots) -> float:
    c = Circuit(n)
    for q in range(n):
        c.h(q)
    counts = backend.run([c], shots).counts[0]
    return max(cut_value(b, edges) for b in counts)


def maxcut(quick: bool) -> pd.DataFrame:
    sizes = [4, 6, 8, 10] if quick else [4, 6, 8, 10, 12, 14]
    graphs = 2 if quick else 3
    rows = []
    variants = {"qaoa_noiseless": get_backend("qiskit"), "qaoa_noisy": get_backend("qiskit", NOISE)}
    emu = get_backend("ibm-emulator")
    for n in sizes:
        for g in range(graphs):
            edges = random_graph(n, 0.5, seed=100 * n + g)
            (opt, t_exact) = timed(maxcut_optimum, n, edges)
            (_, greedy), t_g = timed(maxcut_greedy, n, edges)
            (_, sa), t_sa = timed(maxcut_annealing, n, edges, 400, g)
            base = {"n": n, "graph": g, "edges": len(edges), "optimum": opt}
            rows += [{**base, "method": "exact", "ratio": 1.0, "time_s": t_exact, "kind": "classical"},
                     {**base, "method": "greedy", "ratio": greedy / opt, "time_s": t_g, "kind": "classical"},
                     {**base, "method": "simulated_annealing", "ratio": sa / opt, "time_s": t_sa, "kind": "classical"}]
            rnd = random_sampling_best(variants["qaoa_noiseless"], n, edges, 1000)
            rows.append({**base, "method": "random_sampling", "ratio": rnd / opt, "time_s": 0.0, "kind": "baseline"})
            for name, be in variants.items():
                t0 = time.perf_counter()
                r = run_qaoa(be, n, edges, p=1, shots=1000, maxiter=25, seed=g)
                rows.append({**base, "method": name, "ratio": r.best_cut / opt,
                             "expected_ratio": r.expected_cut / opt, "time_s": time.perf_counter() - t0,
                             "circuit_evals": r.n_circuit_evals, "kind": "hybrid"})
            if n <= (6 if quick else 8) and g == 0:
                t0 = time.perf_counter()
                r = run_qaoa(emu, n, edges, p=1, shots=1000, maxiter=15, seed=g)
                rows.append({**base, "method": "qaoa_ibm_emulator", "ratio": r.best_cut / opt,
                             "expected_ratio": r.expected_cut / opt, "time_s": time.perf_counter() - t0,
                             "circuit_evals": r.n_circuit_evals, "kind": "hybrid"})
    return pd.DataFrame(rows)


def qaoa_all_stacks(n: int = 8) -> pd.DataFrame:
    edges = random_graph(n, 0.5, seed=7)
    opt = maxcut_optimum(n, edges)
    rows = []
    for s in SIMULATOR_STACKS:
        t0 = time.perf_counter()
        r = run_qaoa(get_backend(s), n, edges, p=1, shots=1000, maxiter=20, seed=0)
        rows.append({"stack": s, "n": n, "best_ratio": r.best_cut / opt, "expected_ratio": r.expected_cut / opt,
                     "circuit_evals": r.n_circuit_evals, "time_s": time.perf_counter() - t0})
    return pd.DataFrame(rows)


def kernels(quick: bool) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    rows = []
    dims = [2, 3, 4] if quick else [2, 3, 4, 6]
    seeds = range(2 if quick else 4)
    q_clean, q_noisy = get_backend("qiskit"), get_backend("qiskit", NOISE)
    for d in dims:
        for seed in seeds:
            Xtr, Xte, ytr, yte = kernel_dataset(d, 40, 30, seed)
            base = {"features": d, "seed": seed, "n_train": 40, "n_test": 30}
            for name, model in [("rbf_svm", SVC(kernel="rbf", gamma="scale")),
                                ("logistic", LogisticRegression(max_iter=500))]:
                t0 = time.perf_counter()
                acc = model.fit(Xtr, ytr).score(Xte, yte)
                rows.append({**base, "method": name, "accuracy": acc, "time_s": time.perf_counter() - t0})
            bw = KERNEL_BANDWIDTH(d)
            for name, be in [("quantum_kernel_noiseless", q_clean), ("quantum_kernel_noisy", q_noisy)]:
                t0 = time.perf_counter()
                K, _ = quantum_kernel(be, Xtr * bw, shots=1000)
                Kt, _ = quantum_kernel(be, Xte * bw, Xtr * bw, shots=1000)
                acc = SVC(kernel="precomputed").fit(K, ytr).score(Kt, yte)
                rows.append({**base, "method": name, "accuracy": acc, "time_s": time.perf_counter() - t0})
    return pd.DataFrame(rows)


def tfim(quick: bool) -> pd.DataFrame:
    rows = []
    q_clean, q_noisy = get_backend("qiskit"), get_backend("qiskit", NOISE)
    emu = get_backend("ibm-emulator")
    for n in ([4, 6, 8] if quick else [4, 6, 8, 10, 12]):
        for steps in [2, 5, 10]:
            t0 = time.perf_counter()
            exact = tfim_exact_magnetisation(n, steps)
            rows.append({"n": n, "steps": steps, "method": "classical_statevector", "abs_error": 0.0,
                         "time_s": time.perf_counter() - t0})
            c = tfim_trotter(n, steps)
            backs = [("sim_noiseless", q_clean), ("sim_noisy", q_noisy)]
            if n <= 8:
                backs.append(("ibm_emulator", emu))
            for name, be in backs:
                r = be.run([c], 4000)
                rows.append({"n": n, "steps": steps, "method": name, "abs_error": abs(magnetisation(r.counts[0]) - exact),
                             "time_s": r.wall_time_s, "n_2q": c.n_2q})
    return pd.DataFrame(rows)


def fit_calibration(mc: pd.DataFrame, kn: pd.DataFrame) -> dict:
    exact = mc[mc.method == "exact"]
    # opt/|E| = 0.5 + c / sqrt(n * density); E2 graphs use density 0.5
    c = float((((exact.optimum / exact.edges) - 0.5) * np.sqrt(exact.n * 0.5)).mean())
    sa = mc[mc.method == "simulated_annealing"].groupby("n").ratio.mean()
    acc = kn.groupby("method").accuracy.mean()
    q = mc[mc.method == "qaoa_noiseless"]
    rm = (q.edges / 2) / q.optimum
    gain = ((q.expected_ratio - rm) / (1 - rm)).groupby(q.n).mean().clip(lower=1e-3)
    b, log_a = np.polyfit(np.log(gain.index.values), np.log(gain.values), 1)
    return {"qaoa_gain": {"a": float(np.exp(log_a)), "b": float(-b)},
            "opt_c": c,
            "sa_ratio": {"slope": 0.0, "at": 8, "value": float(sa.mean()), "floor": 0.95},
            "kernel_acc": {"quantum": float(acc["quantum_kernel_noiseless"]), "rbf": float(acc["rbf_svm"])}}


def run(quick: bool = False, out_dir: str = "results") -> dict[str, pd.DataFrame]:
    mc = maxcut(quick)
    kn = kernels(quick)
    mc.to_csv(f"{out_dir}/e2_maxcut.csv", index=False)
    tf = tfim(quick)
    st = qaoa_all_stacks(6 if quick else 8)
    cal = fit_calibration(mc, kn)
    with open(f"{out_dir}/calibration.json", "w") as f:
        json.dump(cal, f, indent=2)
    return {"e2_maxcut": mc, "e2_kernel": kn, "e2_tfim": tf, "e2_qaoa_stacks": st}
