"""E1: the same circuits on every quantum stack.

Checks that Qiskit, PennyLane, Cirq and Braket give the same answer through
the shared interface (total-variation distance to the exact distribution) and
measures how fast each simulator is as circuits grow.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..quantum.algorithms import qaoa_circuit, random_graph, tfim_trotter, zz_feature_map
from ..quantum.backends import SIMULATOR_STACKS, NoiseSpec, get_backend
from ..quantum.backends.qiskit_backend import to_qiskit
from ..quantum.circuit import ghz, probabilities, total_variation


def exact_probs(c) -> dict[str, float]:
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(to_qiskit(c, measure=False))
    p = sv.probabilities_dict()
    return {k[::-1]: v for k, v in p.items() if v > 1e-12}


def mean_z(p: dict[str, float]) -> float:
    """Average <Z> over qubits: an observable that 4000 shots estimate well even at 20 qubits."""
    return sum(v * (1 - 2 * k.count("1") / len(k)) for k, v in p.items())


def families(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "ghz": ghz(n),
        "qaoa_p1": qaoa_circuit(n, random_graph(n, 0.5, seed), [0.6], [0.4]),
        "zz_feature_map": zz_feature_map(rng.uniform(0, np.pi, n), reps=2),
        "tfim_trotter": tfim_trotter(n, 4),
    }


def run(quick: bool = False, shots: int = 4000) -> pd.DataFrame:
    sizes = [2, 4, 6, 8, 10, 12] if quick else [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    rows = []
    backends = {s: get_backend(s) for s in SIMULATOR_STACKS}
    for n in sizes:
        for fam, c in families(n).items():
            ref = exact_probs(c)
            for name, be in backends.items():
                times = []
                for rep in range(2):
                    t0 = time.perf_counter()
                    res = be.run([c], shots)
                    times.append(time.perf_counter() - t0)
                probs = probabilities(res.counts[0])
                tv = total_variation(ref, probs)
                rows.append({"experiment": "noiseless", "stack": name, "family": fam, "n_qubits": n,
                             "depth": c.depth(), "n_2q": c.n_2q, "time_s": min(times), "tv_to_exact": tv,
                             "mean_z_error": abs(mean_z(probs) - mean_z(ref)), "shots": shots})
    # Noisy: identical depolarising model translated into each SDK's own noise API.
    noise = NoiseSpec(p1=1e-3, p2=1e-2)
    noisy = {s: get_backend(s, noise) for s in SIMULATOR_STACKS}
    for n in ([2, 4, 6] if quick else [2, 4, 6, 8]):
        c = families(n)["qaoa_p1"]
        dists = {}
        for name, be in noisy.items():
            t0 = time.perf_counter()
            res = be.run([c], shots)
            dists[name] = probabilities(res.counts[0])
            rows.append({"experiment": "noisy", "stack": name, "family": "qaoa_p1", "n_qubits": n,
                         "depth": c.depth(), "n_2q": c.n_2q, "time_s": time.perf_counter() - t0,
                         "tv_to_exact": total_variation(exact_probs(c), dists[name]), "shots": shots})
        for name in noisy:
            rows[-len(noisy) + list(noisy).index(name)]["tv_to_qiskit_noisy"] = total_variation(dists["qiskit"], dists[name])
    return pd.DataFrame(rows)
