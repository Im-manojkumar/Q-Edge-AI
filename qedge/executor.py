"""Runs real payloads for each workload kind, classically or hybrid.

Classical work runs on this machine (it plays the role of whichever classical
tier was chosen; network transfer for cloud tiers is added from the model,
not slept). Quantum work runs on a real quantum stack: a local simulator from
any of the four SDKs, the IBM device emulator, or real IBM / Braket hardware
when credentials are configured.

``run(w, q, cancel=token)`` hands the quantum path a backend bound to a
``CancelToken``, so race mode can stop it: every circuit job checks the token
first, which ends a QAOA loop or a chunked kernel at its next job.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .classical.solvers import maxcut_annealing
from .quantum.algorithms import (magnetisation, maxcut_optimum, quantum_kernel, random_graph, run_qaoa,
                                 tfim_exact_magnetisation, tfim_trotter)
from .quantum.backends import QuantumBackend, get_backend
from .quantum.backends.base import CancellableBackend, CancelToken

HW_MAX_CIRCUITS = int(os.environ.get("QEDGE_HW_MAX_CIRCUITS", "60"))
HW_MAX_SHOTS = int(os.environ.get("QEDGE_HW_MAX_SHOTS", "2000"))


def KERNEL_BANDWIDTH(d: int) -> float:
    """Feature-map bandwidth 1/d (Shaydulin & Wild, 2022): keeps kernel values away from 0."""
    return 1.0 / d


@dataclass
class ExecResult:
    kind: str
    mode: str                  # "classical" | "hybrid"
    method: str
    quality: float
    quality_metric: str
    compute_s: float
    quantum_s: float = 0.0
    backend: str | None = None
    fallback_from: str | None = None
    details: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _digits():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    return train_test_split(X / 16.0, y, test_size=0.3, random_state=0)


@lru_cache(maxsize=1)
def _pretrained_mlp():
    from sklearn.neural_network import MLPClassifier
    Xtr, Xte, ytr, yte = _digits()
    return MLPClassifier((64,), max_iter=300, random_state=0).fit(Xtr, ytr)


def kernel_dataset(d: int, n_train: int, n_test: int, seed: int = 0):
    """Breast-cancer data reduced to d features (PCA) and scaled to [0, pi]."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[: n_train + n_test]
    Z = PCA(d, random_state=0).fit_transform((X - X.mean(0)) / X.std(0))[idx]
    Z = MinMaxScaler((0, np.pi)).fit_transform(Z)
    return Z[:n_train], Z[n_train:], y[idx][:n_train], y[idx][n_train:]


def representative_circuit(w):
    """One circuit with the structure this workload runs on a QPU. Angles do not
    change qubit placement, so this is what layout and device choice are made on."""
    from .quantum.algorithms import kernel_circuit, qaoa_circuit
    p = w.params
    if w.kind == "optimization":
        n, reps = int(p["n"]), int(p.get("p", 1))
        return qaoa_circuit(n, random_graph(n, p.get("density", 0.3), seed=w.id), [0.4] * reps, [0.3] * reps)
    if w.kind == "kernel_classification":
        d = int(p["features"])
        x = np.linspace(0.1, 1.0, d) * KERNEL_BANDWIDTH(d) * np.pi
        return kernel_circuit(x, x[::-1])
    if w.kind == "quantum_simulation":
        return tfim_trotter(int(p["n"]), int(p["steps"]))
    raise ValueError(f"{w.kind} has no quantum circuit")


class Executor:
    def __init__(self, sim_stack: str = "qiskit", qpu_stack: str = "ibm"):
        self.sim_stack = sim_stack
        self.qpu_stack = qpu_stack
        self._backends: dict[str, QuantumBackend] = {}
        self._lock = threading.Lock()  # race mode calls backend_for from worker threads

    def backend_for(self, quantum_resource: str, cancel: CancelToken | None = None) -> QuantumBackend:
        name = self.sim_stack if quantum_resource == "quantum_sim" else self.qpu_stack
        with self._lock:
            if name not in self._backends:
                self._backends[name] = get_backend(name)
            be = self._backends[name]
        # The shared backend stays unbound; the token only governs this call's work.
        return CancellableBackend(be, cancel) if cancel is not None else be

    # ------------------------------------------------------------------
    def run(self, w, quantum_resource: str | None = None, cancel: CancelToken | None = None) -> ExecResult:
        """Execute ``w`` classically (``quantum_resource=None``) or hybrid. With
        ``cancel``, the quantum path raises ``Cancelled`` at its next circuit job
        once the token fires."""
        fn = getattr(self, f"_{w.kind}")
        return fn(w, quantum_resource, cancel)

    def _guard(self, be: QuantumBackend, n_circ: int, shots: int):
        if be.hardware and (n_circ > HW_MAX_CIRCUITS or shots > HW_MAX_SHOTS):
            raise RuntimeError(f"refusing {n_circ} circuits x {shots} shots on real hardware "
                               f"(limits QEDGE_HW_MAX_CIRCUITS={HW_MAX_CIRCUITS}, QEDGE_HW_MAX_SHOTS={HW_MAX_SHOTS})")

    def _inference(self, w, q, cancel=None):
        Xtr, Xte, ytr, yte = _digits()
        batch = int(w.params.get("batch", 8))
        rng = np.random.default_rng(w.id)
        idx = rng.integers(0, len(Xte), batch)
        model = _pretrained_mlp()
        t0 = time.perf_counter()
        pred = model.predict(Xte[idx])
        dt = time.perf_counter() - t0
        return ExecResult("inference", "classical", "mlp", float((pred == yte[idx]).mean()), "accuracy", dt,
                          details={"batch": batch})

    def _training(self, w, q, cancel=None):
        from sklearn.neural_network import MLPClassifier
        Xtr, Xte, ytr, yte = _digits()
        n = min(int(w.params.get("samples", 800)), len(Xtr))
        t0 = time.perf_counter()
        m = MLPClassifier((64,), max_iter=max(20, 20 * int(w.params.get("epochs", 1))), random_state=w.id)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xtr[:n], ytr[:n])
        dt = time.perf_counter() - t0
        return ExecResult("training", "classical", "mlp-sgd", float(m.score(Xte, yte)), "test accuracy", dt,
                          details={"samples": n})

    def _optimization(self, w, q, cancel=None):
        n = int(w.params["n"])
        edges = random_graph(n, w.params.get("density", 0.3), seed=w.id)
        ref = maxcut_optimum(n, edges) if n <= 20 else maxcut_annealing(n, edges, sweeps=3000, seed=99)[1]
        if q is None:
            t0 = time.perf_counter()
            if n <= 20:
                val, method = maxcut_optimum(n, edges), "exact enumeration"
            else:
                val, method = maxcut_annealing(n, edges)[1], "simulated annealing"
            dt = time.perf_counter() - t0
            return ExecResult("optimization", "classical", method, val / ref, "approximation ratio", dt,
                              details={"n": n, "edges": len(edges), "cut": val, "reference": ref})
        be = self.backend_for(q, cancel)
        iters = 8 if be.hardware else 25
        shots = 512 if be.hardware else 1000
        self._guard(be, 12 + iters + 1, shots)
        t0 = time.perf_counter()
        r = run_qaoa(be, n, edges, p=int(w.params.get("p", 1)), shots=shots, maxiter=iters, seed=w.id)
        dt = time.perf_counter() - t0
        return ExecResult("optimization", "hybrid", "QAOA", r.best_cut / ref, "approximation ratio",
                          r.classical_time_s, r.quantum_time_s, be.name, getattr(be, "fallback_from", None),
                          {"n": n, "edges": len(edges), "cut": r.best_cut, "expected_cut": r.expected_cut,
                           "reference": ref, "circuit_evals": r.n_circuit_evals, "jobs": r.n_jobs, "wall_s": dt})

    def _kernel_classification(self, w, q, cancel=None):
        from sklearn.svm import SVC
        d = int(w.params["features"])
        n_train, n_test = int(w.params["n_train"]), int(w.params["n_test"])
        Xtr, Xte, ytr, yte = kernel_dataset(d, n_train, n_test, seed=w.id)
        if q is None:
            t0 = time.perf_counter()
            acc = SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xtr, ytr).score(Xte, yte)
            return ExecResult("kernel_classification", "classical", "RBF-SVM", float(acc), "test accuracy",
                              time.perf_counter() - t0, details={"features": d, "n_train": n_train})
        be = self.backend_for(q, cancel)
        n_circ = n_train * (n_train - 1) // 2 + n_test * n_train
        self._guard(be, n_circ, 500)
        bw = KERNEL_BANDWIDTH(d)  # narrower feature map avoids an all-diagonal kernel
        Ktr, m1 = quantum_kernel(be, Xtr * bw, shots=500)
        Kte, m2 = quantum_kernel(be, Xte * bw, Xtr * bw, shots=500)
        t0 = time.perf_counter()
        acc = SVC(kernel="precomputed", C=1.0).fit(Ktr, ytr).score(Kte, yte)
        return ExecResult("kernel_classification", "hybrid", "quantum-kernel SVM", float(acc), "test accuracy",
                          time.perf_counter() - t0, m1["quantum_time_s"] + m2["quantum_time_s"], be.name,
                          getattr(be, "fallback_from", None),
                          {"features": d, "n_train": n_train, "circuits": m1["n_circuits"] + m2["n_circuits"]})

    def _quantum_simulation(self, w, q, cancel=None):
        n, steps = int(w.params["n"]), int(w.params["steps"])
        if n > 20:
            raise ValueError("live execution keeps quantum_simulation at <= 20 qubits so it can be verified exactly")
        t0 = time.perf_counter()
        exact = tfim_exact_magnetisation(n, steps)
        t_exact = time.perf_counter() - t0
        if q is None:
            return ExecResult("quantum_simulation", "classical", "state vector", 1.0, "1 - |error| in <Z>",
                              t_exact, details={"n": n, "steps": steps, "magnetisation": exact})
        be = self.backend_for(q, cancel)
        self._guard(be, 1, 4000)
        r = be.run([tfim_trotter(n, steps)], shots=4000)
        m = magnetisation(r.counts[0])
        return ExecResult("quantum_simulation", "hybrid", "Trotter circuit", float(max(0.0, 1 - abs(m - exact))),
                          "1 - |error| in <Z>", 0.0, r.wall_time_s, be.name, getattr(be, "fallback_from", None),
                          {"n": n, "steps": steps, "magnetisation": m, "exact": exact})
