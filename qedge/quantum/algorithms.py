"""Hybrid quantum-classical algorithms, written once over the neutral circuit IR.

Each algorithm keeps the classical part (optimiser, SVM, post-processing) on a
classical resource and sends only the circuit evaluations to the quantum
backend. That split is what the orchestrator places.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .backends import QuantumBackend
from .circuit import Circuit

# --------------------------------------------------------------------------
# QAOA for weighted MaxCut
# --------------------------------------------------------------------------


def qaoa_circuit(n: int, edges: list[tuple[int, int, float]], gammas, betas) -> Circuit:
    c = Circuit(n, name=f"qaoa_p{len(gammas)}")
    for q in range(n):
        c.h(q)
    for g, b in zip(gammas, betas):
        for i, j, w in edges:
            c.rzz(i, j, 2 * g * w)
        for q in range(n):
            c.rx(q, 2 * b)
    return c


def cut_value(bits: str, edges) -> float:
    return sum(w for i, j, w in edges if bits[i] != bits[j])


def expected_cut(counts: dict[str, int], edges) -> float:
    total = sum(counts.values())
    return sum(cut_value(b, edges) * v for b, v in counts.items()) / total


@dataclass
class QAOAResult:
    best_bits: str
    best_cut: float
    expected_cut: float
    params: list[float]
    n_circuit_evals: int
    n_jobs: int
    shots: int
    quantum_time_s: float
    classical_time_s: float
    history: list[float] = field(default_factory=list)
    circuit_profile: dict = field(default_factory=dict)


def run_qaoa(backend: QuantumBackend, n: int, edges, p: int = 1, shots: int = 512,
             maxiter: int = 25, seed: int = 0) -> QAOAResult:
    """Grid-seeded COBYLA loop; one backend job per objective evaluation."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    q_time = 0.0
    evals = 0
    jobs = 0
    history: list[float] = []
    best = ("", -1.0)

    def evaluate(batch_params):
        nonlocal q_time, evals, jobs, best
        circs = [qaoa_circuit(n, edges, x[:p], x[p:]) for x in batch_params]
        res = backend.run(circs, shots)
        q_time += res.wall_time_s
        evals += len(circs)
        jobs += 1
        vals = []
        for counts in res.counts:
            vals.append(expected_cut(counts, edges))
            b = max(counts, key=lambda s: cut_value(s, edges))
            cv = cut_value(b, edges)
            if cv > best[1]:
                best = (b, cv)
        return vals

    t_start = time.perf_counter()
    # Coarse grid batched into one job: cheap on hardware (one queue wait).
    grid = [np.r_[[g] * p, [b] * p] for g in np.linspace(0.1, math.pi / 2, 4) for b in np.linspace(0.1, math.pi / 4, 3)]
    gvals = evaluate(grid)
    x0 = grid[int(np.argmax(gvals))] + rng.normal(0, 0.05, 2 * p)

    def objective(x):
        v = evaluate([x])[0]
        history.append(v)
        return -v

    opt = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
    total = time.perf_counter() - t_start
    final = evaluate([opt.x])[0]
    prof = qaoa_circuit(n, edges, opt.x[:p], opt.x[p:]).profile()
    return QAOAResult(best[0], best[1], final, list(map(float, opt.x)), evals, jobs, shots,
                      q_time, max(0.0, total - q_time), history, prof)


# --------------------------------------------------------------------------
# Quantum kernel (ZZ feature map fidelity kernel)
# --------------------------------------------------------------------------


def zz_feature_map(x: np.ndarray, reps: int = 2) -> Circuit:
    n = len(x)
    c = Circuit(n, name="zz_map")
    for _ in range(reps):
        for q in range(n):
            c.h(q)
            c.rz(q, 2 * x[q])
        for i in range(n - 1):
            c.rzz(i, i + 1, 2 * (math.pi - x[i]) * (math.pi - x[i + 1]))
    return c


def kernel_circuit(x, y, reps: int = 2) -> Circuit:
    c = zz_feature_map(x, reps)
    c.compose(zz_feature_map(y, reps).inverse())
    return c


def quantum_kernel(backend: QuantumBackend, A: np.ndarray, B: np.ndarray | None = None,
                   shots: int = 512, reps: int = 2, chunk: int = 256) -> tuple[np.ndarray, dict]:
    """K[i,j] = |<phi(b_j)|phi(a_i)>|^2 estimated as P(all zeros)."""
    sym = B is None
    B = A if sym else B
    pairs = [(i, j) for i in range(len(A)) for j in range(len(B)) if not sym or j > i]
    K = np.eye(len(A)) if sym else np.zeros((len(A), len(B)))
    q_time, jobs = 0.0, 0
    zero = "0" * A.shape[1]
    for k in range(0, len(pairs), chunk):
        part = pairs[k:k + chunk]
        res = backend.run([kernel_circuit(A[i], B[j], reps) for i, j in part], shots)
        q_time += res.wall_time_s
        jobs += 1
        for (i, j), counts in zip(part, res.counts):
            K[i, j] = counts.get(zero, 0) / shots
            if sym:
                K[j, i] = K[i, j]
    prof = kernel_circuit(A[0], B[0], reps).profile()
    return K, {"n_circuits": len(pairs), "jobs": jobs, "quantum_time_s": q_time, "circuit_profile": prof}


# --------------------------------------------------------------------------
# Quantum-native workload: transverse-field Ising dynamics
# --------------------------------------------------------------------------


def tfim_trotter(n: int, steps: int, dt: float = 0.25, J: float = 1.0, h: float = 1.0) -> Circuit:
    c = Circuit(n, name=f"tfim{n}x{steps}")
    for _ in range(steps):
        for i in range(n - 1):
            c.rzz(i, i + 1, -2 * J * dt)
        for q in range(n):
            c.rx(q, -2 * h * dt)
    return c


def magnetisation(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    return sum(v * np.mean([1 - 2 * int(b) for b in s]) for s, v in counts.items()) / total


def tfim_exact_magnetisation(n: int, steps: int, dt: float = 0.25, J: float = 1.0, h: float = 1.0) -> float:
    """Classical reference: same Trotter circuit on a dense statevector (cost ~2^n)."""
    dim = 2 ** n
    z = np.array([[1 - 2 * ((k >> (n - 1 - q)) & 1) for q in range(n)] for k in range(dim)])
    psi = np.zeros(dim, complex)
    psi[0] = 1
    zz_phase = np.exp(1j * J * dt * sum(z[:, i] * z[:, i + 1] for i in range(n - 1)))
    cx, sx = math.cos(h * dt), 1j * math.sin(h * dt)
    for _ in range(steps):
        psi = psi * zz_phase
        for q in range(n):
            psi = psi.reshape(2 ** q, 2, -1)
            a, b = psi[:, 0, :].copy(), psi[:, 1, :].copy()
            psi[:, 0, :], psi[:, 1, :] = cx * a + sx * b, sx * a + cx * b
            psi = psi.reshape(-1)
    probs = np.abs(psi) ** 2
    return float(probs @ z.mean(axis=1))


def maxcut_optimum(n: int, edges) -> float:
    """Exact optimum by enumeration (vectorised); fine up to ~22 nodes."""
    if n > 24:
        raise ValueError("too large for brute force")
    idx = np.arange(2 ** n, dtype=np.int64)
    best = np.zeros(2 ** n)
    for i, j, w in edges:
        best += w * (((idx >> (n - 1 - i)) & 1) != ((idx >> (n - 1 - j)) & 1))
    return float(best.max())


def random_graph(n: int, p: float = 0.5, seed: int = 0, weighted: bool = False) -> list[tuple[int, int, float]]:
    rng = np.random.default_rng(seed)
    edges = [(i, j, float(rng.uniform(0.5, 1.5)) if weighted else 1.0)
             for i, j in itertools.combinations(range(n), 2) if rng.random() < p]
    if not edges:
        edges = [(0, 1, 1.0)]
    return edges
