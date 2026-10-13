"""Variational Quantum Eigensolver (VQE), written once over the neutral circuit IR.

VQE finds the lowest energy (ground state) of a Hamiltonian. A shallow trainable
circuit (the ansatz) prepares a trial state; the quantum backend estimates its
energy from measurements; a classical optimiser (COBYLA) adjusts the angles.
Only the energy estimates touch the quantum backend, which is the part the
orchestrator places.

Two Hamiltonians are provided:

  * ``h2_hamiltonian()``: the hydrogen molecule at its equilibrium bond length
    (0.735 A, STO-3G basis, parity mapping with two-qubit reduction). These are
    the standard coefficients used in Qiskit's VQE tutorials; the exact ground
    energy is -1.85728 Ha electronic, -1.13731 Ha with nuclear repulsion, which
    matches full configuration interaction.
  * ``tfim_hamiltonian(n)``: the transverse-field Ising chain
    H = -J sum Z_i Z_{i+1} - h sum X_i, the same model the Trotter workload evolves.

A Hamiltonian is a list of ``(coefficient, {qubit: "X" | "Y" | "Z"})`` terms; an
empty dict is the identity. Qubit 0 is the leftmost bit, as everywhere in Q-Edge.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backends import QuantumBackend
from .circuit import Circuit

Term = tuple[float, dict[int, str]]
Hamiltonian = list[Term]

H2_NUCLEAR_REPULSION = 0.7199689944489797  # Ha, at 0.735 A
CHEMICAL_ACCURACY = 1.6e-3                 # Ha (1 kcal/mol)


def h2_hamiltonian() -> Hamiltonian:
    return [(-1.052373245772859, {}),
            (0.39793742484318045, {0: "Z"}),
            (-0.39793742484318045, {1: "Z"}),
            (-0.01128010425623538, {0: "Z", 1: "Z"}),
            (0.18093119978423156, {0: "X", 1: "X"})]


def tfim_hamiltonian(n: int, J: float = 1.0, h: float = 1.0) -> Hamiltonian:
    return ([(-J, {i: "Z", i + 1: "Z"}) for i in range(n - 1)]
            + [(-h, {i: "X"}) for i in range(n)])


def hamiltonian_for(model: str, n: int | None = None) -> tuple[Hamiltonian, int]:
    if model == "h2":
        return h2_hamiltonian(), 2
    if model == "tfim":
        if not n or n < 2:
            raise ValueError("tfim needs n >= 2")
        return tfim_hamiltonian(n), n
    raise ValueError(f"unknown model {model!r}; choose h2 or tfim")


# --------------------------------------------------------------------------
# Measurement: group terms that can share one basis (qubit-wise commuting)
# --------------------------------------------------------------------------


def measurement_groups(H: Hamiltonian) -> list[dict[int, str]]:
    """Greedy qubit-wise-commuting grouping; each group is one circuit per energy estimate.
    The TFIM needs two (all-Z, all-X); H2 needs two (Z-type terms, XX)."""
    groups: list[dict[int, str]] = []
    for _, ops in H:
        if not ops:
            continue
        for g in groups:
            if all(g.get(q, p) == p for q, p in ops.items()):
                g.update(ops)
                break
        else:
            groups.append(dict(ops))
    return groups


def with_basis(c: Circuit, basis: dict[int, str]) -> Circuit:
    out = Circuit(c.n_qubits, list(c.gates), name=c.name)
    for q, p in sorted(basis.items()):
        if p == "X":
            out.h(q)
        elif p == "Y":
            out.sdg(q)
            out.h(q)
    return out


def term_value(bits: str, ops: dict[int, str]) -> float:
    v = 1.0
    for q in ops:
        v *= 1 - 2 * int(bits[q])
    return v


def energy_from_counts(H: Hamiltonian, groups: list[dict[int, str]], counts: list[dict[str, int]]) -> float:
    e = 0.0
    for coeff, ops in H:
        if not ops:
            e += coeff
            continue
        k = next(i for i, g in enumerate(groups) if all(g.get(q) == p for q, p in ops.items()))
        c = counts[k]
        total = sum(c.values())
        e += coeff * sum(term_value(b, ops) * v for b, v in c.items()) / total
    return float(e)


# --------------------------------------------------------------------------
# Ansatz: hardware-efficient RY + CX ladder (real amplitudes)
# --------------------------------------------------------------------------


def n_params(n: int, layers: int) -> int:
    return n * (layers + 1)


def ansatz(n: int, params, layers: int = 1) -> Circuit:
    """RY on every qubit, then ``layers`` x (CX ladder, RY on every qubit).
    Both Hamiltonians here have real ground states, so real amplitudes suffice."""
    params = np.asarray(params, float)
    if len(params) != n_params(n, layers):
        raise ValueError(f"expected {n_params(n, layers)} parameters, got {len(params)}")
    c = Circuit(n, name=f"vqe_ry{layers}")
    k = 0
    for q in range(n):
        c.ry(q, params[k])
        k += 1
    for _ in range(layers):
        for q in range(n - 1):
            c.cx(q, q + 1)
        for q in range(n):
            c.ry(q, params[k])
            k += 1
    return c


# --------------------------------------------------------------------------
# Classical references
# --------------------------------------------------------------------------

_PAULI = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
          "Z": np.array([[1, 0], [0, -1]], complex)}


def hamiltonian_matrix(H: Hamiltonian, n: int):
    """Sparse matrix with qubit 0 as the most significant bit (matches the bitstrings)."""
    from scipy import sparse
    dim = 2 ** n
    M = sparse.csr_matrix((dim, dim), dtype=complex)
    eye = sparse.identity(2, format="csr", dtype=complex)
    for coeff, ops in H:
        t = None
        for q in range(n):
            f = sparse.csr_matrix(_PAULI[ops[q]]) if q in ops else eye
            t = f if t is None else sparse.kron(t, f, format="csr")
        M = M + coeff * t
    return M


def spectrum_bounds(H: Hamiltonian, n: int) -> tuple[float, float]:
    """(ground energy, highest energy): exact diagonalisation, dense up to 10 qubits, Lanczos beyond."""
    M = hamiltonian_matrix(H, n)
    if n <= 10:
        ev = np.linalg.eigvalsh(M.toarray())
        return float(ev[0]), float(ev[-1])
    from scipy.sparse.linalg import eigsh
    lo = eigsh(M, k=1, which="SA", return_eigenvectors=False)[0]
    hi = eigsh(M, k=1, which="LA", return_eigenvectors=False)[0]
    return float(np.real(lo)), float(np.real(hi))


def energy_quality(e: float, e0: float, emax: float) -> float:
    """1 at the ground state, 0 at the top of the spectrum. A fully mixed (pure noise) state
    sits near 0.5 for both models here, so this reads like an accuracy."""
    return float(np.clip(1 - (e - e0) / (emax - e0), 0.0, 1.0))


def exact_energy(H: Hamiltonian, params, n: int, layers: int) -> float:
    """Noiseless expectation of the ansatz state (state vector), for tests and calibration."""
    psi = np.zeros(2 ** n, complex)
    psi[0] = 1
    c = ansatz(n, params, layers)
    for g in c.gates:
        if g.name == "ry":
            t = g.params[0] / 2
            u = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            psi = psi.reshape(2 ** g.qubits[0], 2, -1)
            psi = np.einsum("ab,ibj->iaj", u, psi).reshape(-1)
        elif g.name == "cx":
            a, b = g.qubits
            idx = np.arange(2 ** n)
            ctrl = (idx >> (n - 1 - a)) & 1
            flipped = np.where(ctrl == 1, idx ^ (1 << (n - 1 - b)), idx)
            psi = psi[flipped]
    M = hamiltonian_matrix(H, n)
    return float(np.real(np.vdot(psi, M @ psi)))


# --------------------------------------------------------------------------
# The hybrid loop
# --------------------------------------------------------------------------


@dataclass
class VQEResult:
    energy: float                 # final estimate from shots
    best_energy: float            # lowest estimate seen during the run
    params: list[float]
    n_circuit_evals: int
    n_jobs: int
    shots: int
    quantum_time_s: float
    classical_time_s: float
    history: list[float] = field(default_factory=list)
    circuit_profile: dict = field(default_factory=dict)


def _rotosolve_angle(e_m: float, e_0: float, e_p: float, theta: float) -> float:
    """Energy is a + b cos(t) + c sin(t) in any single RY angle t. Given it at theta - pi/2,
    theta and theta + pi/2, return the angle that minimises it exactly."""
    A = np.array([[np.cos(t), np.sin(t), 1.0] for t in (theta - np.pi / 2, theta, theta + np.pi / 2)])
    b, c, _ = np.linalg.solve(A, [e_m, e_0, e_p])
    return float(np.arctan2(-c, -b))


def run_vqe(backend: QuantumBackend, H: Hamiltonian, n: int, layers: int = 1, shots: int = 1000,
            sweeps: int = 3, n_starts: int = 4, seed: int = 0, optimizer: str = "rotosolve",
            maxiter: int = 60) -> VQEResult:
    """Random starts batched into one job (one queue wait on hardware), then the classical
    optimiser from the best start:

    * ``rotosolve`` (default): sets one angle at a time to its exact minimiser, using two new
      energy estimates per angle sent as one job (Ostaszewski et al., Quantum 5, 391, 2021).
      It needs no step size and copes with shot noise far better than COBYLA here.
    * ``cobyla``: one job per energy evaluation, ``maxiter`` evaluations.
    """
    rng = np.random.default_rng(seed)
    groups = measurement_groups(H)
    k = n_params(n, layers)
    q_time, evals, jobs = 0.0, 0, 0
    history: list[float] = []
    best_seen = np.inf

    def energies(batch):
        nonlocal q_time, evals, jobs, best_seen
        circs = [with_basis(ansatz(n, x, layers), g) for x in batch for g in groups]
        res = backend.run(circs, shots)
        q_time += res.wall_time_s
        evals += len(circs)
        jobs += 1
        m = len(groups)
        out = [energy_from_counts(H, groups, res.counts[i * m:(i + 1) * m]) for i in range(len(batch))]
        best_seen = min(best_seen, *out)
        return out

    t_start = time.perf_counter()
    starts = [rng.uniform(-np.pi, np.pi, k) * 0.5 for _ in range(n_starts)]
    e_starts = energies(starts)
    x = np.array(starts[int(np.argmin(e_starts))])
    e_cur = min(e_starts)
    if optimizer == "rotosolve":
        for _ in range(sweeps):
            for j in range(k):
                xm, xp = x.copy(), x.copy()
                xm[j] -= np.pi / 2
                xp[j] += np.pi / 2
                e_m, e_p = energies([xm, xp])
                x[j] = _rotosolve_angle(e_m, e_cur, e_p, x[j])
                # The closed-form minimum is a prediction; the next step measures it again.
                e_cur = min(e_m, e_cur, e_p)
                history.append(e_cur)
    elif optimizer == "cobyla":
        from scipy.optimize import minimize

        def objective(z):
            v = energies([z])[0]
            history.append(v)
            return v

        x = minimize(objective, x, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.4}).x
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    total = time.perf_counter() - t_start
    final = energies([x])[0]
    prof = with_basis(ansatz(n, x, layers), groups[-1]).profile()
    return VQEResult(final, float(best_seen), list(map(float, x)), evals, jobs, shots,
                     q_time, max(0.0, total - q_time), history, prof)


def vqe_budget(n_params_: int, n_groups: int, sweeps: int = 3, n_starts: int = 4) -> tuple[int, int]:
    """(circuits, jobs) one rotosolve run sends; used by the hardware guard and the cost model."""
    jobs = 1 + sweeps * n_params_ + 1
    circuits = n_groups * (n_starts + 2 * sweeps * n_params_ + 1)
    return circuits, jobs
