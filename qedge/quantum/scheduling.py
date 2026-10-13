"""QAOA for edge/cloud task scheduling: Q-Edge's own placement problem, as a QUBO.

``n`` tasks must each go to the edge device (bit 0) or the cloud (bit 1). The cost
of an assignment, with spins s_i = 1 - 2 b_i (+1 edge, -1 cloud), is

    C(s) = alpha * (sum_i w_i s_i)^2              load imbalance between the two sides
         + sum_{(i,j)} c_ij * [s_i != s_j]         traffic between communicating tasks that are split
         + sum_i a_i * [s_i = -1]                  upload cost of sending task i's data to the cloud

Every term is non-negative, so ``optimum / cost`` is an approximation ratio in (0, 1].
Expanding gives an Ising model with couplings and fields, which QAOA handles the
same way as MaxCut: a cost layer of RZZ and RZ rotations, then an RX mixer. The
classical baselines are exact enumeration (up to ~20 tasks) and simulated annealing.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .backends import QuantumBackend
from .circuit import Circuit


@dataclass
class SchedulingInstance:
    n: int
    durations: list[float]                       # w_i
    comm: list[tuple[int, int, float]]           # (i, j, c_ij)
    upload: list[float]                          # a_i
    alpha: float

    # ---- objective ----------------------------------------------------------
    def cost(self, bits: str) -> float:
        s = [1 - 2 * int(b) for b in bits]
        load = sum(w * si for w, si in zip(self.durations, s))
        split = sum(c for i, j, c in self.comm if bits[i] != bits[j])
        up = sum(a for a, b in zip(self.upload, bits) if b == "1")
        return float(self.alpha * load * load + split + up)

    def all_costs(self) -> np.ndarray:
        """Cost of every assignment, index k = bits with task 0 as the most significant bit."""
        if self.n > 22:
            raise ValueError("too large to enumerate")
        idx = np.arange(2 ** self.n, dtype=np.int64)
        bits = ((idx[:, None] >> (self.n - 1 - np.arange(self.n))) & 1).astype(np.int8)
        s = 1 - 2 * bits
        load = s @ np.asarray(self.durations)
        cost = self.alpha * load ** 2 + bits @ np.asarray(self.upload)
        for i, j, c in self.comm:
            cost = cost + c * (bits[:, i] != bits[:, j])
        return cost

    def optimum(self) -> tuple[str, float]:
        costs = self.all_costs()
        k = int(np.argmin(costs))
        return format(k, f"0{self.n}b"), float(costs[k])

    # ---- Ising form ---------------------------------------------------------
    def ising(self) -> tuple[dict[tuple[int, int], float], list[float], float]:
        """C = const + sum J_ij s_i s_j + sum h_i s_i."""
        J: dict[tuple[int, int], float] = {}
        w = self.durations
        for i, j in itertools.combinations(range(self.n), 2):
            J[(i, j)] = 2 * self.alpha * w[i] * w[j]
        const = self.alpha * sum(x * x for x in w)
        for i, j, c in self.comm:          # c [s_i != s_j] = c/2 (1 - s_i s_j)
            J[(min(i, j), max(i, j))] -= c / 2
            const += c / 2
        h = [-a / 2 for a in self.upload]   # a [s = -1] = a/2 (1 - s)
        const += sum(self.upload) / 2
        return {k: v for k, v in J.items() if abs(v) > 1e-12}, h, const


def scheduling_instance(n: int, density: float = 0.3, seed: int = 0, alpha: float | None = None) -> SchedulingInstance:
    """Random instance: integer durations 1-9, communicating pairs with traffic 1-5,
    upload costs 0-3. ``alpha`` defaults to 1/n so the load term stays comparable with
    the traffic and upload terms as n grows."""
    rng = np.random.default_rng(seed)
    durations = [float(x) for x in rng.integers(1, 10, n)]
    comm = [(i, j, float(rng.integers(1, 6))) for i, j in itertools.combinations(range(n), 2) if rng.random() < density]
    upload = [float(x) for x in rng.integers(0, 4, n)]
    return SchedulingInstance(n, durations, comm, upload, alpha if alpha is not None else 1.0 / n)


# --------------------------------------------------------------------------
# QAOA
# --------------------------------------------------------------------------


def ising_qaoa_circuit(n: int, J: dict, h: list[float], gammas, betas, scale: float = 1.0) -> Circuit:
    """exp(-i gamma C) with C = sum J s s + sum h s, then the RX mixer; ``scale`` normalises
    the coefficients so one gamma range suits every instance."""
    c = Circuit(n, name=f"qaoa_ising_p{len(gammas)}")
    for q in range(n):
        c.h(q)
    for g, b in zip(gammas, betas):
        for (i, j), v in J.items():
            c.rzz(i, j, 2 * g * v / scale)
        for q, v in enumerate(h):
            if v:
                c.rz(q, 2 * g * v / scale)
        for q in range(n):
            c.rx(q, 2 * b)
    return c


@dataclass
class SchedulingResult:
    best_bits: str
    best_cost: float
    expected_cost: float
    params: list[float]
    n_circuit_evals: int
    n_jobs: int
    shots: int
    quantum_time_s: float
    classical_time_s: float
    history: list[float] = field(default_factory=list)
    circuit_profile: dict = field(default_factory=dict)


def run_scheduling_qaoa(backend: QuantumBackend, inst: SchedulingInstance, p: int = 1, shots: int = 1000,
                        maxiter: int = 25, seed: int = 0) -> SchedulingResult:
    """Same structure as ``run_qaoa`` for MaxCut: a coarse (gamma, beta) grid batched into
    one job, then COBYLA minimising the sampled mean cost, keeping the best sample seen."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    J, h, _ = inst.ising()
    scale = max([abs(v) for v in J.values()] + [abs(v) for v in h] + [1e-9])
    q_time, evals, jobs = 0.0, 0, 0
    history: list[float] = []
    best = ("", math.inf)

    def evaluate(batch):
        nonlocal q_time, evals, jobs, best
        circs = [ising_qaoa_circuit(inst.n, J, h, x[:p], x[p:], scale) for x in batch]
        res = backend.run(circs, shots)
        q_time += res.wall_time_s
        evals += len(circs)
        jobs += 1
        out = []
        for counts in res.counts:
            total = sum(counts.values())
            costs = {b: inst.cost(b) for b in counts}
            out.append(sum(costs[b] * v for b, v in counts.items()) / total)
            b = min(costs, key=costs.get)
            if costs[b] < best[1]:
                best = (b, costs[b])
        return out

    t_start = time.perf_counter()
    grid = [np.r_[[g] * p, [b] * p] for g in np.linspace(0.1, math.pi / 2, 4) for b in np.linspace(0.1, math.pi / 4, 3)]
    gvals = evaluate(grid)
    x0 = grid[int(np.argmin(gvals))] + rng.normal(0, 0.05, 2 * p)

    def objective(x):
        v = evaluate([x])[0]
        history.append(v)
        return v

    opt = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
    total = time.perf_counter() - t_start
    final = evaluate([opt.x])[0]
    prof = ising_qaoa_circuit(inst.n, J, h, opt.x[:p], opt.x[p:], scale).profile()
    return SchedulingResult(best[0], best[1], final, list(map(float, opt.x)), evals, jobs, shots,
                            q_time, max(0.0, total - q_time), history, prof)


# --------------------------------------------------------------------------
# Classical baselines
# --------------------------------------------------------------------------


def scheduling_annealing(inst: SchedulingInstance, sweeps: int = 1000, seed: int = 0) -> tuple[str, float]:
    """Single-flip simulated annealing with a geometric schedule. The start temperature is
    the typical cost change of one flip at a random assignment, so it adapts to any instance."""
    rng = np.random.default_rng(seed)
    bits = list(rng.integers(0, 2, inst.n))
    as_str = lambda b: "".join(map(str, b))  # noqa: E731
    cur = inst.cost(as_str(bits))
    deltas = []
    for v in range(inst.n):
        bits[v] ^= 1
        deltas.append(abs(inst.cost(as_str(bits)) - cur))
        bits[v] ^= 1
    t_start = max(float(np.mean(deltas)), 1e-6)
    t_end = 1e-3 * t_start
    best, best_bits = cur, bits.copy()
    for t in range(sweeps):
        T = t_start * (t_end / t_start) ** (t / max(sweeps - 1, 1))
        for v in rng.permutation(inst.n):
            bits[v] ^= 1
            new = inst.cost(as_str(bits))
            if new <= cur or rng.random() < math.exp((cur - new) / T):
                cur = new
                if cur < best:
                    best, best_bits = cur, bits.copy()
            else:
                bits[v] ^= 1
    return as_str(best_bits), float(best)


def scheduling_greedy(inst: SchedulingInstance) -> tuple[str, float]:
    """Longest task first onto whichever side currently costs less to add it to."""
    bits = ["0"] * inst.n
    placed: list[int] = []
    for i in sorted(range(inst.n), key=lambda k: -inst.durations[k]):
        best = None
        for b in "01":
            bits[i] = b
            sub = SchedulingInstance(inst.n, [inst.durations[k] if (k in placed or k == i) else 0.0
                                              for k in range(inst.n)],
                                     [(a, c, w) for a, c, w in inst.comm if a in placed + [i] and c in placed + [i]],
                                     [inst.upload[k] if (k in placed or k == i) else 0.0 for k in range(inst.n)],
                                     inst.alpha)
            v = sub.cost("".join(bits))
            if best is None or v < best[1]:
                best = (b, v)
        bits[i] = best[0]
        placed.append(i)
    s = "".join(bits)
    return s, inst.cost(s)
