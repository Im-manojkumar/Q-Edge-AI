"""Circuit cutting: split a circuit's two-qubit gates across a bipartition, run the
two halves separately and recombine their results classically.

Why it is here: the router (idea 4) may choose to run a circuit as two smaller
pieces, on fewer and better qubits, on a device too small for the whole circuit,
or on an exact classical simulator when each piece fits in memory and the whole
does not. The price is a sampling overhead that grows exponentially with the
number of cut gates, so this module also reports that price.

Method: gate cutting by quasi-probability decomposition (QPD), Mitarai & Fujii,
New J. Phys. 23, 023021 (2021); Piveteau & Sutter, IEEE Trans. Inf. Theory 70,
2734 (2024). For ``rzz(theta) = exp(-i theta/2 Z(x)Z)`` with c = cos(theta/2),
s = sin(theta/2), the channel ``rho -> U rho U^dagger`` equals exactly

    c^2 [I | I] + s^2 [Z | Z] + cs [M | S] - cs [M | Sdg] + cs [S | M] - cs [Sdg | M]

where ``[P | Q]`` applies P on the qubit in piece A and Q on the qubit in piece B,
S / Sdg are the usual phase gates, and M is a Z measurement whose outcome
multiplies the shot's value by +1 / -1 (the identity behind the last four terms is
``i[rho, ZZ] = {Z,.}_A (x) i[Z,.]_B / 2 + i[Z,.]_A (x) {Z,.}_B / 2``). The
one-norm of the coefficients is ``gamma = 1 + 2 |sin(theta)|``: small-angle
Trotter steps are cheap to cut, CZ (and CX, a CZ between Hadamards) is
``rzz(-pi/2)`` up to local RZ gates, so gamma = 3. For several cut gates the joint
terms are products and gamma multiplies. The usual "sampling overhead" gamma^2
bounds the variance of one sample (one shot per piece) against a per-shot variance of
1; for averaged observables such as the magnetisation, whose uncut per-shot variance is
much smaller, the measured shot overhead is far larger (see ``cutting_cost``). Joint decompositions of several gates
with classical communication (Piveteau & Sutter) can beat this product; they are
not implemented.

Mid-circuit measurement: the framework-neutral ``Circuit`` has none, so M is
written by the deferred-measurement principle as a CX from the cut qubit onto a
fresh ancilla that is only read at the end. That is exactly the measure-and-weight
channel (tests/test_cutting.py proves it against an explicit projector-branching
simulator), it runs unchanged on every backend in the repo, and the shot weight
is (-1)^(parity of the ancilla bits). The cost is one extra qubit and one CX per M
in a piece; real mid-circuit measurement would avoid both on IBM hardware.

Observables are Z-diagonal, the quantities the workloads use (<Z_i>, Z_i Z_j and
sums of them), written as a tuple of qubit indices (a Z string; ``()`` is the
identity) or a dict ``{support: coefficient}`` for a weighted sum.

Production equivalent: qiskit-addon-cutting (not installed here; this is an
in-house implementation restricted to RZZ / CZ / CX cuts and Z-diagonal
observables).
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union

import numpy as np

from .circuit import Circuit, Gate

# Local operations a QPD term applies at a cut, on one side.
OPS = ("id", "z", "s", "sdg", "meas")
CUTTABLE = {"rzz", "cz", "cx"}
# Largest piece (data + ancilla qubits) the exact statevector executor will hold.
EXACT_MAX_QUBITS = 26

Observable = Union[Sequence[int], Mapping[Sequence[int], float]]


# --------------------------------------------------------------------------
# Quasi-probability decomposition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QPDTerm:
    coeff: float
    op_a: str
    op_b: str


def rzz_qpd(theta: float, tol: float = 1e-15) -> list[QPDTerm]:
    """The six-term decomposition of rzz(theta); zero-weight terms are dropped
    (theta = 0 leaves only the identity term, theta = pi only Z|Z)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    terms = [QPDTerm(c * c, "id", "id"), QPDTerm(s * s, "z", "z"),
             QPDTerm(c * s, "meas", "s"), QPDTerm(-c * s, "meas", "sdg"),
             QPDTerm(c * s, "s", "meas"), QPDTerm(-c * s, "sdg", "meas")]
    return [t for t in terms if abs(t.coeff) > tol]


def gamma_rzz(theta: float) -> float:
    """One-norm of the rzz(theta) decomposition: 1 + 2|sin(theta)|."""
    return 1.0 + 2.0 * abs(math.sin(theta))


def gamma_cz() -> float:
    return gamma_rzz(-math.pi / 2)   # = 3


@dataclass(frozen=True)
class CutGate:
    """One two-qubit gate crossing the bipartition, rewritten as local gates plus rzz(theta).

    ``pre`` / ``post`` are (global qubit, Gate) pairs applied just before / after the
    QPD operations: CZ = RZ(pi/2) (x) RZ(pi/2) . rzz(-pi/2) up to a global phase, and
    CX(c, t) = H_t CZ H_t."""

    index: int            # position in the original gate list
    name: str
    qubit_a: int          # global index of the qubit in piece A
    qubit_b: int
    theta: float
    pre: tuple = ()
    post: tuple = ()

    @property
    def terms(self) -> list[QPDTerm]:
        return rzz_qpd(self.theta)

    @property
    def gamma(self) -> float:
        return sum(abs(t.coeff) for t in self.terms)


def _cut_gate(index: int, g: Gate, part_a: set[int]) -> CutGate:
    a, b = g.qubits
    qa, qb = (a, b) if a in part_a else (b, a)
    if g.name == "rzz":
        return CutGate(index, g.name, qa, qb, g.params[0])
    if g.name == "cz":
        rz = tuple((q, Gate("rz", (q,), (math.pi / 2,))) for q in (qa, qb))
        return CutGate(index, g.name, qa, qb, -math.pi / 2, pre=rz)
    if g.name == "cx":
        c, t = g.qubits
        h = ((t, Gate("h", (t,))),)
        rz = tuple((q, Gate("rz", (q,), (math.pi / 2,))) for q in (c, t))
        return CutGate(index, g.name, qa, qb, -math.pi / 2, pre=h + rz, post=h)
    raise ValueError(f"cannot cut a {g.name!r} gate (supported: {sorted(CUTTABLE)})")


# --------------------------------------------------------------------------
# Cut plans
# --------------------------------------------------------------------------


@dataclass
class CutPlan:
    circuit: Circuit
    part_a: tuple[int, ...]
    part_b: tuple[int, ...]
    cuts: list[CutGate]

    @property
    def n_cuts(self) -> int:
        return len(self.cuts)

    @property
    def gammas(self) -> list[float]:
        return [c.gamma for c in self.cuts]

    @property
    def gamma(self) -> float:
        return math.prod(self.gammas)

    @property
    def sampling_overhead(self) -> float:
        return self.gamma ** 2

    @property
    def n_terms(self) -> int:
        """Joint QPD terms the exact-term mode evaluates (6 per generic cut)."""
        return math.prod(len(c.terms) for c in self.cuts)

    def side_ops(self, side: str) -> list[tuple[str, ...]]:
        """Distinct local operations each cut needs on one side (5 for a generic angle)."""
        out = []
        for c in self.cuts:
            ops = [t.op_a if side == "a" else t.op_b for t in c.terms]
            out.append(tuple(dict.fromkeys(ops)))
        return out

    def configs(self, side: str) -> list[tuple[str, ...]]:
        return list(itertools.product(*self.side_ops(side)))

    @property
    def n_circuits(self) -> int:
        """Distinct piece circuits the exact-term mode runs (both sides)."""
        return len(self.configs("a")) + len(self.configs("b"))

    @property
    def piece_qubits(self) -> tuple[int, int]:
        return len(self.part_a), len(self.part_b)

    @property
    def piece_widths(self) -> tuple[int, int]:
        """Widest circuit per side, ancillas for deferred measurements included."""
        return tuple(len(p) + max((cfg.count("meas") for cfg in self.configs(s)), default=0)
                     for p, s in ((self.part_a, "a"), (self.part_b, "b")))

    def pieces(self) -> tuple[Circuit, Circuit]:
        """Each side with every cut gate removed (what runs, before QPD operations)."""
        return (build_piece(self, "a", ("id",) * self.n_cuts)[0],
                build_piece(self, "b", ("id",) * self.n_cuts)[0])

    def summary(self) -> dict:
        return {"part_a": list(self.part_a), "part_b": list(self.part_b), "n_cuts": self.n_cuts,
                "cut_gates": [(c.name, c.qubit_a, c.qubit_b, c.theta) for c in self.cuts],
                "gammas": self.gammas, "gamma": self.gamma, "sampling_overhead": self.sampling_overhead,
                "n_terms": self.n_terms, "n_circuits_exact": self.n_circuits,
                "piece_qubits": list(self.piece_qubits), "piece_widths": list(self.piece_widths)}


def plan_cut(circuit: Circuit, part_a: Sequence[int]) -> CutPlan:
    """Cut every two-qubit gate between ``part_a`` and the remaining qubits."""
    sa = set(int(q) for q in part_a)
    if not sa or len(sa) >= circuit.n_qubits or not sa <= set(range(circuit.n_qubits)):
        raise ValueError("part_a must be a non-empty proper subset of the circuit's qubits")
    cuts = [_cut_gate(i, g, sa) for i, g in enumerate(circuit.gates)
            if len(g.qubits) == 2 and (g.qubits[0] in sa) != (g.qubits[1] in sa)]
    part_b = tuple(q for q in range(circuit.n_qubits) if q not in sa)
    return CutPlan(circuit, tuple(sorted(sa)), part_b, cuts)


def find_cut(circuit: Circuit, max_piece_qubits: int) -> CutPlan:
    """Best contiguous bipartition {0..m-1} | {m..n-1} for chain-like circuits.

    Feasible: both pieces have at most ``max_piece_qubits`` data qubits (ancillas
    are extra, see ``CutPlan.piece_widths``) and every crossing gate is cuttable.
    Among those, lowest total gamma, then fewest cuts, then the most balanced
    split. A TFIM Trotter chain cut in the middle has one cut gate per step.
    """
    n = circuit.n_qubits
    best, best_key = None, None
    for m in range(1, n):
        if max(m, n - m) > max_piece_qubits:
            continue
        try:
            plan = plan_cut(circuit, range(m))
        except ValueError:      # a swap (or other uncuttable gate) crosses this bond
            continue
        key = (round(plan.gamma, 12), plan.n_cuts, abs(2 * m - n))
        if best_key is None or key < best_key:
            best, best_key = plan, key
    if best is None:
        raise ValueError(f"no contiguous cut gives pieces of <= {max_piece_qubits} qubits")
    return best


def _local_gate(g: Gate, index: dict[int, int]) -> Gate:
    return Gate(g.name, tuple(index[q] for q in g.qubits), g.params)


def _side_program(plan: CutPlan, side: str):
    """(local index map, k+1 gate segments, local qubit of each cut) for one side.

    Segment j holds the gates between cut j-1's QPD operation and cut j's: a cut's
    ``pre`` gates end segment j, its ``post`` gates start segment j+1."""
    part = plan.part_a if side == "a" else plan.part_b
    index = {q: i for i, q in enumerate(part)}
    by_pos = {c.index: c for c in plan.cuts}
    segments: list[list[Gate]] = [[]]
    cut_qubits = []
    for i, g in enumerate(plan.circuit.gates):
        cut = by_pos.get(i)
        if cut is None:
            if all(q in index for q in g.qubits):
                segments[-1].append(_local_gate(g, index))
            continue
        segments[-1].extend(_local_gate(pg, index) for q, pg in cut.pre if q in index)
        cut_qubits.append(index[cut.qubit_a if side == "a" else cut.qubit_b])
        segments.append([_local_gate(pg, index) for q, pg in cut.post if q in index])
    return index, segments, cut_qubits


def build_piece(plan: CutPlan, side: str, config: Sequence[str]) -> tuple[Circuit, int]:
    """The neutral circuit for one side and one choice of local operation per cut.

    Data qubits keep the side's order (local 0..n-1); each ``meas`` adds an ancilla
    after them, coupled by CX and read at the end. Returns (circuit, n_ancillas)."""
    index, segments, cut_qubits = _side_program(plan, side)
    n_local = len(index)
    n_anc = sum(1 for op in config if op == "meas")
    c = Circuit(n_local + n_anc, name=f"{plan.circuit.name}_{side}_{'-'.join(config) or 'none'}")
    anc = n_local
    for j, seg in enumerate(segments):
        c.gates.extend(seg)
        if j < len(config):
            op, q = config[j], cut_qubits[j]
            if op == "meas":
                c.cx(q, anc)
                anc += 1
            elif op != "id":
                c.add(op, q)
    return c, n_anc


# --------------------------------------------------------------------------
# Observables
# --------------------------------------------------------------------------


def _norm_obs(o: Observable) -> dict[tuple[int, ...], float]:
    if isinstance(o, Mapping):
        out: dict[tuple[int, ...], float] = {}
        for k, v in o.items():
            key = tuple(sorted(int(q) for q in k))
            out[key] = out.get(key, 0.0) + float(v)
        return out
    return {tuple(sorted(int(q) for q in o)): 1.0}


def magnetisation_observable(n: int) -> dict[tuple[int, ...], float]:
    """Average magnetisation (1/n) sum_i Z_i as a weighted Z sum."""
    return {(i,): 1.0 / n for i in range(n)}


def _split(support: tuple[int, ...], index_a: dict[int, int], index_b: dict[int, int]):
    return (tuple(index_a[q] for q in support if q in index_a),
            tuple(index_b[q] for q in support if q in index_b))


_BITS: dict[int, np.ndarray] = {}


def _zsign(n: int, support: tuple[int, ...]) -> np.ndarray:
    """+-1 eigenvalues of Z_support over the 2^n basis states (qubit 0 = most significant bit)."""
    if n not in _BITS:
        _BITS[n] = np.arange(2 ** n, dtype=np.int64)
    idx = _BITS[n]
    parity = np.zeros(2 ** n, dtype=np.int64)
    for q in support:
        parity ^= (idx >> (n - 1 - q)) & 1
    return 1 - 2 * parity


def _zexp(p: np.ndarray, n: int, support: tuple[int, ...]) -> float:
    if not support:
        return float(p.sum())
    if len(support) == 1:   # marginal without building a sign vector
        m = p.reshape(2 ** support[0], 2, -1).sum(axis=(0, 2))
        return float(m[0] - m[1])
    return float(p @ _zsign(n, support))


def z_expectations_from_counts(counts: Mapping[str, int], observables: Sequence[Observable],
                               n_weight_bits: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Shot-noise estimates (means, standard errors) of Z-diagonal observables.

    Bitstrings are qubit 0 first. The last ``n_weight_bits`` bits are deferred-measurement
    ancillas: each shot's value is multiplied by (-1)^(their parity)."""
    obs = [_norm_obs(o) for o in observables]
    strings = sorted({s for o in obs for s in o})
    mean, cov = _string_stats(counts, strings, n_weight_bits)
    vals, ses = [], []
    for o in obs:
        k = np.array([o.get(s, 0.0) for s in strings])
        vals.append(float(k @ mean))
        ses.append(math.sqrt(max(0.0, float(k @ cov @ k))))
    return np.array(vals), np.array(ses)


def _string_stats(counts: Mapping[str, int], strings: Sequence[tuple[int, ...]], n_weight_bits: int):
    """Mean of w * Z_s per shot for every string s, and the covariance of those means."""
    keys = list(counts)
    cnt = np.array([counts[k] for k in keys], dtype=float)
    shots = cnt.sum()
    bits = np.array([[ch == "1" for ch in k] for k in keys], dtype=np.int64).reshape(len(keys), -1)
    n_data = bits.shape[1] - n_weight_bits
    w = 1 - 2 * (bits[:, n_data:].sum(axis=1) % 2)
    vals = np.empty((len(keys), len(strings)))
    for j, s in enumerate(strings):
        vals[:, j] = w * (1 - 2 * (bits[:, list(s)].sum(axis=1) % 2)) if s else w
    mean = cnt @ vals / shots
    second = (vals * cnt[:, None]).T @ vals / shots
    cov = (second - np.outer(mean, mean)) / shots
    return mean, cov


# --------------------------------------------------------------------------
# Exact statevector simulation (the "classical" executor)
# --------------------------------------------------------------------------


def _apply(psi: np.ndarray, n: int, g: Gate) -> np.ndarray:
    """Apply one neutral gate in place-ish to a 2^n statevector (qubit 0 = most significant)."""
    name, q = g.name, g.qubits
    if len(q) == 1:
        v = psi.reshape(2 ** q[0], 2, -1)
        a, b = v[:, 0, :], v[:, 1, :]
        if name in ("z", "s", "sdg", "rz"):
            if name == "rz":
                t = g.params[0]
                a *= np.exp(-0.5j * t)
                b *= np.exp(0.5j * t)
            else:
                b *= {"z": -1, "s": 1j, "sdg": -1j}[name]
            return psi
        if name == "x":
            v[:, [0, 1], :] = v[:, [1, 0], :]
            return psi
        if name == "y":
            a2 = a.copy()
            v[:, 0, :] = -1j * b
            v[:, 1, :] = 1j * a2
            return psi
        if name == "h":
            m = [[1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]]
        elif name == "rx":
            c, s = math.cos(g.params[0] / 2), math.sin(g.params[0] / 2)
            m = [[c, -1j * s], [-1j * s, c]]
        elif name == "ry":
            c, s = math.cos(g.params[0] / 2), math.sin(g.params[0] / 2)
            m = [[c, -s], [s, c]]
        else:  # pragma: no cover - SUPPORTED is closed
            raise ValueError(name)
        a2, b2 = a.copy(), b.copy()
        v[:, 0, :] = m[0][0] * a2 + m[0][1] * b2
        v[:, 1, :] = m[1][0] * a2 + m[1][1] * b2
        return psi
    lo, hi = sorted(q)
    v = psi.reshape(2 ** lo, 2, 2 ** (hi - lo - 1), 2, -1)

    def sl(bq0: int, bq1: int):   # slice with qubit q[0] = bq0 and q[1] = bq1
        bl, bh = (bq0, bq1) if q[0] == lo else (bq1, bq0)
        return (slice(None), bl, slice(None), bh, slice(None))

    if name == "rzz":
        t = g.params[0]
        for x, y in ((0, 0), (1, 1)):
            v[sl(x, y)] *= np.exp(-0.5j * t)
        for x, y in ((0, 1), (1, 0)):
            v[sl(x, y)] *= np.exp(0.5j * t)
    elif name == "cz":
        v[sl(1, 1)] *= -1
    elif name == "cx":
        t10 = v[sl(1, 0)].copy()
        v[sl(1, 0)] = v[sl(1, 1)]
        v[sl(1, 1)] = t10
    elif name == "swap":
        t01 = v[sl(0, 1)].copy()
        v[sl(0, 1)] = v[sl(1, 0)]
        v[sl(1, 0)] = t01
    else:  # pragma: no cover
        raise ValueError(name)
    return psi


def statevector(circuit: Circuit) -> np.ndarray:
    """Dense statevector of a neutral circuit from |0...0> (memory 16 * 2^n bytes)."""
    if circuit.n_qubits > EXACT_MAX_QUBITS:
        raise ValueError(f"{circuit.n_qubits} qubits is too many for the exact executor "
                         f"(limit {EXACT_MAX_QUBITS})")
    psi = np.zeros(2 ** circuit.n_qubits, complex)
    psi[0] = 1.0
    for g in circuit.gates:
        _apply(psi, circuit.n_qubits, g)
    return psi


def exact_expectations(circuit: Circuit, observables: Sequence[Observable]) -> np.ndarray:
    """Uncut reference: exact Z-diagonal expectation values from the whole statevector."""
    p = np.abs(statevector(circuit)) ** 2
    return np.array([sum(k * _zexp(p, circuit.n_qubits, s) for s, k in _norm_obs(o).items())
                     for o in observables])


def _exact_side(plan: CutPlan, side: str, strings: Sequence[tuple[int, ...]]) -> dict:
    """Exact local values for every config of one side, by depth-first branching.

    The state is carried as a signed mixture [(weight, unnormalised vector)]; ``meas``
    replaces each vector v by (+w, P0 v) and (-w, P1 v), which is the measure-and-weight
    channel itself (no ancilla). Shared prefixes are simulated once: each cut multiplies the
    number of vectors by 6 (four unitary branches, and a measurement branch that doubles
    the mixture), so the work is about sum_j 6^j segments instead of 5^k whole circuits."""
    index, segments, cut_qubits = _side_program(plan, side)
    n = len(index)
    if n > EXACT_MAX_QUBITS:
        raise ValueError(f"piece of {n} qubits is too large for the exact executor")
    ops = plan.side_ops(side)
    out: dict[tuple[str, ...], np.ndarray] = {}

    def rec(j: int, mixture: list[tuple[float, np.ndarray]], cfg: tuple[str, ...]):
        for _, psi in mixture:
            for g in segments[j]:
                _apply(psi, n, g)
        if j == len(cut_qubits):
            p = sum(w * np.abs(psi) ** 2 for w, psi in mixture)
            out[cfg] = np.array([_zexp(p, n, s) for s in strings])
            return
        q = cut_qubits[j]
        for op in ops[j]:
            new = []
            for w, psi in mixture:
                if op == "meas":
                    for bit, sign in ((0, 1.0), (1, -1.0)):
                        v = psi.copy()
                        v.reshape(2 ** q, 2, -1)[:, 1 - bit, :] = 0
                        new.append((w * sign, v))
                else:
                    v = psi.copy()
                    if op != "id":
                        _apply(v, n, Gate(op, (q,)))
                    new.append((w, v))
            rec(j + 1, new, cfg + (op,))

    psi0 = np.zeros(2 ** n, complex)
    psi0[0] = 1.0
    rec(0, [(1.0, psi0)], ())
    return out


# --------------------------------------------------------------------------
# Recombination
# --------------------------------------------------------------------------


@dataclass
class CutResult:
    values: np.ndarray
    std_errors: np.ndarray
    mode: str                     # "exact" (every term) or "sample"
    executor: str                 # backend name, or "statevector"
    n_cuts: int
    gammas: list[float]
    gamma: float
    sampling_overhead: float
    n_terms: int                  # joint QPD terms evaluated (exact) or distinct terms sampled
    n_circuits: int               # piece circuits executed
    total_shots: int              # shots over every piece circuit (0 for statevector)
    backend_runs: int             # QuantumBackend.run() calls
    wall_time_s: float
    plan: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _coeff_table(cut: CutGate) -> dict[tuple[str, str], float]:
    return {(t.op_a, t.op_b): t.coeff for t in cut.terms}


def _decompose(plan: CutPlan, observables):
    index_a = {q: i for i, q in enumerate(plan.part_a)}
    index_b = {q: i for i, q in enumerate(plan.part_b)}
    obs = [_norm_obs(o) for o in observables]
    for o in obs:
        for s in o:
            if any(not 0 <= q < plan.circuit.n_qubits for q in s):
                raise ValueError(f"observable support {s} outside the circuit")
    split = [[(k, *_split(s, index_a, index_b)) for s, k in o.items()] for o in obs]
    strings_a = sorted({a for terms in split for _, a, _ in terms})
    strings_b = sorted({b for terms in split for _, _, b in terms})
    return split, strings_a, strings_b


def _bilinear(terms, sa, sb, X, Y, Sx, Sy, C):
    """Value and variance of sum_k kappa_k X[:, a_k]^T C Y[:, b_k] for independent pieces.

    X (configs_a x strings_a) are piece means with per-config covariance Sx; likewise Y, Sy.
    Var = u^T Sx u + v^T Sy v + sum_pq C_pq^2 tr(K^T Sx_p K Sy_q): exact for products of
    independent estimators (plug-in means)."""
    ia = {s: i for i, s in enumerate(sa)}
    ib = {s: i for i, s in enumerate(sb)}
    K = np.zeros((len(sa), len(sb)))
    for k, a, b in terms:
        K[ia[a], ib[b]] += k
    val = float(np.einsum("pa,ab,pq,qb->", X, K, C, Y))
    if Sx is None and Sy is None:
        return val, 0.0
    U = C @ Y @ K.T          # d val / d X[p, a]
    V = C.T @ X @ K          # d val / d Y[q, b]
    var = float(np.einsum("pa,pab,pb->", U, Sx, U) + np.einsum("qa,qab,qb->", V, Sy, V))
    KSyK = np.einsum("ab,qbc,dc->qad", K, Sy, K)       # K Sy_q K^T
    var += float(np.einsum("pq,pad,qda->", C ** 2, Sx, KSyK))
    return val, max(var, 0.0)


def _run_side(backend, circuits_and_anc, shots, strings):
    circs = [c for c, _ in circuits_and_anc]
    res = backend.run(circs, shots)
    means, covs = [], []
    for counts, (_, n_anc) in zip(res.counts, circuits_and_anc):
        m, cv = _string_stats(counts, strings, n_anc)
        means.append(m)
        covs.append(cv)
    return np.array(means), np.array(covs), res


def cut_expectations(circuit: Circuit | CutPlan, observables: Sequence[Observable], *,
                     backend=None, shots: int = 4000, mode: str = "exact",
                     max_piece_qubits: int | None = None, seed: int | None = None) -> CutResult:
    """Estimate Z-diagonal expectation values by cutting ``circuit`` into two pieces.

    ``circuit`` may be a ``CutPlan``; otherwise ``find_cut(circuit, max_piece_qubits)``
    (default: half the qubits, rounded up). ``backend`` is a ``QuantumBackend`` (pieces
    run as circuits with deferred-measurement ancillas; side A's circuits in one
    ``run()`` call and side B's in another, so ``ibm-auto`` places each side on its own
    best region) or ``None`` for the exact statevector executor (no shot noise).

    ``mode="exact"``: every joint QPD term; each distinct piece circuit gets ``shots``
    shots. ``mode="sample"``: ``shots`` joint-term samples drawn with probability
    |c_t| / gamma; each distinct sampled term runs its A and B circuits with as many
    shots as it was drawn, and the estimate is gamma * mean(sign(c_t) * A * B) (the
    product of the two pieces' sample means per term). Standard errors are the
    propagated shot noise (plus, in sampling mode, the term-sampling variance),
    estimated from the data.
    """
    t0 = time.perf_counter()
    plan = circuit if isinstance(circuit, CutPlan) else find_cut(
        circuit, max_piece_qubits or math.ceil(circuit.n_qubits / 2))
    if mode not in ("exact", "sample"):
        raise ValueError("mode must be 'exact' or 'sample'")
    if mode == "sample" and backend is None:
        raise ValueError("sampling mode needs a backend; the statevector executor is exact")
    split, sa, sb = _decompose(plan, observables)
    tables = [_coeff_table(c) for c in plan.cuts]

    def coeff(ca, cb) -> float:
        return math.prod(t.get((x, y), 0.0) for t, x, y in zip(tables, ca, cb))

    common = dict(n_cuts=plan.n_cuts, gammas=plan.gammas, gamma=plan.gamma,
                  sampling_overhead=plan.sampling_overhead, plan=plan.summary())
    if mode == "exact":
        cfg_a, cfg_b = plan.configs("a"), plan.configs("b")
        C = np.array([[coeff(x, y) for y in cfg_b] for x in cfg_a])
        meta: dict = {}
        if backend is None:
            ex_a, ex_b = _exact_side(plan, "a", sa), _exact_side(plan, "b", sb)
            X = np.array([ex_a[c] for c in cfg_a])
            Y = np.array([ex_b[c] for c in cfg_b])
            Sx = Sy = None
            executor, n_circ, total, runs = "statevector", 0, 0, 0
        else:
            X, Sx, ra = _run_side(backend, [build_piece(plan, "a", c) for c in cfg_a], shots, sa)
            Y, Sy, rb = _run_side(backend, [build_piece(plan, "b", c) for c in cfg_b], shots, sb)
            executor, runs = backend.name, 2
            n_circ = len(cfg_a) + len(cfg_b)
            total = shots * n_circ
            meta = {"side_a": ra.meta, "side_b": rb.meta,
                    "quantum_time_s": ra.wall_time_s + rb.wall_time_s}
        vals, ses = [], []
        for terms in split:
            v, var = _bilinear(terms, sa, sb, X, Y, Sx, Sy, C)
            vals.append(v)
            ses.append(math.sqrt(var))
        return CutResult(np.array(vals), np.array(ses), "exact", executor, n_terms=plan.n_terms,
                         n_circuits=n_circ, total_shots=total, backend_runs=runs,
                         wall_time_s=time.perf_counter() - t0, meta=meta, **common)

    # -- sampling mode ------------------------------------------------------
    rng = np.random.default_rng(seed)
    draws = np.empty((shots, plan.n_cuts), dtype=np.int64)
    for j, c in enumerate(plan.cuts):
        w = np.array([abs(t.coeff) for t in c.terms])
        draws[:, j] = rng.choice(len(w), size=shots, p=w / w.sum())
    uniq, counts = np.unique(draws, axis=0, return_counts=True)
    terms_t = [[plan.cuts[j].terms[i] for j, i in enumerate(row)] for row in uniq]
    sign = np.array([math.prod(math.copysign(1.0, t.coeff) for t in tt) for tt in terms_t])
    by_shots: dict[int, list[int]] = {}
    for t, n_t in enumerate(counts):
        by_shots.setdefault(int(n_t), []).append(t)
    est = np.zeros((len(split), len(uniq)))
    var_t = np.zeros_like(est)
    runs, qtime, sides_meta = 0, 0.0, []
    for n_t, idx in sorted(by_shots.items()):
        pa = [build_piece(plan, "a", tuple(t.op_a for t in terms_t[t])) for t in idx]
        pb = [build_piece(plan, "b", tuple(t.op_b for t in terms_t[t])) for t in idx]
        X, Sx, ra = _run_side(backend, pa, n_t, sa)
        Y, Sy, rb = _run_side(backend, pb, n_t, sb)
        runs += 2
        qtime += ra.wall_time_s + rb.wall_time_s
        sides_meta.append({"shots": n_t, "side_a": ra.meta, "side_b": rb.meta})
        for r, t in enumerate(idx):
            for o, terms in enumerate(split):
                v, var = _bilinear(terms, sa, sb, X[r:r + 1], Y[r:r + 1], Sx[r:r + 1], Sy[r:r + 1],
                                   np.ones((1, 1)))
                est[o, t], var_t[o, t] = v, var
    g, N = plan.gamma, shots
    frac = counts / N
    vals = g * (est * sign * frac).sum(axis=1)
    # Var = E[Var(F | draws)] + Var(E[F | draws]); the second is gamma^2 / N times the
    # variance of sign * E_t under the sampling distribution, with E_t^2 estimated
    # without its shot-noise bias.
    within = (g ** 2) * (var_t * frac ** 2).sum(axis=1)
    second = ((est ** 2 - var_t) * frac).sum(axis=1)
    between = (g ** 2 / N) * np.maximum(0.0, second - (vals / g) ** 2)
    ses = np.sqrt(within + between)
    return CutResult(vals, ses, "sample", backend.name, n_terms=len(uniq), n_circuits=2 * len(uniq),
                     total_shots=int(2 * N), backend_runs=runs, wall_time_s=time.perf_counter() - t0,
                     meta={"quantum_time_s": qtime, "groups": sides_meta}, **common)


# --------------------------------------------------------------------------
# Cost model hook
# --------------------------------------------------------------------------


def cutting_cost(n_cuts: int, gammas: Sequence[float] | None = None, shots: int = 1024,
                 piece_qubits: Sequence[int] = (), mode: str = "sample",
                 terms_per_cut: int = 6, configs_per_cut: int = 5,
                 uncut_variance: float = 1.0) -> dict:
    """Pure cost arithmetic for the cost model (no circuits are built).

    ``gammas`` default to 3 per cut (CZ / CX; the worst case for rzz).

    Sampling mode: one sample is a shot on each side, and its value is bounded by gamma in
    magnitude, so its variance is at most gamma^2. ``shots * gamma^2 / uncut_variance``
    samples therefore reach at least the standard error an uncut circuit gets from ``shots``
    shots whose per-shot variance is ``uncut_variance`` (at most 1 for a single Z string,
    1 - <O>^2 exactly). The executed shot count is twice the samples. Leaving
    ``uncut_variance`` at 1 under-states the cost for averaged observables: for the 8-qubit
    TFIM magnetisation, 1 Trotter step, noiseless Aer: 4000 uncut shots gave SE 0.0027
    (per-shot variance 0.03) and 40000 samples gave SE 0.0076, a per-sample variance factor
    of about 80 against gamma^2 = 3.8. Exact-term mode: ``2 * configs_per_cut^n_cuts`` circuits of ``shots``
    shots each. ``piece_widths`` adds the worst-case ancillas (one per cut) to each piece.
    """
    gammas = list(gammas) if gammas is not None else [gamma_cz()] * n_cuts
    if len(gammas) != n_cuts:
        raise ValueError("need one gamma per cut")
    gamma = math.prod(gammas)
    overhead = gamma ** 2
    n_exact = 2 * configs_per_cut ** n_cuts if n_cuts else 1
    if mode == "sample":
        samples = math.ceil(shots * overhead / uncut_variance)
        circuits = min(2 * terms_per_cut ** n_cuts, 2 * samples) if n_cuts else 1
        total = 2 * samples if n_cuts else shots
    elif mode == "exact":
        circuits, total = n_exact, n_exact * shots
    else:
        raise ValueError("mode must be 'sample' or 'exact'")
    return {"n_cuts": n_cuts, "gamma": gamma, "sampling_overhead": overhead, "mode": mode,
            "circuits": circuits, "total_shots": total, "shots_multiplier": total / shots,
            "classical_terms": terms_per_cut ** n_cuts,
            "piece_qubits": list(piece_qubits),
            "piece_widths": [p + n_cuts for p in piece_qubits]}


__all__ = ["QPDTerm", "CutGate", "CutPlan", "CutResult", "rzz_qpd", "gamma_rzz", "gamma_cz",
           "plan_cut", "find_cut", "build_piece", "cut_expectations", "cutting_cost",
           "exact_expectations", "statevector", "z_expectations_from_counts",
           "magnetisation_observable"]
