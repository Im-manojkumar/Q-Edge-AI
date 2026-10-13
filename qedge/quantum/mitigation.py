"""Error mitigation as a placement option: readout correction and zero-noise extrapolation.

The router already decides *where* a circuit runs. This module lets it also decide how
much classical help the QPU gets. Mitigation reduces the *bias* of an expectation value;
it does not create information. Every level that does something costs extra circuits,
shots or depth and widens the error bar, so ``mitigated_expectations`` reports standard
errors and the shots it spent next to the values, and ``mitigation_cost`` gives the cost
model the same multipliers without running anything.

Levels
  * ``none``         - the existing path: one ``backend.run`` of the circuits as given.
  * ``readout``      - tensored readout-error inversion (independent per-qubit assignment
    matrices, as in e.g. Bravyi et al., PRA 103, 042605, 2021). The assignment matrices come
    from two calibration circuits (all-0 and all-1, run in the same job on the same physical
    qubits) or from the device calibration the router already holds (free, but only as fresh
    as the calibration). For Z-diagonal observables the inverse has a closed form per shot:
    a measured bit z' = +-1 on a qubit with P(1|0) = e0 and P(0|1) = e1 has
    E[z' | z] = g z + d with g = 1 - e0 - e1 and d = e1 - e0, so (z' - d) / g is an unbiased
    single-shot estimator of z, and for a product of Z's the product of these factors is
    unbiased because readout flips are independent across qubits (the tensored assumption).
  * ``zne``          - zero-noise extrapolation (Temme, Bravyi & Gambetta, PRL 119, 180509,
    2017) with noise scaled by unitary folding, U -> U (U^dagger U)^k plus a partial fold of
    the last gates for non-odd scales (Giurgica-Tiron et al., IEEE QCE 2020). Folding is done
    on the *executable* circuit, after layout and routing, through the opt-in ``isa_scope``
    hook of the Qiskit backends: the transpiler at optimisation level >= 1 cancels
    U U^dagger pairs, so folding the logical circuit and then transpiling would quietly
    give scale 1 again. The folded circuit is not transpiled again; barriers separate the
    segments so even a later re-transpilation cannot merge them (tested).
  * ``readout+zne``  - both: readout correction applied at every scale, then extrapolation.

Mitiq (ZNE, readout) and Qiskit's M3 / Runtime resilience options are the production
equivalents; they are not used here because the project adds no dependencies.

Observables are Z-diagonal: a linear combination of products of Z on a set of qubits,
written as ``{(i, j, ...): coefficient}`` (``()`` is the constant term). That covers the
workloads' quantities: average magnetisation, ZZ energies of Ising/MaxCut/VQE Hamiltonians
and the per-qubit <Z_i> of the split model's quantum head. Qubit 0 is the leftmost bit,
as everywhere in the repo.

Standard errors are bootstrap estimates: every executed circuit's counts (calibration
circuits included) are resampled multinomially and the whole pipeline (correction and
extrapolation) is recomputed. They therefore include the shot noise of the calibration
circuits, but not drift between calibration and run or a wrong noise model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .circuit import Circuit

LEVELS = ("none", "readout", "zne", "readout+zne")
EXTRAPOLATIONS = ("linear", "richardson", "exponential")
DEFAULT_SCALES = (1.0, 3.0, 5.0)
# Chosen from data (idea-4 hand-off; 6- and 8-qubit TFIM Trotter circuits, 8000 shots, 20 seeds):
# on the emulated IBM device the router picks (fake_aachen), linear least squares over (1, 3, 5)
# had the lowest or equal root-mean-square error (8 qubits, readout+zne: 0.0055 linear, 0.0056
# exponential, 0.0089 Richardson). Richardson removes the most bias but its weights
# (15/8, -5/4, 3/8) give ~1.7x the standard error. Exponential won under heavier uniform noise
# (noisy Aer, 6 qubits: 0.0076 vs 0.0108; fake_brisbane, 8 qubits, 10 seeds: 0.0065 vs 0.0082)
# but lost when the signal was near 0 (0.0130 vs 0.0100) and assumes the noisy limit is 0, so
# it is an option for noisy devices and large signals, not the default.
DEFAULT_EXTRAPOLATION = "linear"
DEFAULT_BOOTSTRAP = 1000
# Gates that are free and noiseless on IBM devices (virtual Z) or not gates at all. They do
# not count towards the noise a fold adds.
_FREE_OPS = {"rz", "barrier", "delay", "id"}

ZObservable = Mapping[tuple[int, ...], float]


# ---- observables -------------------------------------------------------------
def z_observable(obs) -> dict[tuple[int, ...], float]:
    """Normalise an observable: ``(0, 1)`` -> Z0 Z1, ``"ZIZ"`` -> Z0 Z2, or a dict
    ``{(qubits...): coefficient}``."""
    if isinstance(obs, str):
        qs = tuple(i for i, ch in enumerate(obs.upper()) if ch == "Z")
        if any(ch not in "ZI" for ch in obs.upper()):
            raise ValueError(f"only Z and I are supported in {obs!r}")
        return {qs: 1.0}
    if isinstance(obs, Mapping):
        out: dict[tuple[int, ...], float] = {}
        for k, v in obs.items():
            key = tuple(sorted(int(q) for q in (k if isinstance(k, tuple) else (k,))))
            out[key] = out.get(key, 0.0) + float(v)
        return out
    if isinstance(obs, int):
        return {(int(obs),): 1.0}
    return {tuple(sorted(int(q) for q in obs)): 1.0}


def magnetisation_observable(n: int) -> dict[tuple[int, ...], float]:
    """Average magnetisation (1/n) sum_i Z_i (the quantum_simulation workload's output)."""
    return {(i,): 1.0 / n for i in range(n)}


def zz_observable(edges) -> dict[tuple[int, ...], float]:
    """sum_(i,j,w) w Z_i Z_j for edges given as (i, j) or (i, j, w)."""
    out: dict[tuple[int, ...], float] = {}
    for e in edges:
        i, j, w = (e[0], e[1], e[2] if len(e) > 2 else 1.0)
        key = (min(i, j), max(i, j))
        out[key] = out.get(key, 0.0) + float(w)
    return out


def _z_table(counts: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """(outcomes x qubits array of +-1, counts vector)."""
    keys = list(counts)
    z = np.array([[1 - 2 * int(b) for b in k] for k in keys], dtype=float) if keys else np.zeros((0, 0))
    return z, np.array([counts[k] for k in keys], dtype=float)


def _obs_values(zc: np.ndarray, obs: ZObservable) -> np.ndarray:
    """Per-outcome value of the observable from (corrected) per-qubit factors zc."""
    f = np.zeros(zc.shape[0])
    for qs, coeff in obs.items():
        f += coeff * (np.prod(zc[:, list(qs)], axis=1) if qs else 1.0)
    return f


def expectation_from_counts(counts: Mapping[str, int], obs) -> float:
    """Plain (unmitigated) expectation value of a Z-diagonal observable."""
    z, w = _z_table(counts)
    return float(_obs_values(z, z_observable(obs)) @ w / w.sum())


# ---- readout correction ---------------------------------------------------------
@dataclass
class ReadoutCalibration:
    """Per-qubit assignment errors, indexed by *bit position* of the counts they correct
    (qubit 0 = leftmost bit). ``physical_qubits[i]`` is the device qubit bit i was measured on."""

    p01: np.ndarray                       # P(read 1 | prepared 0)
    p10: np.ndarray                       # P(read 0 | prepared 1)
    source: str = "given"                 # "circuits" | "device" | "given"
    physical_qubits: list[int] | None = None

    def __post_init__(self):
        self.p01 = np.asarray(self.p01, dtype=float)
        self.p10 = np.asarray(self.p10, dtype=float)
        if np.any(1 - self.p01 - self.p10 <= 0.05):
            raise ValueError("readout error too large to invert (1 - e0 - e1 <= 0.05)")

    @property
    def gain(self) -> np.ndarray:
        return 1.0 - self.p01 - self.p10

    @property
    def offset(self) -> np.ndarray:
        return self.p10 - self.p01

    def correct_z(self, z: np.ndarray) -> np.ndarray:
        """Unbiased per-shot estimate of each true z from measured z (rows = outcomes)."""
        n = z.shape[1]
        return (z - self.offset[:n]) / self.gain[:n]

    def matrix(self, i: int) -> np.ndarray:
        """Assignment matrix A[measured, prepared] of bit i."""
        e0, e1 = self.p01[i], self.p10[i]
        return np.array([[1 - e0, e1], [e0, 1 - e1]])


def readout_from_counts(zero_counts: Mapping[str, int], one_counts: Mapping[str, int],
                        physical_qubits: list[int] | None = None) -> ReadoutCalibration:
    """Assignment errors from an all-0 and an all-1 calibration circuit."""
    z0, w0 = _z_table(zero_counts)
    z1, w1 = _z_table(one_counts)
    p01 = ((1 - z0) / 2).T @ w0 / w0.sum()     # fraction of shots reading 1 after preparing 0
    p10 = ((1 + z1) / 2).T @ w1 / w1.sum()     # fraction reading 0 after preparing 1
    return ReadoutCalibration(p01, p10, "circuits", physical_qubits)


def readout_from_device(cal, physical_qubits: Sequence[int], asymmetric: bool = False) -> ReadoutCalibration:
    """Assignment errors of the given physical qubits from a ``calibration.DeviceCalibration``.

    ``asymmetric=True`` reads IBM's prob_meas1_prep0 / prob_meas0_prep1 from the backend
    properties (falls back to the symmetric figure per qubit when missing). Measured on the
    snapshots shipped with qiskit-ibm-runtime: Aer's ``AerSimulator.from_backend`` builds its
    readout noise from ``Target["measure"].error`` only, i.e. *symmetric*, so for the local
    emulators the symmetric figure is the one that matches the simulated device; on live
    hardware the asymmetric one is closer to the truth."""
    qs = list(physical_qubits)
    sym = np.array([cal.readout.get(q, 0.0) for q in qs])
    p01, p10 = sym.copy(), sym.copy()
    if asymmetric and cal.backend is not None:
        try:
            props = cal.backend.properties()
        except Exception:
            props = None
        if props is not None:
            for k, q in enumerate(qs):
                try:
                    qp = props.qubit_property(q)
                    p01[k] = float(qp["prob_meas1_prep0"][0])
                    p10[k] = float(qp["prob_meas0_prep1"][0])
                except Exception:
                    pass
    return ReadoutCalibration(p01, p10, "device", qs)


def corrected_expectation(counts: Mapping[str, int], obs, readout: ReadoutCalibration | None) -> float:
    """Expectation value with the tensored readout inverse applied exactly (closed form)."""
    z, w = _z_table(counts)
    zc = readout.correct_z(z) if readout is not None else z
    return float(_obs_values(zc, z_observable(obs)) @ w / w.sum())


def correct_distribution(counts: Mapping[str, int], readout: ReadoutCalibration, max_qubits: int = 14,
                         clip: bool = False) -> dict[str, float]:
    """Full tensored inverse on the probability vector (2^n memory, so small n only).
    The result is a quasi-probability distribution; ``clip=True`` zeroes negative entries
    and renormalises (simple; Smolin, Gambetta & Smith 2012 give the closest-distribution
    projection)."""
    n = len(next(iter(counts)))
    if n > max_qubits:
        raise ValueError(f"{n} qubits: full-distribution correction limited to {max_qubits}")
    total = sum(counts.values())
    p = np.zeros((2,) * n)
    for k, v in counts.items():
        p[tuple(int(b) for b in k)] += v / total
    for i in range(n):
        p = np.moveaxis(np.tensordot(np.linalg.inv(readout.matrix(i)), p, axes=([1], [i])), 0, i)
    flat = p.reshape(-1)
    if clip:
        flat = np.clip(flat, 0, None)
        flat = flat / flat.sum()
    return {format(k, f"0{n}b"): float(v) for k, v in enumerate(flat) if abs(v) > 1e-15}


# ---- folding ----------------------------------------------------------------------
def fold_circuit(c: Circuit, scale: float) -> tuple[Circuit, float]:
    """Framework-neutral fold (gate count scaled by ~``scale``). Only valid where the
    simulator applies the gate list as given (see ``mitigated_expectations``); returns
    (folded circuit, achieved scale)."""
    if scale < 1:
        raise ValueError("scale factors must be >= 1")
    n = len(c.gates)
    if n == 0:
        return Circuit(c.n_qubits, list(c.gates), c.name), 1.0
    k, extra = _fold_plan(scale, n)
    out = Circuit(c.n_qubits, list(c.gates), f"{c.name}_fold{scale:g}")
    inv = c.inverse().gates
    for _ in range(k):
        out.gates.extend(inv)
        out.gates.extend(c.gates)
    if extra:
        tail = c.gates[n - extra:]
        out.gates.extend(Circuit(c.n_qubits, list(tail)).inverse().gates)
        out.gates.extend(tail)
    return out, len(out.gates) / n


def _fold_plan(scale: float, n_noisy: int) -> tuple[int, int]:
    """(global folds k, number of trailing noisy gates to fold once more)."""
    k = int((scale - 1) // 2)
    extra = int(round((scale - 1 - 2 * k) / 2 * n_noisy))
    return k, min(extra, n_noisy)


def measured_physical_qubits(qc) -> list[int]:
    """Physical qubit measured into each classical bit (index = bit = our qubit position)."""
    out: dict[int, int] = {}
    for ins in qc.data:
        if ins.operation.name == "measure":
            out[qc.find_bit(ins.clbits[0]).index] = qc.find_bit(ins.qubits[0]).index
    return [out[i] for i in sorted(out)]


def two_qubit_gate_count(qc) -> int:
    return sum(1 for ins in qc.data if ins.operation.num_qubits == 2 and ins.operation.name != "barrier")


def _inverse_ops(ops):
    """Inverse of a list of (operation, qubits) in the device's basis. SX^dagger is not an
    IBM basis gate; it equals RZ(pi) SX RZ(pi) up to a global phase (checked in the tests)."""
    from qiskit.circuit.library import RZGate, SXGate

    out = []
    for op, qs in reversed(ops):
        if op.name == "sx":
            out += [(RZGate(math.pi), qs), (SXGate(), qs), (RZGate(math.pi), qs)]
        elif op.name in ("measure", "reset") or op.num_clbits:
            raise ValueError(f"cannot fold through {op.name!r}")
        else:
            out.append((op.inverse(), qs))
    return out


def fold_isa(qc, scale: float):
    """Fold an executable (transpiled) circuit: body U -> U (U^dg U)^k [L^dg L], with L the
    trailing gates that make up the fractional part, then the original measurements.
    Barriers separate the segments. Returns (folded circuit, achieved noisy-gate scale)."""
    from qiskit import QuantumCircuit

    if scale < 1:
        raise ValueError("scale factors must be >= 1")
    body, meas, seen_meas = [], [], False
    for ins in qc.data:
        name = ins.operation.name
        if name == "measure":
            seen_meas = True
            meas.append(ins)
        elif name == "barrier":
            continue
        elif seen_meas:
            raise ValueError("folding needs all measurements at the end of the circuit")
        else:
            body.append((ins.operation, tuple(ins.qubits)))
    noisy = [i for i, (op, _) in enumerate(body) if op.name not in _FREE_OPS]
    n_noisy = len(noisy)
    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=f"{qc.name}_fold{scale:g}")
    out.global_phase = qc.global_phase
    active = sorted({q for _, qs in body for q in qs} | {ins.qubits[0] for ins in meas},
                    key=lambda q: qc.find_bit(q).index)

    def emit(ops):
        for op, qs in ops:
            out.append(op, qs)

    if n_noisy == 0 or scale == 1:
        emit(body)
        achieved = 1.0
    else:
        k, extra = _fold_plan(scale, n_noisy)
        inv = _inverse_ops(body)
        emit(body)
        for _ in range(k):
            out.barrier(active)
            emit(inv)
            out.barrier(active)
            emit(body)
        if extra:
            tail = body[noisy[n_noisy - extra]:]
            out.barrier(active)
            emit(_inverse_ops(tail))
            out.barrier(active)
            emit(tail)
        achieved = (n_noisy * (1 + 2 * k) + 2 * extra) / n_noisy
    out.barrier(active)
    for ins in meas:
        out.append(ins.operation, ins.qubits, ins.clbits)
    return out, achieved


def readout_calibration_circuits(nq: int, physical_qubits: Sequence[int], qregs=None):
    """All-0 and all-1 circuits measuring ``physical_qubits`` (bit k <- qubit k of the list)
    on an ``nq``-qubit device register, in the device basis (x is native on IBM devices)."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    out = []
    for flip in (False, True):
        regs = list(qregs) if qregs else [QuantumRegister(nq, "q")]
        qc = QuantumCircuit(*regs, ClassicalRegister(len(physical_qubits), "meas"),
                            name=f"readout_cal_{int(flip)}")
        if flip:
            for q in physical_qubits:
                qc.x(q)
        qc.barrier(list(physical_qubits))
        for k, q in enumerate(physical_qubits):
            qc.measure(q, k)
        out.append(qc)
    return out


# ---- extrapolation ----------------------------------------------------------------------
def extrapolation_weights(scales: Sequence[float], method: str) -> np.ndarray:
    """Linear extrapolations are a weighted sum of the per-scale values; these are the weights.
    sqrt(sum w^2) is how much the standard error grows (equal shots and per-shot variance)."""
    x = np.asarray(scales, dtype=float)
    if method == "richardson":
        # Lagrange polynomial through all points, evaluated at 0.
        return np.array([np.prod([x[j] / (x[j] - x[i]) for j in range(len(x)) if j != i])
                         for i in range(len(x))])
    if method == "linear":
        if len(x) < 2:
            return np.ones(1)
        X = np.column_stack([np.ones_like(x), x])
        return np.linalg.pinv(X)[0]          # intercept row of the least-squares solution
    raise ValueError(f"no linear weights for {method!r}")


def extrapolate(scales: Sequence[float], values: Sequence[float], method: str = DEFAULT_EXTRAPOLATION) -> float:
    """Zero-noise value from values measured at the given noise scale factors.

    ``exponential`` fits y = a exp(-b x) (asymptote 0, the fully mixed value of a traceless
    observable) by least squares on log|y|; it falls back to ``linear`` when the values
    change sign or touch 0, where the model cannot hold."""
    y = np.asarray(values, dtype=float)
    if len(y) == 1:
        return float(y[0])
    if method == "exponential":
        if np.all(y > 0) or np.all(y < 0):
            sgn = np.sign(y[0])
            slope, intercept = np.polyfit(np.asarray(scales, dtype=float), np.log(np.abs(y)), 1)
            return float(sgn * math.exp(intercept))
        method = "linear"
    return float(extrapolation_weights(scales, method) @ y)


# ---- the main entry point --------------------------------------------------------------
@dataclass
class MitigationResult:
    values: np.ndarray                    # (circuits, observables) mitigated estimates
    stderrs: np.ndarray                   # bootstrap standard errors, same shape
    raw_values: np.ndarray                # unmitigated scale-1 estimates from the same run
    raw_stderrs: np.ndarray
    per_scale: np.ndarray                 # (scales, circuits, observables), readout-corrected if asked
    level: str
    metadata: dict = field(default_factory=dict)


def _unwrap(backend):
    inner = getattr(backend, "inner", None)
    return inner if inner is not None else backend


def _isa_capable(backend) -> bool:
    from .backends.qiskit_backend import CalibratedIBMBackend, QiskitAerBackend, QiskitFakeIBMBackend
    return isinstance(_unwrap(backend), (QiskitAerBackend, QiskitFakeIBMBackend, CalibratedIBMBackend))


def _device_calibration(backend, run_meta: dict):
    from . import calibration as C
    from .backends.qiskit_backend import CalibratedIBMBackend, QiskitFakeIBMBackend
    be = _unwrap(backend)
    if isinstance(be, CalibratedIBMBackend):
        return next(d for d in be.devices if d.name == run_meta["device"])
    if isinstance(be, QiskitFakeIBMBackend):
        return C.from_backend(be._get(), "snapshot")
    raise ValueError(f"{backend.name} has no device calibration; use readout_source='circuits'")


def mitigated_expectations(backend, circuits: Sequence[Circuit], observables, shots: int = 4000,
                           level: str = "none", scale_factors: Sequence[float] = DEFAULT_SCALES,
                           extrapolation: str = DEFAULT_EXTRAPOLATION, readout_source: str = "circuits",
                           asymmetric_readout: bool | None = None, n_boot: int = DEFAULT_BOOTSTRAP,
                           seed: int | None = None, allow_circuit_folding: bool | None = None) -> MitigationResult:
    """Expectation values of Z-diagonal observables with the chosen mitigation level.

    Everything (folded copies at every scale and the readout calibration circuits) goes in
    ONE backend job with ``shots`` shots per executed circuit, so mitigation adds circuits and
    shots but not queue waits. ``observables`` is one observable or a list (see
    ``z_observable``); every observable is evaluated on every circuit.

    ``readout_source``: ``"circuits"`` (two calibration circuits on the same physical
    qubits, measured now) or ``"device"`` (the router's calibration data; no extra
    circuits; ``asymmetric_readout`` defaults to True on live hardware and False on
    emulators, matching how Aer builds their readout noise).

    ``seed`` fixes the Aer simulator seed (Qiskit backends) and the bootstrap. On non-Qiskit
    stacks ZNE folds the framework-neutral circuit; that is allowed by default only for local
    simulators, which apply the gate list as given with a noise channel after every gate
    (Cirq's ``to_cirq`` appends ``depolarize`` after each operation), so nothing can cancel.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    if extrapolation not in EXTRAPOLATIONS:
        raise ValueError(f"extrapolation must be one of {EXTRAPOLATIONS}")
    if readout_source not in ("circuits", "device"):
        raise ValueError("readout_source must be 'circuits' or 'device'")
    circuits = list(circuits)
    obs_list = [observables] if isinstance(observables, (Mapping, str)) else list(observables)
    if obs_list and isinstance(obs_list[0], int):   # a bare tuple of qubits = one product observable
        obs_list = [tuple(obs_list)]
    obs = [z_observable(o) for o in obs_list]
    use_ro = "readout" in level
    use_zne = "zne" in level
    scales = tuple(float(s) for s in scale_factors) if use_zne else (1.0,)
    if use_zne and (len(scales) < 2 or scales[0] != 1.0 or any(s < 1 for s in scales)):
        raise ValueError("ZNE needs at least two scale factors, the first equal to 1")
    cal_circuits = use_ro and readout_source == "circuits"
    m = len(circuits)
    isa = _isa_capable(backend)
    meta: dict = {"level": level, "backend": backend.name, "shots_per_circuit": shots,
                  "scale_factors": list(scales), "extrapolation": extrapolation if use_zne else None,
                  "readout_source": readout_source if use_ro else None, "jobs": 1,
                  "stderr_method": f"bootstrap ({n_boot} resamples)"}

    record: dict = {}
    if level == "none" and not (isa and seed is not None):
        res = backend.run(circuits, shots)                    # exactly the existing path
        counts = list(res.counts)
        meta["fold_level"] = None
    elif isa:
        from .backends.qiskit_backend import isa_scope

        def transform(isa_list, be):
            record["measured"] = [measured_physical_qubits(qc) for qc in isa_list]
            record["two_q"], record["depth"], record["achieved"] = [], [], []
            out = []
            for s in scales:
                row_2q, row_d, row_a = [], [], []
                for qc in isa_list:
                    f, a = fold_isa(qc, s) if s != 1.0 else (qc, 1.0)
                    out.append(f)
                    row_2q.append(two_qubit_gate_count(f))
                    row_d.append(f.depth())
                    row_a.append(a)
                record["two_q"].append(row_2q)
                record["depth"].append(row_d)
                record["achieved"].append(row_a)
            if cal_circuits:
                union = sorted({q for ms in record["measured"] for q in ms})
                record["union"] = union
                out += readout_calibration_circuits(isa_list[0].num_qubits, union, isa_list[0].qregs)
            return out

        run_opts = {} if seed is None else {"seed_simulator": int(seed)}
        with isa_scope(transform, **run_opts):
            res = backend.run(circuits, shots)
        counts = list(res.counts)
        meta["fold_level"] = "isa" if use_zne else None
    else:
        if use_zne:
            ok = allow_circuit_folding if allow_circuit_folding is not None else not getattr(backend, "hardware", False)
            if not ok:
                raise ValueError(f"{backend.name}: folding the logical circuit is only known to survive on "
                                 "local simulators; pass allow_circuit_folding=True to force it")
        submit, achieved = [], []
        for s in scales:
            row = []
            for c in circuits:
                f, a = fold_circuit(c, s) if s != 1.0 else (c, 1.0)
                submit.append(f)
                row.append(a)
            achieved.append(row)
        record["achieved"] = achieved
        record["depth"] = [[c.depth() for c in submit[i * m:(i + 1) * m]] for i in range(len(scales))]
        record["two_q"] = [[c.n_2q for c in submit[i * m:(i + 1) * m]] for i in range(len(scales))]
        record["measured"] = [list(range(c.n_qubits)) for c in circuits]   # logical = physical here
        if cal_circuits:
            nmax = max(c.n_qubits for c in circuits)
            one = Circuit(nmax, name="readout_cal_1")
            for q in range(nmax):
                one.x(q)
            submit += [Circuit(nmax, name="readout_cal_0"), one]
            record["union"] = list(range(nmax))
        res = backend.run(submit, shots)
        counts = list(res.counts)
        meta["fold_level"] = "circuit" if use_zne else None

    n_exec = len(counts)
    meta.update(circuits_run=n_exec, total_shots=n_exec * shots,
                backend_meta={k: v for k, v in res.meta.items() if k != "transpiled_depths"})
    if "measured" in record:
        meta["measured_physical_qubits"] = record["measured"]
    if use_zne or use_ro:
        meta["depth_per_scale"] = record.get("depth")
        meta["two_qubit_gates_per_scale"] = record.get("two_q")
        meta["achieved_scale_factors"] = record.get("achieved")

    # Readout model, as (physical qubit -> (p01, p10)) so every circuit picks its own bits.
    ro_phys: dict[int, tuple[float, float]] | None = None
    cal_idx: tuple[int, int] | None = None
    if use_ro:
        if cal_circuits:
            cal_idx = (n_exec - 2, n_exec - 1)
            rc = readout_from_counts(counts[cal_idx[0]], counts[cal_idx[1]], record["union"])
            ro_phys = {q: (rc.p01[k], rc.p10[k]) for k, q in enumerate(record["union"])}
        else:
            dcal = _device_calibration(backend, res.meta)
            asym = asymmetric_readout if asymmetric_readout is not None else bool(getattr(backend, "hardware", False))
            qs = sorted({q for ms in record["measured"] for q in ms})
            rc = readout_from_device(dcal, qs, asym)
            ro_phys = {q: (rc.p01[k], rc.p10[k]) for k, q in enumerate(qs)}
            meta["readout_asymmetric"] = asym
        meta["readout_errors"] = {int(q): (round(float(a), 5), round(float(b), 5)) for q, (a, b) in ro_phys.items()}

    main = counts[: len(scales) * m]
    x_scales = np.array(record["achieved"]) if "achieved" in record else np.ones((1, m))
    tables = [_z_table(c) for c in main]
    cal_tables = [_z_table(counts[i]) for i in cal_idx] if cal_idx else None

    def estimate(weights_main, weights_cal):
        """(mitigated (m, n_obs), per-scale (k, m, n_obs)) from (possibly resampled) counts."""
        ro = ro_phys
        if cal_tables is not None:
            (z0, _), (z1, _) = cal_tables
            p01 = ((1 - z0) / 2).T @ weights_cal[0] / weights_cal[0].sum()
            p10 = ((1 + z1) / 2).T @ weights_cal[1] / weights_cal[1].sum()
            ro = {q: (p01[k], p10[k]) for k, q in enumerate(record["union"])}
        per = np.zeros((len(scales), m, len(obs)))
        for idx, ((z, _), w) in enumerate(zip(tables, weights_main)):
            si, ci = divmod(idx, m)
            zc = z
            if ro is not None:
                qs = record["measured"][ci]
                e0 = np.array([ro[q][0] for q in qs])
                e1 = np.array([ro[q][1] for q in qs])
                zc = (z - (e1 - e0)) / np.maximum(1 - e0 - e1, 0.05)
            for oi, o in enumerate(obs):
                per[si, ci, oi] = _obs_values(zc, o) @ w / w.sum()
        if not use_zne:
            return per[0], per
        out = np.zeros((m, len(obs)))
        for ci in range(m):
            for oi in range(len(obs)):
                out[ci, oi] = extrapolate(x_scales[:, ci], per[:, ci, oi], extrapolation)
        return out, per

    base_w = [w for _, w in tables]
    base_cal = [w for _, w in cal_tables] if cal_tables else None
    values, per_scale = estimate(base_w, base_cal)
    raw = np.array([[_obs_values(tables[ci][0], o) @ base_w[ci] / base_w[ci].sum() for o in obs] for ci in range(m)])

    rng = np.random.default_rng(seed)
    boot, boot_raw = [], []
    for _ in range(n_boot):
        bw = [rng.multinomial(int(w.sum()), w / w.sum()).astype(float) for w in base_w]
        bc = [rng.multinomial(int(w.sum()), w / w.sum()).astype(float) for w in base_cal] if base_cal else None
        v, _ = estimate(bw, bc)
        boot.append(v)
        boot_raw.append([[_obs_values(tables[ci][0], o) @ bw[ci] / bw[ci].sum() for o in obs] for ci in range(m)])
    stderrs = np.std(np.array(boot), axis=0, ddof=1) if n_boot > 1 else np.full_like(values, np.nan)
    raw_se = np.std(np.array(boot_raw), axis=0, ddof=1) if n_boot > 1 else np.full_like(raw, np.nan)
    return MitigationResult(values, stderrs, raw, raw_se, per_scale, level, meta)


# ---- cost model ----------------------------------------------------------------------
def mitigation_cost(level: str, depth: int, shots: int, n_circuits: int = 1,
                    scale_factors: Sequence[float] = DEFAULT_SCALES, extrapolation: str = DEFAULT_EXTRAPOLATION,
                    readout_source: str = "circuits") -> dict:
    """What a mitigation level costs, as multipliers on the unmitigated job (pure arithmetic).

    * ``circuits_multiplier`` / ``shots_multiplier``: executed circuits and total shots
      relative to running the ``n_circuits`` circuits once with ``shots`` shots each
      (calibration circuits get the same shots, as in ``mitigated_expectations``).
    * ``depth_multiplier``: depth of the deepest executed circuit (it sets the fidelity),
      ``mean_depth_multiplier`` the shot-weighted mean (it sets gate time per shot).
    * ``stderr_multiplier``: growth of the standard error at equal shots per circuit,
      sqrt(sum w_i^2) of the extrapolation weights. It assumes the per-shot variance is the
      same at every scale and ignores readout correction's 1/(1-e0-e1) factor (about 1.03 at
      1.5% readout error), so it is a lower bound; ``None`` for exponential (nonlinear).
    * ``shots_for_equal_stderr``: total shots needed to match the unmitigated standard error.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    k = len(scale_factors) if "zne" in level else 1
    scales = [float(s) for s in scale_factors] if "zne" in level else [1.0]
    extra_cal = 2 if ("readout" in level and readout_source == "circuits") else 0
    executed = k * n_circuits + extra_cal
    se = 1.0
    if "zne" in level:
        se = None if extrapolation == "exponential" else float(np.sqrt(np.sum(extrapolation_weights(scales, extrapolation) ** 2)))
    shots_mult = executed / n_circuits
    return {"level": level,
            "circuits_multiplier": executed / n_circuits,
            "extra_circuits": executed - n_circuits,
            "shots_multiplier": shots_mult,
            "total_shots": executed * shots,
            "depth_multiplier": max(scales),
            "mean_depth_multiplier": float(np.mean(scales)),
            "max_depth": int(math.ceil(depth * max(scales))),
            "stderr_multiplier": se,
            "shots_for_equal_stderr": None if se is None else int(math.ceil(shots_mult * se ** 2 * n_circuits * shots))}
