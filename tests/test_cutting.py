"""Circuit cutting (qedge/quantum/cutting.py): QPD algebra, exact recombination against an
independent simulator (Qiskit's statevector), the deferred-measurement formulation, the
TFIM cut finder, pieces on the repo's Qiskit backend, sampling-mode unbiasedness and the
honesty of its error bars, and the cost arithmetic."""
import itertools
import math

import numpy as np
import pytest

from qedge.quantum import cutting as K
from qedge.quantum.algorithms import tfim_exact_magnetisation, tfim_trotter
from qedge.quantum.backends import QuantumBackend, get_backend
from qedge.quantum.backends.qiskit_backend import to_qiskit
from qedge.quantum.circuit import Circuit

ANGLES = [0.3, -0.5, 1.2, math.pi / 2, -2.0, 2.9]

# -- helpers ---------------------------------------------------------------

_I = np.eye(2)
_Z = np.diag([1.0, -1.0])
_S = np.diag([1.0, 1j])
_P = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]
_LOCAL = {
    "id": lambda r: r,
    "z": lambda r: _Z @ r @ _Z,
    "s": lambda r: _S @ r @ _S.conj().T,
    "sdg": lambda r: _S.conj().T @ r @ _S,
    "meas": lambda r: _P[0] @ r @ _P[0] - _P[1] @ r @ _P[1],
}


def _product_map(fa, fb, rho):
    """(fa (x) fb)(rho) for a two-qubit rho, by linearity over the product basis."""
    out = np.zeros((4, 4), complex)
    for i, j in itertools.product(range(4), repeat=2):
        ea = np.zeros((2, 2)); ea[i // 2, j // 2] = 1
        eb = np.zeros((2, 2)); eb[i % 2, j % 2] = 1
        out += rho[i, j] * np.kron(fa(ea), fb(eb))
    return out


def _qiskit_expectations(c: Circuit, observables):
    """Z-diagonal expectations from Qiskit's statevector (little-endian): independent of cutting.py."""
    from qiskit.quantum_info import Statevector
    p = Statevector(to_qiskit(c, measure=False)).probabilities()
    idx = np.arange(2 ** c.n_qubits)
    out = []
    for o in observables:
        items = o.items() if isinstance(o, dict) else [(tuple(o), 1.0)]
        tot = 0.0
        for s, k in items:
            par = np.zeros_like(idx)
            for q in s:
                par ^= (idx >> q) & 1
            tot += k * float(p @ (1 - 2 * par))
        out.append(tot)
    return np.array(out)


def _random_circuit(n, crossing, seed):
    """Random local layers with the given crossing gates on the middle bond (n//2-1, n//2)."""
    r = np.random.default_rng(seed)
    c = Circuit(n, name="rand")
    a, b = n // 2 - 1, n // 2
    for q in range(n):
        c.ry(q, r.uniform(0, 3)).rz(q, r.uniform(0, 3))
    for name, theta in crossing:
        for q in range(n - 1):
            if q != a:
                c.cx(q, q + 1) if r.random() < 0.5 else c.rzz(q, q + 1, r.uniform(-2, 2))
        {"rzz": lambda: c.rzz(a, b, theta), "cz": lambda: c.cz(b, a), "cx": lambda: c.cx(b, a)}[name]()
        for q in range(n):
            c.rx(q, r.uniform(0, 3))
            c.h(q) if r.random() < 0.3 else c.sdg(q)
    return c


# -- QPD algebra -----------------------------------------------------------


@pytest.mark.parametrize("theta", ANGLES)
def test_qpd_reproduces_the_rzz_channel(theta):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    u = np.diag(np.exp(-0.5j * theta * np.array([1, -1, -1, 1])))
    got = sum(t.coeff * _product_map(_LOCAL[t.op_a], _LOCAL[t.op_b], rho) for t in K.rzz_qpd(theta))
    assert np.abs(got - u @ rho @ u.conj().T).max() < 1e-12


def test_gamma_formula():
    for theta in ANGLES + [0.0, math.pi]:
        assert sum(abs(t.coeff) for t in K.rzz_qpd(theta)) == pytest.approx(1 + 2 * abs(math.sin(theta)))
        assert K.gamma_rzz(theta) == pytest.approx(1 + 2 * abs(math.sin(theta)))
    assert K.gamma_cz() == pytest.approx(3.0)
    assert len(K.rzz_qpd(0.0)) == 1 and len(K.rzz_qpd(math.pi)) == 1   # zero terms dropped
    c = Circuit(2).rzz(0, 1, 0.4).cz(0, 1).cx(1, 0)
    plan = K.plan_cut(c, [0])
    assert plan.gammas == pytest.approx([1 + 2 * math.sin(0.4), 3.0, 3.0])
    assert plan.gamma == pytest.approx(9 * (1 + 2 * math.sin(0.4)))
    assert plan.sampling_overhead == pytest.approx(plan.gamma ** 2)
    assert plan.n_terms == 6 ** 3 and plan.n_circuits == 2 * 5 ** 3


# -- exact recombination ---------------------------------------------------

CASES = [[("rzz", 0.4)], [("rzz", -1.3)], [("rzz", 2.9)], [("cz", 0)], [("cx", 0)],
         [("rzz", 0.7), ("cz", 0)], [("rzz", 2.5), ("rzz", -0.2)], [("cx", 0), ("rzz", math.pi)]]


@pytest.mark.parametrize("crossing", CASES)
def test_exact_recombination_matches_uncut(crossing):
    n = 5
    c = _random_circuit(n, crossing, seed=len(crossing) * 7 + 1)
    obs = ([(i,) for i in range(n)] + [(1, 2), (0, 4), (1, 3), (0, 1), (2, 3, 4), ()]
           + [K.magnetisation_observable(n), {(1, 2): 0.5, (0,): -2.0}])
    plan = K.find_cut(c, 3)
    assert plan.part_a == (0, 1) and plan.n_cuts == len(crossing)
    res = K.cut_expectations(plan, obs)
    assert np.abs(res.values - _qiskit_expectations(c, obs)).max() < 1e-9
    assert np.all(res.std_errors == 0) and res.total_shots == 0
    assert np.abs(K.exact_expectations(c, obs) - _qiskit_expectations(c, obs)).max() < 1e-9


def test_deferred_measurement_equals_projector_branching():
    """The ancilla circuits (what backends run) give exactly the measure-and-weight values."""
    c = tfim_trotter(6, 2)
    plan = K.find_cut(c, 3)
    obs = [(0,), (2,), (0, 2), (2, 3), ()]
    _, sa, _ = K._decompose(plan, obs)
    branch = K._exact_side(plan, "a", sa)
    for cfg in plan.configs("a"):
        circ, n_anc = K.build_piece(plan, "a", cfg)
        assert n_anc == cfg.count("meas") and circ.n_qubits == 3 + n_anc
        p = np.abs(K.statevector(circ)) ** 2
        n, nl = circ.n_qubits, circ.n_qubits - n_anc
        local = (p * K._zsign(n, tuple(range(nl, n)))).reshape(2 ** nl, -1).sum(axis=1)
        vals = [K._zexp(local, nl, s) for s in sa]
        assert np.abs(np.array(vals) - branch[cfg]).max() < 1e-12


def test_exact_executor_on_a_16_qubit_chain():
    c = tfim_trotter(16, 3)
    res = K.cut_expectations(c, [K.magnetisation_observable(16)])
    assert res.n_cuts == 3 and res.plan["piece_qubits"] == [8, 8]
    assert res.values[0] == pytest.approx(tfim_exact_magnetisation(16, 3), abs=1e-9)


def test_statevector_matches_qiskit_for_every_gate():
    c = Circuit(4)
    for q in range(4):
        c.h(q).rx(q, 0.3 + q).ry(q, 0.7 * q).rz(q, -0.4 * q)
    c.x(0).y(1).z(2).s(3).sdg(0).cx(2, 0).cz(1, 3).rzz(3, 0, 0.9).swap(0, 2).swap(3, 1).cx(1, 2)
    obs = [(0,), (1,), (2,), (3,), (0, 3), (1, 2), (0, 1, 2, 3)]
    assert np.abs(K.exact_expectations(c, obs) - _qiskit_expectations(c, obs)).max() < 1e-12


def test_no_crossing_gates_means_no_overhead():
    c = Circuit(4).h(0).cx(0, 1).h(2).cz(2, 3)
    plan = K.find_cut(c, 2)
    assert plan.n_cuts == 0 and plan.gamma == 1 and plan.n_circuits == 2
    assert K.cut_expectations(plan, [(0, 1), (0, 2)]).values == pytest.approx([1.0, 0.0], abs=1e-12)


# -- cut finder --------------------------------------------------------------


@pytest.mark.parametrize("n,steps", [(8, 1), (8, 3), (10, 2), (12, 3)])
def test_find_cut_on_tfim_cuts_the_middle_bond_once_per_step(n, steps):
    plan = K.find_cut(tfim_trotter(n, steps), n // 2)
    assert plan.n_cuts == steps
    assert plan.part_a == tuple(range(n // 2)) and plan.part_b == tuple(range(n // 2, n))
    assert all((g.qubit_a, g.qubit_b) == (n // 2 - 1, n // 2) for g in plan.cuts)
    assert plan.gamma == pytest.approx((1 + 2 * math.sin(0.5)) ** steps)
    assert plan.piece_widths == (n // 2 + steps, n // 2 + steps)
    a, b = plan.pieces()
    assert a.n_qubits == n // 2 and a.n_2q == steps * (n // 2 - 1)


def test_find_cut_prefers_cheap_bonds_and_rejects_impossible_ones():
    c = Circuit(4).cz(0, 1).rzz(1, 2, 0.1).cz(2, 3)
    assert K.find_cut(c, 3).part_a in ((0, 1),)          # gamma 1.2 beats 3
    with pytest.raises(ValueError):
        K.find_cut(c, 1)
    s = Circuit(4).swap(1, 2).cz(0, 1).cz(2, 3)
    with pytest.raises(ValueError):
        K.find_cut(s, 2)                                  # only bond 1-2 balances; a swap crosses it
    with pytest.raises(ValueError):
        K.plan_cut(s, [0, 1])


# -- pieces on the repo's Qiskit backend --------------------------------------


@pytest.fixture(scope="module")
def aer():
    return get_backend("qiskit")


def test_pieces_run_on_the_qiskit_backend(aer):
    c = tfim_trotter(6, 2)
    obs = [K.magnetisation_observable(6), (2,), (3,), (2, 3), (0, 5), (1, 2)]
    exact = K.exact_expectations(c, obs)
    res = K.cut_expectations(c, obs, backend=aer, shots=3000)
    assert res.backend_runs == 2 and res.n_circuits == 50 and res.total_shots == 150_000
    assert np.all(res.std_errors > 0)
    assert np.all(np.abs(res.values - exact) < 5 * res.std_errors)   # cross-cut ZZ included


class _SeededSampler(QuantumBackend):
    """Exact probabilities sampled with a seeded RNG: deterministic, fast shot noise, so the
    estimator's statistics can be checked over many repeats without flakiness."""

    name = "seeded-sampler"

    def __init__(self, seed):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def _run(self, circuits, shots):
        out = []
        for c in circuits:
            p = np.abs(K.statevector(c)) ** 2
            draws = self.rng.multinomial(shots, p / p.sum())
            out.append({format(i, f"0{c.n_qubits}b"): int(k) for i, k in enumerate(draws) if k})
        return out, {}


def test_sampling_mode_is_unbiased_with_honest_error_bars():
    c = tfim_trotter(6, 2)
    obs = [(2,), (3,), (2, 3), (0, 5), K.magnetisation_observable(6)]
    exact = K.exact_expectations(c, obs)
    be = _SeededSampler(11)
    vals, ses = [], []
    for seed in range(120):
        r = K.cut_expectations(c, obs, backend=be, shots=2000, mode="sample", seed=seed)
        vals.append(r.values)
        ses.append(r.std_errors)
    vals, ses = np.array(vals), np.array(ses)
    spread = vals.std(axis=0, ddof=1)
    assert np.all(np.abs(vals.mean(axis=0) - exact) < 4 * spread / math.sqrt(len(vals)))
    # Reported SE tracks the run-to-run spread (120 repeats: the ratio is known to ~+-13% at 2 sigma).
    ratio = ses.mean(axis=0) / spread
    assert np.all((ratio > 0.8) & (ratio < 1.25)), ratio
    assert r.total_shots == 4000 and r.n_terms <= 36
    # The cost of cutting: the spread is well above the uncut shot noise for the same number
    # of samples, and within the gamma / sqrt(N) bound (each sample is bounded by gamma).
    uncut_sd = np.sqrt((1 - exact[:4] ** 2) / 2000)
    assert np.all(spread[:4] > 1.5 * uncut_sd) and np.all(spread < 1.15 * r.gamma / math.sqrt(2000))


def test_counts_estimator_applies_ancilla_weights():
    # Bits: q0, q1, ancilla. Ancilla 1 flips the sign of the shot.
    counts = {"000": 30, "101": 10, "011": 20, "110": 40}
    vals, ses = K.z_expectations_from_counts(counts, [(0,), (1,), (0, 1), ()], n_weight_bits=1)
    # w*Z0 per shot: 000:+1, 101:+1 (Z0=-1,w=-1), 011:-1 (Z0=+1,w=-1), 110:-1
    # w*Z0Z1 is +1 on every shot here; w alone is the identity observable's estimate.
    assert vals == pytest.approx([(30 + 10 - 20 - 40) / 100, (30 - 10 + 20 - 40) / 100,
                                  1.0, (30 - 10 - 20 + 40) / 100])
    m = vals[0]
    assert ses[0] == pytest.approx(math.sqrt((1 - m * m) / 100))


def test_errors():
    c = tfim_trotter(4, 1)
    with pytest.raises(ValueError):
        K.cut_expectations(c, [(0,)], mode="sample")              # needs a backend
    with pytest.raises(ValueError):
        K.cut_expectations(c, [(7,)])
    with pytest.raises(ValueError):
        K.plan_cut(c, [0, 1, 2, 3])


# -- cost arithmetic ----------------------------------------------------------


def test_cutting_cost():
    g = K.gamma_rzz(-0.5)
    s = K.cutting_cost(3, [g] * 3, shots=1000, piece_qubits=[4, 4])
    assert s["gamma"] == pytest.approx(g ** 3) and s["sampling_overhead"] == pytest.approx(g ** 6)
    assert s["total_shots"] == 2 * math.ceil(1000 * g ** 6)
    assert s["shots_multiplier"] == pytest.approx(s["total_shots"] / 1000)
    assert s["circuits"] == 2 * 6 ** 3 and s["piece_widths"] == [7, 7] and s["classical_terms"] == 216
    assert K.cutting_cost(1, [2.0], shots=100, uncut_variance=0.25)["total_shots"] == 2 * 1600
    e = K.cutting_cost(2, shots=500, mode="exact")
    assert e["gamma"] == pytest.approx(9.0) and e["circuits"] == 50 and e["total_shots"] == 25_000
    z = K.cutting_cost(0, shots=500)
    assert z["gamma"] == 1 and z["total_shots"] == 500 and z["circuits"] == 1
    with pytest.raises(ValueError):
        K.cutting_cost(2, [3.0])
