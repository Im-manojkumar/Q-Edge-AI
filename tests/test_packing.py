"""Job packing: allocation, joint circuit, result splitting, emulation and accounting.

Allocation is checked on the Torino snapshot (133 qubits, 11 broken couplers).
Emulation is checked on a 27-qubit snapshot so the noise model builds in well
under a second; the code is device-agnostic.
"""
import copy

import numpy as np
import pytest
from scipy.stats import chi2_contingency

from qedge.quantum import calibration as C
from qedge.quantum import packing as P
from qedge.quantum.algorithms import tfim_trotter
from qedge.quantum.backends.qiskit_backend import qiskit_counts, to_qiskit
from qedge.quantum.circuit import Circuit, ghz
from qedge.resources import load_catalogue
from qedge.workloads import make_workload


@pytest.fixture(scope="module")
def torino():
    return C.snapshot("FakeTorino")


@pytest.fixture(scope="module")
def small():
    return C.snapshot("FakeMumbaiV2")


def _copy(cal):
    out = copy.deepcopy(cal)
    out.backend = cal.backend
    return out


def _path(cal, n):
    """n physical qubits forming a simple path of good couplers."""
    nb = {q: sorted(b for (a, b), e in cal.two_q.items() if a == q and e < C.BROKEN) for q in range(cal.num_qubits)}
    def dfs(p):
        if len(p) == n:
            return p
        for b in nb[p[-1]]:
            if b not in p:
                r = dfs(p + [b])
                if r:
                    return r
        return None
    return next(r for q in range(cal.num_qubits) if (r := dfs([q])))


@pytest.fixture(scope="module")
def adjacent_pair(small):
    """Calibration confined to a 6-qubit path whose middle coupler is broken: two
    3-qubit programs must sit on the two halves, physically adjacent."""
    path = _path(small, 6)
    cal = _copy(small)
    for q in range(cal.num_qubits):
        if q not in path:
            cal.readout[q] = 1.0
    a, b = path[2], path[3]
    cal.two_q[(a, b)] = cal.two_q[(b, a)] = 1.0
    return cal, path


def _hellinger(ideal, counts):
    shots = sum(counts.values())
    return sum(np.sqrt(ideal.get(k, 0.0) * counts.get(k, 0) / shots) for k in set(ideal) | set(counts)) ** 2


def _ideal(circ):
    from qiskit.quantum_info import Statevector
    sv = Statevector(to_qiskit(circ).remove_final_measurements(inplace=False))
    return {k[::-1]: v for k, v in sv.probabilities_dict().items()}


# ---- allocation ------------------------------------------------------------------
def _check_valid(cal, alloc):
    seen = set()
    for p in alloc:
        assert not seen & set(p.qubits), "programs overlap"
        seen |= set(p.qubits)
        assert not set(p.qubits) & P.unusable_qubits(cal)
        for a, b in p.couplers:
            assert cal.edge_error(a, b) < C.BROKEN
        _, couplers = P._touched(p.circuit)
        assert set(couplers) == set(p.couplers)


def test_allocation_buffer_one_keeps_programs_apart(torino):
    alloc = P.allocate(torino, [tfim_trotter(4, 2)] * 40, buffer=1)
    _check_valid(torino, alloc)
    nb = P.neighbours(torino)
    for i, p in enumerate(alloc):
        near = {n for q in p.qubits for n in nb[q]}
        for o in alloc.placements[i + 1:]:
            assert not near & set(o.qubits), "buffer 1 left two programs adjacent"
    assert all(not v for v in P.exposed_couplers(torino, alloc.placements).values())
    # capacity is reported, not silently dropped
    assert alloc.unplaced and len(alloc) + len(alloc.unplaced) == 40
    with pytest.raises(P.CapacityError):
        P.allocate(torino, [tfim_trotter(4, 2)] * 40, buffer=1, strict=True)
    with pytest.raises(P.CapacityError):
        P.pack(torino, [tfim_trotter(4, 2)] * 40, buffer=1)


def test_allocation_buffer_zero_packs_more_and_can_touch(torino):
    a1 = P.allocate(torino, [tfim_trotter(4, 2)] * 40, buffer=1)
    a0 = P.allocate(torino, [tfim_trotter(4, 2)] * 40, buffer=0)
    _check_valid(torino, a0)
    assert len(a0) > len(a1)
    assert any(P.exposed_couplers(torino, a0.placements).values())


def test_first_program_matches_router_and_largest_goes_first(torino):
    alloc = P.allocate(torino, [tfim_trotter(4, 2), tfim_trotter(6, 2)], buffer=1)
    assert [p.order for p in alloc] == [1, 0]
    solo = C.best_layout(torino, to_qiskit(tfim_trotter(6, 2)))
    assert alloc[1].esp >= solo.esp - 1e-12 and alloc[1].exhaustive
    assert alloc[0].esp <= P.allocate(torino, [tfim_trotter(4, 2)])[0].esp + 1e-12


def test_allocation_avoids_broken_coupler(torino):
    first = P.allocate(torino, [tfim_trotter(4, 2)])[0]
    a, b = first.couplers[0]
    cal = _copy(torino)
    cal.two_q[(a, b)] = cal.two_q[(b, a)] = 1.0
    again = P.allocate(cal, [tfim_trotter(4, 2)])[0]
    assert (a, b) not in again.couplers and again.esp < first.esp
    _check_valid(cal, [again])


def test_capacity_search(torino):
    cap = P.capacity(torino, lambda k: [tfim_trotter(4, 2)] * k, buffer=1)
    assert P.allocate(torino, [tfim_trotter(4, 2)] * cap, buffer=1).complete
    assert not P.allocate(torino, [tfim_trotter(4, 2)] * (cap + 1), buffer=1).complete


# ---- packing and splitting ---------------------------------------------------------
def test_split_counts_bit_order(torino):
    from qiskit_aer import AerSimulator
    a = Circuit(3, name="a")
    a.x(0).cx(0, 1)                     # -> "110" (qubit 0 first)
    b = Circuit(2, name="b")
    b.x(1)                              # -> "01"
    job = P.pack(torino, [a, b], buffer=1)
    assert job.registers == ["p0", "p1"] and job.circuit.num_qubits == torino.num_qubits
    raw = AerSimulator().run(job.circuit, shots=64).result().get_counts()
    assert P.split_counts(job, raw) == [{"110": 64}, {"01": 64}]
    # same answer from SamplerV2-style per-register counts and from keys without spaces
    assert P.split_counts(job, {"p0": {"011": 5}, "p1": {"10": 5}}) == [{"110": 5}, {"01": 5}]
    assert P.split_counts(job, {"10011": 7}) == [{"110": 7}, {"01": 7}]
    with pytest.raises(ValueError):
        P.split_counts(job, {"1 0": 1})


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_split_counts_from_sampler_v2_registers(small):
    """The live path reads SamplerV2's per-register counts; check that bit order on a
    real SamplerV2 result (local mode, ideal Aer), plus its joined bitstrings."""
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import SamplerV2
    a = Circuit(3, name="a")
    a.x(0).cx(0, 1)
    b = Circuit(2, name="b")
    b.x(1)
    job = P.pack(small, [a, b], buffer=1)
    pub = SamplerV2(mode=AerSimulator()).run([job.circuit], shots=32).result()[0]
    per_reg = {r: getattr(pub.data, r).get_counts() for r in job.registers}
    assert P.split_counts(job, per_reg) == [{"110": 32}, {"01": 32}]
    assert P.split_counts(job, pub.join_data().get_counts()) == [{"110": 32}, {"01": 32}]


# ---- emulation -------------------------------------------------------------------
def test_per_program_emulation_matches_joint(small):
    job = P.pack(small, [ghz(3), tfim_trotter(3, 2)], buffer=0)
    shots = 20000
    per = P.emulate(job, shots, seed=3)
    joint = P.emulate_joint(job, shots, seed=4)
    for x, y in zip(per, joint):
        keys = sorted(set(x) | set(y))
        tab = np.array([[x.get(k, 0) for k in keys], [y.get(k, 0) for k in keys]])
        assert chi2_contingency(tab)[1] > 1e-3
    # the joint distribution factorises: programs' outcomes are independent shot by shot
    sim = P._simulator(small, P.base_noise_model(small), 5)
    raw = sim.run(job.circuit, shots=shots).result().get_counts()
    pairs = {}
    for key, v in raw.items():
        f1, f0 = key.split()
        pairs[(f0, f1)] = pairs.get((f0, f1), 0) + v
    r0, r1 = sorted({k[0] for k in pairs}), sorted({k[1] for k in pairs})
    tab = np.array([[pairs.get((u, w), 0) for w in r1] for u in r0])
    assert chi2_contingency(tab)[1] > 1e-3


def test_crosstalk_factor_one_is_base_noise_and_larger_lowers_fidelity(adjacent_pair):
    from qiskit_aer import AerSimulator
    cal, path = adjacent_pair
    progs = [tfim_trotter(3, 4), tfim_trotter(3, 4)]
    job = P.pack(cal, progs, buffer=0)
    assert all(set(p.qubits) <= set(path) for p in job.placements)
    assert all(job.exposed[p.index] for p in job.placements)
    # buffer 1 on the same region cannot hold both: the halves are adjacent
    assert P.allocate(cal, progs, buffer=1).unplaced == [1]
    shots = 20000
    base = P.emulate(job, shots, crosstalk_factor=1.0, seed=9)
    ref = AerSimulator.from_backend(cal.backend, seed_simulator=9).run(
        [p.circuit for p in job.placements], shots=shots).result()
    assert base == [qiskit_counts(ref.get_counts(i), p.n_clbits) for i, p in enumerate(job.placements)]
    hot = P.emulate(job, shots, crosstalk_factor=5.0, seed=9)
    ideal = _ideal(progs[0])
    for b, h in zip(base, hot):
        assert _hellinger(ideal, h) < _hellinger(ideal, b) - 0.01


def test_crosstalk_composes_with_base_error(adjacent_pair):
    from qiskit.quantum_info import average_gate_fidelity
    cal, _ = adjacent_pair
    job = P.pack(cal, [tfim_trotter(3, 4), tfim_trotter(3, 4)], buffer=0)
    a, b = job.exposed[job.placements[0].index][0]
    gate, c = cal.two_q_gate, 5.0
    nm = P.crosstalk_noise_model(cal, [(a, b)], c)
    # Private dict read on purpose: it is the only way to see the stored channel.
    r_base = 1 - average_gate_fidelity(P.base_noise_model(cal)._local_quantum_errors[gate][(a, b)].to_quantumchannel())
    r_new = 1 - average_gate_fidelity(nm._local_quantum_errors[gate][(a, b)].to_quantumchannel())
    lam = P.extra_depolarising(cal.edge_error(a, b), c)
    f_base = 1 - r_base * 5 / 4
    expect = 1 - ((1 - lam) * f_base + lam / 16) * 4 / 5 - 1 / 5      # composed, not replaced
    assert r_new == pytest.approx(expect, rel=1e-6)
    assert r_new == pytest.approx(c * cal.edge_error(a, b), rel=0.05)
    replaced = 1 - (1 - lam + lam / 16) * 4 / 5 - 1 / 5
    assert abs(r_new - replaced) > 0.1 * replaced
    assert P.crosstalk_noise_model(cal, [(a, b)], 1.0) is P.base_noise_model(cal)
    assert P.extra_depolarising(0.01, 1.0) == 0.0


# ---- accounting and the live-path helper -----------------------------------------------
def test_packing_cost():
    q = load_catalogue()["qpu"]
    one = P.packing_cost([40], shots=4000)
    four = P.packing_cost([40, 60, 30, 50], shots=4000)
    t60 = 4000 * (q.shot_overhead_s + 60 * q.layer_time_s)
    assert four.qpu_s_job == pytest.approx(t60) and four.qpu_s_per_program == pytest.approx(t60 / 4)
    assert four.price_per_program == pytest.approx((q.cost_per_task + 4000 * q.cost_per_shot) / 4)
    assert four.price_per_program_solo == pytest.approx(q.cost_per_task + 4000 * q.cost_per_shot)
    assert four.programs_per_free_plan == pytest.approx(600 / (t60 / 4))
    assert one.qpu_s_per_program == pytest.approx(one.qpu_s_per_program_solo)
    assert one.programs_per_free_plan == pytest.approx(one.programs_per_free_plan_solo)
    assert four.qpu_s_per_program_with_overhead == pytest.approx((t60 + q.job_overhead_s) / 4)
    with pytest.raises(ValueError):
        P.packing_cost([])


def test_run_packed_workloads_skips_and_scores():
    from qedge.quantum.backends.qiskit_backend import CalibratedIBMBackend
    ws = [make_workload("quantum_simulation", 0, seed=0, n=3, steps=2),
          make_workload("quantum_simulation", 1, seed=1, n=4, steps=2),
          make_workload("optimization", 2, seed=2, n=6),
          make_workload("inference", 3, seed=3)]
    be = CalibratedIBMBackend(devices=["FakeMumbaiV2"])
    r = P.run_packed_workloads(be, ws, shots=2000, buffer=1)
    assert r.emulated and r.calibration == "snapshot" and r.device == "fake_mumbai"
    assert [row["workload"] for row in r.results] == [0, 1] and not r.unplaced
    assert {s["id"] for s in r.skipped} == {2, 3} and all(s["reason"] for s in r.skipped)
    assert all(0.5 < row["quality"] <= 1.0 for row in r.results)
    assert r.cost.k == 2 and r.results[0]["qpu_s_share"] == pytest.approx(r.cost.qpu_s_job / 2)
    with pytest.raises(ValueError):
        P.run_packed_workloads(be, ws[2:])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_run_packed_workloads_live_branch_in_local_mode(small):
    """A calibration marked live takes the SamplerV2 branch; with a fake backend
    qiskit-ibm-runtime runs it locally, so the submission and split are exercised."""
    cal = copy.copy(small)
    cal.source = "live"
    ws = [make_workload("quantum_simulation", 0, seed=0, n=3, steps=2),
          make_workload("quantum_simulation", 1, seed=1, n=4, steps=2)]
    with pytest.raises(RuntimeError, match="QEDGE_HW_MAX_SHOTS"):
        P.run_packed_workloads(cal, ws, shots=10 ** 6)
    r = P.run_packed_workloads(cal, ws, shots=1000, buffer=1)
    assert not r.emulated and r.calibration == "live" and len(r.job_ids) == 1
    assert [row["workload"] for row in r.results] == [0, 1]
    assert all(0.5 < row["quality"] <= 1.0 for row in r.results)


def test_loss_standard_error_counts_shared_solo_draw_once():
    import pandas as pd
    from qedge.experiments import e7_packing as E
    se = E.fidelity_se(0.99, 4000)
    assert se == pytest.approx(np.sqrt(0.99 * 0.01 / 4000))
    # k copies of one circuit: k independent packed draws, ONE shared solo draw
    k = 4
    g = pd.DataFrame({"fidelity_packed_se": [se] * k, "fidelity_solo_se": [se] * k, "solo_id": [0] * k})
    assert E.loss_se(g) == pytest.approx(np.sqrt(k * se ** 2 + k ** 2 * se ** 2) / k)
    # k distinct circuits: the solo term averages down like the packed one
    g["solo_id"] = range(k)
    assert E.loss_se(g) == pytest.approx(np.sqrt(2 * k * se ** 2) / k)
