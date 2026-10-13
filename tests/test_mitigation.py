import math

import numpy as np
import pytest

from qedge.quantum import mitigation as M
from qedge.quantum.algorithms import magnetisation, tfim_exact_magnetisation, tfim_trotter
from qedge.quantum.backends import NoiseSpec
from qedge.quantum.backends.base import QuantumBackend
from qedge.quantum.backends.qiskit_backend import CalibratedIBMBackend, QiskitAerBackend, to_qiskit
from qedge.quantum.circuit import Circuit


# ---- readout correction ----------------------------------------------------------
def _noisy_counts(probs: dict[str, float], p01, p10, shots, rng) -> dict[str, int]:
    keys = list(probs)
    true = np.array([[int(b) for b in k] for k in rng.choice(keys, size=shots, p=[probs[k] for k in keys])])
    u = rng.random(true.shape)
    flip = np.where(true == 0, u < np.asarray(p01), u < np.asarray(p10))
    meas = true ^ flip
    out: dict[str, int] = {}
    for row in meas:
        k = "".join(map(str, row))
        out[k] = out.get(k, 0) + 1
    return out


def _exact_z(probs, qs):
    return sum(p * np.prod([1 - 2 * int(k[q]) for q in qs]) for k, p in probs.items())


def test_readout_inverse_recovers_true_expectations_on_synthetic_counts():
    rng = np.random.default_rng(3)
    probs = {"000": 0.40, "011": 0.25, "101": 0.20, "110": 0.10, "111": 0.05}
    p01, p10 = [0.02, 0.05, 0.08], [0.04, 0.03, 0.12]       # asymmetric, different per qubit
    shots = 200_000
    counts = _noisy_counts(probs, p01, p10, shots, rng)
    ro = M.ReadoutCalibration(p01, p10)
    raw_err = []
    for qs in [(0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)]:
        true = _exact_z(probs, qs)
        raw = M.expectation_from_counts(counts, qs)
        fixed = M.corrected_expectation(counts, qs, ro)
        # 1/sqrt(shots) = 0.0022; the correction divides by (1-e0-e1) per qubit, so allow ~5 sigma.
        assert abs(fixed - true) < 0.012, (qs, fixed, true)
        raw_err.append(abs(raw - true))
    assert max(raw_err) > 0.1          # the flips did bias the raw values
    q = M.correct_distribution(counts, ro)
    assert sum(abs(q.get(k, 0) - probs.get(k, 0)) for k in set(q) | set(probs)) / 2 < 0.01
    # The calibration circuits' counts give back the flip probabilities.
    zero = _noisy_counts({"000": 1.0}, p01, p10, shots, rng)
    one = _noisy_counts({"111": 1.0}, p01, p10, shots, rng)
    est = M.readout_from_counts(zero, one)
    assert np.allclose(est.p01, p01, atol=0.003) and np.allclose(est.p10, p10, atol=0.003)


def test_plain_expectation_matches_existing_magnetisation():
    counts = {"0101": 30, "1111": 12, "0000": 50, "1000": 8}
    assert M.expectation_from_counts(counts, M.magnetisation_observable(4)) == pytest.approx(magnetisation(counts), abs=1e-12)
    assert M.z_observable("ZIZ") == {(0, 2): 1.0}
    assert M.zz_observable([(1, 0, 2.0), (1, 2)]) == {(0, 1): 2.0, (1, 2): 1.0}


# ---- folding -------------------------------------------------------------------------
def _small_isa(basis, n=4, steps=2):
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=basis,
                                      coupling_map=CouplingMap.from_line(n + 1), seed_transpiler=1)
    return pm.run(to_qiskit(tfim_trotter(n, steps)))


def test_sx_dagger_translation_is_exact():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import SXdgGate
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(1)
    for op, qs in M._inverse_ops([(QuantumCircuit(1).sx(0)[0].operation, (qc.qubits[0],))]):
        qc.append(op, qs)
    assert Operator(qc).equiv(Operator(SXdgGate()))


@pytest.mark.parametrize("basis", [["cz", "rz", "sx", "x"], ["ecr", "rz", "sx", "x"]])
@pytest.mark.parametrize("scale", [3.0, 5.0, 2.0])
def test_fold_isa_preserves_the_ideal_circuit(basis, scale):
    from qiskit.quantum_info import Operator, Statevector
    isa = _small_isa(basis)
    folded, achieved = M.fold_isa(isa, scale)
    assert achieved == pytest.approx(scale, abs=0.05)
    assert set(folded.count_ops()) - {"barrier"} <= set(basis) | {"measure"}
    a, b = isa.remove_final_measurements(inplace=False), folded.remove_final_measurements(inplace=False)
    assert Operator(a).equiv(Operator(b))
    # Noiseless simulation: same distribution, and the bits are still read from the same qubits.
    assert np.allclose(Statevector(a).probabilities(), Statevector(b).probabilities(), atol=1e-9)
    assert M.measured_physical_qubits(folded) == M.measured_physical_qubits(isa)
    two = M.two_qubit_gate_count(isa)
    assert M.two_qubit_gate_count(folded) == pytest.approx(scale * two, rel=0.15)


def test_folded_circuit_survives_re_transpilation():
    # Barriers between segments stop optimisation level 1 from cancelling U U^dagger.
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    isa = _small_isa(["cz", "rz", "sx", "x"])
    folded, _ = M.fold_isa(isa, 3.0)
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["cz", "rz", "sx", "x"],
                                      coupling_map=CouplingMap.from_line(5), seed_transpiler=1)
    again = pm.run(folded)
    assert M.two_qubit_gate_count(again) >= 2.9 * M.two_qubit_gate_count(isa)
    # Without the barriers the same pass manager removes the folds: the reason for ISA folding.
    no_bar = folded.copy_empty_like()
    for ins in folded.data:
        if ins.operation.name != "barrier":
            no_bar.append(ins)
    assert M.two_qubit_gate_count(pm.run(no_bar)) < 1.5 * M.two_qubit_gate_count(isa)


def test_fold_circuit_neutral_preserves_state():
    from qiskit.quantum_info import Statevector
    c = tfim_trotter(3, 2)
    for s in (3.0, 2.0):
        f, a = M.fold_circuit(c, s)
        assert a == pytest.approx(s, abs=0.1)
        p = Statevector(to_qiskit(c, measure=False)).probabilities()
        q = Statevector(to_qiskit(f, measure=False)).probabilities()
        assert np.allclose(p, q, atol=1e-9)


# ---- extrapolation and cost -------------------------------------------------------------
def test_extrapolation_methods_are_exact_on_their_own_models():
    x = [1.0, 3.0, 5.0]
    assert M.extrapolate(x, [0.9 - 0.05 * s for s in x], "linear") == pytest.approx(0.9)
    assert M.extrapolate(x, [0.9 - 0.05 * s + 0.004 * s * s for s in x], "richardson") == pytest.approx(0.9)
    assert M.extrapolate(x, [0.8 * math.exp(-0.1 * s) for s in x], "exponential") == pytest.approx(0.8)
    assert np.allclose(M.extrapolation_weights(x, "richardson"), [15 / 8, -5 / 4, 3 / 8])
    assert np.allclose(M.extrapolation_weights(x, "linear"), [13 / 12, 1 / 3, -5 / 12])
    # Sign change: the exponential model cannot hold, falls back to linear.
    assert M.extrapolate(x, [0.1, -0.05, -0.2], "exponential") == pytest.approx(M.extrapolate(x, [0.1, -0.05, -0.2], "linear"))


def test_mitigation_cost_arithmetic():
    none = M.mitigation_cost("none", depth=40, shots=1000)
    assert none["shots_multiplier"] == 1 and none["extra_circuits"] == 0 and none["stderr_multiplier"] == 1.0
    ro = M.mitigation_cost("readout", depth=40, shots=1000, n_circuits=4)
    assert ro["extra_circuits"] == 2 and ro["shots_multiplier"] == pytest.approx(1.5) and ro["depth_multiplier"] == 1
    assert M.mitigation_cost("readout", 40, 1000, readout_source="device")["extra_circuits"] == 0
    z = M.mitigation_cost("zne", depth=40, shots=1000, n_circuits=2, extrapolation="richardson")
    assert z["circuits_multiplier"] == 3 and z["extra_circuits"] == 4 and z["total_shots"] == 6000
    assert z["depth_multiplier"] == 5 and z["mean_depth_multiplier"] == 3 and z["max_depth"] == 200
    assert z["stderr_multiplier"] == pytest.approx(math.sqrt((15 / 8) ** 2 + (5 / 4) ** 2 + (3 / 8) ** 2))
    assert z["shots_for_equal_stderr"] == math.ceil(3 * z["stderr_multiplier"] ** 2 * 2000)
    both = M.mitigation_cost("readout+zne", depth=10, shots=100)
    assert both["extra_circuits"] == 4 and both["total_shots"] == 500
    assert M.mitigation_cost("zne", 10, 100, extrapolation="exponential")["stderr_multiplier"] is None


# ---- end to end ----------------------------------------------------------------------------
class _Recorder(QuantumBackend):
    name = "recorder"

    def __init__(self):
        super().__init__()
        self.calls = []

    def _run(self, circuits, shots):
        self.calls.append((list(circuits), shots))
        return [{"0" * c.n_qubits: shots // 2, "1" * c.n_qubits: shots - shots // 2 - 7, "1" + "0" * (c.n_qubits - 1): 7}
                for c in circuits], {}


def test_level_none_is_the_existing_path():
    be = _Recorder()
    circs = [tfim_trotter(3, 1), tfim_trotter(3, 2)]
    r = M.mitigated_expectations(be, circs, M.magnetisation_observable(3), shots=1000, level="none")
    assert len(be.calls) == 1 and be.calls[0][0] == circs and be.calls[0][1] == 1000
    direct = be.run(circs, 1000).counts
    assert np.allclose(r.values[:, 0], [magnetisation(c) for c in direct])
    assert np.allclose(r.values, r.raw_values) and r.metadata["circuits_run"] == 2 and r.metadata["total_shots"] == 2000


@pytest.fixture(scope="module")
def noisy_aer_runs():
    be = QiskitAerBackend(NoiseSpec(p1=1e-3, p2=1e-2))
    c = tfim_trotter(6, 3)
    obs = M.magnetisation_observable(6)
    runs = {lvl: M.mitigated_expectations(be, [c], obs, shots=20000, level=lvl, seed=7, n_boot=300)
            for lvl in ("none", "zne")}
    runs["richardson"] = M.mitigated_expectations(be, [c], obs, shots=20000, level="zne", seed=7, n_boot=300,
                                                  extrapolation="richardson")
    return runs, tfim_exact_magnetisation(6, 3)


def test_zne_reduces_error_on_noisy_aer(noisy_aer_runs):
    runs, exact = noisy_aer_runs
    e_none = abs(runs["none"].values[0, 0] - exact)
    e_zne = abs(runs["zne"].values[0, 0] - exact)
    # Measured with seed 7: |error| 0.0331 (se 0.0031) unmitigated, 0.0082 (se 0.0036) with ZNE
    # (linear), 0.0016 (se 0.0073) with Richardson. Over 20 seeds at 8000 shots the means were
    # 0.0329 vs 0.0089, so the margin below is ~4 standard errors of the difference.
    assert e_zne < e_none - 0.015
    assert abs(runs["richardson"].values[0, 0] - exact) < e_none - 0.015
    assert runs["zne"].metadata["fold_level"] == "isa"
    assert runs["zne"].metadata["two_qubit_gates_per_scale"] == [[15], [45], [75]]


def test_stderr_grows_with_zne(noisy_aer_runs):
    runs, _ = noisy_aer_runs
    se_none, se_zne, se_rich = (runs[k].stderrs[0, 0] for k in ("none", "zne", "richardson"))
    # Weights predict sqrt(sum w^2) = 1.21x (linear) and 2.29x (Richardson) at equal per-shot variance;
    # measured (seed 7) 0.0031 -> 0.0036 -> 0.0073. Higher scales sit closer to 0, so slightly more.
    assert 1.1 * se_none < se_zne < se_rich and se_rich > 2.0 * se_none
    # And the shots it cost: three circuits instead of one.
    assert runs["zne"].metadata["total_shots"] == 3 * runs["none"].metadata["total_shots"]


@pytest.fixture(scope="module")
def torino_auto():
    return CalibratedIBMBackend(devices=["FakeTorino"])


def test_folding_survives_on_the_ibm_auto_path(torino_auto, monkeypatch):
    """Count the two-qubit gates of the circuits Aer actually executes."""
    from qiskit_aer import AerSimulator
    executed = []
    orig = AerSimulator.run

    def spy(self, circuits, *a, **k):
        executed.extend(circuits if isinstance(circuits, list) else [circuits])
        return orig(self, circuits, *a, **k)

    monkeypatch.setattr(AerSimulator, "run", spy)
    c = tfim_trotter(4, 2)
    r = M.mitigated_expectations(torino_auto, [c], M.magnetisation_observable(4), shots=500,
                                 level="readout+zne", seed=1, n_boot=20)
    assert len(executed) == 5 and r.metadata["circuits_run"] == 5    # 3 scales + 2 calibration circuits
    base = M.two_qubit_gate_count(executed[0])
    assert base == M.two_qubit_gate_count(torino_auto.choose(c).circuit) > 0
    assert [M.two_qubit_gate_count(q) for q in executed[:3]] == [base, 3 * base, 5 * base]
    assert [M.two_qubit_gate_count(q) for q in executed[3:]] == [0, 0]
    meas = M.measured_physical_qubits(torino_auto.choose(c).circuit)
    assert all(M.measured_physical_qubits(q) == meas for q in executed[:3])
    assert sorted(M.measured_physical_qubits(executed[3])) == sorted(meas)


def test_readout_correction_uses_the_measured_physical_qubits(torino_auto):
    c = tfim_trotter(5, 1)
    meas = M.measured_physical_qubits(torino_auto.choose(c).circuit)
    cal = torino_auto.devices[0]
    dev = M.mitigated_expectations(torino_auto, [c], (0,), shots=200, level="readout", readout_source="device", n_boot=10)
    assert dev.metadata["measured_physical_qubits"] == [meas]
    assert dev.metadata["readout_errors"] == {q: (round(cal.readout[q], 5), round(cal.readout[q], 5)) for q in meas}
    # Calibration circuits on those qubits measure what the emulator's noise model holds for them
    # (Aer uses the symmetric Target error), qubit by qubit.
    run = M.mitigated_expectations(torino_auto, [c], (0,), shots=20000, level="readout", seed=3, n_boot=10)
    for q in meas:
        e0, e1 = run.metadata["readout_errors"][q]
        tol = 4 * math.sqrt(cal.readout[q] / 20000) + 0.004   # shot noise plus x-gate / relaxation error
        assert abs(e0 - cal.readout[q]) < tol and abs(e1 - cal.readout[q]) < tol, (q, e0, e1, cal.readout[q])


def test_circuit_level_folding_on_cirq_scales_the_noise():
    from qedge.quantum.backends.cirq_backend import CirqBackend
    be = CirqBackend(NoiseSpec(p1=2e-3, p2=2e-2))
    c = tfim_trotter(3, 3)
    r = M.mitigated_expectations(be, [c], M.magnetisation_observable(3), shots=4000, level="zne", n_boot=50)
    assert r.metadata["fold_level"] == "circuit" and r.metadata["two_qubit_gates_per_scale"] == [[6], [18], [30]]
    exact = tfim_exact_magnetisation(3, 3)
    err = np.abs(r.per_scale[:, 0, 0] - exact)
    # No cancellation: the error at scale 5 is clearly larger than at scale 1.
    assert err[2] > err[0] + 0.03
