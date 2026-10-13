"""VQE and QAOA task scheduling: physics checks, baselines, and wiring into the orchestrator."""
import itertools

import numpy as np
import pytest

from qedge.quantum.backends import SIMULATOR_STACKS, get_backend
from qedge.quantum.scheduling import (ising_qaoa_circuit, run_scheduling_qaoa, scheduling_annealing,
                                      scheduling_instance)
from qedge.quantum.vqe import (H2_NUCLEAR_REPULSION, _rotosolve_angle, ansatz, energy_from_counts, energy_quality,
                               exact_energy, h2_hamiltonian, measurement_groups, n_params, run_vqe,
                               spectrum_bounds, tfim_hamiltonian, vqe_budget, with_basis)
from qedge.workloads import make_workload


def test_h2_ground_energy_matches_full_ci():
    e0, _ = spectrum_bounds(h2_hamiltonian(), 2)
    assert e0 == pytest.approx(-1.857275, abs=1e-5)
    assert e0 + H2_NUCLEAR_REPULSION == pytest.approx(-1.137306, abs=1e-5)


def test_tfim_lanczos_matches_dense_diagonalisation():
    from qedge.quantum.vqe import hamiltonian_matrix
    H = tfim_hamiltonian(11)
    e0, emax = spectrum_bounds(H, 11)  # 11 qubits takes the Lanczos path
    ev = np.linalg.eigvalsh(hamiltonian_matrix(H, 11).toarray())
    assert (e0, emax) == (pytest.approx(ev[0]), pytest.approx(ev[-1]))
    assert emax == pytest.approx(-e0)  # the open TFIM spectrum is symmetric, so pure noise scores 0.5
    assert len(measurement_groups(H)) == 2 and len(measurement_groups(h2_hamiltonian())) == 2


@pytest.mark.parametrize("stack", SIMULATOR_STACKS)
def test_energy_estimate_from_shots_matches_state_vector(stack):
    H, n = tfim_hamiltonian(3), 3
    x = np.random.default_rng(0).uniform(-1, 1, n_params(n, 1))
    groups = measurement_groups(H)
    r = get_backend(stack).run([with_basis(ansatz(n, x, 1), g) for g in groups], 40000)
    assert energy_from_counts(H, groups, r.counts) == pytest.approx(exact_energy(H, x, n, 1), abs=0.06)


def test_rotosolve_step_is_exact_on_a_sinusoid():
    f = lambda t: 0.3 + 1.2 * np.cos(t - 0.7)  # noqa: E731  minimum at 0.7 + pi
    t = _rotosolve_angle(f(0.2 - np.pi / 2), f(0.2), f(0.2 + np.pi / 2), 0.2)
    assert np.cos(t - (0.7 + np.pi)) == pytest.approx(1.0, abs=1e-9)


def test_vqe_reaches_h2_ground_state_noiselessly():
    H = h2_hamiltonian()
    e0, emax = spectrum_bounds(H, 2)
    r = run_vqe(get_backend("qiskit"), H, 2, 1, shots=4000, sweeps=5, seed=0)
    assert abs(exact_energy(H, r.params, 2, 1) - e0) < 5e-3  # the angles found, without shot noise
    assert energy_quality(r.energy, e0, emax) > 0.98
    assert (r.n_circuit_evals, r.n_jobs) == vqe_budget(n_params(2, 1), 2, sweeps=5)


def test_scheduling_ising_form_equals_cost_everywhere():
    inst = scheduling_instance(6, 0.4, seed=3)
    J, h, const = inst.ising()
    costs = inst.all_costs()
    for k, bits in enumerate("".join(b) for b in itertools.product("01", repeat=6)):
        s = [1 - 2 * int(b) for b in bits]
        e = const + sum(v * s[i] * s[j] for (i, j), v in J.items()) + sum(a * b for a, b in zip(h, s))
        assert e == pytest.approx(inst.cost(bits)) == pytest.approx(costs[k])
    assert ising_qaoa_circuit(6, J, h, [0.1], [0.2]).n_2q == len(J)


def test_scheduling_baselines_and_qaoa_find_good_assignments():
    inst = scheduling_instance(8, 0.3, seed=1)
    _, opt = inst.optimum()
    assert scheduling_annealing(inst, 1000, 0)[1] == pytest.approx(opt)
    r = run_scheduling_qaoa(get_backend("qiskit"), inst, p=1, shots=1000, maxiter=15, seed=0)
    assert opt / r.best_cost >= 0.9 and r.best_cost == pytest.approx(inst.cost(r.best_bits))


@pytest.mark.parametrize("kind,kw", [("ground_state", {"model": "h2"}), ("ground_state", {"model": "tfim", "n": 30}),
                                     ("scheduling", {"n": 10}), ("scheduling", {"n": 40})])
def test_new_workloads_have_quantum_profiles(kind, kw):
    from qedge.executor import representative_circuit
    w = make_workload(kind, seed=0, **kw)
    assert w.quantum_capable and w.quantum.n_qubits == w.params["n"]
    if w.params["n"] <= 12:
        assert representative_circuit(w).n_qubits == w.params["n"]


def test_live_orchestrator_runs_new_kinds(tmp_path):
    from qedge.orchestrator.engine import Orchestrator
    orch = Orchestrator(log_path=str(tmp_path / "log.jsonl"))
    for kind, kw in [("ground_state", {"model": "h2"}), ("ground_state", {"model": "tfim", "n": 4}),
                     ("scheduling", {"n": 6})]:
        rep = orch.submit(kind, **kw)
        assert rep.error is None and rep.result.quality == pytest.approx(1.0)
        rep = orch.submit(kind, force="cloud_cpu+quantum_sim", **kw)
        assert rep.error is None and rep.result.mode == "hybrid" and rep.result.quality > 0.85


def test_engine_keeps_large_chains_classical():
    from qedge.orchestrator.engine import Orchestrator
    orch = Orchestrator(log_path=None)
    _, d = orch.plan("ground_state", model="tfim", n=60, deadline_s=3600)
    assert "+" not in d.placement.label  # DMRG handles a 1D chain; the quantum gate should not open
