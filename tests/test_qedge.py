import numpy as np
import pytest

from qedge.orchestrator.estimator import CostModel, Placement
from qedge.orchestrator.policies import POLICIES, QEdgeCostPolicy, make_policy
from qedge.quantum.algorithms import (magnetisation, maxcut_optimum, quantum_kernel, random_graph,
                                      tfim_exact_magnetisation, tfim_trotter)
from qedge.quantum.backends import SIMULATOR_STACKS, NoiseSpec, get_backend
from qedge.quantum.circuit import Circuit, ghz, probabilities, total_variation
from qedge.resources import ResourceState, load_catalogue, scale
from qedge.simulation.simulator import SCENARIOS, Simulator, summarise
from qedge.workloads import KINDS, make_workload


def mixed_circuit():
    c = Circuit(3)
    c.h(0).rx(1, 0.7).ry(2, 1.1).cx(0, 1).rzz(1, 2, 0.9).rz(2, 0.4).cz(0, 2).swap(0, 1).s(1).sdg(2).h(2).x(0)
    return c


@pytest.mark.parametrize("stack", SIMULATOR_STACKS)
def test_stacks_agree_with_exact(stack):
    from qedge.experiments.e1_stacks import exact_probs
    c = mixed_circuit()
    res = get_backend(stack).run([c], 20000)
    assert total_variation(exact_probs(c), probabilities(res.counts[0])) < 0.03


@pytest.mark.parametrize("stack", SIMULATOR_STACKS)
def test_bit_order_is_qubit0_first(stack):
    c = Circuit(3).x(0)
    assert get_backend(stack).run([c], 50).counts[0] == {"100": 50}


def test_noise_is_equivalent_across_stacks():
    c = mixed_circuit()
    noise = NoiseSpec(0.02, 0.05)
    dists = [probabilities(get_backend(s, noise).run([c], 20000).counts[0]) for s in SIMULATOR_STACKS]
    for d in dists[1:]:
        assert total_variation(dists[0], d) < 0.03


def test_ghz_counts():
    counts = get_backend("qiskit").run([ghz(4)], 2000).counts[0]
    assert set(counts) <= {"0000", "1111"}


def test_inverse_gives_identity():
    c = mixed_circuit()
    c2 = Circuit(3).compose(c).compose(c.inverse())
    assert get_backend("qiskit").run([c2], 200).counts[0] == {"000": 200}


def test_tfim_matches_exact():
    r = get_backend("cirq").run([tfim_trotter(5, 3)], 40000)
    assert abs(magnetisation(r.counts[0]) - tfim_exact_magnetisation(5, 3)) < 0.02


def test_maxcut_optimum_small():
    assert maxcut_optimum(3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]) == 2.0


def test_kernel_symmetric_unit_diagonal():
    X = np.random.default_rng(0).uniform(0, 1, (5, 2))
    K, _ = quantum_kernel(get_backend("qiskit"), X, shots=256)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1)


def test_hardware_falls_back_without_credentials(monkeypatch):
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    monkeypatch.delenv("BRAKET_DEVICE_ARN", raising=False)
    from qedge.quantum.backends.ibm_backend import IBMQuantumBackend
    monkeypatch.setattr(IBMQuantumBackend, "credentials_present", staticmethod(lambda: False))
    assert getattr(get_backend("ibm"), "fallback_from", None) == "ibm"
    assert getattr(get_backend("braket"), "fallback_from", None) == "braket"
    with pytest.raises(RuntimeError):
        get_backend("ibm", allow_fallback=False)


# ---------------------------------------------------------------- orchestration

def test_classical_only_kinds_never_go_quantum():
    res = load_catalogue()
    model = CostModel(res)
    w = make_workload("inference", seed=0)
    assert not model.estimate(w, Placement("cloud_cpu", "qpu")).feasible


def test_privacy_keeps_data_on_edge():
    res = load_catalogue()
    pol = QEdgeCostPolicy(res)
    w = make_workload("inference", seed=0, privacy=True, batch=4)
    d = pol.decide(w, ResourceState())
    assert res[d.placement.classical].is_edge and d.placement.quantum is None


def test_too_many_qubits_blocks_quantum():
    res = load_catalogue()
    w = make_workload("quantum_simulation", seed=0, n=200, steps=4)
    assert not CostModel(res).estimate(w, Placement("cloud_cpu", "qpu")).feasible


def test_today_qpu_rejected_better_qpu_accepted_for_deep_wide_simulation():
    w = make_workload("quantum_simulation", seed=0, n=64, steps=36, deadline_s=3600)
    today = QEdgeCostPolicy(load_catalogue()).decide(w, ResourceState())
    future = QEdgeCostPolicy(scale(load_catalogue(), qpu__p1=0.01, qpu__p2=0.01, qpu__readout_err=0.01,
                                   qpu__cost_per_task=0.1, qpu__cost_per_shot=0.1)).decide(w, ResourceState())
    assert today.placement.quantum != "qpu"
    assert future.placement.quantum == "qpu"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("policy", list(POLICIES))
def test_every_policy_returns_a_placement(kind, policy):
    pol = make_policy(policy, load_catalogue())
    d = pol.decide(make_workload(kind, seed=1), ResourceState())
    assert d.placement.classical in load_catalogue()


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_simulator_runs(scenario):
    sim = Simulator(scenario, seed=0)
    wl = sim.workloads(60)
    df = sim.run("qedge_adaptive", wl)
    assert len(df) == 60 and df.latency_s.ge(0).all()
    assert summarise(df).deadline_hit_rate.between(0, 1).all()


def test_qedge_beats_naive_baselines_on_deadlines():
    sim = Simulator("baseline", seed=3)
    wl = sim.workloads(400)
    hit = {p: sim.run(p, wl).deadline_met.mean() for p in ["qedge_static", "cloud_only", "quantum_first", "random"]}
    assert hit["qedge_static"] > max(hit["cloud_only"], hit["quantum_first"], hit["random"])


def test_live_orchestrator_runs_each_kind(tmp_path):
    from qedge.orchestrator.engine import Orchestrator
    orch = Orchestrator(log_path=str(tmp_path / "log.jsonl"))
    for kind, kw in [("inference", {"batch": 4}), ("optimization", {"n": 6}),
                     ("kernel_classification", {"features": 2, "n_train": 12}),
                     ("quantum_simulation", {"n": 5, "steps": 2})]:
        rep = orch.submit(kind, **kw)
        assert rep.error is None and 0 <= rep.result.quality <= 1
    rep = orch.submit("optimization", force="cloud_cpu+quantum_sim", n=5)
    assert rep.result.mode == "hybrid" and rep.result.quality > 0.7
