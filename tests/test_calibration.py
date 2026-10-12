import copy
import json

import numpy as np
import pytest

from qedge.orchestrator.estimator import CostModel, Placement
from qedge.quantum import calibration as C
from qedge.quantum.backends import get_backend
from qedge.quantum.backends.qiskit_backend import to_qiskit
from qedge.quantum.circuit import ghz
from qedge.resources import ResourceState, load_catalogue
from qedge.workloads import make_workload


@pytest.fixture(scope="module")
def torino():
    return C.snapshot("FakeTorino")


@pytest.fixture(scope="module")
def two_devices():
    return [C.snapshot("FakeTorino"), C.snapshot("FakeAachen")]


def test_snapshot_has_per_qubit_data(torino):
    s = torino.summary()
    assert torino.source == "snapshot" and torino.num_qubits == 133
    assert len(torino.readout) == 133 and len(torino.two_q) > 100
    assert 1e-4 < s["median_2q_error"] < 0.05 and 1e-3 < s["median_readout_error"] < 0.1


def test_json_roundtrip(torino):
    back = C.DeviceCalibration.from_json(json.loads(json.dumps(torino.to_json())))
    assert back.two_q == torino.two_q and back.readout == torino.readout


def test_esp_counts_errors_and_rejects_broken(torino):
    (a, b), _ = min(torino.two_q.items(), key=lambda kv: kv[1])
    one = C.esp(torino, [("cz", (a, b))], [a, b])
    two = C.esp(torino, [("cz", (a, b))] * 2, [a, b])
    assert 0 < two < one < 1
    broken = copy.deepcopy(torino)
    broken.backend = torino.backend
    broken.two_q[(a, b)] = broken.two_q[(b, a)] = 1.0
    assert C.esp(broken, [("cz", (a, b))], [a, b]) == 0.0


def test_best_layout_beats_or_matches_default_and_is_executable(torino):
    qc = to_qiskit(ghz(5))
    choice = C.best_layout(torino, qc)
    _, l1 = C.default_layout_circuit(torino, qc)
    assert choice.esp >= choice.default_esp - 1e-12 and choice.esp >= l1 - 1e-12
    for ins in choice.circuit.data:
        if len(ins.qubits) == 2:
            qs = tuple(choice.circuit.find_bit(q).index for q in ins.qubits)
            assert qs in torino.two_q and torino.two_q[qs] < C.BROKEN
    assert len(choice.layout) == 5


def test_select_device_prefers_better_device(two_devices):
    best, allc = C.select_device(to_qiskit(ghz(6)), two_devices)
    assert best.esp == max(c.esp for c in allc)
    assert best.device == "fake_aachen"  # its median 2q error is less than half of Torino's


def test_region_figures_and_best_device(two_devices):
    f = C.region_figures(two_devices[0], 6)
    assert len(f["region"]) == 6 and 0 < f["p2"] < 0.05 and 0 < f["readout_err"] < 0.2
    w = make_workload("quantum_simulation", 0, seed=0, n=6, steps=2)
    best = C.best_device_figures(w.quantum, two_devices)
    assert best["device"] == "fake_aachen" and 0 < best["fidelity"] <= 1


def test_drift_flags_worse_coupler(torino):
    new = copy.deepcopy(torino)
    new.backend = torino.backend
    (a, b) = next(k for k, e in torino.two_q.items() if k[0] < k[1] and 0.005 < e < 0.05)
    new.two_q[(a, b)] = new.two_q[(b, a)] = 0.9
    d = C.drift(torino, new)
    assert (a, b) in d["newly_broken_couplers"] and d["degraded"]
    assert not C.drift(torino, torino)["degraded"]


def test_calibrated_backend_runs_on_chosen_device():
    be = get_backend("ibm-auto")
    be._device_names = ["FakeTorino", "FakeAachen"]
    be.refresh()
    res = be.run([ghz(4)], 2000)
    c = res.counts[0]
    assert res.meta["device"] == "fake_aachen" and res.meta["calibration"] == "snapshot"
    assert (c.get("0000", 0) + c.get("1111", 0)) / 2000 > 0.85
    assert not res.hardware


def test_cost_model_uses_calibrated_figures(two_devices):
    cat = load_catalogue()
    w = make_workload("quantum_simulation", 0, seed=0, n=8, steps=3)
    p = Placement("cloud_cpu", "qpu")
    static = CostModel(cat).estimate(w, p, ResourceState())
    st = ResourceState(qpu_figures={"qpu": C.best_device_figures(w.quantum, two_devices)})
    cal = CostModel(cat).estimate(w, p, st)
    assert static.qpu_device is None and cal.qpu_device.startswith("fake_aachen")
    assert cal.fidelity != pytest.approx(static.fidelity)


def test_trace_scenario(tmp_path):
    from qedge.simulation.simulator import Simulator, trace_scenario
    path = tmp_path / "trace.jsonl"
    with pytest.raises(FileNotFoundError):
        trace_scenario(str(path))
    path.write_text("".join(json.dumps({"t": i, "name": "d", "pending_jobs": j}) + "\n"
                            for i, j in enumerate([1, 5, 30, 200])))
    sc = trace_scenario(str(path))
    assert sc.queue_trace == [20.0, 100.0, 600.0, 4000.0]
    sim = Simulator(sc)
    df = sim.run("qedge_adaptive", sim.workloads(40))
    assert len(df) == 40 and np.isfinite(df.latency_s).all()
