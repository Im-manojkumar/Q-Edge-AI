"""Idea 3 live: split_inference executed for real (executor, orchestrator, race mode, CLI).

Uses quick zoo models (few epochs; trained once into a temporary cache), so the
accuracies seen here are not the reported ones. The tests check what runs where:
the cut, the bytes that leave the edge, the quantum head's backend path,
cancellation, and that a race fallback runs its own placement. Slow and failing
"QPUs" are fake backends, as in tests/test_race.py; the packed path runs on the
local IBM device emulator (``ibm-auto`` without credentials).
"""
import copy
import threading
import time

import numpy as np
import pytest

from qedge.cli import main
from qedge.executor import Executor
from qedge.ml import nets, zoo
from qedge.ml.qhead import QuantumHead
from qedge.orchestrator import split
from qedge.orchestrator.engine import Orchestrator
from qedge.orchestrator.estimator import Placement
from qedge.quantum.backends import QuantumBackend
from qedge.quantum.backends.base import Cancelled, CancelToken
from qedge.workloads import make_workload


class FakeQPU(QuantumBackend):
    """Sleeps ``delay`` per run() and answers |0...0> (every <Z_i> = +1)."""

    name = "fake-qpu"
    framework = "fake"

    def __init__(self, delay=0.0):
        super().__init__()
        self.delay, self.calls = delay, 0

    def _run(self, circuits, shots):
        self.calls += 1
        time.sleep(self.delay)
        return [{"0" * c.n_qubits: shots} for c in circuits], {}


@pytest.fixture(scope="module")
def quick_cache(tmp_path_factory):
    cache = str(tmp_path_factory.mktemp("models"))
    zoo.get_model(4, cache_dir=cache, quick=True)
    return cache


@pytest.fixture
def live(quick_cache, monkeypatch):
    """Executors use the quick k=4 model; the profiles are restored afterwards
    (``Orchestrator()`` reloads them from results/split_profile.json)."""
    monkeypatch.setattr(Executor, "model_cache", quick_cache)
    monkeypatch.setattr(Executor, "quick_models", True)
    saved = copy.deepcopy(split.PROFILES)
    yield zoo.get_model(4, cache_dir=quick_cache, quick=True)
    split.PROFILES.clear()
    split.PROFILES.update(saved)


def split_w(i=0, **kw):
    kw = {"profile": "digits", "k": 4, "batch": 6, "privacy": False, "deadline_s": 30.0, **kw}
    return make_workload("split_inference", seed=i, **kw)


def run(ex, w, label):
    p = Placement.parse(label)
    return ex.run(w, p.quantum, placement=p)


def batch(r):
    data = zoo.data()
    idx = np.array(r.details["images"])
    return data[4][idx], data[5][idx]


# ---------------------------------------------------------------- the zoo
def test_zoo_saves_every_head_and_reloads_identical_predictions(quick_cache):
    tr = zoo.get_model(4, cache_dir=quick_cache, quick=True)
    assert zoo.get_model(4, cache_dir=quick_cache, quick=True) is tr              # held in memory
    loaded, info = zoo.load(zoo.model_path(4, cache_dir=quick_cache, quick=True))
    assert info["settings"] == zoo.settings(quick=True) and info["quick"] and info["train_seconds"] > 0
    assert set(loaded.heads) == set(nets.HEADS)
    X = zoo.data()[4][:40]
    Z = tr.model.embed(X)
    assert np.array_equal(loaded.model.embed(X), Z)
    for h in nets.HEADS:
        assert np.array_equal(loaded.accuracy(h, X, zoo.data()[5][:40]), tr.accuracy(h, X, zoo.data()[5][:40]))
    q, lq = tr.heads["quantum"], loaded.heads["quantum"]
    assert isinstance(lq, QuantumHead) and lq.layers == 2
    for key in ("w", "a", "b"):                                                 # the circuit angles are cached too
        assert np.array_equal(lq.params[key], q.params[key])
    assert np.array_equal(lq.predict_exact(Z), q.predict_exact(Z))
    assert loaded.fits["quantum"].best_val_acc == tr.fits["quantum"].best_val_acc
    with pytest.raises(ValueError, match="k must be"):
        zoo.get_model(9, cache_dir=quick_cache, quick=True)


def test_zoo_trains_once_under_concurrency_and_retrains_on_new_settings(quick_cache, tmp_path, monkeypatch):
    """Training is stubbed (the cached quick model is returned after a pause) to count calls."""
    tr = zoo.get_model(4, cache_dir=quick_cache, quick=True)
    calls = []

    def fake_train(data, k, **kw):
        calls.append(kw)
        time.sleep(0.3)
        return nets.TransferResult(copy.deepcopy(tr.model), tr.heads, tr.fits, tr.backbone_fit)

    monkeypatch.setattr(nets, "train_transfer", fake_train)
    cache = str(tmp_path / "m")
    got = []
    threads = [threading.Thread(target=lambda: got.append(zoo.get_model(4, cache_dir=cache, quick=True)))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len({id(g) for g in got}) == 1                  # one training run, one model
    assert got[0].model.meta["zoo"] == "trained" and calls[0]["backbone_kw"] == zoo.QUICK_BACKBONE
    zoo.clear_memory()
    assert zoo.get_model(4, cache_dir=cache, quick=True).model.meta["zoo"] == "loaded"
    assert len(calls) == 1
    # A file trained with other settings is not passed off as the current model.
    zoo.clear_memory()
    monkeypatch.setattr(zoo, "QUICK_HEAD", {**zoo.QUICK_HEAD, "epochs": 11})
    assert zoo.get_model(4, cache_dir=cache, quick=True).model.meta["zoo"] == "trained" and len(calls) == 2
    zoo.clear_memory()


# ---------------------------------------------------------------- the executor
def test_executor_runs_each_classical_placement_and_counts_what_leaves_the_edge(live):
    ex = Executor()
    w = split_w(3, batch=10)
    rs = {lab: run(ex, w, lab) for lab in ("edge_gpu", "cloud_gpu", "edge_gpu>conv1>cloud_gpu",
                                          "edge_cpu>conv2>cloud_cpu", "edge_gpu>embedding>cloud_gpu")}
    expect = {"edge_gpu": ([], 0), "cloud_gpu": (["input"], 256), "edge_gpu>conv1>cloud_gpu": (["conv1"], 512),
              "edge_cpu>conv2>cloud_cpu": (["conv2"], 256), "edge_gpu>embedding>cloud_gpu": (["embedding"], 16)}
    X, y = batch(rs["edge_gpu"])
    whole = live.with_head(w.params["head"]).predict(X)
    for lab, r in rs.items():
        d = r.details
        assert r.kind == "split_inference" and r.mode == "classical" and r.backend is None
        assert d["placement"] == lab and (d["left_edge"], d["bytes_up_per_image"]) == expect[lab]
        assert d["bytes_up_total"] == 10 * expect[lab][1] and d["raw_data_left_edge"] == (lab == "cloud_gpu")
        assert d["images"] == rs["edge_gpu"].details["images"] and len(set(d["images"])) == 10
        assert d["head"] == w.params["head"] and d["model"]["quick"]
        # the remote half starts from the float32 tensor the edge would send; the answer is the same
        assert d["agree_unsplit"] == 1.0 and r.quality == pytest.approx(np.mean(whole == y))
    assert rs["edge_gpu>conv1>cloud_gpu"].details["cut"] == "conv1" and rs["cloud_gpu"].details["cut"] is None
    assert rs["edge_gpu"].details["remote_s"] == 0.0 and rs["edge_gpu"].details["bytes_down_per_image"] == 0
    assert rs["edge_gpu>conv1>cloud_gpu"].details["bytes_down_per_image"] == 40   # float32 logits come back
    # No placement: the whole model on the edge, as before placements existed.
    assert Executor().run(w).details["left_edge"] == []
    with pytest.raises(ValueError, match="assumption profile"):
        Executor().run(split_w(profile="camera"), placement=Placement("edge_gpu"))


def test_quantum_head_on_the_qiskit_simulator(live):
    ex = Executor(sim_stack="qiskit")
    w = split_w(5, batch=6, shots=4000)
    r = run(ex, w, "edge_cpu>embedding>quantum_sim")
    d = r.details
    assert r.mode == "hybrid" and r.backend == "qiskit-aer" and r.quantum_s > 0
    assert d["left_edge"] == ["embedding"] and d["bytes_up_per_image"] == 16 and not d["raw_data_left_edge"]
    assert d["path"] == "one circuit per image" and (d["jobs"], d["circuits"]) == (1, 6) and d["qpu_s"] is None
    assert d["head"] == "quantum" and d["bytes_down_per_image"] == 16
    assert d["agree_exact_quantum"] >= 0.8                     # shot noise can flip a near tie
    X, y = batch(r)
    q = live.heads["quantum"]
    exact = q.predict_exact(live.model.embed(X))
    assert d["exact_quantum_accuracy"] == pytest.approx(np.mean(exact == y))
    # The L = 0 ablation runs when the workload asks for it.
    r0 = run(ex, split_w(5, batch=3, qhead="quantum_l0"), "edge_cpu>embedding>quantum_sim")
    assert r0.details["head"] == "quantum_l0" and r0.method.endswith("(4 qubits, 0 layers)")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_quantum_head_on_ibm_auto_is_packed(live):
    ex = Executor(qpu_stack="ibm-auto")
    w = split_w(7, batch=3, shots=1000)
    r = run(ex, w, "edge_gpu>embedding>qpu")
    d = r.details
    assert r.mode == "hybrid" and r.backend == "ibm-auto" and r.method.endswith(", packed")
    assert d["path"] == "packed" and d["circuits"] == 3 and d["jobs"] == 1 < 3      # 3 heads side by side
    assert d["device"] and d["packing"]["per_job"] == 8 and d["packing"]["jobs"][0]["emulated"]
    assert 0 < d["qpu_s"] < d["qpu_s_unpacked"] and d["price_usd"] > 0 and "modelled" in d["qpu_s_note"]
    assert d["left_edge"] == ["embedding"] and 0.0 <= d["agree_exact_quantum"] <= 1.0
    # One image is not packed: one circuit through the calibrated backend, QPU time still modelled.
    r1 = run(ex, split_w(8, batch=1), "edge_gpu>embedding>qpu")
    assert r1.details["path"] == "one circuit per image" and r1.details["device"] and r1.details["qpu_s"] > 0


def test_quantum_head_respects_cancellation(live, monkeypatch):
    ex = Executor(qpu_stack="ibm-auto")
    tok = CancelToken()
    tok.cancel("test")
    for label in ("edge_gpu>embedding>qpu", "edge_gpu>embedding>quantum_sim"):
        with pytest.raises(Cancelled):
            p = Placement.parse(label)
            ex.run(split_w(batch=4), p.quantum, cancel=tok, placement=p)
    # Packed: the token is checked before every job, so a cancel between jobs stops the rest.
    from qedge.quantum import calibration as C
    from qedge.quantum import packing as P
    head = live.heads["quantum"]
    tok2, emulated = CancelToken(), []
    real = P.emulate

    def emulate_then_cancel(*a, **kw):
        emulated.append(1)
        tok2.cancel("deadline")
        return real(*a, **kw)

    monkeypatch.setattr(P, "emulate", emulate_then_cancel)
    Z = live.model.embed(zoo.data()[4][:2])
    with pytest.raises(Cancelled):
        head.predict_packed(C.snapshot("FakeTorino"), Z, shots=200, per_job=1, cancel=tok2)
    assert emulated == [1]


# ---------------------------------------------------------------- through the orchestrator
def test_submit_runs_the_decided_and_the_forced_split(live, tmp_path):
    o = Orchestrator("qedge_static", log_path=str(tmp_path / "log.jsonl"))
    rep = o.submit("split_inference", k=4, batch=5, privacy=False, deadline_s=30)
    assert rep.error is None and rep.result.details["placement"] == rep.decision.placement.label
    assert "split:" in rep.explain()
    rep = o.submit("split_inference", force="edge_gpu>conv2>cloud_cpu", k=4, batch=5, privacy=False)
    d = rep.result.details
    assert rep.error is None and rep.decision.policy.endswith("(forced)")
    assert (d["placement"], d["cut"], d["left_edge"], d["bytes_up_total"]) == ("edge_gpu>conv2>cloud_cpu", "conv2",
                                                                              ["conv2"], 5 * 256)
    assert rep.measured_latency_s >= rep.decision.chosen.breakdown["transfer"] > 0    # modelled upload added
    text = rep.explain()
    assert "left the edge: the conv2 activation, 256 B per image" in text and "raw data left the edge: no" in text
    assert "split: edge_gpu runs up to conv2, cloud_cpu runs the rest" in text


def test_router_decides_a_quantum_head_split_and_it_runs(live, tmp_path):
    """What-if (an assumption, not a measurement): the quantum head is taken to be more accurate
    than the classical heads. The router then sends the embedding to the state-vector simulator,
    and that decision executes on the real (quick) model."""
    o = Orchestrator("qedge_static", log_path=str(tmp_path / "log.jsonl"))
    prof = copy.deepcopy(split.get_profile("digits"))
    prof["embedding"]["4"]["accuracy"] = {"linear": 0.80, "mlp": 0.80, "quantum": 0.99, "quantum_l0": 0.70}
    split.PROFILES["digits"] = prof
    rep = o.submit("split_inference", k=4, batch=4, privacy=False, deadline_s=60)
    p = rep.decision.placement
    assert p.split == "embedding" and p.quantum == "quantum_sim" and p.classical.startswith("edge_")
    assert rep.error is None and rep.result.mode == "hybrid" and rep.result.backend == "qiskit-aer"
    assert rep.result.details["placement"] == p.label and rep.result.details["bytes_up_per_image"] == 16
    assert "the quantum head ran on quantum_sim" in rep.explain()


def test_forced_placement_outside_the_candidates_is_costed_and_run(live, tmp_path):
    o = Orchestrator("qedge_static", log_path=str(tmp_path / "log.jsonl"))
    o.executor._backends[o.executor.qpu_stack] = FakeQPU()
    rep = o.submit("split_inference", force="cloud_cpu+qpu", k=4, batch=3, privacy=False)
    assert rep.decision.chosen.placement == Placement("cloud_cpu", "qpu")         # its own estimate
    d = rep.result.details
    assert rep.error is None and d["left_edge"] == ["input"] and d["raw_data_left_edge"]
    assert d["bytes_up_per_image"] == 256 and rep.result.backend == "fake-qpu"
    assert "cloud_cpu ran the backbone on the uploaded images" in rep.explain()


def test_race_quantum_head_split_falls_back_to_its_own_classical_placement(live, tmp_path):
    o = Orchestrator(log_path=str(tmp_path / "log.jsonl"))
    o.executor._backends[o.executor.qpu_stack] = FakeQPU(delay=1.5)
    rep = o.submit("split_inference", force="edge_gpu>embedding>qpu", race=True, deadline_s=0.6, k=4, batch=4,
                   privacy=False)
    r = rep.race
    assert rep.raced and rep.winner == "classical" and rep.error is None
    assert rep.result.mode == "classical" and r.fallback != "edge_gpu>embedding>qpu"
    assert not Placement.parse(r.fallback).is_hybrid
    assert rep.result.details["placement"] == r.fallback                           # the fallback's own placement
    assert r.wait_settled(10) and r.quantum_status in ("cancelled", "late")
    if r.quantum_result is not None:                                               # the QPU path ran the head
        q = r.quantum_result.details
        assert q["placement"] == "edge_gpu>embedding>qpu" and q["cut"] == "embedding"
        assert q["images"] == rep.result.details["images"]                         # same batch on both paths
    # A fast QPU wins and delivers the quantum head's answer.
    o.executor._backends[o.executor.qpu_stack] = FakeQPU(delay=0.0)
    rep = o.submit("split_inference", force="edge_gpu>embedding>qpu", race=True, deadline_s=30, k=4, batch=4,
                   privacy=False)
    assert rep.winner == "quantum" and rep.result.details["placement"] == "edge_gpu>embedding>qpu"
    assert rep.race.wait_settled(10) and rep.race.classical_result.details["placement"] == rep.race.fallback


# ---------------------------------------------------------------- CLI
def test_cli_split_plan_lists_every_candidate(capsys):
    saved = copy.deepcopy(split.PROFILES)
    try:
        main(["split", "plan", "k=4", "batch=8", "privacy=true", "deadline_s=2", "--bandwidth", "1",
              "--edge", "edge_cpu"])
    finally:
        split.PROFILES.clear()
        split.PROFILES.update(saved)
    out = capsys.readouterr().out
    assert "uplink Mbps: cloud_cpu 1, cloud_gpu 1, quantum_sim 1, qpu 1; edge device: edge_cpu" in out
    labels = ["edge_cpu", "cloud_cpu", "cloud_gpu", "edge_cpu>conv1>cloud_gpu", "edge_cpu>conv2>cloud_cpu",
              "edge_cpu>embedding>cloud_gpu", "edge_cpu>embedding>quantum_sim", "edge_cpu>embedding>qpu"]
    assert all(lab in out for lab in labels) and "edge_gpu" not in out.split("qedge_adaptive")[0]
    assert "no: privacy: the raw input leaves the edge" in out
    assert "qedge_adaptive: workload 0 (split_inference) -> " in out and "split: " in out
    for col in ("latency_s", "edge_energy_j", "cost_usd", "quality", "leaves edge (per image)"):
        assert col in out


def test_cli_split_profile_and_plan_only_submit(capsys):
    saved = copy.deepcopy(split.PROFILES)
    try:
        main(["split", "profile"])
        out = capsys.readouterr().out
        assert "profile digits" in out and "profile camera" in out and "source:" in out
        assert "against the attacks tried only" in out and "quantum_l0" in out
        main(["split", "profile", "--name", "camera"])
        assert "profile digits" not in capsys.readouterr().out
        main(["submit", "split_inference", "k=4", "batch=4", "privacy=false", "--plan-only"])
        out = capsys.readouterr().out
        assert "(split_inference) -> " in out and "  split: " in out and "quantum gate for qpu" in out
    finally:
        split.PROFILES.clear()
        split.PROFILES.update(saved)
