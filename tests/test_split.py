"""Split placements (idea 3): labels, candidates, cost arithmetic, privacy, the quantum gate, profiles.

Tests register their own profiles (hand-written accuracy and leakage numbers, not
measurements) so they do not depend on whether results/split_profile.json exists.
"""
import copy
import hashlib
import json

import pytest

from qedge.orchestrator import split
from qedge.orchestrator.estimator import CostModel, Placement, candidate_placements
from qedge.orchestrator.policies import EXTRA_POLICIES, POLICIES, QEdgeCostPolicy, make_policy
from qedge.resources import ResourceState, load_catalogue, scale
from qedge.workloads import DEFAULT_MIX, KINDS, QUANTUM_CAPABLE, generate, make_workload

ACC = {"linear": 0.90, "mlp": 0.91, "quantum": 0.85, "quantum_l0": 0.60}


def hand_profile(name="t_digits", acc=None) -> dict:
    """The digits defaults with hand-set accuracy and leakage (fixture values, not measured)."""
    p = split.digits_default()
    p.update(name=name, source="test fixture")
    leak = {"conv1": (0.97, 0.45, False), "conv2": (0.30, 0.20, False)}
    for c in p["cuts"]:
        if c["name"] in leak:
            c["reid"], c["r2_beyond_label"], c["private_ok"] = leak[c["name"]]
    for e in p["embedding"].values():
        e.update(reid=0.01, r2_beyond_label=0.02, private_ok=True, accuracy=dict(acc or ACC))
    return p


@pytest.fixture
def profiles(monkeypatch):
    """Register "t_digits" and "t_camera" (camera inherits t_digits' embedding verdicts)."""
    def register(acc=None):
        d = hand_profile(acc=acc)
        c = split.camera_profile(d)
        c["name"] = "t_camera"
        monkeypatch.setitem(split.PROFILES, "t_digits", d)
        monkeypatch.setitem(split.PROFILES, "t_camera", c)
        return d, c
    register()
    return register


def split_w(**kw):
    kw.setdefault("profile", "t_digits")
    kw.setdefault("batch", 8)
    return make_workload("split_inference", seed=0, **kw)


def by_label(ests):
    return {e.placement.label: e for e in ests}


# ---------------------------------------------------------------- placements

def test_placement_labels_round_trip_and_stay_backward_compatible(profiles):
    res = load_catalogue()
    assert Placement("edge_gpu").label == "edge_gpu"
    assert Placement("cloud_cpu", "qpu").label == "cloud_cpu+qpu"
    assert Placement("cloud_cpu", "qpu") == Placement("cloud_cpu", quantum="qpu") and Placement("cloud_cpu", "qpu").is_hybrid
    s = Placement("edge_gpu", split="conv2", back="cloud_gpu")
    q = Placement("edge_gpu", "qpu", split="embedding")
    assert (s.label, s.is_hybrid) == ("edge_gpu>conv2>cloud_gpu", False)
    assert (q.label, q.is_hybrid) == ("edge_gpu>embedding>qpu", True)
    for w in (split_w(), split_w(profile="t_camera"), make_workload("optimization", seed=0, n=6),
              make_workload("inference", seed=0)):
        for p in candidate_placements(w, res):
            assert Placement.parse(p.label, res) == p
            assert Placement.parse(p.label) == p        # built-in names resolve without a catalogue
    # A custom quantum resource name needs the catalogue to be told apart from a classical back.
    custom = {**res, "my_qpu": copy.copy(res["qpu"])}
    assert Placement.parse("edge_cpu>embedding>my_qpu", custom) == Placement("edge_cpu", "my_qpu", split="embedding")
    for bad in ("edge_gpu>conv2", "edge_gpu>conv2>nowhere", "edge_gpu>>cloud_gpu", "a>b>c>d"):
        with pytest.raises(ValueError):
            Placement.parse(bad)
    with pytest.raises(ValueError):
        Placement("edge_gpu", split="conv2")                       # a cut needs somewhere to send it
    with pytest.raises(ValueError):
        Placement("edge_gpu", "qpu", split="embedding", back="cloud_gpu")
    with pytest.raises(ValueError):
        Placement("edge_gpu", back="cloud_gpu")
    # A split whose last resource sits in the wrong field would not round-trip, so it is refused.
    with pytest.raises(ValueError, match="quantum resource, not a classical back"):
        Placement("edge_gpu", split="embedding", back="qpu")
    with pytest.raises(ValueError, match="classical resource, not a quantum resource"):
        Placement("edge_gpu", "cloud_gpu", split="embedding")


def test_force_parses_split_labels(profiles):
    from qedge.orchestrator.engine import Orchestrator
    orch = Orchestrator(log_path=None)
    rep = orch.submit("split_inference", force="edge_gpu>conv2>cloud_gpu", profile="t_digits", batch=4,
                      privacy=False)
    p = Placement("edge_gpu", split="conv2", back="cloud_gpu")
    assert rep.decision.placement == p and rep.decision.chosen.placement == p
    assert rep.decision.policy.endswith("(forced)")


# ---------------------------------------------------------------- candidates

def test_candidates_for_split_inference(profiles):
    res = load_catalogue()
    labels = [p.label for p in candidate_placements(split_w(), res)]
    whole = ["edge_cpu", "edge_gpu", "cloud_cpu", "cloud_gpu"]
    splits = [f"{f}>{c}>{b}" for f in ("edge_cpu", "edge_gpu") for c in ("conv1", "conv2", "embedding")
              for b in ("cloud_cpu", "cloud_gpu")]
    heads = [f"{f}>embedding>{q}" for f in ("edge_cpu", "edge_gpu") for q in ("quantum_sim", "qpu")]
    assert labels == whole + splits + heads                        # no "input" split, no host+q hybrid
    cam = {p.split for p in candidate_placements(split_w(profile="t_camera"), res) if p.split}
    assert cam == {"stem", "last_conv", "pool", "embedding"}
    res["edge_gpu"].available = False
    res["qpu"].available = False
    labels = [p.label for p in candidate_placements(split_w(), res)]
    assert not any("edge_gpu" in x or "qpu" in x for x in labels) and "edge_cpu>embedding>quantum_sim" in labels


def test_candidates_unchanged_for_other_kinds():
    res = load_catalogue()
    classical = ["edge_cpu", "edge_gpu", "cloud_cpu", "cloud_gpu"]
    hybrid = ["edge_cpu+quantum_sim", "cloud_cpu+quantum_sim", "edge_cpu+qpu", "cloud_cpu+qpu"]
    for kind in KINDS:
        if kind == "split_inference":
            continue
        labels = [p.label for p in candidate_placements(make_workload(kind, seed=0), res)]
        assert labels == (classical + hybrid if kind in QUANTUM_CAPABLE else classical), kind


# ---------------------------------------------------------------- cost

def toy_profile() -> dict:
    """Round numbers so the arithmetic can be checked by hand: 100 kB input, a 25 kB cut "mid"
    halfway through 12 MFLOP, a 4-number embedding."""
    return {
        "name": "toy", "label": "toy", "source": "test fixture", "classes": 10, "input_bytes": 100_000,
        "cuts": [split._cut("input", 25_000, 100_000, 0, 12e6, reid=1.0),
                 split._cut("mid", 6_250, 25_000, 6e6, 6e6, reid=0.5, r2b=0.3)],
        "embedding": {"4": {"bytes": 16, "front_flop": 12e6 - 80, "back_flop": 80, "reid": 0.01,
                            "r2_beyond_label": 0.0, "private_ok": True, "accuracy": dict(ACC),
                            "quantum_circuit": split.head_circuit_stats(4, 2),
                            "params": split.head_params(4, 2, 10)}},
    }


def test_cost_arithmetic_hand_checked(monkeypatch):
    monkeypatch.setitem(split.PROFILES, "toy", toy_profile())
    assert split.check_profile(toy_profile()) == []
    res = load_catalogue()
    w = make_workload("split_inference", seed=0, profile="toy", k=4, batch=8, shots=1000, privacy=False)
    assert w.gflop == pytest.approx(8 * 12e6 / 1e9) and w.data_mb == pytest.approx(0.8)
    assert w.result_mb == pytest.approx(8 * 40 / 1e6)
    m = CostModel(res)
    # edge_gpu (600 GFLOPS, 15 W) runs 6 MFLOP/image; 8 x 25 kB = 0.2 MB up at 20 Mbps = 0.08 s,
    # 8 x 40 B of logits down at 100 Mbps = 2.56e-5 s, plus 0.04 s RTT; cloud_gpu (12 TFLOPS,
    # 350 W, $3/h) runs the other 6 MFLOP/image.
    e = m.estimate(w, Placement("edge_gpu", split="mid", back="cloud_gpu"))
    assert e.feasible
    assert e.breakdown["compute"] == pytest.approx(8 * 6e6 / 600e9)          # 8e-5 s
    assert e.breakdown["transfer"] == pytest.approx(0.04 + 0.08 + 2.56e-5)
    assert e.breakdown["back_compute"] == pytest.approx(8 * 6e6 / 12000e9)   # 4e-6 s
    assert e.latency_s == pytest.approx(0.1200256 + 8e-5 + 4e-6)
    assert e.edge_energy_j == pytest.approx(8e-5 * 15 + 0.1200256 * 2.0)     # compute + radio
    assert e.energy_j == pytest.approx(e.edge_energy_j + 4e-6 * 350)
    assert e.cost_usd == pytest.approx(4e-6 * 3.0 / 3600)
    assert e.quality == pytest.approx(0.91)                                  # best classical head (mlp)
    # Whole model in the cloud: 0.8 MB raw input up = 0.32 s.
    c = m.estimate(w, Placement("cloud_gpu"))
    assert c.latency_s == pytest.approx(0.04 + 0.32 + 2.56e-5 + 8 * 12e6 / 12000e9)
    assert m.estimate(w, Placement("edge_gpu")).latency_s == pytest.approx(8 * 12e6 / 600e9)
    # Quantum head on the QPU: 8 x 16 B up and 8 x 16 B (k expectation values) down, bytes only;
    # the RTT is the per-round hop. Readout (2 x 4 x 10 FLOP/image) runs on the edge.
    # The circuit is the routed one: 8 logical CZs become 17 on heavy-hex, depth 15 + 9.
    qp = w.quantum
    assert (qp.n_qubits, qp.n_circuits, qp.shots, qp.n_rounds, qp.depth, qp.n_1q, qp.n_2q) == (4, 8, 1000, 1, 24, 28, 17)
    q = m.estimate(w, Placement("edge_gpu", "qpu", split="embedding"))
    xfer = 128e-6 * 8 / 20 + 128e-6 * 8 / 100
    exec_t = 8 * 1000 * (2.5e-4 + 24 * 6.6e-7)
    assert q.breakdown["transfer"] == pytest.approx(xfer)
    assert q.breakdown["compute"] == pytest.approx(8 * 12e6 / 600e9)
    assert q.breakdown["q_exec"] == pytest.approx(exec_t)
    assert q.latency_s == pytest.approx(xfer + 8 * 12e6 / 600e9 + 120 + 3.0 + 0.04 + exec_t)
    assert q.cost_usd == pytest.approx(0.30 + 8000 * 0.00145)
    f = qp.fidelity(3e-4, 3e-3, 1.5e-2)
    assert q.fidelity == pytest.approx(f) and q.quality == pytest.approx(f * 0.85 + (1 - f) * 0.1)
    sim = m.estimate(w, Placement("edge_gpu", "quantum_sim", split="embedding"))
    assert sim.fidelity == 1.0 and sim.quality == pytest.approx(0.85)


def test_split_flops_add_up_to_the_whole_model():
    for prof in (split.digits_default(), split.camera_profile(split.digits_default())):
        for k in split.K_RANGE:
            total = split.total_flop(prof, k)
            for cut in [c["name"] for c in prof["cuts"]] + ["embedding"]:
                c = split.cut_info(prof, cut, k)
                assert c["front_flop"] + c["back_flop"] == pytest.approx(total)
    d = split.digits_default()
    assert split.total_flop(d, 4) == 9216 + 36864 + 512 + 80
    assert [c["bytes"] for c in d["cuts"]] == [256, 512, 256]               # conv1 is bigger than the input
    cam = split.camera_profile(d)
    assert split.total_flop(cam, 4) == pytest.approx(0.6e9, rel=0.01)       # make_inference's MobileNet figure


def test_quantum_head_is_costed_after_routing_onto_heavy_hex(profiles):
    # Logical ring: 8 CZs at k = 4, L = 2. Routed (measured on FakeTorino): 17, depth 15 + 9.
    logical = split.head_circuit_stats(4, 2)
    assert split.routed_circuit(logical) == {**logical, "n_2q": 17, "depth": 24}
    assert split.routed_circuit(split.head_circuit_stats(2, 2))["n_2q"] == 2      # k = 2 has no ring
    assert split.routed_circuit(split.head_circuit_stats(4, 0))["n_2q"] == 0      # L = 0: no CZs
    assert split.PROFILES["t_digits"]["embedding"]["4"]["quantum_circuit"]["n_2q"] == 8   # profile stays logical
    w = split_w(k=4)
    assert (w.quantum.n_2q, w.quantum.depth, w.quantum.n_1q) == (17, 24, 28)
    # At the catalogue's error rates routing lowers the estimated circuit fidelity of the head.
    res = load_catalogue()
    q = res["qpu"]
    assert w.quantum.fidelity(q.p1, q.p2, q.readout_err) < 0.85 < 0.88 < copy.replace(
        w.quantum, n_2q=8).fidelity(q.p1, q.p2, q.readout_err)


def test_head_circuit_and_parameter_matching():
    assert split.head_circuit_stats(2, 3) == {"n_qubits": 2, "layers": 3, "depth": 13, "n_1q": 20, "n_2q": 3}
    s = split.head_circuit_stats(4, 2)
    assert (s["n_1q"], s["n_2q"]) == (4 + 2 * 12, 2 * 4)                    # ring wraps for k > 2
    assert split.head_circuit_stats(4, 0) == {"n_qubits": 4, "layers": 0, "depth": 1, "n_1q": 4, "n_2q": 0}
    p = split.head_params(4, 2, 10)
    assert p["linear"] == 50 and p["quantum"] == 74
    assert p["mlp"] == p["mlp_hidden"] * 15 + 10                                # k -> h -> 10 with biases
    assert abs(p["mlp"] - 74) == min(abs(h * 15 + 10 - 74) for h in range(1, 50))


# ---------------------------------------------------------------- privacy

def test_privacy_raw_and_conv1_infeasible_embedding_feasible(profiles):
    m = CostModel(load_catalogue())
    w = split_w(privacy=True)
    ests = by_label([m.estimate(w, p) for p in candidate_placements(w, m.resources)])
    assert not ests["cloud_gpu"].feasible and "raw input" in ests["cloud_gpu"].reasons[0]
    assert not ests["edge_gpu>conv1>cloud_gpu"].feasible
    assert "conv1 activations re-identify 97% of inputs" in ests["edge_gpu>conv1>cloud_gpu"].reasons[0]
    assert not ests["edge_gpu>conv2>cloud_gpu"].feasible
    for label in ("edge_gpu", "edge_gpu>embedding>cloud_gpu", "edge_gpu>embedding>qpu", "edge_cpu>embedding>quantum_sim"):
        assert ests[label].feasible, (label, ests[label].reasons)
    # Not private data: every placement is allowed.
    w2 = split_w(privacy=False)
    assert all(m.estimate(w2, p).feasible for p in candidate_placements(w2, m.resources))
    # The built-in defaults have no leakage measurement, so nothing transmitted counts as private.
    w3 = make_workload("split_inference", seed=0, profile="digits", batch=8, privacy=True)
    e3 = m.estimate(w3, Placement("edge_gpu", split="embedding", back="cloud_gpu"))
    assert not e3.feasible and "not measured" in e3.reasons[0]
    # A host+quantum hybrid with a cloud host uploads the raw input first.
    assert not m.estimate(w, Placement("cloud_cpu", "qpu")).feasible
    assert split.transmitted(Placement("cloud_cpu", "qpu"), m.resources) == ["input"]


# ---------------------------------------------------------------- the quantum gate

def test_gate_closed_today_because_the_quantum_head_is_less_accurate(profiles):
    w = split_w(privacy=True, deadline_s=3600)
    d = QEdgeCostPolicy(load_catalogue()).decide(w, ResourceState())
    assert not d.placement.is_hybrid
    for q in ("qpu", "quantum_sim"):
        checks = {name: (ok, why) for name, ok, why in d.quantum_checks[q]}
        ok, why = checks["data may leave edge"]
        assert ok and "only the 4-number embedding leaves the edge" in why
        assert checks["meets deadline"][0]
        ok, why = checks["expected quality gain"]
        assert not ok and "accuracy" in why and "0.910" in why
    assert "ideal 0.850" in dict((n, y) for n, _, y in d.quantum_checks["qpu"])["expected quality gain"]
    # Same workload, data not private: the gate still closes on accuracy alone.
    d2 = QEdgeCostPolicy(load_catalogue()).decide(split_w(privacy=False, deadline_s=3600), ResourceState())
    failed = [n for n, ok, _ in d2.quantum_checks["qpu"] if not ok]
    assert failed == ["expected quality gain"]


def test_gate_shown_for_a_split_is_the_variant_the_router_evaluated(profiles):
    # Quantum head as accurate as the best classical head (fixture assumption). The edge_cpu front
    # misses a 3.5 s deadline on the simulator, the edge_gpu front makes it: the explanation must
    # show the edge_gpu variant, not the first candidate.
    profiles(acc={**ACC, "quantum": 0.91})
    w = split_w(profile="t_camera", batch=32, privacy=False, deadline_s=3.5)
    pol = QEdgeCostPolicy(load_catalogue())
    d = pol.decide(w, ResourceState())
    ests = by_label(d.estimates)
    slow, fast = ests["edge_cpu>embedding>quantum_sim"], ests["edge_gpu>embedding>quantum_sim"]
    assert slow.latency_s > 3.5 > fast.latency_s and "gate: meets deadline" in slow.reasons
    assert all(ok for _, ok, _ in d.quantum_checks["quantum_sim"])
    assert "quantum gate for quantum_sim: open" in d.explain()
    # When the router picks a quantum head, its own checks are the ones shown (what-if numbers,
    # as in the test below: quantum head assumed more accurate, future QPU).
    profiles(acc={**ACC, "quantum": 0.99})
    d = QEdgeCostPolicy(cheap_future_qpu(sim=False)).decide(split_w(privacy=True, deadline_s=600), ResourceState())
    assert d.placement == Placement("edge_gpu", "qpu", split="embedding")
    shown = {n: y for n, _, y in d.quantum_checks["qpu"]}["meets deadline"]
    assert shown == f"{d.chosen.latency_s:.3g}s vs 600s"
    assert d.chosen.latency_s < by_label(d.estimates)["edge_cpu>embedding>qpu"].latency_s


def test_camera_says_its_privacy_and_accuracy_are_inherited_from_digits(profiles):
    _, cam = profiles()
    e = cam["embedding"]["4"]
    assert e["private_ok"] and e["reid"] is None and e["r2_beyond_label"] is None    # no attack ran on camera
    assert e["inherited_leakage"] == {"reid": 0.01, "r2_beyond_label": 0.02} and e["inherited_from"] == "t_digits"
    w = split_w(profile="t_camera", batch=32, privacy=True, deadline_s=3600)
    d = QEdgeCostPolicy(load_catalogue()).decide(w, ResourceState())
    checks = {name: why for name, _, why in d.quantum_checks["qpu"]}
    why = checks["data may leave edge"]
    assert why.startswith("only the 4-number embedding leaves the edge; privacy verdict inherited from t_digits")
    assert "re-identify 1.0% of t_digits inputs" in why and "not measured on this model" in why
    assert "head accuracies inherited from t_digits, not measured on this model" in checks["expected quality gain"]
    # The digits gate text is unchanged and claims the measurement as its own.
    wd = split_w(privacy=True, deadline_s=3600)
    dd = QEdgeCostPolicy(load_catalogue()).decide(wd, ResourceState())
    why = {n: y for n, _, y in dd.quantum_checks["qpu"]}["data may leave edge"]
    assert why == "only the 4-number embedding leaves the edge; it re-identifies 1.0% of inputs against the attacks tried"
    # Built-in camera: digits' placeholder verdict (not private, not measured) is inherited, and says so.
    wb = make_workload("split_inference", seed=0, profile="camera", batch=4, privacy=True)
    ok, why = split.privacy_verdict(wb, Placement("edge_gpu", "qpu", split="embedding"), load_catalogue())
    assert not ok and why == ("privacy: embedding counts as not private: privacy verdict inherited from digits "
                              "(its leakage was not measured on digits either), not measured on this model")


def cheap_future_qpu(sim: bool):
    """What-if hardware: 100x lower errors, 10x shorter queue, free shots (not today's QPU)."""
    res = scale(load_catalogue(), qpu__p1=0.01, qpu__p2=0.01, qpu__readout_err=0.01, qpu__queue_mean_s=0.1,
                qpu__cost_per_task=0.0, qpu__cost_per_shot=0.0)
    res["quantum_sim"].available = sim
    return res


def test_what_if_accurate_quantum_head_and_cheap_qpu_is_picked_for_a_relaxed_batch(profiles):
    # Assumption, not a measurement: the quantum head is MORE accurate than both classical heads.
    profiles(acc={**ACC, "quantum": 0.99})
    w = split_w(privacy=True, deadline_s=600)
    d = QEdgeCostPolicy(cheap_future_qpu(sim=False)).decide(w, ResourceState())
    assert d.placement == Placement("edge_gpu", "qpu", split="embedding")
    assert all(ok for _, ok, _ in d.quantum_checks["qpu"])
    # A tight deadline sends it back to the edge, since the queue alone is longer.
    d = QEdgeCostPolicy(cheap_future_qpu(sim=False)).decide(split_w(privacy=True, deadline_s=2.0), ResourceState())
    assert not d.placement.is_hybrid
    # With a state-vector simulator on offer, simulating a 4-qubit head is cheaper than running it.
    d = QEdgeCostPolicy(cheap_future_qpu(sim=True)).decide(w, ResourceState())
    assert d.placement == Placement("edge_gpu", "quantum_sim", split="embedding")
    # Equal accuracy is not enough: the gate opens but the quantum head only adds latency and energy.
    profiles(acc={**ACC, "quantum": 0.91})
    d = QEdgeCostPolicy(cheap_future_qpu(sim=False)).decide(w, ResourceState())
    assert all(ok for _, ok, _ in d.quantum_checks["qpu"]) and not d.placement.is_hybrid


def test_heavy_camera_batch_goes_to_the_cloud_unless_private(profiles):
    # Weak edge (no GPU, CPU at 5 GFLOPS) and a fast network (1 Gbps up).
    res = scale(load_catalogue(), edge_cpu__gflops=0.2, cloud_cpu__uplink_mbps=50, cloud_gpu__uplink_mbps=50,
                qpu__uplink_mbps=50, quantum_sim__uplink_mbps=50)
    res["edge_gpu"].available = False
    pol = QEdgeCostPolicy(res)
    w = split_w(profile="t_camera", batch=32, deadline_s=1.0, privacy=False)
    d = pol.decide(w, ResourceState())
    assert d.placement == Placement("cloud_gpu") and d.chosen.latency_s < 1.0
    wp = split_w(profile="t_camera", batch=32, deadline_s=1.0, privacy=True)
    d = pol.decide(wp, ResourceState())
    assert res[d.placement.classical].is_edge
    ok, _ = split.privacy_verdict(wp, d.placement, res)
    assert ok and d.placement == Placement("edge_cpu")
    assert d.chosen.latency_s > 1.0                  # privacy costs the deadline here, and the model says so
    ests = by_label(d.estimates)
    assert not ests["edge_cpu>stem>cloud_gpu"].feasible and "stem leakage not measured" in ests["edge_cpu>stem>cloud_gpu"].reasons[0]


# ---------------------------------------------------------------- profiles

@pytest.fixture
def restore_profiles():
    saved = copy.deepcopy(split.PROFILES)
    yield
    split.PROFILES.clear()
    split.PROFILES.update(saved)


def test_builtin_profiles_match_the_schema():
    for name, p in split._defaults().items():
        assert split.check_profile(p) == [], name
        assert set(p["embedding"]) == {str(k) for k in split.K_RANGE}
    cam = split.camera_profile(split.digits_default())
    assert all(not c["private_ok"] for c in cam["cuts"])
    assert all("inherited_from" in e for e in cam["embedding"].values())
    bad = split.digits_default()
    del bad["embedding"]["4"]["accuracy"]["quantum"]
    bad["cuts"][0]["private_ok"] = "yes"
    problems = split.check_profile(bad)
    assert any("accuracy" in x for x in problems) and any("private_ok" in x for x in problems)


def test_load_profiles_merges_measured_over_defaults(tmp_path, restore_profiles):
    measured = hand_profile(name="digits")
    measured["source"] = "measured (E8)"
    measured["embedding"] = {"4": measured["embedding"]["4"]}            # only k = 4 was measured
    path = tmp_path / "split_profile.json"
    path.write_text(json.dumps(measured))
    profs = split.load_profiles(str(path))
    d = profs["digits"]
    assert d["source"] == "measured (E8)" and d["cuts"][1]["reid"] == 0.97
    assert d["embedding"]["4"]["private_ok"] and d["embedding"]["4"]["accuracy"]["mlp"] == 0.91
    assert d["embedding"]["2"]["source"] == split.PLACEHOLDER_NOTE and not d["embedding"]["2"]["private_ok"]
    assert profs["camera"]["embedding"]["4"]["private_ok"]                # camera inherits the measurement
    assert profs["camera"]["embedding"]["4"]["accuracy"]["mlp"] == 0.91
    assert all(split.check_profile(p) == [] for p in profs.values())
    w = make_workload("split_inference", seed=0, k=4, batch=2)
    assert w.classical_quality == 0.91 and w.classical_method == "split CNN, mlp head"
    # {name: profile} and lists load too; an extra model is added as is.
    extra = toy_profile()
    path.write_text(json.dumps({"digits": measured, "toy": extra}))
    assert set(split.load_profiles(str(path))) == {"digits", "camera", "toy"}
    path.write_text(json.dumps([extra]))
    assert split.load_profiles(str(path))["digits"] == split.digits_default()
    # A malformed profile is refused, not half-loaded.
    broken = dict(extra, embedding={"4": {"bytes": 16}})
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="embedding"):
        split.load_profiles(str(path))
    # A missing file resets to the built-in defaults.
    assert split.load_profiles(str(tmp_path / "missing.json")) == split._defaults()


def test_workload_fields_come_from_the_profile(profiles):
    w = split_w(batch=8, k=4, shots=500)
    assert w.kind == "split_inference" and w.quantum_capable
    assert w.classical_quality == 0.91 and w.params["head"] == "mlp"
    assert w.quantum.algorithm == "quantum classifier head (4 qubits, 2 layers)" and w.quantum.shots == 500
    assert split_w(head="linear").classical_quality == 0.90
    l0 = split_w(qhead="quantum_l0")
    assert (l0.quantum.n_2q, l0.quantum.depth) == (0, 1)
    assert CostModel.quantum_quality(l0, 1.0) == pytest.approx(0.60)
    with pytest.raises(ValueError, match="k=9"):
        split_w(k=9)
    with pytest.raises(ValueError):
        split_w(head="quantum")
    with pytest.raises(KeyError, match="unknown split profile"):
        make_workload("split_inference", seed=0, profile="nope")


# ---------------------------------------------------------------- policies and the simulator

@pytest.mark.parametrize("policy", list(POLICIES) + list(EXTRA_POLICIES))
def test_every_policy_returns_a_candidate_for_split_inference(policy, profiles):
    res = load_catalogue()
    for kw in ({"privacy": True}, {"privacy": False}, {"profile": "t_camera", "batch": 32},
               {"profile": "digits", "privacy": True}):
        w = split_w(**kw)
        d = make_policy(policy, res).decide(w, ResourceState())
        assert d.placement in candidate_placements(w, res)
        if w.privacy and d.chosen.feasible:
            assert split.privacy_verdict(w, d.placement, res)[0]


def test_quantum_first_sends_split_inference_to_the_quantum_head(profiles):
    res = load_catalogue()
    d = make_policy("quantum_first", res).decide(split_w(privacy=False), ResourceState())
    assert d.placement == Placement("edge_gpu", "qpu", split="embedding")
    # Not private_ok (the placeholder digits embedding): it falls back as before.
    d = make_policy("quantum_first", res).decide(split_w(profile="digits", privacy=True), ResourceState())
    assert d.chosen.feasible and not d.placement.is_hybrid
    assert make_policy("quantum_first", res).decide(make_workload("optimization", seed=0, n=6),
                                                    ResourceState()).placement.label == "cloud_cpu+qpu"


def fingerprint(ws) -> str:
    """Hash of every workload field the generator's random stream drives (classical_quality and
    gflop are left out: they also depend on CALIBRATION, which other tests load)."""
    def norm(x):
        if isinstance(x, float):
            return float(f"{x:.10g}")
        if isinstance(x, dict):
            return {k: norm(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x
    keep = ("id", "kind", "arrival_s", "deadline_s", "data_mb", "result_mb", "privacy", "min_quality",
            "memory_gb", "parallel", "classical_method", "quantum", "params")
    rows = [{k: norm(w.to_dict()[k]) for k in keep} for w in ws]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


def test_default_workload_stream_unchanged():
    assert "split_inference" not in DEFAULT_MIX and KINDS[-1] == "split_inference"
    ws = generate(200, seed=0)
    assert not any(w.kind == "split_inference" for w in ws)
    # Recorded from a clean export of base commit f52c8f6, before split_inference existed.
    assert fingerprint(ws) == "21aaa365654629e5f74921c43ea03a2967d04ba8e71070a3ba7e5efbd38b963b"


def test_simulator_runs_split_inference_when_asked():
    from qedge.simulation.simulator import Simulator
    wl = generate(60, mix={"split_inference": 0.5, "inference": 0.25, "optimization": 0.25}, seed=0)
    sim = Simulator("baseline", seed=0)
    for policy in ("qedge_adaptive", "qedge_hedged", "random"):
        df = sim.run(policy, wl)
        assert len(df) == 60 and df.latency_s.ge(0).all()


def test_simulator_frees_the_edge_front_once_its_share_of_a_split_is_done(profiles):
    """A split holds the edge front for its front compute only, not for the upload after it."""
    from qedge.orchestrator.policies import Decision, Policy
    from qedge.simulation.simulator import Scenario, Simulator

    class Fixed(Policy):
        name = "fixed"

        def decide(self, w, st):
            p = Placement.parse(w.params["force"], self.resources)
            e = self.model.estimate(w, p, st)
            return Decision(w, p, e, [e], self.name)

    cat = load_catalogue()
    cat["edge_cpu"].slots = 1
    sim = Simulator(Scenario("flat", "no noise", noise_sigma=1e-9), catalogue=cat, seed=0)
    ws = []
    for i, (t, force) in enumerate([(0.0, "edge_cpu>stem>cloud_gpu"), (0.5, "edge_cpu")]):
        w = split_w(profile="t_camera", batch=32, privacy=False, deadline_s=100)
        w.id, w.arrival_s, w.params["force"] = i, t, force
        ws.append(w)
    df = sim.run(Fixed(sim.nominal), ws)
    model = CostModel(sim.nominal)
    first = model.estimate(ws[0], Placement.parse("edge_cpu>stem>cloud_gpu"))
    assert first.breakdown["transfer"] > 10 and first.breakdown["compute"] < 0.5   # 51 MB of stem activations
    alone = model.estimate(ws[1], Placement("edge_cpu")).latency_s
    assert df.latency_s.iloc[1] == pytest.approx(alone, rel=1e-6)    # no wait behind task 0's upload
