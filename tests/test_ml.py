"""Idea 3, the real split model: backbone, heads, quantum head export and paths, leakage, profile.

One quick transfer run at k=4 (few epochs) is shared by most tests, so the
file stays fast; the full-size numbers come from the E8 run, not from here.
Packed inference runs on the Torino snapshot (133 qubits) as the design asks;
the live SamplerV2 branch is exercised on a 27-qubit snapshot in local mode.
"""
import copy
import json

import numpy as np
import pytest

from qedge.ml import leakage as L
from qedge.ml import nets
from qedge.ml import profile as PR
from qedge.ml.data import load_digits_split
from qedge.ml.qhead import QuantumHead, expz_from_counts, quantum_params, ring, to_circuits
from qedge.quantum import calibration as C
from qedge.quantum import packing as P
from qedge.quantum.backends.qiskit_backend import to_qiskit

QUICK_BACKBONE = dict(epochs=8, lr=0.01, batch=32, patience=8)
QUICK_HEAD = dict(epochs=10, lr=0.1, batch=128, patience=10)


@pytest.fixture(scope="module")
def data():
    return load_digits_split(0)


@pytest.fixture(scope="module")
def quick(data):
    return nets.train_transfer(data, 4, seed=0, backbone_kw=QUICK_BACKBONE, head_kw=QUICK_HEAD)


@pytest.fixture(scope="module")
def head4():
    """An untrained but fixed 4-qubit, 2-layer head (random angles)."""
    h = QuantumHead(4, layers=2)
    h.params = h.init(np.random.default_rng(5))
    return h


def _exact_expz_qiskit(c):
    """<Z_i> of an exported circuit from Qiskit's statevector: an independent simulator."""
    from qiskit.quantum_info import Statevector
    probs = Statevector(to_qiskit(c, measure=False)).probabilities_dict()
    return expz_from_counts({k[::-1]: v for k, v in probs.items()}, c.n_qubits)   # Qiskit: qubit 0 rightmost


# ---- data and backbone -----------------------------------------------------------
def test_digits_split_is_stratified_and_reproducible(data):
    Xtr, ytr, Xval, yval, Xte, yte = data
    assert Xtr.shape[1:] == (1, 8, 8) and Xtr.dtype == np.float64
    assert (len(Xtr), len(Xval), len(Xte)) == (1078, 359, 360)
    assert 0.0 <= Xtr.min() and Xtr.max() <= 1.0
    for y in (ytr, yval, yte):
        frac = np.bincount(y, minlength=10) / len(y)
        assert np.all(np.abs(frac - 0.1) < 0.02)
    again = load_digits_split(0)
    assert all(np.array_equal(a, b) for a, b in zip(data, again))
    assert not np.array_equal(load_digits_split(1)[0], Xtr)


def test_shapes_and_cut_stats():
    k = 3
    rng = np.random.default_rng(0)
    m = nets.SplitCNN(k, nets.init_backbone(k, rng), nets.LinearHead(k))
    m.head.params = m.head.init(rng)
    X = rng.uniform(0, 1, (5, 1, 8, 8))
    shapes = {"input": (5, 1, 8, 8), "conv1": (5, 8, 4, 4), "conv2": (5, 16, 2, 2), "embedding": (5, k)}
    full = m.forward_from("input", X)
    assert full.shape == (5, 10)
    for cut, shape in shapes.items():
        H = m.forward_to(cut, X)
        assert H.shape == shape and H.dtype == np.float32
        # the remote half continues from exactly what was sent (float32 rounding only)
        assert np.allclose(m.forward_from(cut, H), full, atol=1e-5)
        assert np.allclose(m.forward_from(cut, H.reshape(5, -1)), full, atol=1e-5)
    assert np.all(np.abs(m.embed(X)) <= 1.0)
    stats = {r["name"]: r for r in m.cut_stats()}
    assert [r["name"] for r in m.cut_stats()] == list(nets.CUTS)
    assert {c: stats[c]["values"] for c in nets.CUTS} == {"input": 64, "conv1": 128, "conv2": 64, "embedding": k}
    assert stats["conv1"]["bytes"] == 512 > stats["input"]["bytes"] == 256      # the Neurosurgeon observation
    conv1, conv2, dense, head = 2 * 64 * 9 * 8, 2 * 16 * 72 * 16, 2 * 64 * k, 2 * k * 10
    assert stats["conv1"]["front_flop"] == conv1 and stats["conv2"]["front_flop"] == conv1 + conv2
    assert stats["embedding"]["front_flop"] == conv1 + conv2 + dense and stats["embedding"]["back_flop"] == head
    assert all(r["front_flop"] + r["back_flop"] == conv1 + conv2 + dense + head for r in stats.values())


def test_conv_matches_correlation_and_its_gradient_matches_finite_differences():
    import autograd.numpy as anp
    from autograd import grad
    from scipy.signal import correlate
    rng = np.random.default_rng(1)
    x, w, b = rng.normal(size=(2, 2, 4, 4)), rng.normal(size=(3, 2, 3, 3)), rng.normal(size=3)
    out = nets.conv3x3(x, w, b)
    ref = np.stack([np.stack([sum(correlate(x[n, c], w[o, c], mode="same") for c in range(2)) + b[o]
                              for o in range(3)]) for n in range(2)])
    assert np.allclose(out, ref)
    R = rng.normal(size=out.shape)
    f = lambda x, w, b: anp.sum(nets.conv3x3(x, w, b) * R)   # noqa: E731
    eps = 1e-6
    for argnum, arr in enumerate((x, w, b)):
        g = grad(f, argnum)(x, w, b)
        fd = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            args_p, args_m = [x.copy(), w.copy(), b.copy()], [x.copy(), w.copy(), b.copy()]
            args_p[argnum][idx] += eps
            args_m[argnum][idx] -= eps
            fd[idx] = (f(*args_p) - f(*args_m)) / (2 * eps)
        assert np.allclose(g, fd, rtol=1e-5, atol=1e-6)


def test_backbone_loss_gradient_matches_finite_differences():
    """Through conv, ReLU, max-pool, dense and tanh, on a few random coordinates."""
    from autograd import grad
    from autograd.misc import flatten
    rng = np.random.default_rng(2)
    head = nets.LinearHead(3)
    params = {"backbone": nets.init_backbone(3, rng), "head": head.init(rng)}
    X, y = rng.uniform(0, 1, (4, 1, 8, 8)), np.array([0, 3, 7, 9])
    flat, unflat = flatten(params)
    loss = lambda fp: nets.cross_entropy(head.logits(unflat(fp)["head"], nets.backbone(unflat(fp)["backbone"], X)), y)  # noqa: E731
    g = grad(loss)(flat)
    eps = 1e-6
    for i in rng.choice(len(flat), 25, replace=False):
        e = np.zeros_like(flat)
        e[i] = eps
        assert g[i] == pytest.approx((loss(flat + e) - loss(flat - e)) / (2 * eps), rel=1e-4, abs=1e-7)


def test_training_is_deterministic(data):
    small = (data[0][:200], data[1][:200], data[2][:100], data[3][:100])
    a = nets.train_end_to_end(small, 2, seed=3, epochs=2)
    b = nets.train_end_to_end(small, 2, seed=3, epochs=2)
    c = nets.train_end_to_end(small, 2, seed=4, epochs=2)
    assert all(np.array_equal(a.backbone[n], b.backbone[n]) for n in a.backbone)
    assert all(np.array_equal(a.head.params[n], b.head.params[n]) for n in a.head.params)
    assert not np.array_equal(a.backbone["conv1_w"], c.backbone["conv1_w"])


def test_quick_training_reaches_sane_accuracy(quick, data):
    """Quick settings (8 backbone epochs, 10 head epochs) measured at 0.92 backbone test
    accuracy and 0.85 for this quantum head; thresholds leave a wide margin."""
    Xte, yte = data[4], data[5]
    assert quick.model.accuracy(Xte, yte) > 0.8
    acc = {h: quick.accuracy(h, Xte, yte) for h in nets.HEADS}
    assert acc["linear"] > 0.8 and acc["mlp"] > 0.6 and acc["quantum_l0"] > 0.8
    assert acc["quantum"] > 0.6                       # well above chance (0.1); not compared with classical here
    for name, fit in quick.fits.items():
        assert fit.best_val_acc == max(h["val_acc"] for h in fit.history)


def test_mlp_is_parameter_matched_to_the_quantum_head(quick):
    q, m = quick.heads["quantum"], quick.heads["mlp"]
    assert q.n_params == quantum_params(4, 2) == 3 * 4 * 2 + 4 * 10 + 10 == 74
    assert m.n_params == nets.mlp_params(4, m.hidden) == 70 and m.hidden == 4
    for k in range(2, 9):
        target = quantum_params(k, 2)
        h = nets.mlp_hidden_for(k, target)
        assert all(abs(nets.mlp_params(k, h) - target) <= abs(nets.mlp_params(k, j) - target) for j in range(1, 200))
    assert quick.heads["linear"].n_params == 50 and quick.heads["quantum_l0"].n_params == 50


def test_end_to_end_backprop_reaches_the_backbone_through_the_qnode(data):
    """Fairness mode: gradients flow from the quantum head's loss into the conv filters."""
    from autograd import grad
    rng = np.random.default_rng(0)
    h = QuantumHead(2, layers=1)
    p = {"backbone": nets.init_backbone(2, rng), "head": h.init(rng)}
    X, y = data[0][:16], data[1][:16]
    g = grad(lambda p: nets.cross_entropy(h.logits(p["head"], nets.backbone(p["backbone"], X)), y))(p)
    assert np.abs(g["backbone"]["conv1_w"]).sum() > 0 and np.abs(g["head"]["w"]).sum() > 0
    small = (data[0][:120], data[1][:120], data[2][:60], data[3][:60])
    m = nets.train_end_to_end(small, 2, "quantum", seed=0, layers=1, epochs=1, batch=60)
    assert isinstance(m.head, QuantumHead) and m.meta["mode"] == "end-to-end" and m.meta["steps"] == 2


def test_save_load_roundtrip(quick, data, tmp_path):
    path = str(tmp_path / "m.npz")
    quick.model.save(path)
    back = nets.SplitCNN.load(path)
    X = data[4][:30]
    assert np.array_equal(back.forward_from("input", X), quick.model.forward_from("input", X))
    q = quick.with_head("quantum")
    q.save(str(tmp_path / "q.npz"))
    qb = nets.SplitCNN.load(str(tmp_path / "q.npz"))
    assert isinstance(qb.head, QuantumHead) and qb.head.layers == 2
    assert np.allclose(qb.forward_from("conv2", q.forward_to("conv2", X)), q.forward_from("input", X), atol=1e-5)
    # np.savez appends .npz; load accepts the path as given to save
    written = quick.model.save(str(tmp_path / "nosuffix"))
    assert written.endswith("nosuffix.npz")
    again = nets.SplitCNN.load(str(tmp_path / "nosuffix"))
    assert np.array_equal(again.forward_from("input", X), quick.model.forward_from("input", X))


def test_training_defaults_are_the_measured_settings():
    """The reported numbers used these settings; calling the API without keywords must reproduce them."""
    assert nets.BACKBONE_FIT == {"epochs": 60, "lr": 0.003, "batch": 32, "patience": 10}
    assert nets.HEAD_FIT == {"epochs": 60, "lr": 0.03, "batch": 32, "patience": 10}
    assert nets._fit_kw(None, nets.BACKBONE_FIT) == nets.BACKBONE_FIT
    assert nets._fit_kw({"epochs": 2, "seed": 9}, nets.HEAD_FIT) == {**nets.HEAD_FIT, "epochs": 2}


# ---- quantum head ----------------------------------------------------------------
@pytest.mark.parametrize("k,layers", [(2, 2), (3, 1), (5, 2)])
def test_qnode_matches_exported_circuit(k, layers):
    h = QuantumHead(k, layers=layers)
    h.params = h.init(np.random.default_rng(k))
    Z = np.random.default_rng(10 + k).uniform(-1, 1, (4, k))
    E = h.expvals_exact(Z)
    circuits = h.to_circuits(Z)
    assert len(circuits) == 4
    for c, e in zip(circuits, E):
        assert {g.name for g in c.gates} <= {"ry", "rz", "cx"}
        assert np.allclose(_exact_expz_qiskit(c), e, atol=1e-6)
    prof = h.circuit_profile()
    assert prof == {"n_qubits": k, "layers": layers, "depth": circuits[0].depth(),
                    "n_1q": k + 3 * k * layers, "n_2q": layers * len(ring(k))}
    assert len(ring(2)) == 1 and len(ring(4)) == 4


def test_l0_head_is_the_fixed_cosine_encoding():
    h = QuantumHead(3, layers=0, name="quantum_l0")
    h.params = h.init(np.random.default_rng(0))
    Z = np.random.default_rng(1).uniform(-1, 1, (6, 3))
    assert np.allclose(h.expvals_exact(Z), -np.sin(np.pi * Z / 2), atol=1e-12)
    assert h.n_circuit_params == 0 and to_circuits(Z, h.params["w"])[0].n_2q == 0


def test_predict_shots_converges_to_exact(head4):
    Z = np.random.default_rng(3).uniform(-1, 1, (150, 4))
    E = head4.expvals_exact(Z)
    errs = [np.abs(head4.expvals_shots(Z, s, seed=1) - E).mean() for s in (100, 10_000, 1_000_000)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 0.003
    assert np.array_equal(head4.expvals_shots(Z, 500, seed=7), head4.expvals_shots(Z, 500, seed=7))
    assert np.mean(head4.predict_shots(Z, 1_000_000, seed=1) == head4.predict_exact(Z)) > 0.97


def test_predict_on_backend_qiskit_simulator(quick, data):
    from qedge.quantum.backends import get_backend
    q = quick.heads["quantum"]
    Z = quick.model.embed(data[4][:12])
    r = q.predict_on_backend(get_backend("qiskit"), Z, shots=8000)
    assert r.predictions.shape == (12,) and r.n_circuits == 12 and r.n_jobs == 1 and r.backend == "qiskit-aer"
    E = q.expvals_exact(Z)
    delta = np.abs(r.expvals - E).max()
    assert delta < 0.08                                                     # ~7 standard errors at 8000 shots
    # Wherever the exact logit margin exceeds what an error of delta in every <Z_i> can
    # move, the backend's prediction must be the exact one (a readout consistency check).
    logits = E @ q.params["a"] + q.params["b"]
    top2 = np.sort(logits, axis=1)
    a = q.params["a"]
    spread = max(np.abs(a[:, c] - a[:, d]).sum() for c in range(10) for d in range(10))
    safe = top2[:, -1] - top2[:, -2] > delta * spread
    assert safe.sum() >= 6 and np.array_equal(r.predictions[safe], np.argmax(logits, axis=1)[safe])


def test_predict_on_backend_respects_hardware_limits(head4, monkeypatch):
    from qedge import executor
    from qedge.quantum.backends.base import QuantumBackend

    class FakeHW(QuantumBackend):
        name, hardware = "fake-hw", True

        def _run(self, circuits, shots):  # pragma: no cover - must never be reached
            raise AssertionError("ran on hardware past the limits")

    monkeypatch.setattr(executor, "HW_MAX_CIRCUITS", 3)
    with pytest.raises(RuntimeError, match="QEDGE_HW_MAX_CIRCUITS"):
        head4.predict_on_backend(FakeHW(), np.zeros((4, 4)), shots=100)
    with pytest.raises(RuntimeError, match="QEDGE_HW_MAX_SHOTS"):
        head4.predict_on_backend(FakeHW(), np.zeros((2, 4)), shots=10 ** 6)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_predict_packed_on_torino_snapshot(head4):
    torino = C.snapshot("FakeTorino")
    Z = np.random.default_rng(6).uniform(-1, 1, (3, 4))
    r = head4.predict_packed(torino, Z, shots=2000, per_job=2)
    assert r.predictions.shape == (3,) and r.n_jobs == 2 < 3 and r.meta["devices"] == ["fake_torino"]
    assert [j["k"] for j in r.meta["jobs"]] == [2, 1] and sorted(sum((j["samples"] for j in r.meta["jobs"]), [])) == [0, 1, 2]
    lay = r.meta["layout"]
    assert len(lay) == 3 and not set(lay[0]) & set(lay[1])                  # same job: disjoint regions
    cost = P.packing_cost([r.meta["jobs"][0]["depth"]] * 2, 2000)
    assert r.meta["jobs"][0]["price"] == pytest.approx(cost.price_job)
    assert r.meta["price"] == pytest.approx(sum(j["price"] for j in r.meta["jobs"]))
    assert r.meta["price"] < r.meta["price_solo"]
    # noisy device: the estimates follow the ideal ones, damped by noise
    E = head4.expvals_exact(Z)
    assert np.corrcoef(r.expvals.ravel(), E.ravel())[0, 1] > 0.8


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_predict_packed_live_branch_in_local_mode(head4, monkeypatch):
    """A calibration marked live goes through SamplerV2; with a fake backend
    qiskit-ibm-runtime runs it locally, so submission and the per-register split run."""
    cal = copy.copy(C.snapshot("FakeMumbaiV2"))
    cal.source = "live"
    Z = np.random.default_rng(7).uniform(-1, 1, (2, 4))
    with pytest.raises(RuntimeError, match="QEDGE_HW_MAX_SHOTS"):
        head4.predict_packed(cal, Z, shots=10 ** 6)
    from qedge import executor
    monkeypatch.setattr(executor, "HW_MAX_CIRCUITS", 1)
    with pytest.raises(RuntimeError, match="refusing 2 packed jobs"):
        head4.predict_packed(cal, Z, shots=100, per_job=1)
    monkeypatch.undo()
    # per_job larger than the chip holds: 4 of these programs fit on 27 qubits with buffer=1, so
    # 6 samples need 2 jobs although ceil(6 / 6) = 1. The limit must count the planned jobs and
    # refuse before any submission.
    import qiskit_ibm_runtime as qir
    submitted = []
    monkeypatch.setattr(qir.SamplerV2, "run", lambda self, *a, **kw: submitted.append(1))
    monkeypatch.setattr(executor, "HW_MAX_CIRCUITS", 1)
    Z6 = np.random.default_rng(8).uniform(-1, 1, (6, 4))
    with pytest.raises(RuntimeError, match="refusing 2 packed jobs"):
        head4.predict_packed(cal, Z6, shots=100, per_job=6)
    assert submitted == []
    monkeypatch.undo()
    r = head4.predict_packed(cal, Z, shots=1000, per_job=2)
    assert r.n_jobs == 1 and not r.meta["jobs"][0]["emulated"] and r.meta["jobs"][0]["job_id"]
    assert np.corrcoef(r.expvals.ravel(), head4.expvals_exact(Z).ravel())[0, 1] > 0.8


# ---- leakage -----------------------------------------------------------------------
def test_reid_and_r2_metrics():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (50, 64))
    assert L.reid_rates(X, X) == (1.0, 1.0) and L.r2_pixels(X, X) == 1.0
    top1, top5 = L.reid_rates(X, X[rng.permutation(50)])
    assert top1 <= 0.1 and top5 <= 0.3
    assert L.r2_pixels(X, np.tile(X.mean(0), (50, 1))) == pytest.approx(0.0)


def test_private_ok_rule():
    base = {"chance": 1 / 360, "reid_label": 10 / 360}
    emb = {**base, "reid": 13 / 360, "reid_beyond_label": 3 / 360, "r2_beyond_label": 0.01}
    assert L.private_ok(emb)                                    # 3 images beyond the label: <= 5 x chance
    assert not L.private_ok(emb, relative_to_label=False)       # the label alone is already 10 x chance
    assert not L.private_ok({**emb, "reid": 0.9, "reid_beyond_label": 0.88})
    assert not L.private_ok({**emb, "r2_beyond_label": 0.2})
    assert L.private_ok({**emb, "reid": 0.9, "reid_beyond_label": 0.88}, max_reid=1.0)
    # Fewer hits than the label but 7 different images: the rate difference is negative,
    # the per-image count is 7 > 5 and fails.
    assert not L.private_ok({**emb, "reid": 8 / 360, "reid_beyond_label": 7 / 360})
    with pytest.raises(KeyError, match="reid_beyond_label"):
        L.private_ok({k: v for k, v in emb.items() if k != "reid_beyond_label"})
    assert L.private_ok({k: v for k, v in emb.items() if k != "reid_beyond_label"}, relative_to_label=False,
                        max_reid=0.05)
    reason = L.privacy_reason({**emb, "reid": 0.9, "reid_beyond_label": 0.88}, "conv1")
    assert "re-identify 90%" in reason and "88.0% that the label does not" in reason


def test_reid_beyond_label_is_counted_per_image():
    """A decoder that re-identifies exactly the set S: beyond the label is |S minus label hits|,
    not the difference of the two rates."""
    rng = np.random.default_rng(4)
    n, d = 60, 64
    proto = rng.uniform(0.2, 0.8, (10, d))                       # class structure, so the label helps
    ytr, yval, yte = (rng.integers(0, 10, m) for m in (300, 100, n))
    Xtr, Xval, Xte = (np.clip(proto[y] + rng.normal(0, 0.1, (len(y), d)), 0, 1) for y in (ytr, yval, yte))
    label_hits = set(np.flatnonzero(L.reid_ranks(Xte, L.label_baseline(Xtr, ytr, yte)) == 0))
    assert label_hits
    S = set(sorted(label_hits)[:1]) | set(range(0, n, 4))       # one label hit plus some others
    others = [i for i in range(n) if i not in S]
    Hte = Xte.copy()
    Hte[others] = Xte[np.roll(others, 1)]                         # these rows decode to another original
    r = L.attack(Xtr, Xval, Hte, Xtr, Xval, Xte, ytr, yte, attacks=("ridge",))
    assert r["reid"] == pytest.approx(len(S) / n)
    assert r["reid_beyond_label"] == pytest.approx(len(S - label_hits) / n)
    assert r["reid_with_label"] == pytest.approx(len(S | label_hits) / n)
    assert r["reid_minus_label"] == pytest.approx((len(S) - len(label_hits)) / n)
    assert r["reid_beyond_label"] > r["reid_minus_label"]


def test_knn_decoder_is_tuned_on_validation_per_metric():
    rng = np.random.default_rng(5)
    Xtr = rng.uniform(0, 1, (200, 16))
    Xval = np.clip(Xtr[:80] + rng.normal(0, 0.01, (80, 16)), 0, 1)   # near-duplicates: one neighbour is best
    decs, info = L.knn_decoder(Xtr * 3.0, Xtr, Xval * 3.0, Xval)
    assert set(decs) == set(L.METRICS) == set(info["chosen"])
    for m, c in info["chosen"].items():
        assert c["neighbours"] in L.KNN_NEIGHBOURS and isinstance(c["standardised"], bool)
    # activations are (scaled) images: one neighbour returns the training image
    assert np.allclose(decs["reid"](Xtr[:5] * 3.0), Xtr[:5]) and info["chosen"]["reid"]["neighbours"] == 1


def test_leakage_ordering_and_label_baseline(quick, data):
    rows = {r["cut"]: r for r in L.leakage_report(quick.model, data, cuts=("input", "conv2", "embedding"),
                                                  attacks=("ridge", "knn"))}
    n = len(data[4])
    inp, emb = rows["input"], rows["embedding"]
    assert inp["reid"] >= 0.99 and inp["r2"] > 0.99 and not inp["private_ok"]
    assert inp["reid"] > rows["conv2"]["reid"] > emb["reid"]
    # label-only baseline: one image per class is the nearest to its class mean
    assert inp["reid_label"] == pytest.approx(10 / n, abs=2 / n) and 0.2 < inp["r2_label"] < 0.7
    assert all(r["chance"] == 1 / n and r["r2_beyond_label"] == pytest.approx(r["r2"] - r["r2_label"])
               for r in rows.values())
    assert emb["r2_beyond_label"] < inp["r2_beyond_label"]
    for r in rows.values():
        assert set(r["attacks"]) == {"ridge", "knn"}
        assert r["reid"] == max(a["reid"] for a in r["attacks"].values())
        assert r["reid_beyond_label"] == max(a["reid_beyond_label"] for a in r["attacks"].values())
        assert r["reid_with_label"] == pytest.approx(r["reid_label"] + r["reid_beyond_label"])
        assert r["reid_beyond_label"] >= r["reid_minus_label"] - 1e-12


# ---- profile -----------------------------------------------------------------------
def _fake_leak(cut, reid, r2b, n=360):
    return {"cut": cut, "chance": 1 / n, "reid": reid, "reid_label": 10 / n,
            "reid_beyond_label": max(0.0, reid - 10 / n), "r2_beyond_label": r2b, "attacks": {"ridge": {}}}


def test_measured_profile_matches_schema(quick, data):
    leak = [_fake_leak("input", 1.0, 0.58), _fake_leak("conv1", 1.0, 0.5), _fake_leak("conv2", 0.9, 0.4),
            _fake_leak("embedding", 12 / 360, 0.01)]
    k4 = PR.summarise(quick, data[4], data[5], leak)
    k2 = copy.deepcopy(k4)
    m2 = nets.SplitCNN(2, nets.init_backbone(2, np.random.default_rng(0)), nets.LinearHead(2))
    k2["cut_stats"] = m2.cut_stats()
    k2["leakage"] = [_fake_leak("input", 1.0, 0.58), _fake_leak("conv1", 1.0, 0.5), _fake_leak("conv2", 0.95, 0.45),
                     _fake_leak("embedding", 0.2, 0.05)]
    prof = PR.measured_profile({4: k4, 2: k2})
    PR.validate_profile(prof)
    assert json.loads(json.dumps(prof)) == prof
    assert prof["input_bytes"] == 256 and prof["classes"] == 10 and prof["source"] == "measured (E8)"
    cuts = {c["name"]: c for c in prof["cuts"]}
    assert list(cuts) == ["input", "conv1", "conv2"]
    assert cuts["conv2"]["reid"] == 0.95 and not cuts["conv2"]["private_ok"]    # worst over k
    e4 = prof["embedding"]["4"]
    assert e4["bytes"] == 16 and e4["private_ok"] and not prof["embedding"]["2"]["private_ok"]
    assert e4["params"] == {"linear": 50, "mlp": 70, "quantum": 74}
    assert e4["quantum_circuit"] == quick.heads["quantum"].circuit_profile()
    val = k4["val_accuracy"]
    assert val == {h: quick.fits[h].best_val_acc for h in nets.HEADS}
    best = max(("linear", "mlp"), key=lambda h: (val[h], h == "linear"))
    assert e4["classical_head"] == best and e4["classical_selected_on"] == "validation"
    assert e4["back_flop"] == quick.heads[best].flop and e4["classical_accuracy"] == e4["accuracy"][best]
    # the classical head is chosen on validation even when the test split would pick the other
    k4b = copy.deepcopy(k4)
    k4b["val_accuracy"].update(linear=0.90, mlp=0.95)
    k4b["accuracy"].update(linear=0.97, mlp=0.93)
    e = PR.measured_profile({4: k4b})["embedding"]["4"]
    assert e["classical_head"] == "mlp" and e["classical_accuracy"] == 0.93 and e["back_flop"] == quick.heads["mlp"].flop
    # conv2's back FLOPs are the dense layer plus the linear head: 148 k, kept per k
    assert cuts["conv2"]["back_flop"] == 148 * 4 and cuts["conv2"]["back_flop_by_k"] == {"2": 296.0, "4": 592.0}
    assert prof["privacy"]["attacks"] == ["ridge"] and "per image" in prof["privacy"]["rule"]
    assert e4["accuracy"]["quantum"] == quick.accuracy("quantum", data[4], data[5])
    bad = copy.deepcopy(prof)
    del bad["embedding"]["4"]["accuracy"]["quantum_l0"]
    with pytest.raises(ValueError):
        PR.validate_profile(bad)
