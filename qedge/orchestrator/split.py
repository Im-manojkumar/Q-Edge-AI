"""Split placements: cut a neural network between the edge and a remote resource (idea 3).

A ``split_inference`` workload runs a batch of images through a model of the form

    image -> CNN backbone -> k-dim embedding -> head -> class scores

and the router chooses where to cut it (Neurosurgeon-style partitioning, Kang et al.,
ASPLOS 2017) and where each part runs. The edge runs the network up to a named cut point
and uploads that activation; a cloud CPU/GPU runs the rest with a classical head, or the
quantum head (a k-qubit variational circuit plus a linear readout) runs on a cloud
state-vector simulator or a QPU. Placement labels:

    edge_gpu                    whole model on the edge (nothing leaves it)
    cloud_gpu                   whole model in the cloud (the raw input is uploaded)
    edge_gpu>conv2>cloud_gpu    edge up to conv2, the conv2 activation is uploaded, cloud runs the rest
    edge_gpu>embedding>qpu      edge up to the embedding, the k numbers go to the QPU's quantum head

A *split profile* (schema in ``check_profile``) describes one model: bytes and FLOPs at
every cut, head accuracies per embedding size k, the quantum head's circuit, and how much
an attacker can recover from each transmitted tensor. "Private" is a measured property of
a cut against a stated attack (``qedge.ml.leakage``: decoders that see the activation and
reconstruct / re-identify the input), never a guarantee.

Assumptions (all of them, so they can be checked against the report):

* ``digits`` defaults: bytes and FLOPs are computed analytically from the architecture
  (multiply-add = 2 FLOPs; bias, ReLU and pooling not counted; tensors as float32). The
  head accuracies and every leakage figure are PLACEHOLDERS, not measurements: pessimistic
  round numbers (quantum head below the classical heads, nothing transmitted is private)
  that ``load_profiles`` replaces with the measured profile from experiment E8.
* ``camera`` is an assumption profile: MobileNetV2 (1.0, 224x224) with a k-dim bottleneck
  before the head. FLOPs from the published total of about 300 M multiply-adds per image
  (Sandler et al., 2018), minus its 1000-way classifier; activation sizes from the
  architecture; input 0.15 MB per image as in ``make_inference``. Its leakage is NOT
  measured: every cut but the embedding is marked not private, and the embedding's
  accuracies and privacy verdict are inherited from the digits profile (labelled so).
* A cut's ``back_flop`` is recomputed per k as total(k) - front, so a split never adds
  or loses work against the whole model (the backbone's last layer and the head depend on k).
* The classical back returns float32 logits (4 x classes bytes per image); the quantum
  head returns k expectation values (4 x k bytes per image) and its linear readout runs on
  the edge. Upload to a quantum resource pays bytes / uplink here; its round trip is the
  per-round hop already in ``CostModel._quantum``.
* Quantum head quality = f x (ideal quantum accuracy) + (1 - f) / classes, f = estimated
  circuit fidelity: an error anywhere randomises the answer. This is a model (pessimistic
  for errors that do not flip the argmax); E8 checks it against emulated-device runs.
* Shot noise of the quantum head's <Z_i> estimates is not modelled in its quality.
"""
from __future__ import annotations

import copy
import json
import os
from typing import TYPE_CHECKING

from ..quantum.circuit import Circuit
from ..resources import RADIO_POWER_W, Resource

if TYPE_CHECKING:
    from ..resources import ResourceState
    from ..workloads import Workload
    from .estimator import CostModel, Estimate, Placement

DEFAULT_K = 4
K_RANGE = range(2, 9)
DEFAULT_LAYERS = 2            # trainable layers of the quantum head in the built-in profiles
HEADS = ("linear", "mlp")              # classical heads
QHEADS = ("quantum", "quantum_l0")     # trained circuit, and the L=0 ablation (encoding + readout only)
BYTES_PER_VALUE = 4                    # float32 activations

# Placeholder accuracies for the digits defaults, the same for every k. NOT measured:
# pessimistic round numbers with the quantum head below the classical heads, so the router
# never favours the quantum head before E8 has measured it.
PLACEHOLDER_ACCURACY = {"linear": 0.80, "mlp": 0.80, "quantum": 0.75, "quantum_l0": 0.70}
PLACEHOLDER_NOTE = "built-in placeholder (accuracy and leakage not measured)"


# ---------------------------------------------------------------------------
# Head shapes
# ---------------------------------------------------------------------------

def head_circuit(k: int, layers: int) -> Circuit:
    """The quantum head's circuit structure in the repo gate set: RY(pi/2 (z_i + 1)) encoding,
    then ``layers`` x (RZ RY RZ on every qubit, CZ ring; no wrap-around for k = 2). Angles are
    zero here: gate counts and depth, which the cost model uses, do not depend on them."""
    c = Circuit(k, name=f"qhead_k{k}_L{layers}")
    for i in range(k):
        c.ry(i, 0.0)
    for _ in range(layers):
        for i in range(k):
            c.rz(i, 0.0).ry(i, 0.0).rz(i, 0.0)
        for i in range(k - 1):
            c.cz(i, i + 1)
        if k > 2:
            c.cz(k - 1, 0)
    return c


def head_circuit_stats(k: int, layers: int) -> dict:
    c = head_circuit(k, layers)
    return {"n_qubits": k, "layers": layers, "depth": c.depth(), "n_1q": c.n_1q, "n_2q": c.n_2q}


def head_params(k: int, layers: int, classes: int) -> dict:
    """Parameter counts: linear k->classes; quantum = 3 angles per qubit and layer plus its
    linear readout; mlp k->h->classes with h chosen to match the quantum head's count."""
    linear = k * classes + classes
    quantum = 3 * k * layers + linear
    h = min(range(1, 257), key=lambda h: (abs(h * (k + 1 + classes) + classes - quantum), h))
    return {"linear": linear, "mlp": h * (k + 1 + classes) + classes, "quantum": quantum, "mlp_hidden": h}


def _cut(name: str, values: int, nbytes: int, front: float, back: float, reid=None, r2b=None,
         private_ok: bool = False) -> dict:
    return {"name": name, "values": int(values), "bytes": int(nbytes), "front_flop": float(front),
            "back_flop": float(back), "reid": reid, "r2_beyond_label": r2b, "private_ok": bool(private_ok)}


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

def digits_default(layers: int = DEFAULT_LAYERS) -> dict:
    """The Part A model on 8x8 digits, with analytic bytes/FLOPs and placeholder accuracy/leakage.

    input 1x8x8 -> conv3x3 1->8 pad 1, ReLU, maxpool 2 -> conv1 (8x4x4) -> conv3x3 8->16 pad 1,
    ReLU, maxpool 2 -> conv2 (16x2x2) -> dense 64->k, tanh -> embedding (k) -> head (10 classes)."""
    classes = 10
    conv1 = 2 * (8 * 8) * 8 * (3 * 3 * 1)      # 9 216: 8x8 outputs x 8 channels x 9 taps
    conv2 = conv1 + 2 * (4 * 4) * 16 * (3 * 3 * 8)   # + 36 864
    emb = {}
    for k in K_RANGE:
        front = conv2 + 2 * 64 * k
        emb[str(k)] = {"bytes": BYTES_PER_VALUE * k, "front_flop": float(front),
                       "back_flop": float(2 * k * classes),          # linear head
                       "reid": None, "r2_beyond_label": None, "private_ok": False,
                       "accuracy": dict(PLACEHOLDER_ACCURACY),
                       "quantum_circuit": head_circuit_stats(k, layers),
                       "params": head_params(k, layers, classes), "source": PLACEHOLDER_NOTE}
    total = emb[str(DEFAULT_K)]["front_flop"] + emb[str(DEFAULT_K)]["back_flop"]
    return {
        "name": "digits", "label": "8x8 digits, small CNN split model",
        "source": "built-in defaults (bytes and FLOPs from the architecture; accuracy and leakage are "
                  "placeholders until E8 writes results/split_profile.json)",
        "classes": classes, "input_bytes": 64 * BYTES_PER_VALUE,
        "cuts": [
            # Sending the input is sending the image: the attacker's copy is exact.
            _cut("input", 64, 64 * BYTES_PER_VALUE, 0.0, total, reid=1.0),
            _cut("conv1", 128, 128 * BYTES_PER_VALUE, conv1, total - conv1),
            _cut("conv2", 64, 64 * BYTES_PER_VALUE, conv2, total - conv2),
        ],
        "embedding": emb,
    }


# MobileNetV2 1.0 at 224x224: ~300 M multiply-adds per image (Sandler et al., 2018), which
# includes the 1280 -> 1000 classifier that the k-dim bottleneck replaces.
MOBILENET_V2_MADDS = 300e6
CAMERA_INPUT_BYTES = 150_000   # 0.15 MB per image, as make_inference assumes


def camera_profile(digits: dict) -> dict:
    """Assumption profile for an edge camera model, inheriting head accuracies, the number of
    classes they were measured on, and the embedding's privacy verdict from ``digits`` (see the
    module docstring)."""
    stem = 2 * (112 * 112) * 32 * (3 * 3 * 3)               # first conv, stride 2: 21.7 MFLOP
    last_conv = 2 * (MOBILENET_V2_MADDS - 1280 * 1000)      # through the final 1x1 conv to 7x7x1280
    pool = last_conv + 7 * 7 * 1280                         # global average pooling (adds)
    classes = digits["classes"]
    emb = {}
    for k, e in digits["embedding"].items():
        kk = int(k)
        emb[k] = {**copy.deepcopy(e), "bytes": BYTES_PER_VALUE * kk,
                  "front_flop": float(pool + 2 * 1280 * kk), "back_flop": float(2 * kk * classes),
                  "inherited_from": "digits: accuracy, params, quantum circuit and privacy verdict"}
    ref = emb.get(str(DEFAULT_K)) or next(iter(emb.values()))
    total = ref["front_flop"] + ref["back_flop"]
    not_measured = "not measured, assumed not private"
    return {
        "name": "camera", "label": "MobileNetV2 224x224 edge camera with a k-dim bottleneck",
        "source": "assumption: MobileNetV2 published totals and activation sizes; leakage " + not_measured
                  + " except the embedding, whose verdict and accuracies are inherited from digits",
        "classes": classes, "input_bytes": CAMERA_INPUT_BYTES,
        "cuts": [
            _cut("input", 224 * 224 * 3, CAMERA_INPUT_BYTES, 0.0, total, reid=1.0),
            _cut("stem", 112 * 112 * 32, 112 * 112 * 32 * BYTES_PER_VALUE, stem, total - stem),
            _cut("last_conv", 7 * 7 * 1280, 7 * 7 * 1280 * BYTES_PER_VALUE, last_conv, total - last_conv),
            _cut("pool", 1280, 1280 * BYTES_PER_VALUE, pool, total - pool),
        ],
        "embedding": emb,
    }


def _defaults() -> dict[str, dict]:
    d = digits_default()
    return {"digits": d, "camera": camera_profile(d)}


# Placeholders overwritten by results/split_profile.json when E8 has run (same pattern as
# CALIBRATION / load_calibration in workloads.py).
PROFILES: dict[str, dict] = _defaults()


# ---------------------------------------------------------------------------
# Schema, loading, lookups
# ---------------------------------------------------------------------------

def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_profile(p: dict) -> list[str]:
    """Problems with a split profile (empty list = matches the schema):

    {"name", "label", "source", "classes": int, "input_bytes": int,
     "cuts": [{"name", "values", "bytes", "front_flop", "back_flop",
               "reid": float|None, "r2_beyond_label": float|None, "private_ok": bool}],
     "embedding": {"<k>": {"bytes", "front_flop", "back_flop", "reid", "r2_beyond_label", "private_ok",
                           "accuracy": {"linear", "mlp", "quantum", "quantum_l0"},
                           "quantum_circuit": {"n_qubits", "layers", "depth", "n_1q", "n_2q"},
                           "params": {"linear", "mlp", "quantum"}}}}

    Extra keys are allowed (e.g. "source" or "inherited_from" on an embedding entry)."""
    out = []
    if not isinstance(p, dict):
        return ["not a dict"]
    for key in ("name", "label", "source"):
        if not isinstance(p.get(key), str):
            out.append(f"{key} must be a string")
    for key in ("classes", "input_bytes"):
        if not _num(p.get(key)) or p.get(key) <= 0:
            out.append(f"{key} must be a positive number")

    def leak(where: str, d: dict):
        for key in ("reid", "r2_beyond_label"):
            if key not in d or not (d[key] is None or _num(d[key])):
                out.append(f"{where}.{key} must be a number or null")
        if not isinstance(d.get("private_ok"), bool):
            out.append(f"{where}.private_ok must be true/false")

    cuts = p.get("cuts")
    if not isinstance(cuts, list) or not cuts:
        out.append("cuts must be a non-empty list")
        cuts = []
    for i, c in enumerate(cuts):
        where = f"cuts[{i}]"
        if not isinstance(c, dict) or not isinstance(c.get("name"), str):
            out.append(f"{where} needs a name")
            continue
        for key in ("values", "bytes", "front_flop", "back_flop"):
            if not _num(c.get(key)) or c[key] < 0:
                out.append(f"{where}.{key} must be a number >= 0")
        leak(where, c)
    emb = p.get("embedding")
    if not isinstance(emb, dict) or not emb:
        return out + ["embedding must be a non-empty dict keyed by k"]
    for k, e in emb.items():
        where = f"embedding[{k}]"
        if not str(k).isdigit() or not isinstance(e, dict):
            out.append(f"{where}: key must be k as a string, value a dict")
            continue
        for key in ("bytes", "front_flop", "back_flop"):
            if not _num(e.get(key)) or e[key] < 0:
                out.append(f"{where}.{key} must be a number >= 0")
        leak(where, e)
        for sub, keys in (("accuracy", HEADS + QHEADS), ("quantum_circuit", ("n_qubits", "layers", "depth", "n_1q", "n_2q")),
                          ("params", ("linear", "mlp", "quantum"))):
            d = e.get(sub)
            if not isinstance(d, dict) or any(not _num(d.get(x)) for x in keys):
                out.append(f"{where}.{sub} needs numbers for {', '.join(keys)}")
        qc = e.get("quantum_circuit")
        if isinstance(qc, dict) and _num(qc.get("n_qubits")) and qc["n_qubits"] != int(k):
            out.append(f"{where}.quantum_circuit.n_qubits must equal k")
    return out


def merge_profile(default: dict, measured: dict | None) -> dict:
    """Measured values over a default profile: top-level keys and the cut list are replaced,
    embedding entries are replaced per k. A k only the default has keeps its placeholder
    entry, which says so in its "source"."""
    if measured is None:
        return copy.deepcopy(default)
    out = {**copy.deepcopy(default), **copy.deepcopy(measured)}
    out["embedding"] = {**copy.deepcopy(default.get("embedding", {})), **copy.deepcopy(measured.get("embedding", {}))}
    return out


def load_profiles(path: str = "results/split_profile.json") -> dict[str, dict]:
    """Rebuild ``PROFILES`` from the built-in defaults and merge measured profiles over them.

    The file holds one profile, a list of profiles, or {name: profile}. A measured digits
    profile also updates what ``camera`` inherits, unless the file has its own camera
    profile. A missing file resets ``PROFILES`` to the defaults."""
    measured: dict[str, dict] = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "cuts" in data:
            items = [data]
        else:
            items = data if isinstance(data, list) else list(data.values())
        for p in items:
            problems = check_profile(p)
            if problems:
                raise ValueError(f"{path}: profile {p.get('name') if isinstance(p, dict) else p!r}: "
                                 + "; ".join(problems))
            measured[p["name"]] = p
    digits = merge_profile(digits_default(), measured.pop("digits", None))
    camera = merge_profile(camera_profile(digits), measured.pop("camera", None))
    PROFILES.clear()
    PROFILES.update({"digits": digits, "camera": camera, **measured})
    return PROFILES


def get_profile(name: str) -> dict:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown split profile {name!r}; known: {', '.join(PROFILES)}") from None


def embedding(prof: dict, k: int) -> dict:
    e = prof["embedding"].get(str(k))
    if e is None:
        raise ValueError(f"profile {prof['name']!r} has no k={k} embedding; available: "
                         f"{', '.join(sorted(prof['embedding'], key=int))}")
    return e


def total_flop(prof: dict, k: int) -> float:
    """FLOPs per sample of the whole model with the classical head."""
    e = embedding(prof, k)
    return e["front_flop"] + e["back_flop"]


def cut_info(prof: dict, cut: str, k: int) -> dict:
    """Bytes, FLOPs and leakage of one cut for embedding size k (``back_flop`` = total(k) - front)."""
    if cut == "embedding":
        return {"name": "embedding", "values": k, **embedding(prof, k)}
    for c in prof["cuts"]:
        if c["name"] == cut:
            return {**c, "back_flop": max(0.0, total_flop(prof, k) - c["front_flop"])}
    raise KeyError(f"profile {prof['name']!r} has no cut {cut!r}; cuts: "
                   f"{', '.join(c['name'] for c in prof['cuts'])}, embedding")


def result_bytes(prof: dict) -> int:
    """Per image, back to the edge from a classical back: float32 logits."""
    return BYTES_PER_VALUE * int(prof["classes"])


def head_quality(w: Workload, fid: float) -> float:
    """Modelled accuracy of the quantum head at circuit fidelity ``fid`` (see the module docstring)."""
    prof = get_profile(w.params["profile"])
    acc = embedding(prof, w.params["k"])["accuracy"][w.params.get("qhead", "quantum")]
    return fid * acc + (1 - fid) / prof["classes"]


# ---------------------------------------------------------------------------
# Candidates and privacy
# ---------------------------------------------------------------------------

def split_cuts(prof: dict, k: int) -> list[str]:
    """Cuts a split may use: every named cut except the input (that is all-cloud) and any cut
    with nothing left to run after it (the output), then the embedding."""
    named = [c["name"] for c in prof["cuts"] if c["name"] not in ("input", "embedding")
             and cut_info(prof, c["name"], k)["back_flop"] > 0]
    return named + ["embedding"]


def candidate_split_placements(w: Workload, resources: dict[str, Resource]) -> list[Placement]:
    """Whole model on every classical resource, every (edge front, cut, cloud back), and every
    (edge front, embedding, quantum resource)."""
    from .estimator import Placement
    prof = get_profile(w.params["profile"])
    classical = [n for n, r in resources.items() if not r.is_quantum and r.available]
    edge = [n for n in classical if resources[n].is_edge]
    cloud = [n for n in classical if not resources[n].is_edge]
    quantum = [n for n, r in resources.items() if r.is_quantum and r.available]
    out = [Placement(n) for n in classical]
    out += [Placement(f, split=c, back=b) for f in edge for c in split_cuts(prof, w.params["k"]) for b in cloud]
    out += [Placement(f, q, split="embedding") for f in edge for q in quantum]
    return out


def transmitted(p: Placement, resources: dict[str, Resource]) -> list[str]:
    """Names of the tensors that leave the edge under placement ``p``."""
    front = resources[p.classical]
    sent = [] if front.is_edge else ["input"]
    if p.split is not None or p.quantum is not None:
        if front.is_edge:
            sent.append(p.split or "embedding")
    return sent


def _pct(x: float) -> str:
    return f"{x:.0%}" if x >= 0.1 else f"{x:.1%}"


def leak_text(c: dict) -> str:
    if c["name"] == "input":
        return "the raw input leaves the edge"
    if c.get("reid") is None:
        return f"{c['name']} leakage not measured, so it counts as not private"
    s = f"{c['name']} activations re-identify {_pct(c['reid'])} of inputs"
    if c.get("r2_beyond_label") is not None:
        s += f" and explain {c['r2_beyond_label']:.2f} of pixel variance beyond the label"
    return s


def privacy_verdict(w: Workload, p: Placement, resources: dict[str, Resource]) -> tuple[bool, str]:
    """(ok, why): ok iff everything transmitted is at a cut the profile marks private_ok.
    Judged whether or not the workload is private; the caller decides whether it matters."""
    prof, k = get_profile(w.params["profile"]), w.params["k"]
    sent = transmitted(p, resources)
    if not sent:
        return True, "nothing leaves the edge"
    for name in sent:
        try:
            c = cut_info(prof, name, k)
        except KeyError as e:
            return False, f"privacy: {e.args[0]}"
        if name == "input" or not c["private_ok"]:
            return False, "privacy: " + leak_text(c)
    c = cut_info(prof, sent[-1], k)
    what = f"the {k}-number embedding" if sent[-1] == "embedding" else f"the {sent[-1]} activation"
    extra = f"; it re-identifies {_pct(c['reid'])} of inputs against the attacks tried" if c.get("reid") is not None else ""
    return True, f"only {what} leaves the edge{extra}"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def estimate_split(model: CostModel, w: Workload, p: Placement, st: ResourceState) -> Estimate:
    """Latency, energy, money and quality of ``p`` for a split_inference workload.

    Whole-model placements use the same code as every other kind (raw input uploaded to a
    cloud resource). A split pays: front compute on the edge (batch x front FLOPs / rate),
    the upload of batch x activation bytes and download of the results, then either the
    back compute on a cloud CPU/GPU or the quantum head through ``CostModel._quantum``
    (queue, job overhead, round trip, shots, fidelity, calibration figures). A hybrid
    without a cut (e.g. ``cloud_cpu+qpu``) is read as host -> embedding -> quantum head."""
    from .estimator import Estimate, transfer_time
    res = model.resources
    est = Estimate(p)
    prof, k, batch = get_profile(w.params["profile"]), int(w.params["k"]), int(w.params["batch"])
    front = res[p.classical]
    if w.privacy:
        ok, why = privacy_verdict(w, p, res)
        if not ok:
            est.feasible = False
            est.reasons.append(why)
    if p.split is None and p.quantum is None:
        model._classical(w, front, st, est, w.gflop, w.data_mb, w.result_mb)
        est.quality = w.classical_quality
        return est
    cut = p.split or "embedding"
    back = res[p.back] if p.back is not None else None
    bad = None
    if p.quantum is not None and cut != "embedding":
        bad = f"the quantum head takes the embedding, not {cut}"
    elif back is not None and (not front.is_edge or back.is_edge or back.is_quantum):
        bad = "a split runs its front on an edge resource and its back on a cloud classical resource"
    if bad is None:
        try:
            c = cut_info(prof, cut, k)
        except KeyError as e:
            bad = e.args[0]
    if bad is not None:
        est.feasible = False
        est.reasons.append(bad)
        return est

    front_flop = c["front_flop"]
    if p.quantum is not None:
        front_flop += 2 * k * prof["classes"]     # the quantum head's linear readout, on the edge
    for r in (front, back):
        if r is not None and w.memory_gb > r.memory_gb:
            est.feasible = False
            est.reasons.append(f"needs {w.memory_gb:.1f} GB > {r.memory_gb:.0f} GB on {r.name}")
    rate = front.gflops if w.parallel else front.serial_gflops
    t_front = batch * front_flop / 1e9 / max(rate, 1e-9)
    wait = st.queue_wait_s.get(front.name, 0.0)
    up_mb = batch * c["bytes"] / 1e6
    if not front.is_edge:     # host+quantum with a cloud host: the raw input goes up first
        xfer = transfer_time(front, w.data_mb, w.result_mb, st.rtt_s.get(front.name))
    elif back is not None:
        xfer = transfer_time(back, up_mb, w.result_mb, st.rtt_s.get(back.name))
    else:                     # bytes only; the round trip is _quantum's per-round hop
        xfer = transfer_time(res[p.quantum], up_mb, batch * BYTES_PER_VALUE * k / 1e6, rtt=0.0)
    est.breakdown.update(queue=wait, transfer=xfer, compute=t_front)
    est.latency_s += wait + xfer + t_front
    est.energy_j += t_front * front.power_w + xfer * RADIO_POWER_W
    est.edge_energy_j += (t_front * front.power_w if front.is_edge else 0.0) + xfer * RADIO_POWER_W
    est.cost_usd += t_front * front.cost_per_s
    if back is not None:
        rate_b = back.gflops if w.parallel else back.serial_gflops
        t_back = batch * c["back_flop"] / 1e9 / max(rate_b, 1e-9)
        wait_b = st.queue_wait_s.get(back.name, 0.0)
        est.breakdown.update(back_queue=wait_b, back_compute=t_back)
        est.latency_s += wait_b + t_back
        est.energy_j += t_back * back.power_w
        est.cost_usd += t_back * back.cost_per_s
        est.quality = w.classical_quality
        return est
    fid = model._quantum(w, front, res[p.quantum], st, est, check_privacy=False)
    est.quality = head_quality(w, fid)
    return est


def quality_note(w: Workload, e: Estimate) -> str:
    """What the quantum-gate quality check prints for a quantum head."""
    prof = get_profile(w.params["profile"])
    acc = embedding(prof, w.params["k"])["accuracy"][w.params.get("qhead", "quantum")]
    f = e.fidelity if e.fidelity is not None else 1.0
    if f >= 1.0:
        return f"quantum head accuracy {e.quality:.3f}"
    return f"quantum head accuracy {e.quality:.3f} (ideal {acc:.3f} at circuit fidelity {f:.3f})"
