"""The split model: a small CNN backbone with named cut points, and its heads.

``8x8 image -> conv backbone (edge) -> k-dim embedding -> head -> 10 logits``.
The network can be cut after any named stage; the edge device runs the part
before the cut and transmits that activation, a remote resource runs the rest.
This is "Neurosurgeon"-style partitioning (Kang et al., ASPLOS 2017), here
with a quantum tier for the head (``qhead.py``).

Backbone stages and the activation at each cut point (values per sample)::

    input      1x8x8  = 64    the raw image
    conv1      8x4x4  = 128   conv3x3 1->8, pad 1, ReLU, maxpool 2
    conv2      16x2x2 = 64    conv3x3 8->16, pad 1, ReLU, maxpool 2
    embedding  k              dense 64->k, tanh: k values in [-1, 1]

conv1 is twice the size of the input: cutting early can cost more upload
than sending the raw image, the classic Neurosurgeon observation.

Everything is written in ``autograd.numpy`` so gradients come from autograd
(PyTorch is not available here). The convolution is im2col + one matmul.
FLOPs are counted analytically as 2 per multiply-add of the conv and dense
layers; bias adds, activations and pooling are not counted.

Heads on the embedding: ``LinearHead`` (k -> 10), ``MLPHead`` (k -> h -> 10,
tanh, h chosen to match the quantum head's parameter count) and
``qhead.QuantumHead``. Two training modes:

* transfer (default; Mari et al. 2020, "Transfer learning in hybrid
  classical-quantum neural networks"): train backbone + linear head end to
  end, freeze the backbone, then train every head on the frozen embeddings
  (the linear head too, from the same embeddings, for symmetry);
* end to end (fairness check): backbone + head from scratch for each head,
  because a backbone trained with a linear head yields an embedding that
  favours a linear head.

Adam, minibatches, softmax cross-entropy, early stopping on validation
accuracy (the best epoch's parameters are kept). One seed fixes the
initialisation and the minibatch order: same seed, same result.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad
from autograd.misc import flatten
from autograd.scipy.special import logsumexp

CLASSES = 10
CUTS = ("input", "conv1", "conv2", "embedding")
SHAPES = {"input": (1, 8, 8), "conv1": (8, 4, 4), "conv2": (16, 2, 2)}   # embedding: (k,)
FLOAT_BYTES = 4                                                          # activations are sent as float32


# ---- layers --------------------------------------------------------------------
def conv3x3(x, w, b):
    """3x3 convolution, stride 1, zero padding 1, as im2col + one matmul.

    x: (N, C, H, W), w: (O, C, 3, 3), b: (O,). The nine shifted views of the
    padded input become the columns (N, C*9, H*W); with w reshaped to (O, C*9)
    the convolution is one matrix product. Column index c*9 + 3*i + j matches
    the row-major reshape of w."""
    n, c, h, wd = x.shape
    xp = anp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    cols = anp.stack([xp[:, :, i:i + h, j:j + wd] for i in range(3) for j in range(3)], axis=2)
    cols = anp.reshape(cols, (n, c * 9, h * wd))
    out = anp.matmul(anp.reshape(w, (w.shape[0], c * 9)), cols) + b[None, :, None]
    return anp.reshape(out, (n, w.shape[0], h, wd))


def relu(x):
    return anp.maximum(x, 0.0)


def maxpool2(x):
    n, c, h, w = x.shape
    return anp.max(anp.reshape(x, (n, c, h // 2, 2, w // 2, 2)), axis=(3, 5))


def _stage(cut: str, p: dict, x):
    """The layer block that produces the activation at ``cut`` from the previous one."""
    if cut == "conv1":
        return maxpool2(relu(conv3x3(x, p["conv1_w"], p["conv1_b"])))
    if cut == "conv2":
        return maxpool2(relu(conv3x3(x, p["conv2_w"], p["conv2_b"])))
    if cut == "embedding":
        return anp.tanh(anp.dot(anp.reshape(x, (x.shape[0], -1)), p["fc_w"]) + p["fc_b"])
    raise ValueError(f"no stage produces {cut!r}")


def backbone(p: dict, x, start: str = "input", stop: str = "embedding"):
    """Run the backbone from the activation at ``start`` to the one at ``stop``."""
    i, j = CUTS.index(start), CUTS.index(stop)
    if j < i:
        raise ValueError(f"cannot run backwards from {start} to {stop}")
    for cut in CUTS[i + 1:j + 1]:
        x = _stage(cut, p, x)
    return x


def init_backbone(k: int, rng: np.random.Generator) -> dict:
    """He-normal convolutions (ReLU), fan-in-scaled dense layer (tanh), zero biases."""
    return {"conv1_w": rng.normal(0.0, math.sqrt(2 / 9), (8, 1, 3, 3)), "conv1_b": np.zeros(8),
            "conv2_w": rng.normal(0.0, math.sqrt(2 / 72), (16, 8, 3, 3)), "conv2_b": np.zeros(16),
            "fc_w": rng.normal(0.0, math.sqrt(1 / 64), (64, k)), "fc_b": np.zeros(k)}


def stage_flops(k: int) -> dict[str, int]:
    """FLOPs per sample of the stage that produces each cut (2 per multiply-add)."""
    return {"conv1": 2 * (8 * 8) * (1 * 9) * 8,      # 64 output positions x 9 taps x 8 filters
            "conv2": 2 * (4 * 4) * (8 * 9) * 16,     # 16 positions x 72 taps x 16 filters
            "embedding": 2 * 64 * k}


# ---- heads ---------------------------------------------------------------------
@dataclass
class LinearHead:
    """z -> z W + b."""

    k: int
    classes: int = CLASSES
    name: str = "linear"
    params: dict | None = None

    def init(self, rng: np.random.Generator) -> dict:
        return {"w": rng.normal(0.0, math.sqrt(1 / self.k), (self.k, self.classes)), "b": np.zeros(self.classes)}

    def logits(self, p: dict, z):
        return anp.dot(z, p["w"]) + p["b"]

    @property
    def n_params(self) -> int:
        return self.k * self.classes + self.classes

    @property
    def flop(self) -> int:
        return 2 * self.k * self.classes

    def config(self) -> dict:
        return {}


@dataclass
class MLPHead:
    """z -> tanh(z W1 + b1) W2 + b2 with ``hidden`` units."""

    k: int
    hidden: int
    classes: int = CLASSES
    name: str = "mlp"
    params: dict | None = None

    def init(self, rng: np.random.Generator) -> dict:
        return {"w1": rng.normal(0.0, math.sqrt(1 / self.k), (self.k, self.hidden)), "b1": np.zeros(self.hidden),
                "w2": rng.normal(0.0, math.sqrt(1 / self.hidden), (self.hidden, self.classes)),
                "b2": np.zeros(self.classes)}

    def logits(self, p: dict, z):
        return anp.dot(anp.tanh(anp.dot(z, p["w1"]) + p["b1"]), p["w2"]) + p["b2"]

    @property
    def n_params(self) -> int:
        return mlp_params(self.k, self.hidden, self.classes)

    @property
    def flop(self) -> int:
        return 2 * (self.k * self.hidden + self.hidden * self.classes)

    def config(self) -> dict:
        return {"hidden": self.hidden}


def mlp_params(k: int, hidden: int, classes: int = CLASSES) -> int:
    return k * hidden + hidden + hidden * classes + classes


def mlp_hidden_for(k: int, target_params: int, classes: int = CLASSES) -> int:
    """Hidden width whose parameter count is closest to ``target_params`` (ties: smaller)."""
    return min(range(1, 4 * target_params + 2), key=lambda h: (abs(mlp_params(k, h, classes) - target_params), h))


HEADS = ("linear", "mlp", "quantum", "quantum_l0")


def make_head(name: str, k: int, layers: int = 2, hidden: int | None = None):
    """A fresh head. The MLP's width matches an L=``layers`` quantum head's parameter count."""
    from .qhead import QuantumHead, quantum_params
    if name == "linear":
        return LinearHead(k)
    if name == "mlp":
        return MLPHead(k, hidden or mlp_hidden_for(k, quantum_params(k, layers)))
    if name == "quantum":
        return QuantumHead(k, layers=layers)
    if name == "quantum_l0":
        return QuantumHead(k, layers=0, name="quantum_l0")
    raise ValueError(f"unknown head {name!r}; choose from {HEADS}")


# ---- training ------------------------------------------------------------------
def cross_entropy(logits, y) -> float:
    """Mean softmax cross-entropy of integer labels ``y``."""
    return -anp.mean(logits[np.arange(len(y)), y] - logsumexp(logits, axis=1))


@dataclass
class FitResult:
    params: dict
    history: list[dict]            # one row per epoch: train loss, validation accuracy, seconds
    best_epoch: int
    best_val_acc: float
    seconds: float
    steps: int


def fit(params: dict, logits_fn, Xtr, ytr, Xval, yval, *, epochs: int = 40, batch: int = 32, lr: float = 0.01,
        patience: int = 8, seed: int = 0) -> FitResult:
    """Adam on minibatches of softmax cross-entropy; early stopping on validation accuracy.

    ``logits_fn(params, X)`` must be written in autograd.numpy (a PennyLane
    QNode with the autograd interface qualifies). The parameters of the best
    validation epoch are returned (ties keep the earlier epoch); training stops
    after ``patience`` epochs without improvement. ``seed`` fixes the minibatch
    order."""
    flat, unflatten = flatten(params)
    flat = np.array(flat, dtype=np.float64)

    def loss(fp, xb, yb):
        return cross_entropy(logits_fn(unflatten(fp), xb), yb)

    step_fn = value_and_grad(loss)
    rng = np.random.default_rng(seed)
    m, v = np.zeros_like(flat), np.zeros_like(flat)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_flat, best_acc, best_epoch, wait, t, hist = flat.copy(), -1.0, 0, 0, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(Xtr))
        losses = []
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            val, g = step_fn(flat, Xtr[idx], ytr[idx])
            t += 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            flat = flat - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            losses.append(float(val))
        acc = float(np.mean(np.argmax(np.asarray(logits_fn(unflatten(flat), Xval)), axis=1) == yval))
        hist.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_acc": acc,
                     "seconds": time.perf_counter() - t0})
        if acc > best_acc:
            best_flat, best_acc, best_epoch, wait = flat.copy(), acc, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return FitResult(unflatten(best_flat), hist, best_epoch, best_acc, time.perf_counter() - t0, t)


# ---- the split model -----------------------------------------------------------
@dataclass
class SplitCNN:
    """Backbone parameters plus a trained head, with the edge/remote split exposed."""

    k: int
    backbone: dict
    head: object                       # LinearHead | MLPHead | QuantumHead, with .params set
    meta: dict = field(default_factory=dict)

    def shape(self, cut: str) -> tuple[int, ...]:
        return (self.k,) if cut == "embedding" else SHAPES[cut]

    def forward_to(self, cut: str, X) -> np.ndarray:
        """Edge part: input -> activation at ``cut``, as the float32 tensor that is transmitted."""
        return np.asarray(backbone(self.backbone, np.asarray(X, dtype=np.float64), "input", cut), dtype=np.float32)

    def forward_from(self, cut: str, H) -> np.ndarray:
        """Remote part: activation at ``cut`` (any shape with the right size per sample) -> logits."""
        H = np.asarray(H, dtype=np.float64)
        H = H.reshape((len(H),) + self.shape(cut))
        z = backbone(self.backbone, H, cut, "embedding")
        return np.asarray(self.head.logits(self.head.params, z))

    def embed(self, X) -> np.ndarray:
        return self.forward_to("embedding", X)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.forward_from("input", X), axis=1)

    def accuracy(self, X, y) -> float:
        return float(np.mean(self.predict(X) == y))

    def cut_stats(self) -> list[dict]:
        """Per cut: values and float32 bytes per sample, FLOPs before (front) and after
        (back, including this model's head) the cut. A quantum head's FLOPs are its
        classical readout layer only; the circuit is not counted in FLOPs."""
        fl = stage_flops(self.k)
        head = int(self.head.flop)
        rows, front = [], 0
        total = sum(fl.values())
        for cut in CUTS:
            front += fl.get(cut, 0)
            values = int(np.prod(self.shape(cut)))
            rows.append({"name": cut, "values": values, "bytes": values * FLOAT_BYTES, "front_flop": float(front),
                         "back_flop": float(total - front + head), "head_flop": float(head)})
        return rows

    def save(self, path: str) -> None:
        arrays = {f"backbone__{n}": a for n, a in self.backbone.items()}
        arrays.update({f"head__{n}": np.asarray(a) for n, a in self.head.params.items()})
        info = {"k": self.k, "head": self.head.name, "head_config": self.head.config(), "meta": self.meta}
        np.savez(path, __info__=np.array(json.dumps(info, default=float)), **arrays)

    @classmethod
    def load(cls, path: str) -> "SplitCNN":
        with np.load(path, allow_pickle=False) as f:
            info = json.loads(str(f["__info__"]))
            bb = {n.split("__", 1)[1]: f[n] for n in f.files if n.startswith("backbone__")}
            hp = {n.split("__", 1)[1]: f[n] for n in f.files if n.startswith("head__")}
        cfg = info["head_config"]
        name = info["head"]
        head = make_head(name, info["k"], layers=cfg.get("layers", 2), hidden=cfg.get("hidden"))
        head.params = hp
        return cls(info["k"], bb, head, info.get("meta", {}))


def _fit_kw(kw: dict | None) -> dict:
    return {k: v for k, v in (kw or {}).items() if k in ("epochs", "batch", "lr", "patience")}


def train_end_to_end(data, k: int, head="linear", *, seed: int = 0, layers: int = 2, **fit_kw) -> SplitCNN:
    """Backbone + head trained jointly from scratch (for a quantum head: autograd
    backprop through the PennyLane QNode into the backbone). With the linear head
    this is also step 1 of transfer learning."""
    Xtr, ytr, Xval, yval = data[:4]
    h = make_head(head, k, layers=layers) if isinstance(head, str) else head
    rng = np.random.default_rng(seed)
    params = {"backbone": init_backbone(k, rng), "head": h.init(rng)}

    def logits_fn(p, x):
        return h.logits(p["head"], backbone(p["backbone"], x))

    res = fit(params, logits_fn, Xtr, ytr, Xval, yval, seed=seed + 1, **_fit_kw(fit_kw))
    h.params = res.params["head"]
    meta = {"mode": "end-to-end", "seed": seed, "best_epoch": res.best_epoch, "val_acc": res.best_val_acc,
            "seconds": res.seconds, "steps": res.steps, "epochs_run": len(res.history)}
    return SplitCNN(k, res.params["backbone"], h, meta)


def train_head(head, Ztr, ytr, Zval, yval, *, seed: int = 0, **fit_kw) -> FitResult:
    """Train ``head`` on frozen embeddings; sets ``head.params`` and returns the fit."""
    rng = np.random.default_rng(seed)
    res = fit(head.init(rng), head.logits, np.asarray(Ztr, np.float64), ytr, np.asarray(Zval, np.float64), yval,
              seed=seed + 1, **_fit_kw(fit_kw))
    head.params = res.params
    return res


@dataclass
class TransferResult:
    model: SplitCNN                    # backbone + its own linear head (step 1)
    heads: dict                        # name -> trained head on the frozen embedding
    fits: dict                         # name -> FitResult
    backbone_fit: dict

    def accuracy(self, name: str, X, y) -> float:
        """Accuracy of head ``name`` on images ``X`` through the frozen backbone."""
        z = self.model.embed(X).astype(np.float64)
        h = self.heads[name]
        return float(np.mean(np.argmax(np.asarray(h.logits(h.params, z)), axis=1) == y))

    def with_head(self, name: str) -> SplitCNN:
        return SplitCNN(self.model.k, self.model.backbone, self.heads[name], {**self.model.meta, "head": name})


def train_transfer(data, k: int, heads=HEADS, *, seed: int = 0, layers: int = 2,
                   backbone_kw: dict | None = None, head_kw: dict | None = None) -> TransferResult:
    """Transfer mode: backbone + linear head end to end, then every head on the frozen embedding."""
    Xtr, ytr, Xval, yval = data[:4]
    model = train_end_to_end(data, k, "linear", seed=seed, **(backbone_kw or {}))
    Ztr, Zval = model.embed(Xtr), model.embed(Xval)
    trained, fits = {}, {}
    for name in heads:
        h = make_head(name, k, layers=layers)
        fits[name] = train_head(h, Ztr, ytr, Zval, yval, seed=seed, **(head_kw or {}))
        trained[name] = h
    return TransferResult(model, trained, fits, dict(model.meta))
