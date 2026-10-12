"""Trained split models, trained once and then loaded from disk.

The executor runs ``split_inference`` workloads through a real model (the
digits CNN of ``nets`` with every head of ``nets.HEADS``), and training one
takes half a minute to a few minutes (measured below; mostly the quantum
head), so it must not happen per request.
``get_model(k)`` trains a transfer-mode model on first use with the default
settings (``nets.BACKBONE_FIT`` / ``nets.HEAD_FIT``, the settings behind the
reported accuracies), writes it to ``results/models/`` and loads it from there
afterwards. The quantum head's circuit angles and readout are part of the file,
so a reloaded model gives the same predictions as the one that was trained.

* One file per (k, seed, quick): ``digits_k4_seed0.npz``. The file records the
  training settings; if they no longer match the code's, the model is retrained
  and the file replaced, so a stale cache cannot pass for the current model.
* The data split is always ``load_digits_split(0)`` (the split the profile is
  measured on); ``seed`` only fixes the initialisation and minibatch order.
* ``quick=True`` trains with few epochs (``QUICK_BACKBONE`` / ``QUICK_HEAD``,
  the settings tests/test_ml.py uses) into a separate file
  (``digits_k4_seed0_quick.npz``). For tests and demos only: its accuracies are
  not the reported ones.
* Thread-safe: concurrent callers for the same model wait for one training run
  (race mode runs the quantum and classical paths in two threads). Files are
  written to a temporary name and renamed, so a reader never sees half a file.
  Processes do not coordinate: two processes starting cold may both train.

Training time, measured on this machine (4 CPU cores, Python 3.13, seed 0, first
call with nothing cached, ``get_model`` wall clock): k=4 27.6 s (the quantum head
15.7 s of it), k=8 146 s (quantum head 133 s), k=4 quick 6.1 s. Loading a cached
model took 0.002-0.003 s. The reloaded k=4 and k=8 models give the same test
accuracies as when they were trained (k=4: linear 0.956, mlp 0.950, quantum
0.944, quantum_l0 0.958; k=8: 0.969, 0.961, 0.922, 0.964 on the 360 test
images) and the same quantum-head predictions.
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

from . import nets
from .data import load_digits_split

DATA_SEED = 0
CACHE_DIR = "results/models"
QUICK_BACKBONE = {"epochs": 8, "lr": 0.01, "batch": 32, "patience": 8}
QUICK_HEAD = {"epochs": 10, "lr": 0.1, "batch": 128, "patience": 10}
LAYERS = 2

_lock = threading.Lock()
_key_locks: dict[tuple, threading.Lock] = {}
_models: dict[tuple, nets.TransferResult] = {}


def settings(quick: bool = False) -> dict:
    """Everything that determines the trained weights besides k and the seed."""
    return {"data_seed": DATA_SEED, "layers": LAYERS, "heads": list(nets.HEADS),
            "backbone_kw": dict(QUICK_BACKBONE if quick else nets.BACKBONE_FIT),
            "head_kw": dict(QUICK_HEAD if quick else nets.HEAD_FIT)}


def model_path(k: int, seed: int = 0, cache_dir: str = CACHE_DIR, quick: bool = False) -> str:
    return os.path.join(cache_dir, f"digits_k{int(k)}_seed{int(seed)}{'_quick' if quick else ''}.npz")


_data_cache: list = []


def data():
    """The digits split every zoo model is trained and evaluated on (loaded once)."""
    with _lock:
        if not _data_cache:
            _data_cache.append(load_digits_split(DATA_SEED))
        return _data_cache[0]


def get_model(k: int = 4, seed: int = 0, cache_dir: str = CACHE_DIR, quick: bool = False) -> nets.TransferResult:
    """The transfer-mode model for embedding size ``k`` with every head trained.

    Returned from memory, else loaded from ``cache_dir``, else trained and saved.
    ``result.meta`` (on ``result.model.meta``) says which: ``"zoo": "trained" |
    "loaded"`` and the seconds it took."""
    if not 2 <= int(k) <= 8:
        raise ValueError(f"k must be 2..8 (one qubit per embedding value), got {k}")
    path = model_path(k, seed, cache_dir, quick)
    key = (os.path.abspath(path), int(k), int(seed), bool(quick))
    with _lock:
        if key in _models:
            return _models[key]
        klock = _key_locks.setdefault(key, threading.Lock())
    with klock:
        if key in _models:          # another thread finished while this one waited
            return _models[key]
        want = settings(quick)
        tr = None
        t0 = time.perf_counter()
        if os.path.exists(path):
            try:
                tr, info = load(path)
                if info.get("settings") != want or info.get("k") != int(k) or info.get("seed") != int(seed):
                    tr = None       # trained with other settings: retrain
            except (OSError, ValueError, KeyError) as e:   # unreadable cache: retrain over it
                import logging
                logging.getLogger(__name__).warning("model cache %s unreadable (%s); retraining", path, e)
                tr = None
            if tr is not None:
                tr.model.meta.update(zoo="loaded", zoo_seconds=time.perf_counter() - t0, zoo_path=path)
        if tr is None:
            tr = nets.train_transfer(data(), int(k), seed=int(seed), layers=LAYERS,
                                     backbone_kw=want["backbone_kw"], head_kw=want["head_kw"])
            secs = time.perf_counter() - t0
            tr.model.meta.update(zoo="trained", zoo_seconds=secs, zoo_path=path, quick=bool(quick))
            save(tr, path, {"settings": want, "k": int(k), "seed": int(seed), "quick": bool(quick),
                            "train_seconds": secs})
        with _lock:
            _models[key] = tr
        return tr


def clear_memory() -> None:
    """Forget models held in memory (the files stay)."""
    with _lock:
        _models.clear()


# ---- file format -----------------------------------------------------------------
def save(tr: nets.TransferResult, path: str, info: dict) -> str:
    """Backbone, the backbone's own linear head (step 1 of transfer) and every trained
    head, with their configs and fit summaries, in one .npz; written atomically."""
    arrays = {f"backbone__{n}": np.asarray(a) for n, a in tr.model.backbone.items()}
    arrays.update({f"step1__{n}": np.asarray(a) for n, a in tr.model.head.params.items()})
    heads = {}
    for name, h in tr.heads.items():
        arrays.update({f"head__{name}__{n}": np.asarray(a) for n, a in h.params.items()})
        f = tr.fits.get(name)
        heads[name] = {"config": h.config(),
                       "fit": None if f is None else {"best_epoch": f.best_epoch, "best_val_acc": f.best_val_acc,
                                                      "seconds": f.seconds, "steps": f.steps,
                                                      "epochs_run": len(f.history)}}
    meta = {k: v for k, v in tr.model.meta.items() if not k.startswith("zoo")}
    full = {**info, "heads": heads, "backbone_meta": meta, "backbone_fit": tr.backbone_fit}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
    np.savez(tmp, __info__=np.array(json.dumps(full, default=float)), **arrays)
    os.replace(tmp, path)
    return path


def load(path: str) -> tuple[nets.TransferResult, dict]:
    """Inverse of ``save``: (TransferResult, info). Fit histories are not stored, so each
    ``FitResult`` carries its summary (best epoch, validation accuracy, seconds) only."""
    with np.load(path, allow_pickle=False) as f:
        info = json.loads(str(f["__info__"]))
        arrays = {n: f[n] for n in f.files if n != "__info__"}
    k = int(info["k"])
    layers = int(info["settings"]["layers"])

    def group(prefix: str) -> dict:
        return {n[len(prefix):]: a for n, a in arrays.items() if n.startswith(prefix)}

    step1 = nets.LinearHead(k)
    step1.params = group("step1__")
    model = nets.SplitCNN(k, group("backbone__"), step1, dict(info.get("backbone_meta", {})))
    heads, fits = {}, {}
    for name, h in info["heads"].items():
        head = nets.make_head(name, k, layers=h["config"].get("layers", layers), hidden=h["config"].get("hidden"))
        head.params = group(f"head__{name}__")
        heads[name] = head
        if h["fit"] is not None:
            s = h["fit"]
            fits[name] = nets.FitResult(head.params, [], s["best_epoch"], s["best_val_acc"], s["seconds"], s["steps"])
    return nets.TransferResult(model, heads, fits, dict(info.get("backbone_fit", {}))), info
