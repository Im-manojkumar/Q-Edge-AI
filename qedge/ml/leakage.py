"""How much of the input can be recovered from what a split model transmits.

"Raw data never leaves the edge" is not a privacy property: an intermediate
activation can carry the whole image. This module replaces the slogan with a
measurement against a stated attack. Every number it produces is "against
this attack"; a stronger attacker could do better, so a low number is
evidence, never a guarantee.

Attacker model (honest-but-curious remote party):

* it sees the tensor transmitted at one cut point for many samples (the
  remote half of a split model receives exactly this);
* it has its own labelled images from the same distribution (we give it the
  training split) and the matching activations, which it can get by running
  the edge half on them (it knows the model);
* it fits a decoder activation -> image and applies it to activations of
  held-out inputs (the test split).

Decoders: ridge regression (closed form on standardised activations, penalty
chosen on the validation split) and scikit-learn's ``MLPRegressor`` with one
hidden layer. Both know pixels lie in [0, 1] and clip to it. Each metric is
reported for the strongest decoder on that metric ("strongest of the attacks
we tried"), which is the conservative reading for privacy.

Metrics on the N test images:

* ``r2``: fraction of pixel variance explained by the reconstruction
  (pooled over pixels, i.e. variance-weighted).
* ``r2_label``: the same for the label-only baseline, the class-mean training
  image of the true label: what anyone who sees only the model's output could
  reconstruct. ``r2_beyond_label = r2 - r2_label`` is what the transmitted
  tensor reveals beyond the label.
* ``reid``: re-identification rate. A test image is re-identified when its own
  original is the nearest (pixel L2) of all N test originals to its
  reconstruction; chance is 1/N. ``reid_top5`` is the top-5 rate.
  ``reid_label`` is the rate of the label-only baseline and
  ``reid_beyond_label = reid - reid_label``.

``private_ok`` is the router's feasibility rule. Its thresholds are policy
parameters (an assumption, not a standard): by default r2_beyond_label <= 0.10
and reid_beyond_label <= 5 x chance. The reid limit is applied beyond the label
for the same reason r2 is: whoever runs the remote half computes the output,
so it learns the label whatever the cut. The label alone re-identifies one
image per class (the one nearest its class mean), about classes / N: 10/360 =
2.8% on the digits test split, 10x chance. An absolute "reid <= 5 x chance"
rule can therefore never pass for any placement that sends work off the edge;
it is kept as ``private_ok(..., relative_to_label=False)`` and reported next to
the default verdict as ``private_ok_absolute``.
"""
from __future__ import annotations

import warnings

import numpy as np

ATTACKS = ("ridge", "mlp")
RIDGE_ALPHAS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_REID_X_CHANCE = 5.0      # policy assumption: at most 5x the chance re-identification rate
DEFAULT_MAX_R2_BEYOND_LABEL = 0.10


def _flat(A) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    return A.reshape(len(A), -1)


def r2_pixels(X, Xhat) -> float:
    """1 - SSE / SST over all pixels, SST around each pixel's mean over the evaluated images."""
    X, Xhat = _flat(X), _flat(Xhat)
    sst = float(((X - X.mean(axis=0)) ** 2).sum())
    return 1.0 - float(((X - Xhat) ** 2).sum()) / sst


def reid_rates(X, Xhat, top: int = 5) -> tuple[float, float]:
    """(top-1, top-``top``) rate at which an image's own original is among the nearest
    originals to its reconstruction. Only strictly closer originals count against it."""
    from scipy.spatial.distance import cdist
    X, Xhat = _flat(X), _flat(Xhat)
    d = cdist(Xhat, X)
    own = np.diag(d)
    rank = (d < own[:, None]).sum(axis=1)
    return float(np.mean(rank == 0)), float(np.mean(rank < top))


def label_baseline(Xtr, ytr, yte) -> np.ndarray:
    """Class-mean training image for each test label."""
    Xtr = _flat(Xtr)
    means = {c: Xtr[ytr == c].mean(axis=0) for c in np.unique(ytr)}
    return np.stack([means[c] for c in yte])


class _Standardise:
    def __init__(self, H):
        self.mu = H.mean(axis=0)
        sd = H.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)   # dead units (all-zero ReLU) stay zero

    def __call__(self, H):
        return (H - self.mu) / self.sd


def ridge_decoder(Htr, Xtr, Hval, Xval, alphas=RIDGE_ALPHAS):
    """Closed-form ridge on standardised activations; penalty with the best validation r2.
    Returns (decode function, info)."""
    Htr, Xtr, Hval, Xval = _flat(Htr), _flat(Xtr), _flat(Hval), _flat(Xval)
    sc = _Standardise(Htr)
    A, xm = sc(Htr), Xtr.mean(axis=0)
    gram, rhs = A.T @ A, A.T @ (Xtr - xm)
    best = None
    for a in alphas:
        W = np.linalg.solve(gram + a * np.eye(len(gram)), rhs)
        r2 = r2_pixels(Xval, np.clip(sc(Hval) @ W + xm, 0.0, 1.0))
        if best is None or r2 > best[0]:
            best = (r2, a, W)
    r2v, alpha, W = best
    return (lambda H: np.clip(sc(_flat(H)) @ W + xm, 0.0, 1.0)), {"alpha": alpha, "val_r2": r2v}


def mlp_decoder(Htr, Xtr, Hval, Xval, seed: int = 0, hidden: int = 256, max_iter: int = 400):
    """One-hidden-layer MLPRegressor on standardised activations. Returns (decode function, info)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
    Htr, Xtr, Hval, Xval = _flat(Htr), _flat(Xtr), _flat(Hval), _flat(Xval)
    sc = _Standardise(Htr)
    reg = MLPRegressor(hidden_layer_sizes=(hidden,), alpha=1e-4, max_iter=max_iter, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        reg.fit(sc(Htr), Xtr)
    dec = lambda H: np.clip(reg.predict(sc(_flat(H))), 0.0, 1.0)   # noqa: E731
    return dec, {"hidden": hidden, "iterations": int(reg.n_iter_), "val_r2": r2_pixels(Xval, dec(Hval))}


def attack(Htr, Hval, Hte, Xtr, Xval, Xte, ytr, yte, attacks=ATTACKS, seed: int = 0) -> dict:
    """Fit every decoder on (Htr, Xtr), evaluate on the test split; strongest per metric."""
    n = len(Xte)
    base = label_baseline(Xtr, ytr, yte)
    r2_label = r2_pixels(Xte, base)
    reid_label, _ = reid_rates(Xte, base)
    per = {}
    for name in attacks:
        if name == "ridge":
            dec, info = ridge_decoder(Htr, Xtr, Hval, Xval)
        elif name == "mlp":
            dec, info = mlp_decoder(Htr, Xtr, Hval, Xval, seed=seed)
        else:
            raise ValueError(f"unknown attack {name!r}; choose from {ATTACKS}")
        rec = dec(Hte)
        top1, top5 = reid_rates(Xte, rec)
        per[name] = {"r2": r2_pixels(Xte, rec), "reid": top1, "reid_top5": top5, **info}
    best = {m: max(per, key=lambda a: (per[a][m], a == "ridge")) for m in ("r2", "reid", "reid_top5")}
    r2 = per[best["r2"]]["r2"]
    return {"values": int(_flat(Hte).shape[1]), "n_test": n, "chance": 1.0 / n,
            "r2": r2, "r2_label": r2_label, "r2_beyond_label": r2 - r2_label,
            "reid": per[best["reid"]]["reid"], "reid_top5": per[best["reid_top5"]]["reid_top5"],
            "reid_label": reid_label, "reid_beyond_label": per[best["reid"]]["reid"] - reid_label,
            "strongest": best, "attacks": per}


def leakage_report(model, data, cuts=("input", "conv1", "conv2", "embedding"), attacks=ATTACKS,
                   seed: int = 0, max_reid: float | None = None,
                   max_r2_beyond_label: float = DEFAULT_MAX_R2_BEYOND_LABEL) -> list[dict]:
    """One row per cut of a trained ``SplitCNN``: the attack results plus the ``private_ok`` verdict."""
    Xtr, ytr, Xval, yval, Xte, yte = data
    rows = []
    for cut in cuts:
        Htr, Hval, Hte = (model.forward_to(cut, X) for X in (Xtr, Xval, Xte))
        r = {"cut": cut, "k": model.k, **attack(Htr, Hval, Hte, Xtr, Xval, Xte, ytr, yte, attacks, seed)}
        r["private_ok"] = private_ok(r, max_reid, max_r2_beyond_label)
        r["private_ok_absolute"] = private_ok(r, max_reid, max_r2_beyond_label, relative_to_label=False)
        rows.append(r)
    return rows


def private_ok(stats: dict, max_reid: float | None = None,
               max_r2_beyond_label: float = DEFAULT_MAX_R2_BEYOND_LABEL, relative_to_label: bool = True) -> bool:
    """Feasibility rule for sending this tensor off the edge (policy thresholds, an assumption):
    r2_beyond_label <= ``max_r2_beyond_label`` and re-identification <= ``max_reid``
    (default 5 x chance), measured beyond the label-only baseline unless
    ``relative_to_label=False`` (see the module docstring for why)."""
    limit = DEFAULT_REID_X_CHANCE * stats["chance"] if max_reid is None else max_reid
    reid = stats["reid"] - stats["reid_label"] if relative_to_label else stats["reid"]
    return bool(reid <= limit and stats["r2_beyond_label"] <= max_r2_beyond_label)


def privacy_reason(stats: dict, cut: str | None = None) -> str:
    """One line saying what the attack achieved at this cut, for decision logs."""
    name = cut or stats.get("cut", "this cut")
    return (f"{name} activations re-identify {stats['reid']:.0%} of inputs (label alone {stats['reid_label']:.1%}, "
            f"chance {stats['chance']:.1%}) and explain {stats['r2_beyond_label']:+.2f} of pixel variance beyond "
            f"the label, against the strongest of the attacks we tried")
