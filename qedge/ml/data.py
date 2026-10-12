"""The data the split model is trained and judged on.

scikit-learn's ``load_digits`` (1797 handwritten digits, 8x8 pixels, 17 grey
levels) is small enough to train a CNN, a quantum head and every leakage
attack on a laptop CPU in minutes, and it ships with scikit-learn, so nothing
is downloaded. It is a toy next to a camera stream; the "camera" profile in
the orchestrator covers that case as an assumption.

Splits are stratified 60/20/20. The validation split is used for early
stopping and model selection only (and by the leakage attacker to pick its
ridge penalty); the test split is touched once per reported number.
"""
from __future__ import annotations

import numpy as np

SPLIT = (0.6, 0.2, 0.2)


def load_digits_split(seed: int = 0):
    """(Xtr, ytr, Xval, yval, Xte, yte): pixels scaled to [0, 1], shape (N, 1, 8, 8) float64."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).reshape(-1, 1, 8, 8).astype(np.float64)
    y = y.astype(np.int64)
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, train_size=SPLIT[0], stratify=y, random_state=seed)
    Xval, Xte, yval, yte = train_test_split(Xrest, yrest, train_size=SPLIT[1] / (SPLIT[1] + SPLIT[2]),
                                            stratify=yrest, random_state=seed)
    return Xtr, ytr, Xval, yval, Xte, yte
