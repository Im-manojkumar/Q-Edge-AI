"""Idea 3: a real split hybrid model (edge CNN backbone + classical or quantum head).

* ``data``     digits splits (train / validation / test)
* ``nets``     autograd CNN backbone with named cut points, classical heads, training
* ``qhead``    the k-qubit quantum head, its export to repo Circuits and its inference paths
* ``leakage``  what an attacker can reconstruct from the tensor sent at each cut
* ``profile``  the measured split profile the orchestrator's router reads

PennyLane is imported only when a quantum head is built or run.

Assumptions and choices (in the style of docs/assumptions.md):

| Figure / rule | Value | Basis |
|---|---|---|
| Data | scikit-learn digits, 8x8, pixels / 16, stratified 60/20/20, seed 0 | small enough to train everything on a CPU; a toy next to a camera stream |
| Transmitted tensor | float32, 4 bytes per value | no compression or quantisation is modelled |
| FLOPs | 2 per multiply-add of conv and dense layers; bias, activations, pooling not counted | analytic, per sample |
| Quantum head | RY(pi/2 (z+1)) encoding, L x (RZ RY RZ + CNOT ring), <Z_i>, linear k -> 10 | dressed quantum circuit (Mari et al. 2020); L = 2 unless stated |
| MLP width | hidden units whose parameter count is closest to the quantum head's total | parameter-matched comparison; the quantum total includes k redundant angles |
| Training | Adam, softmax cross-entropy, early stopping on validation accuracy | validation only for selection; the test split is touched once per number |
| Attacker | honest-but-curious remote party with same-distribution labelled images (the training split) | ridge and one-hidden-layer MLP decoders; strongest per metric; "against this attack" only |
| Privacy rule | reid beyond the label-only baseline <= 5 x chance, r2 beyond the label <= 0.10 | policy thresholds, not a standard; the absolute reid rule is reported too |
| Packed QPU cost | ``packing_cost``: catalogue QPU figures, Braket-style per-task / per-shot prices | same assumptions as E7 |
"""
