"""The quantum head: a k-qubit variational circuit on the k-dim embedding.

This is the "dressed quantum circuit" of Mari et al. (2020): a small
parametrised circuit sandwiched between the classical backbone and a
trainable linear readout. It is the realistic shape of quantum machine
learning today: a small layer inside a classical model. We expect it to
match, not beat, a classical head of similar size, and report what the
numbers say.

Circuit on k qubits (k = 2..8)::

    encode   RY(pi/2 * (z_i + 1)) on qubit i       z in [-1, 1] -> angle in [0, pi]
    L times  RZ RY RZ (trainable) on every qubit, then a CNOT ring
             (i -> i+1, and k-1 -> 0 when k > 2)
    readout  <Z_i> for every qubit, then a trainable linear layer k -> 10 + bias

Every gate maps 1:1 onto the repo gate set (ry, rz, cx), so ``to_circuits``
exports the head for any backend: the four simulator stacks, ``ibm-auto``
(best IBM device, emulated with its real calibration without a token) or real
IBM / Braket hardware with credentials.

Notes on the parameter count (stated, not hidden): the last RZ of the last
layer is diagonal and only basis permutations (CNOTs) follow it, so it cannot
change any <Z_i>; k of the 3kL circuit angles are therefore redundant. They
are kept so every layer has the same form, and ``quantum_params`` counts them
(the parameter-matched MLP is matched to the total, which favours the MLP
slightly). The ring wrap needs SWAPs on heavy-hex chips (no short cycles);
the transpiled depth on a device is reported by the packed path.

``layers=0`` is an ablation: fixed encoding + linear readout only, i.e. the
linear head applied to <Z_i> = cos(pi/2 (z_i + 1)) = -sin(pi z_i / 2). The
gap between L=0 and L>0 is what the trainable circuit itself adds.

Training uses PennyLane ``default.qubit`` with the autograd interface,
backprop and parameter broadcasting over the minibatch, exact expectations
(no shots). Inference has four paths: ``predict_exact``; ``predict_shots``
(samples from the exact probabilities, fast); ``predict_on_backend`` (one
circuit per sample through any repo backend); ``predict_packed`` (several
samples per QPU job with ``qedge.quantum.packing``, i.e. idea 2's packing
applied to this head).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import autograd.numpy as anp
import numpy as np

from ..quantum.circuit import Circuit

CLASSES = 10


def ring(k: int) -> list[tuple[int, int]]:
    """CNOT pairs of one entangling layer: a chain, closed into a ring when k > 2."""
    return [(i, i + 1) for i in range(k - 1)] + ([(k - 1, 0)] if k > 2 else [])


def quantum_params(k: int, layers: int, classes: int = CLASSES) -> int:
    """Trainable parameters: 3 angles per qubit per layer, plus the k x classes readout and its bias."""
    return 3 * k * layers + k * classes + classes


def encode_angles(Z) -> np.ndarray:
    return np.pi / 2 * (np.asarray(Z, dtype=np.float64) + 1.0)


@lru_cache(maxsize=None)
def _qnodes(k: int, layers: int):
    """(expectations, probabilities) QNodes for one (k, L); built once, reused."""
    import pennylane as qml

    dev = qml.device("default.qubit", wires=k)
    pairs = ring(k)

    def body(z, w):
        for i in range(k):
            qml.RY(anp.pi / 2 * (z[:, i] + 1.0), wires=i)
        for layer in range(layers):
            for i in range(k):
                qml.RZ(w[layer, i, 0], wires=i)
                qml.RY(w[layer, i, 1], wires=i)
                qml.RZ(w[layer, i, 2], wires=i)
            for a, b in pairs:
                qml.CNOT(wires=[a, b])

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def expz(z, w):
        body(z, w)
        return [qml.expval(qml.PauliZ(i)) for i in range(k)]

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def probs(z, w):
        body(z, w)
        return qml.probs(wires=list(range(k)))

    return expz, probs


def to_circuits(Z, weights) -> list[Circuit]:
    """One repo ``Circuit`` per sample (gates ry, rz, cx), identical to the QNode.
    Measured on every qubit by the backends; qubit 0 is the leftmost bit."""
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    w = np.asarray(weights, dtype=np.float64)
    k, layers = Z.shape[1], w.shape[0]
    pairs = ring(k)
    out = []
    for s, ang in enumerate(encode_angles(Z)):
        c = Circuit(k, name=f"qhead_k{k}_L{layers}_{s}")
        for i in range(k):
            c.ry(i, ang[i])
        for layer in range(layers):
            for i in range(k):
                c.rz(i, w[layer, i, 0]).ry(i, w[layer, i, 1]).rz(i, w[layer, i, 2])
            for a, b in pairs:
                c.cx(a, b)
        out.append(c)
    return out


def expz_from_counts(counts: dict[str, int], k: int) -> np.ndarray:
    """<Z_i> = mean(1 - 2 bit_i) from counts keyed qubit 0 first."""
    total = sum(counts.values())
    e = np.zeros(k)
    for key, v in counts.items():
        e += v * (1.0 - 2.0 * np.array([int(b) for b in key[:k]]))
    return e / total


@dataclass
class HeadRun:
    """Predictions of one inference path plus what it cost."""

    predictions: np.ndarray
    expvals: np.ndarray                # (N, k) estimated <Z_i>
    path: str                          # "backend" | "packed"
    shots: int
    backend: str
    n_circuits: int
    n_jobs: int
    wall_s: float
    fallback_from: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class QuantumHead:
    """k-qubit dressed quantum circuit head; same training interface as the classical heads."""

    k: int
    layers: int = 2
    classes: int = CLASSES
    name: str = "quantum"
    params: dict | None = None          # {"w": (L, k, 3) angles, "a": (k, classes), "b": (classes,)}

    def init(self, rng: np.random.Generator) -> dict:
        """Angles uniform in [-pi, pi]; readout fan-in scaled, like the linear head."""
        return {"w": rng.uniform(-np.pi, np.pi, (self.layers, self.k, 3)),
                "a": rng.normal(0.0, np.sqrt(1 / self.k), (self.k, self.classes)), "b": np.zeros(self.classes)}

    # ---- the model -----------------------------------------------------------
    def expvals(self, w, Z):
        """Exact <Z_i>, shape (N, k), differentiable with autograd."""
        return anp.stack(_qnodes(self.k, self.layers)[0](Z, w), axis=1)

    def logits(self, p: dict, Z):
        return anp.dot(self.expvals(p["w"], Z), p["a"]) + p["b"]

    def readout(self, E) -> np.ndarray:
        """Class predictions from (estimated) expectations."""
        p = self._trained()
        return np.argmax(np.asarray(E) @ p["a"] + p["b"], axis=1)

    @property
    def n_params(self) -> int:
        return quantum_params(self.k, self.layers, self.classes)

    @property
    def n_circuit_params(self) -> int:
        return 3 * self.k * self.layers

    @property
    def flop(self) -> int:
        return 2 * self.k * self.classes          # classical readout only

    def config(self) -> dict:
        return {"layers": self.layers}

    def _trained(self) -> dict:
        if self.params is None:
            raise RuntimeError("quantum head has no parameters yet: train it or set .params")
        return self.params

    # ---- export --------------------------------------------------------------
    def to_circuits(self, Z) -> list[Circuit]:
        return to_circuits(Z, self._trained()["w"])

    def circuit_profile(self) -> dict:
        """Size of the logical circuit (before any device transpilation)."""
        c = to_circuits(np.zeros((1, self.k)), np.zeros((self.layers, self.k, 3)))[0]
        return {"n_qubits": self.k, "layers": self.layers, "depth": c.depth(), "n_1q": c.n_1q, "n_2q": c.n_2q}

    # ---- inference paths -----------------------------------------------------
    def expvals_exact(self, Z) -> np.ndarray:
        return np.asarray(self.expvals(self._trained()["w"], np.atleast_2d(np.asarray(Z, np.float64))))

    def probabilities(self, Z) -> np.ndarray:
        """Exact outcome probabilities (N, 2^k); PennyLane orders wire 0 as the most significant bit."""
        Z = np.atleast_2d(np.asarray(Z, np.float64))
        return np.asarray(_qnodes(self.k, self.layers)[1](Z, self._trained()["w"])).reshape(len(Z), -1)

    def expvals_shots(self, Z, shots: int, seed: int = 0) -> np.ndarray:
        """<Z_i> estimated from ``shots`` samples of the exact distribution of each input."""
        P = self.probabilities(Z)
        P = np.clip(P, 0.0, None)
        P /= P.sum(axis=1, keepdims=True)
        rng = np.random.default_rng(seed)
        counts = np.stack([rng.multinomial(shots, p) for p in P])               # (N, 2^k)
        idx = np.arange(2 ** self.k)
        bits = (idx[:, None] >> (self.k - 1 - np.arange(self.k))[None, :]) & 1  # wire 0 = leftmost bit
        return counts @ (1.0 - 2.0 * bits) / shots

    def predict_exact(self, Z) -> np.ndarray:
        return self.readout(self.expvals_exact(Z))

    def predict_shots(self, Z, shots: int, seed: int = 0) -> np.ndarray:
        return self.readout(self.expvals_shots(Z, shots, seed))

    def predict_on_backend(self, backend, Z, shots: int = 1000) -> HeadRun:
        """One circuit per sample in a single ``backend.run`` call. On real hardware the
        executor's limits (QEDGE_HW_MAX_CIRCUITS / QEDGE_HW_MAX_SHOTS) apply, as in
        ``Executor._guard``."""
        from ..executor import HW_MAX_CIRCUITS, HW_MAX_SHOTS

        circuits = self.to_circuits(Z)
        if backend.hardware and (len(circuits) > HW_MAX_CIRCUITS or shots > HW_MAX_SHOTS):
            raise RuntimeError(f"refusing {len(circuits)} circuits x {shots} shots on real hardware "
                               f"(limits QEDGE_HW_MAX_CIRCUITS={HW_MAX_CIRCUITS}, QEDGE_HW_MAX_SHOTS={HW_MAX_SHOTS})")
        t0 = time.perf_counter()
        res = backend.run(circuits, shots)
        wall = time.perf_counter() - t0
        E = np.stack([expz_from_counts(c, self.k) for c in res.counts])
        return HeadRun(self.readout(E), E, "backend", shots, res.backend, len(circuits), 1, wall,
                       getattr(backend, "fallback_from", None) or res.fallback_from,
                       {"hardware": res.hardware, "job_ids": res.job_ids, **res.meta})

    def predict_packed(self, cal_or_backend, Z, shots: int = 1000, per_job: int = 8, buffer: int = 1,
                       crosstalk_factor: float = 1.0, seed: int = 17) -> HeadRun:
        """Several samples' circuits side by side in one QPU job (``qedge.quantum.packing``).

        Samples are taken ``per_job`` at a time; for each group the device is
        chosen (``select_packing_device``), the circuits are packed with
        ``buffer`` free hops between them, run, and split back per sample. A
        sample that does not fit on the chip goes first into the next job. A
        calibration marked live is submitted through SamplerV2 (exercised in
        tests only in qiskit-ibm-runtime's local mode, never on a real account);
        anything else runs on the emulator of the chosen device, where the
        labelled ``crosstalk_factor`` applies. On live devices the executor's
        limits apply before anything is submitted: at most QEDGE_HW_MAX_SHOTS
        shots and QEDGE_HW_MAX_CIRCUITS packed jobs (one circuit each). QPU
        seconds and price come from ``packing_cost`` (catalogue figures and
        Braket-style pricing, both assumptions)."""
        from ..executor import HW_MAX_CIRCUITS, HW_MAX_SHOTS
        from ..quantum import packing as P

        devices, live = P._devices_from(cal_or_backend)
        circuits = self.to_circuits(Z)
        n = len(circuits)
        if live and (shots > HW_MAX_SHOTS or -(-n // per_job) > HW_MAX_CIRCUITS):
            raise RuntimeError(f"refusing {-(-n // per_job)} packed jobs x {shots} shots on real hardware "
                               f"(limits QEDGE_HW_MAX_CIRCUITS={HW_MAX_CIRCUITS}, QEDGE_HW_MAX_SHOTS={HW_MAX_SHOTS})")
        E = np.zeros((n, self.k))
        regions: list[list[int] | None] = [None] * n
        jobs: list[dict] = []
        pending = list(range(n))
        t0 = time.perf_counter()
        while pending:
            idx = pending[:per_job]
            cal, alloc = P.select_packing_device(devices, [circuits[i] for i in idx], buffer)
            job = P.pack(cal, [circuits[i] for i in idx], buffer, allow_partial=True, allocation=alloc)
            job_id = None
            if live and cal.source == "live":  # tested in local mode only (fake backend), never on a real account
                from qiskit_ibm_runtime import SamplerV2
                rjob = SamplerV2(mode=cal.backend).run([job.circuit], shots=shots)
                pub = rjob.result()[0]
                counts = P.split_counts(job, {r: getattr(pub.data, r).get_counts() for r in job.registers})
                job_id = rjob.job_id()
            else:
                counts = P.emulate(job, shots, crosstalk_factor, seed + len(jobs))
            cost = P.packing_cost([p.depth for p in job.placements], shots)
            for p, cnt in zip(job.placements, counts):
                E[idx[p.index]] = expz_from_counts(cnt, self.k)
                regions[idx[p.index]] = list(p.qubits)
            placed = {idx[p.index] for p in job.placements}
            pending = [i for i in idx if i not in placed] + pending[per_job:]
            jobs.append({"device": cal.name, "calibration": cal.source, "emulated": job_id is None,
                         "job_id": job_id, "samples": sorted(placed), "k": job.k,
                         "unplaced": len(idx) - len(placed), "depth": job.depth,
                         "esp": [p.esp for p in job.placements], "qpu_s": cost.qpu_s_job, "price": cost.price_job,
                         "qpu_s_per_sample": cost.qpu_s_per_program, "price_per_sample": cost.price_per_program,
                         "price_per_sample_solo": cost.price_per_program_solo,
                         "qpu_s_per_sample_solo": cost.qpu_s_per_program_solo})
        wall = time.perf_counter() - t0
        devices_used = sorted({j["device"] for j in jobs})
        meta = {"devices": devices_used, "jobs": jobs, "layout": regions, "buffer": buffer,
                "crosstalk_factor": crosstalk_factor, "per_job": per_job,
                "qpu_s": sum(j["qpu_s"] for j in jobs), "price": sum(j["price"] for j in jobs),
                "qpu_s_solo": sum(j["qpu_s_per_sample_solo"] * j["k"] for j in jobs),
                "price_solo": sum(j["price_per_sample_solo"] * j["k"] for j in jobs)}
        return HeadRun(self.readout(E), E, "packed", shots, ",".join(devices_used), n, len(jobs), wall,
                       None, meta)
