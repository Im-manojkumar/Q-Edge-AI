"""AI workloads, their profiles, and a synthetic workload generator.

Five workload kinds cover the spectrum the project asks about:

  inference            classical only (CNN-style batch inference)
  training             classical only (on-device fine-tuning)
  optimization         MaxCut-style combinatorial problem: classical solver, or hybrid QAOA
  kernel_classification small-data classifier: RBF-SVM, or hybrid quantum-kernel SVM
  quantum_simulation   spin-chain dynamics: classical state vector / MPS, or run natively on a QPU

Only the last three have a quantum formulation; the orchestrator never sends
the first two to a QPU.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

KINDS = ["inference", "training", "optimization", "kernel_classification", "quantum_simulation"]
QUANTUM_CAPABLE = {"optimization", "kernel_classification", "quantum_simulation"}


@dataclass
class QuantumProfile:
    n_qubits: int
    depth: int
    n_1q: int
    n_2q: int
    n_circuits: int
    shots: int
    n_rounds: int          # sequential classical<->quantum round trips
    algorithm: str

    def fidelity(self, p1: float, p2: float, readout: float = 0.0) -> float:
        """Probability that no error hits a circuit (standard NISQ estimate)."""
        log_f = (self.n_1q * math.log1p(-p1) + 2 * self.n_2q * math.log1p(-p2)
                 + self.n_qubits * math.log1p(-readout))
        return math.exp(log_f)


@dataclass
class Workload:
    id: int
    kind: str
    arrival_s: float = 0.0
    deadline_s: float = 1.0
    data_mb: float = 0.1          # input that must reach the executor
    result_mb: float = 0.01
    privacy: bool = False         # raw data must not leave the edge
    min_quality: float = 0.0
    gflop: float = 1.0            # classical work
    memory_gb: float = 0.1
    parallel: bool = True         # benefits from GPU-style parallelism
    classical_method: str = ""
    classical_quality: float = 1.0
    quantum: QuantumProfile | None = None
    params: dict = field(default_factory=dict)

    @property
    def quantum_capable(self) -> bool:
        return self.quantum is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quantum"] = asdict(self.quantum) if self.quantum else None
        return d


# ---------------------------------------------------------------------------
# Quality calibration. Defaults below were produced by `qedge calibrate`
# (experiments/e2_quantum_vs_classical.py) and are overwritten by
# results/calibration.json when that file exists.
# ---------------------------------------------------------------------------
CALIBRATION = {
    "qaoa_ratio": {"slope": -0.012, "at": 8, "value": 0.97, "floor": 0.70},
    "random_ratio": {"slope": -0.015, "at": 8, "value": 0.93, "floor": 0.60},
    "sa_ratio": {"slope": -0.0008, "at": 8, "value": 1.0, "floor": 0.95},
    "kernel_acc": {"quantum": 0.86, "rbf": 0.90},
    "mps_chi_max": 1024,
    "entanglement_bits_per_step": 0.5,
}


def load_calibration(path: str = "results/calibration.json") -> dict:
    import json
    import os

    if os.path.exists(path):
        with open(path) as f:
            CALIBRATION.update(json.load(f))
    return CALIBRATION


def _lin(c: dict, n: int) -> float:
    return float(np.clip(c["value"] + c["slope"] * (n - c["at"]), c["floor"], 1.0))


def qaoa_quality(n: int, fidelity: float = 1.0) -> float:
    """Best-sampled approximation ratio, degraded toward random sampling by noise."""
    q, r = _lin(CALIBRATION["qaoa_ratio"], n), _lin(CALIBRATION["random_ratio"], n)
    return fidelity * q + (1 - fidelity) * r


def sa_quality(n: int) -> float:
    return _lin(CALIBRATION["sa_ratio"], n)


def kernel_quality(fidelity: float = 1.0, n_classes: int = 2) -> float:
    acc = CALIBRATION["kernel_acc"]["quantum"]
    return fidelity * acc + (1 - fidelity) * (1 / n_classes)


def tfim_mps(n: int, steps: int) -> tuple[float, float]:
    """(gflop, quality) for a matrix-product-state simulation with capped bond dimension."""
    chi_need = 2 ** min(CALIBRATION["entanglement_bits_per_step"] * steps, n / 2)
    chi = min(chi_need, CALIBRATION["mps_chi_max"])
    gflop = n * steps * 8 * chi ** 3 * 1e-9 * 4
    quality = 1.0 if chi_need <= chi else float(chi / chi_need) ** 0.5
    return gflop, quality


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_inference(i, rng, **kw) -> Workload:
    batch = int(kw.get("batch", rng.integers(1, 17)))
    model_gflop = float(kw.get("model_gflop", rng.choice([0.6, 1.8, 4.1])))  # MobileNet / EfficientNet / ResNet50
    return Workload(i, "inference", deadline_s=float(kw.get("deadline_s", rng.uniform(0.1, 2.0))),
                    data_mb=batch * 0.15, result_mb=0.001 * batch, privacy=bool(kw.get("privacy", rng.random() < 0.15)),
                    gflop=batch * model_gflop, memory_gb=0.5, parallel=True, classical_method="cnn",
                    params={"batch": batch, "model_gflop": model_gflop})


def make_training(i, rng, **kw) -> Workload:
    samples = int(kw.get("samples", rng.integers(500, 20000)))
    epochs = int(kw.get("epochs", rng.integers(1, 5)))
    model_gflop = float(kw.get("model_gflop", 0.6))
    return Workload(i, "training", deadline_s=float(kw.get("deadline_s", rng.uniform(60, 900))),
                    data_mb=samples * 0.05, result_mb=20, privacy=bool(kw.get("privacy", rng.random() < 0.25)),
                    gflop=3 * epochs * samples * model_gflop, memory_gb=4, parallel=True, classical_method="sgd",
                    params={"samples": samples, "epochs": epochs})


def make_optimization(i, rng, **kw) -> Workload:
    n = int(kw.get("n", rng.integers(6, 64)))
    density = float(kw.get("density", 0.3))
    n_edges = max(1, int(density * n * (n - 1) / 2))
    p = int(kw.get("p", 2))
    if n <= 22:
        method, gflop, quality = "exact", (2 ** n) * n_edges * 4e-9, 1.0
    else:
        method, gflop, quality = "simulated_annealing", 400 * n_edges * 2 * 50e-9, sa_quality(n)
    avg_deg = 2 * n_edges / n
    routing = 1 + 0.25 * avg_deg  # SWAPs added mapping a dense graph onto sparse hardware coupling
    iters = 25
    qp = QuantumProfile(n_qubits=n, depth=int(p * (avg_deg + 2) * routing) + 1, n_1q=n + n * p,
                        n_2q=int(n_edges * p * routing), n_circuits=12 + iters, shots=1000,
                        n_rounds=1 + iters, algorithm=f"QAOA(p={p})")
    return Workload(i, "optimization", deadline_s=float(kw.get("deadline_s", rng.uniform(2, 600))),
                    data_mb=n_edges * 1e-5, result_mb=1e-4, privacy=False, gflop=gflop, memory_gb=0.01 + 2 ** min(n, 22) * 8e-9,
                    parallel=False, classical_method=method, classical_quality=quality,
                    min_quality=float(kw.get("min_quality", 0.8)), quantum=qp,
                    params={"n": n, "edges": n_edges, "p": p, "density": density})


def make_kernel(i, rng, **kw) -> Workload:
    d = int(kw.get("features", rng.integers(2, 10)))
    n_train = int(kw.get("n_train", rng.integers(20, 120)))
    n_test = max(5, n_train // 3)
    reps = 2
    n_circ = n_train * (n_train - 1) // 2 + n_test * n_train
    qp = QuantumProfile(n_qubits=d, depth=2 * reps * (d + 2), n_1q=4 * reps * d, n_2q=2 * reps * (d - 1),
                        n_circuits=n_circ, shots=500, n_rounds=1, algorithm="ZZ quantum kernel")
    return Workload(i, "kernel_classification", deadline_s=float(kw.get("deadline_s", rng.uniform(1, 900))),
                    data_mb=(n_train + n_test) * d * 8e-6, result_mb=1e-4, privacy=bool(kw.get("privacy", rng.random() < 0.2)),
                    gflop=(n_train ** 2) * d * 1e-6, memory_gb=0.05, parallel=False, classical_method="rbf_svm",
                    classical_quality=CALIBRATION["kernel_acc"]["rbf"], min_quality=float(kw.get("min_quality", 0.7)),
                    quantum=qp, params={"features": d, "n_train": n_train, "n_test": n_test})


def make_qsim(i, rng, **kw) -> Workload:
    n = int(kw.get("n", rng.integers(8, 80)))
    steps = int(kw.get("steps", rng.integers(2, 30)))
    points = 5
    sv_mem = 16 * 2 ** n / 1e9
    gates = (2 * n - 1) * steps
    if sv_mem <= 64:
        method, gflop, quality, mem = "statevector", points * gates * 2 ** n * 8e-9, 1.0, sv_mem
    else:
        g, quality = tfim_mps(n, steps)
        method, gflop, mem = "mps", points * g, 2.0
    qp = QuantumProfile(n_qubits=n, depth=3 * steps, n_1q=n * steps, n_2q=(n - 1) * steps,
                        n_circuits=points, shots=4000, n_rounds=1, algorithm="Trotterised TFIM")
    return Workload(i, "quantum_simulation", deadline_s=float(kw.get("deadline_s", rng.uniform(10, 1800))),
                    data_mb=1e-3, result_mb=1e-3, privacy=False, gflop=gflop, memory_gb=mem, parallel=True,
                    classical_method=method, classical_quality=quality,
                    min_quality=float(kw.get("min_quality", 0.5)), quantum=qp, params={"n": n, "steps": steps})


BUILDERS = {"inference": make_inference, "training": make_training, "optimization": make_optimization,
            "kernel_classification": make_kernel, "quantum_simulation": make_qsim}

DEFAULT_MIX = {"inference": 0.45, "training": 0.10, "optimization": 0.2,
               "kernel_classification": 0.15, "quantum_simulation": 0.10}


def make_workload(kind: str, i: int = 0, seed: int | None = None, **kw) -> Workload:
    return BUILDERS[kind](i, np.random.default_rng(seed), **kw)


def generate(n: int, rate_per_s: float = 0.5, mix: dict | None = None, seed: int = 0) -> list[Workload]:
    """Poisson arrival stream with a configurable workload mix."""
    rng = np.random.default_rng(seed)
    mix = mix or DEFAULT_MIX
    kinds, probs = list(mix), np.array(list(mix.values()), float)
    probs /= probs.sum()
    t = 0.0
    out = []
    for i in range(n):
        t += rng.exponential(1 / rate_per_s)
        w = BUILDERS[rng.choice(kinds, p=probs)](i, rng)
        w.arrival_s = t
        out.append(w)
    return out
