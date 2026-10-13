"""AI workloads, their profiles, and a synthetic workload generator.

Seven workload kinds cover the spectrum the project asks about:

  inference            classical only (CNN-style batch inference)
  training             classical only (on-device fine-tuning)
  optimization         MaxCut-style combinatorial problem: classical solver, or hybrid QAOA
  kernel_classification small-data classifier: RBF-SVM, or hybrid quantum-kernel SVM
  quantum_simulation   spin-chain dynamics: classical state vector / MPS, or run natively on a QPU
  ground_state         lowest energy of H2 or a spin chain: exact diagonalisation / DMRG, or hybrid VQE
  scheduling           edge/cloud task assignment (a QUBO): exact search / annealing, or hybrid QAOA

Only the last five have a quantum formulation; the orchestrator never sends
the first two to a QPU.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

KINDS = ["inference", "training", "optimization", "kernel_classification", "quantum_simulation",
         "ground_state", "scheduling"]
QUANTUM_CAPABLE = {"optimization", "kernel_classification", "quantum_simulation", "ground_state", "scheduling"}


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
    # QAOA(p=1) sample mean, as the fraction of the gap between random sampling and the optimum
    # it closes: gain(n) = a * n^-b (fitted in E2). The best of S shots is then modelled from
    # that mean with an extreme-value estimate (see best_of_shots).
    "qaoa_gain": {"a": 1.3, "b": 0.9},
    # MaxCut optimum on G(n, density): opt / |E| ~ 0.5 + opt_c / sqrt(n * density) (fitted in E2)
    "opt_c": 0.55,
    # Set only by what-if experiments: assumed noiseless QAOA best-cut ratio.
    "qaoa_override": None,
    "sa_ratio": {"slope": 0.0, "at": 8, "value": 1.0, "floor": 0.95},
    "kernel_acc": {"quantum": 0.86, "rbf": 0.90},
    "mps_chi_max": 1024,
    "entanglement_bits_per_step": 0.5,
    # VQE (RY ansatz, Rotosolve), noiseless energy quality 1 - (E - E0)/(Emax - E0) by qubit count,
    # and what DMRG reaches on 1D chains too large for exact diagonalisation (fitted in E8).
    "vqe_quality": {"slope": -0.01, "at": 2, "value": 0.995, "floor": 0.85},
    "dmrg_quality": 0.999,
    # Scheduling QUBO: best-of-shots optimum/cost of uniform random sampling by task count, and how
    # much noiseless QAOA(p=1) adds over random sampling with the same number of samples (E8 measured
    # none); simulated annealing quality (fitted in E8).
    "sched_random": {"slope": -0.03, "at": 10, "value": 1.0, "floor": 0.1},
    "sched_qaoa_advantage": 0.0,
    "sched_sa": {"slope": 0.0, "at": 10, "value": 1.0, "floor": 0.9},
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


def maxcut_opt_per_edge(n: int, density: float) -> float:
    return min(1.0, 0.5 + CALIBRATION["opt_c"] / math.sqrt(max(n * density, 1e-9)))


def best_of_shots(mean_ratio: float, n_edges: int, opt: float, shots: int) -> float:
    """Expected best cut ratio among `shots` samples whose cut has the given mean.

    A sample's cut is a sum over |E| edges, so its spread is about sqrt(|E|)/2; the
    maximum of S roughly independent draws sits ~sqrt(2 ln S) spreads above the mean.
    For small graphs this reaches the optimum (what E2 measures, for QAOA *and* for random
    sampling); for large graphs it does not, which is why the mean matters there."""
    sigma = math.sqrt(n_edges) / 2 / opt
    return float(min(1.0, mean_ratio + sigma * math.sqrt(2 * math.log(max(shots, 2)))))


def qaoa_quality(n: int, fidelity: float = 1.0, density: float = 0.3, shots: int = 1000) -> float:
    """Best sampled cut / optimum for QAOA on a noisy device: noise mixes QAOA's output
    distribution toward uniform random sampling in proportion to (1 - fidelity)."""
    n_edges = max(1, int(density * n * (n - 1) / 2))
    opt = n_edges * maxcut_opt_per_edge(n, density)
    random_mean = (n_edges / 2) / opt
    override = CALIBRATION.get("qaoa_override")
    if override is not None:
        # What-if for a hard instance class: assumed noiseless quality, degraded by noise toward
        # 0.5 (a random assignment cuts half the edges; the conservative floor on such instances).
        return fidelity * override + (1 - fidelity) * 0.5
    g = CALIBRATION["qaoa_gain"]
    gain = min(1.0, g["a"] * n ** (-g["b"]))
    qaoa_mean = random_mean + gain * (1 - random_mean)
    mean = fidelity * qaoa_mean + (1 - fidelity) * random_mean
    return best_of_shots(mean, n_edges, opt, shots)


def sa_quality(n: int) -> float:
    return _lin(CALIBRATION["sa_ratio"], n)


def kernel_quality(fidelity: float = 1.0, n_classes: int = 2) -> float:
    acc = CALIBRATION["kernel_acc"]["quantum"]
    return fidelity * acc + (1 - fidelity) * (1 / n_classes)


def vqe_quality(n: int, fidelity: float = 1.0) -> float:
    """Noise pulls the VQE energy toward the fully mixed state, which scores ~0.5 on both models."""
    return fidelity * _lin(CALIBRATION["vqe_quality"], n) + (1 - fidelity) * 0.5


def scheduling_quality(n: int, fidelity: float = 1.0) -> float:
    """Uniform random sampling plus QAOA's measured advantage over it; noise removes the advantage."""
    return min(1.0, _lin(CALIBRATION["sched_random"], n) + fidelity * CALIBRATION.get("sched_qaoa_advantage", 0.0))


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


def make_ground_state(i, rng, **kw) -> Workload:
    """VQE workload. model=h2 (2 qubits, the hydrogen molecule) or model=tfim with n spins."""
    from .quantum.vqe import hamiltonian_for, measurement_groups, n_params, vqe_budget
    model = str(kw.get("model", "tfim" if "n" in kw else rng.choice(["h2", "tfim"])))
    n_req = int(kw.get("n", rng.integers(4, 40))) if model == "tfim" else 2
    layers = int(kw.get("layers", 1))
    H, n = hamiltonian_for(model, n_req)
    groups = len(measurement_groups(H))
    k = n_params(n, layers)
    circuits, jobs = vqe_budget(k, groups)
    if n <= 20:
        # Lanczos on a sparse matrix with ~n nonzeros per row, ~100 matrix-vector products
        method, gflop, quality, mem = "exact_diagonalisation", 100 * 2 ** n * (n + 1) * 8e-9, 1.0, 3 * 16 * 2 ** n / 1e9
    else:
        # 1D chain: DMRG at bond dimension ~64, a few sweeps
        method, gflop, quality, mem = "dmrg", 10 * n * 64 ** 3 * 4 * 8e-9, CALIBRATION["dmrg_quality"], 0.1
    qp = QuantumProfile(n_qubits=n, depth=2 + layers * n, n_1q=k + n, n_2q=layers * (n - 1),
                        n_circuits=circuits, shots=1000, n_rounds=jobs, algorithm=f"VQE(RY x{layers}, Rotosolve)")
    return Workload(i, "ground_state", deadline_s=float(kw.get("deadline_s", rng.uniform(10, 1800))),
                    data_mb=1e-3, result_mb=1e-4, privacy=False, gflop=gflop, memory_gb=mem, parallel=False,
                    classical_method=method, classical_quality=quality,
                    min_quality=float(kw.get("min_quality", 0.8)), quantum=qp,
                    params={"model": model, "n": n, "layers": layers})


def make_scheduling(i, rng, **kw) -> Workload:
    """QAOA workload: assign n tasks to edge or cloud (load balance + split traffic + upload cost)."""
    n = int(kw.get("n", rng.integers(6, 40)))
    density = float(kw.get("density", 0.3))
    p = int(kw.get("p", 1))
    n_pairs = n * (n - 1) // 2  # the load-balance term couples every pair of tasks
    if n <= 20:
        method, gflop, quality = "exact", (2 ** n) * (n + density * n_pairs) * 4e-9, 1.0
    else:
        method, gflop, quality = "simulated_annealing", 2000 * n * n * 50e-9, _lin(CALIBRATION["sched_sa"], n)
    routing = 1 + 0.25 * (n - 1)  # all-to-all couplings need many SWAPs on sparse hardware
    iters = 25
    qp = QuantumProfile(n_qubits=n, depth=int(p * (n + 2) * routing) + 1, n_1q=n + 2 * n * p,
                        n_2q=int(n_pairs * p * routing), n_circuits=12 + iters, shots=1000,
                        n_rounds=1 + iters, algorithm=f"QAOA scheduling (p={p})")
    return Workload(i, "scheduling", deadline_s=float(kw.get("deadline_s", rng.uniform(2, 600))),
                    data_mb=n * 1e-5, result_mb=1e-4, privacy=False, gflop=gflop,
                    memory_gb=0.01 + 2 ** min(n, 22) * n * 1e-9, parallel=False, classical_method=method,
                    classical_quality=quality, min_quality=float(kw.get("min_quality", 0.8)), quantum=qp,
                    params={"n": n, "density": density, "p": p})


BUILDERS = {"inference": make_inference, "training": make_training, "optimization": make_optimization,
            "kernel_classification": make_kernel, "quantum_simulation": make_qsim,
            "ground_state": make_ground_state, "scheduling": make_scheduling}

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
