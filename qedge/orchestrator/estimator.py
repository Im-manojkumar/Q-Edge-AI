"""Cost model: predicts latency, energy, money and result quality of a placement.

A *placement* is either purely classical (``Placement("edge_gpu")``) or hybrid
(``Placement("cloud_cpu", "qpu")``), where the classical host runs the
optimiser / SVM / post-processing and only circuit execution goes to the
quantum resource. The same model is used by the scheduler (with the nominal,
believed parameters) and by the simulator (with the true, hidden ones).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..resources import RADIO_POWER_W, Resource, ResourceState
from ..workloads import Workload, kernel_quality, qaoa_quality, scheduling_quality, vqe_quality


@dataclass(frozen=True)
class Placement:
    classical: str
    quantum: str | None = None

    @property
    def label(self) -> str:
        return self.classical if self.quantum is None else f"{self.classical}+{self.quantum}"

    @property
    def is_hybrid(self) -> bool:
        return self.quantum is not None


@dataclass
class Estimate:
    placement: Placement
    feasible: bool = True
    reasons: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    energy_j: float = 0.0
    edge_energy_j: float = 0.0
    cost_usd: float = 0.0
    quality: float = 1.0
    fidelity: float | None = None
    breakdown: dict = field(default_factory=dict)
    score: float | None = None
    qpu_device: str | None = None      # device chosen by calibration-aware routing

    def row(self) -> dict:
        return {"placement": self.placement.label, "feasible": self.feasible, "latency_s": self.latency_s,
                "energy_j": self.energy_j, "edge_energy_j": self.edge_energy_j, "cost_usd": self.cost_usd,
                "quality": self.quality, "fidelity": self.fidelity, "score": self.score,
                "qpu_device": self.qpu_device,
                "reasons": "; ".join(self.reasons), **{f"t_{k}": v for k, v in self.breakdown.items()}}


def transfer_time(r: Resource, up_mb: float, down_mb: float, rtt: float | None = None) -> float:
    if r.is_edge:
        return 0.0
    return (r.rtt_s if rtt is None else rtt) + up_mb * 8 / r.uplink_mbps + down_mb * 8 / r.downlink_mbps


def candidate_placements(w: Workload, resources: dict[str, Resource]) -> list[Placement]:
    classical = [n for n, r in resources.items() if not r.is_quantum and r.available]
    out = [Placement(n) for n in classical]
    if w.quantum_capable:
        hosts = [n for n in classical if n in ("edge_cpu", "cloud_cpu")] or classical[:1]
        for q, r in resources.items():
            if r.is_quantum and r.available:
                out.extend(Placement(h, q) for h in hosts)
    return out


class CostModel:
    def __init__(self, resources: dict[str, Resource]):
        self.resources = resources

    # ---- classical execution ------------------------------------------------
    def _classical(self, w: Workload, r: Resource, st: ResourceState, est: Estimate, gflop: float,
                   up_mb: float, down_mb: float):
        if w.memory_gb > r.memory_gb:
            est.feasible = False
            est.reasons.append(f"needs {w.memory_gb:.1f} GB > {r.memory_gb:.0f} GB on {r.name}")
        rate = r.gflops if w.parallel else r.serial_gflops
        compute = gflop / max(rate, 1e-9)
        xfer = transfer_time(r, up_mb, down_mb, st.rtt_s.get(r.name))
        wait = st.queue_wait_s.get(r.name, 0.0)
        est.breakdown.update(queue=wait, transfer=xfer, compute=compute)
        est.latency_s += wait + xfer + compute
        est.energy_j += compute * r.power_w
        radio = (xfer if not r.is_edge else 0.0) * RADIO_POWER_W
        est.edge_energy_j += (compute * r.power_w if r.is_edge else 0.0) + radio
        est.energy_j += radio
        est.cost_usd += compute * r.cost_per_s

    # ---- quantum execution --------------------------------------------------
    def _quantum(self, w: Workload, host: Resource, q: Resource, st: ResourceState, est: Estimate):
        qp = w.quantum
        cal = st.qpu_figures.get(q.name)
        if cal and not q.sim_gflops:
            # Today's calibration for the best region of the best device replaces the
            # static catalogue figures for this workload.
            q = copy.copy(q)
            for k in ("p1", "p2", "readout_err", "max_qubits"):
                if cal.get(k) is not None:
                    setattr(q, k, cal[k])
            src = "live calibration" if cal.get("source") == "live" else "calibration snapshot"
            est.qpu_device = f"{cal.get('device')} ({src})"
        if qp.n_qubits > q.max_qubits:
            est.feasible = False
            est.reasons.append(f"{qp.n_qubits} qubits > {q.max_qubits} on {q.name}")
        if w.privacy:
            est.feasible = False
            est.reasons.append("privacy: data may not leave the edge")
        # host <-> quantum service round trip, paid once per sequential round
        hop = st.rtt_s.get(q.name, q.rtt_s) if host.is_edge else 0.01
        queue = st.qpu_queue_s.get(q.name, q.queue_mean_s) + st.queue_wait_s.get(q.name, 0.0)
        if q.sim_gflops:  # classical state-vector simulator offered as a quantum service
            per_circ = (qp.n_1q + qp.n_2q) * 2 ** qp.n_qubits * 8e-9 / q.sim_gflops
            exec_t = qp.n_circuits * per_circ + qp.n_circuits * qp.shots * 1e-6
            fid = 1.0
            money = exec_t * q.cost_per_s
        else:
            per_shot = q.shot_overhead_s + qp.depth * q.layer_time_s
            exec_t = qp.n_circuits * qp.shots * per_shot
            fid = qp.fidelity(q.p1, q.p2, q.readout_err)
            money = qp.n_rounds * q.cost_per_task + qp.n_circuits * qp.shots * q.cost_per_shot
        rounds = qp.n_rounds * (q.job_overhead_s + hop)
        est.breakdown.update(q_queue=queue, q_rounds=rounds, q_exec=exec_t)
        est.latency_s += queue + rounds + exec_t
        est.energy_j += exec_t * q.power_w
        if host.is_edge:
            est.edge_energy_j += qp.n_rounds * hop * RADIO_POWER_W
        est.cost_usd += money
        est.fidelity = fid
        return fid

    # ---- public -------------------------------------------------------------
    def estimate(self, w: Workload, p: Placement, st: ResourceState | None = None) -> Estimate:
        st = st or ResourceState()
        est = Estimate(p)
        host = self.resources[p.classical]
        if w.privacy and not host.is_edge:
            est.feasible = False
            est.reasons.append("privacy: data may not leave the edge")
        if not p.is_hybrid:
            self._classical(w, host, st, est, w.gflop, w.data_mb, w.result_mb)
            est.quality = w.classical_quality
            return est
        if not w.quantum_capable:
            est.feasible = False
            est.reasons.append(f"{w.kind} has no quantum formulation")
            return est
        # Hybrid: small classical share on the host (optimiser, SVM fit, post-processing)
        self._classical(w, host, st, est, gflop=0.02 * w.gflop + 0.01, up_mb=w.data_mb, down_mb=w.result_mb)
        fid = self._quantum(w, host, self.resources[p.quantum], st, est)
        est.quality = self.quantum_quality(w, fid)
        return est

    @staticmethod
    def quantum_quality(w: Workload, fid: float) -> float:
        if w.kind == "optimization":
            return qaoa_quality(w.params["n"], fid, w.params.get("density", 0.3), w.quantum.shots)
        if w.kind == "kernel_classification":
            return kernel_quality(fid)
        if w.kind == "quantum_simulation":
            return fid
        if w.kind == "ground_state":
            return vqe_quality(w.params["n"], fid)
        if w.kind == "scheduling":
            return scheduling_quality(w.params["n"], fid)
        return 0.0
