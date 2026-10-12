"""Compute resources the orchestrator can place work on, and their live state.

Numbers in ``DEFAULT_CATALOGUE`` are order-of-magnitude figures for a typical
deployment (an edge board with a small GPU, a cloud VM, a cloud GPU, a managed
state-vector simulator and a superconducting QPU). They are *assumptions*,
listed in one place so they can be changed from ``configs/default.yaml`` and
so the report can state them. Sources for the QPU figures are given in
docs/assumptions.md.
"""
from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field, fields

import yaml


@dataclass
class Resource:
    name: str
    kind: str                    # "classical" | "quantum"
    tier: str                    # "edge" | "cloud"
    label: str = ""
    gflops: float = 0.0          # sustained FP32 throughput for well-parallelised work
    serial_gflops: float = 0.0   # throughput for work that does not parallelise (e.g. on a GPU)
    power_w: float = 0.0         # active power attributed to the task
    memory_gb: float = 0.0
    rtt_s: float = 0.0           # round trip from the edge device
    uplink_mbps: float = 1e9
    downlink_mbps: float = 1e9
    cost_per_s: float = 0.0      # USD per second of use
    slots: int = 1               # tasks it can run concurrently
    # quantum-only
    max_qubits: int = 0
    p1: float = 0.0              # 1-qubit gate Pauli error
    p2: float = 0.0              # 2-qubit gate Pauli error (per qubit)
    readout_err: float = 0.0
    layer_time_s: float = 0.0    # time per circuit layer
    shot_overhead_s: float = 0.0 # reset / repetition delay per shot
    job_overhead_s: float = 0.0  # submit, compile, load per job
    queue_mean_s: float = 0.0    # advertised mean wait from other users
    cost_per_task: float = 0.0
    cost_per_shot: float = 0.0
    sim_gflops: float = 0.0      # quantum simulators: state-vector throughput
    available: bool = True

    @property
    def is_quantum(self) -> bool:
        return self.kind == "quantum"

    @property
    def is_edge(self) -> bool:
        return self.tier == "edge"


DEFAULT_CATALOGUE: dict[str, dict] = {
    "edge_cpu": dict(kind="classical", tier="edge", label="Edge CPU (ARM board)", gflops=25, serial_gflops=8,
                     power_w=6, memory_gb=8, slots=2),
    "edge_gpu": dict(kind="classical", tier="edge", label="Edge GPU (Jetson-class)", gflops=600, serial_gflops=4,
                     power_w=15, memory_gb=8, slots=1),
    "cloud_cpu": dict(kind="classical", tier="cloud", label="Cloud CPU (32 vCPU VM)", gflops=600, serial_gflops=40,
                      power_w=180, memory_gb=128, rtt_s=0.04, uplink_mbps=20, downlink_mbps=100,
                      cost_per_s=1.5 / 3600, slots=8),
    "cloud_gpu": dict(kind="classical", tier="cloud", label="Cloud GPU (A100-class)", gflops=12000, serial_gflops=30,
                      power_w=350, memory_gb=80, rtt_s=0.04, uplink_mbps=20, downlink_mbps=100,
                      cost_per_s=3.0 / 3600, slots=2),
    "quantum_sim": dict(kind="quantum", tier="cloud", label="Cloud quantum simulator (state vector)",
                        power_w=350, memory_gb=80, rtt_s=0.04, uplink_mbps=20, downlink_mbps=100,
                        max_qubits=34, sim_gflops=2000, job_overhead_s=1.0, queue_mean_s=2.0,
                        cost_per_s=0.075 / 60, slots=4),
    "qpu": dict(kind="quantum", tier="cloud", label="Cloud QPU (superconducting, NISQ)",
                power_w=25000, rtt_s=0.04, uplink_mbps=20, downlink_mbps=100,
                max_qubits=133, p1=3e-4, p2=3e-3, readout_err=1.5e-2,
                layer_time_s=6.6e-7, shot_overhead_s=2.5e-4, job_overhead_s=3.0, queue_mean_s=120.0,
                cost_per_task=0.30, cost_per_shot=0.00145, slots=1),
}

# Energy an edge radio spends per second of transmission.
RADIO_POWER_W = 2.0


def load_catalogue(path: str | None = None) -> dict[str, Resource]:
    cat = copy.deepcopy(DEFAULT_CATALOGUE)
    path = path or os.environ.get("QEDGE_CONFIG")
    if path and os.path.exists(path):
        with open(path) as f:
            override = (yaml.safe_load(f) or {}).get("resources", {})
        for name, vals in override.items():
            cat.setdefault(name, {}).update(vals)
    valid = {f.name for f in fields(Resource)}
    return {n: Resource(name=n, **{k: v for k, v in d.items() if k in valid}) for n, d in cat.items()}


@dataclass
class ResourceState:
    """What the monitor can observe at decision time."""

    now: float = 0.0
    queue_wait_s: dict[str, float] = field(default_factory=dict)   # wait for a free slot
    rtt_s: dict[str, float] = field(default_factory=dict)          # measured round trip
    qpu_queue_s: dict[str, float] = field(default_factory=dict)    # provider-reported wait
    utilisation: dict[str, float] = field(default_factory=dict)
    online: dict[str, bool] = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


def scale(res: dict[str, Resource], **factors) -> dict[str, Resource]:
    """Copy a catalogue with attributes scaled, e.g. scale(cat, qpu__queue_mean_s=10)."""
    out = copy.deepcopy(res)
    for key, f in factors.items():
        name, attr = key.split("__")
        targets = out.values() if name == "all" else [out[name]]
        for r in targets:
            setattr(r, attr, getattr(r, attr) * f)
    return out
