"""Discrete-event simulation of an edge-cloud-quantum deployment.

The scheduler only sees the *nominal* resource catalogue plus what a monitor
can observe (slot queues, ping RTT, provider-reported QPU queue). The
simulator executes every decision against a hidden *true* catalogue, with
per-task randomness, so a policy is rewarded for decisions that hold up in
reality rather than on paper.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..orchestrator.estimator import CostModel
from ..orchestrator.policies import Policy, make_policy
from ..resources import Resource, ResourceState, load_catalogue, scale
from ..workloads import Workload, generate


@dataclass
class Scenario:
    name: str
    description: str
    true_scale: dict = field(default_factory=dict)      # hidden from the scheduler
    belief_scale: dict = field(default_factory=dict)    # applied to both (known change)
    rate_per_s: float = 0.2
    mix: dict | None = None
    qpu_report_bias: float = 1.0                         # provider queue estimate / true
    noise_sigma: float = 0.15


SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario("baseline", "Nominal conditions; reality matches the catalogue up to random noise."),
    Scenario("network_congested", "Uplink drops to 20% of nominal and RTT x4; ping RTT is observed, bandwidth is not.",
             true_scale={"cloud_cpu__uplink_mbps": 0.2, "cloud_gpu__uplink_mbps": 0.2, "quantum_sim__uplink_mbps": 0.2,
                         "qpu__uplink_mbps": 0.2, "cloud_cpu__rtt_s": 4, "cloud_gpu__rtt_s": 4,
                         "quantum_sim__rtt_s": 4, "qpu__rtt_s": 4}),
    Scenario("edge_overload", "Arrival rate x4 and the edge board thermally throttles to 60% speed.",
             true_scale={"edge_cpu__gflops": 0.6, "edge_cpu__serial_gflops": 0.6, "edge_gpu__gflops": 0.6},
             rate_per_s=0.8),
    Scenario("qpu_busy", "QPU queue is 10x the advertised mean and the provider under-reports it by 3x.",
             true_scale={"qpu__queue_mean_s": 10}, qpu_report_bias=1 / 3),
    Scenario("free_qpu_tier", "QPU access is free (e.g. IBM Open Plan minutes), everything else nominal.",
             belief_scale={"qpu__cost_per_task": 0.0, "qpu__cost_per_shot": 0.0}),
    Scenario("future_qpu", "Hypothetical early fault-tolerant QPU: 100x lower gate and readout errors, 10x shorter queue, "
             "10x cheaper shots. Not today's hardware.",
             belief_scale={"qpu__p1": 0.01, "qpu__p2": 0.01, "qpu__readout_err": 0.01, "qpu__queue_mean_s": 0.1,
                           "qpu__cost_per_shot": 0.1, "qpu__cost_per_task": 0.1}),
    Scenario("future_qpu_busy", "Hypothetical future QPU, but its real queue is 50x the advertised mean and "
             "under-reported 3x: tests whether the feedback loop notices.",
             belief_scale={"qpu__p1": 0.01, "qpu__p2": 0.01, "qpu__readout_err": 0.01, "qpu__queue_mean_s": 0.1,
                           "qpu__cost_per_shot": 0.1, "qpu__cost_per_task": 0.1},
             true_scale={"qpu__queue_mean_s": 50}, qpu_report_bias=1 / 50),
]}


class Simulator:
    def __init__(self, scenario: Scenario | str = "baseline", catalogue: dict[str, Resource] | None = None,
                 seed: int = 0):
        self.sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        base = catalogue or load_catalogue()
        self.nominal = scale(base, **self.sc.belief_scale)
        self.true = scale(self.nominal, **self.sc.true_scale)
        self.true_model = CostModel(self.true)
        self.seed = seed

    def workloads(self, n: int) -> list[Workload]:
        return generate(n, self.sc.rate_per_s, self.sc.mix, seed=self.seed)

    def run(self, policy: Policy | str, workloads: list[Workload]) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + 1)
        if isinstance(policy, str):
            policy = make_policy(policy, copy.deepcopy(self.nominal), seed=self.seed)
        free = {n: [0.0] * r.slots for n, r in self.true.items()}
        rows = []
        for w in workloads:
            t = w.arrival_s
            qpu_true_mean = {n: r.queue_mean_s for n, r in self.true.items() if r.is_quantum}
            observed = ResourceState(
                now=t,
                queue_wait_s={n: max(0.0, min(s) - t) for n, s in free.items()},
                rtt_s={n: r.rtt_s * float(rng.lognormal(0, 0.1)) for n, r in self.true.items() if not r.is_edge},
                qpu_queue_s={n: m * self.sc.qpu_report_bias for n, m in qpu_true_mean.items()},
            )
            d = policy.decide(w, observed)
            # --- ground truth ---------------------------------------------------
            truth_state = copy.copy(observed)
            truth_state.rtt_s = {n: r.rtt_s for n, r in self.true.items() if not r.is_edge}
            truth_state.qpu_queue_s = {n: float(rng.exponential(m)) if m > 0 else 0.0 for n, m in qpu_true_mean.items()}
            truth = self.true_model.estimate(w, d.placement, truth_state)
            noise = float(rng.lognormal(0, self.sc.noise_sigma))
            wait = truth.breakdown.get("queue", 0.0)
            latency = wait + (truth.latency_s - wait) * noise
            quality = float(np.clip(truth.quality + rng.normal(0, 0.01), 0, 1))
            busy_res = d.placement.quantum or d.placement.classical
            busy_for = (truth.breakdown.get("q_rounds", 0) + truth.breakdown.get("q_exec", 0)) if d.placement.quantum \
                else truth.breakdown.get("compute", 0.0) * noise
            slots = free[busy_res]
            k = int(np.argmin(slots))
            start = max(t, slots[k]) + (truth.breakdown.get("transfer", 0.0) if not d.placement.quantum else 0.0)
            slots[k] = start + busy_for
            outcome = {"latency_s": latency, "quality": quality}
            policy.observe(d, outcome)
            rows.append({
                "policy": policy.name, "scenario": self.sc.name, "id": w.id, "kind": w.kind, "arrival_s": t,
                "placement": d.placement.label, "hybrid": d.placement.is_hybrid,
                "quantum_resource": d.placement.quantum or "", "explored": d.explored,
                "feasible": truth.feasible, "deadline_s": w.deadline_s, "latency_s": latency,
                "predicted_latency_s": d.chosen.latency_s, "deadline_met": latency <= w.deadline_s,
                "edge_energy_j": truth.edge_energy_j * noise, "energy_j": truth.energy_j * noise,
                "cost_usd": truth.cost_usd, "quality": quality, "min_quality": w.min_quality,
                "quality_met": quality >= w.min_quality, "privacy": w.privacy,
                "classical_quality": w.classical_quality,
            })
        return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["scenario", "policy"])
    out = pd.DataFrame({
        "tasks": g.size(),
        "mean_latency_s": g.latency_s.mean(),
        "p95_latency_s": g.latency_s.quantile(0.95),
        "deadline_hit_rate": g.deadline_met.mean(),
        "edge_energy_j": g.edge_energy_j.mean(),
        "total_energy_kj": g.energy_j.mean() / 1e3,
        "cost_usd_per_task": g.cost_usd.mean(),
        "mean_quality": g.quality.mean(),
        "quality_hit_rate": g.quality_met.mean(),
        "quantum_offload_rate": g.hybrid.mean(),
        "infeasible_rate": 1 - g.feasible.mean(),
    })
    return out.reset_index()


def run_matrix(policies: list[str], scenarios: list[str], n_tasks: int = 600, seeds=(0, 1, 2)) -> pd.DataFrame:
    frames = []
    for sc in scenarios:
        for seed in seeds:
            sim = Simulator(sc, seed=seed)
            wl = sim.workloads(n_tasks)
            for p in policies:
                df = sim.run(p, wl)
                df["seed"] = seed
                frames.append(df)
    return pd.concat(frames, ignore_index=True)
