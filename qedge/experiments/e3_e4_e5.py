"""E3 policy comparison, E4 decision boundaries, E5 live prototype run."""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from ..orchestrator.engine import Orchestrator
from ..orchestrator.policies import POLICIES, QEdgeCostPolicy
from ..resources import ResourceState, load_catalogue, scale
from ..simulation.simulator import SCENARIOS, run_matrix, summarise
from ..workloads import load_calibration, make_optimization, make_qsim


# ---------------------------------------------------------------- E3
def e3(quick: bool = False) -> dict[str, pd.DataFrame]:
    load_calibration()
    raw = run_matrix(list(POLICIES), list(SCENARIOS), n_tasks=300 if quick else 800,
                     seeds=(0,) if quick else (0, 1, 2))
    summ = summarise(raw)
    per_kind = (raw.groupby(["scenario", "policy", "kind"])
                .agg(deadline_hit_rate=("deadline_met", "mean"), mean_latency_s=("latency_s", "mean"),
                     mean_quality=("quality", "mean"), quantum_offload_rate=("hybrid", "mean"))
                .reset_index())
    placements = (raw[raw.policy.isin(["qedge_static", "qedge_adaptive"])]
                  .groupby(["scenario", "policy", "kind", "placement"]).size().rename("tasks").reset_index())
    return {"e3_summary": summ, "e3_per_kind": per_kind, "e3_placements": placements}


# ---------------------------------------------------------------- E4
HARD_CLASSICAL = 0.70

def _choice(policy, w):
    d = policy.decide(w, ResourceState())
    return d.placement.label


def e4() -> dict[str, pd.DataFrame]:
    """Where in workload space does Q-Edge send work to the QPU?"""
    load_calibration()
    today = load_catalogue()
    futures = {
        "today": today,
        "10x_better_qpu": scale(today, qpu__p1=0.1, qpu__p2=0.1, qpu__readout_err=0.1),
        "100x_better_qpu": scale(today, qpu__p1=0.01, qpu__p2=0.01, qpu__readout_err=0.01,
                                 qpu__queue_mean_s=0.1, qpu__cost_per_shot=0.1, qpu__cost_per_task=0.1),
    }
    rows = []
    for regime, cat in futures.items():
        pol = QEdgeCostPolicy(copy.deepcopy(cat))
        for n in range(8, 129, 8):
            for steps in range(2, 41, 2):
                w = make_qsim(0, np.random.default_rng(0), n=n, steps=steps, deadline_s=3600)
                d = pol.decide(w, ResourceState())
                qpu_est = next((e for e in d.estimates if e.placement.label == "cloud_cpu+qpu"), None)
                rows.append({"regime": regime, "workload": "quantum_simulation", "n_qubits": n, "steps": steps,
                             "choice": d.placement.label, "uses_qpu": d.placement.quantum == "qpu",
                             "uses_quantum": d.placement.is_hybrid, "classical_method": w.classical_method,
                             "classical_quality": w.classical_quality,
                             "qpu_fidelity": qpu_est.fidelity if qpu_est else None})
    qsim = pd.DataFrame(rows)

    # Optimisation: how big a quality edge would QAOA need, given the queue, before Q-Edge offloads?
    # Measured instances leave no headroom (annealing is ~0.98 of optimum), so this models a hard
    # instance class where the classical heuristic reaches only HARD_CLASSICAL of the optimum.
    rows = []
    from .. import workloads as W
    base = copy.deepcopy(W.CALIBRATION)
    fut = futures["100x_better_qpu"]
    for adv in np.linspace(0, 0.3, 16):
        for price in [0.0, 0.25, 1.0, 4.0]:  # multiplier on the 100x-better QPU's per-task/per-shot price
            cat = scale(fut, qpu__cost_per_task=price, qpu__cost_per_shot=price)
            pol = QEdgeCostPolicy(copy.deepcopy(cat))
            for n in [30, 60, 100]:
                w = make_optimization(0, np.random.default_rng(0), n=n, deadline_s=600)
                w.classical_quality = HARD_CLASSICAL
                # hypothetical: noiseless QAOA beats the classical heuristic by `adv`
                W.CALIBRATION["qaoa_override"] = min(1.0, HARD_CLASSICAL + adv)
                d = pol.decide(w, ResourceState())
                qpu = next(e for e in d.estimates if e.placement.label == "cloud_cpu+qpu")
                rows.append({"workload": "optimization", "n": n, "quality_advantage": round(adv, 3),
                             "price_multiplier": price, "qpu_cost_usd": qpu.cost_usd, "qpu_fidelity": qpu.fidelity,
                             "choice": d.placement.label, "uses_qpu": d.placement.quantum == "qpu",
                             "target": "QPU" if d.placement.quantum == "qpu" else
                             ("quantum simulator" if d.placement.quantum else "classical")})
    W.CALIBRATION.clear()
    W.CALIBRATION.update(base)
    return {"e4_qsim_map": qsim, "e4_qaoa_advantage": pd.DataFrame(rows)}


# ---------------------------------------------------------------- E5
LIVE_JOBS = [
    ("inference", {"batch": 4}), ("inference", {"batch": 16}), ("training", {"samples": 800, "epochs": 1}),
    ("optimization", {"n": 6}), ("optimization", {"n": 10}), ("optimization", {"n": 30}),
    ("kernel_classification", {"features": 2, "n_train": 30}), ("kernel_classification", {"features": 4, "n_train": 30}),
    ("quantum_simulation", {"n": 6, "steps": 4}), ("quantum_simulation", {"n": 12, "steps": 8}),
]


def e5(quick: bool = False, qpu_stack: str = "ibm") -> dict[str, pd.DataFrame]:
    """Real execution: the orchestrator's choice, then every quantum-capable job
    forced through each path so measured quality and time can be compared."""
    orch = Orchestrator("qedge_adaptive", qpu_stack=qpu_stack, log_path="results/live_log.jsonl")
    rows = []

    def record(rep, tag):
        r = rep.result
        rows.append({"tag": tag, "kind": rep.workload.kind, "params": str({k: v for k, v in rep.workload.params.items()
                                                                           if k in ("batch", "n", "steps", "features", "n_train", "samples")}),
                     "placement": rep.decision.placement.label, "predicted_latency_s": rep.decision.chosen.latency_s,
                     "predicted_quality": rep.decision.chosen.quality, "measured_latency_s": rep.measured_latency_s,
                     "quality": r.quality if r else None, "quality_metric": r.quality_metric if r else None,
                     "method": r.method if r else None, "backend": (r.backend if r else None),
                     "fallback_from": r.fallback_from if r else None, "error": rep.error,
                     "predicted_cost_usd": rep.decision.chosen.cost_usd})

    jobs = LIVE_JOBS[:6] if quick else LIVE_JOBS
    for kind, kw in jobs:
        record(orch.submit(kind, **kw), "orchestrator")
    for kind, kw in jobs:
        if kind in ("inference", "training"):
            continue
        if kind == "optimization" and kw["n"] > 12:
            continue
        for force in ["edge_cpu", "cloud_cpu+quantum_sim", "cloud_cpu+qpu"]:
            if force.endswith("qpu") and (kind == "kernel_classification" or kw.get("n", 0) > 10):
                continue  # keep the device emulator runs short
            record(orch.submit(kind, force=force, **kw), f"forced:{force}")
    return {"e5_live": pd.DataFrame(rows)}
