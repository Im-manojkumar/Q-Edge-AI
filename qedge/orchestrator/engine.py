"""The live Q-Edge orchestrator: profile -> monitor -> decide -> execute -> evaluate -> feedback.

    from qedge.orchestrator.engine import Orchestrator
    orch = Orchestrator(qpu_stack="ibm-auto")     # best real IBM device if QISKIT_IBM_TOKEN is set
    report = orch.submit("optimization", n=8)
    print(report.explain())
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import psutil

from ..executor import ExecResult, Executor
from ..quantum.backends import CalibratedIBMBackend
from ..resources import Resource, ResourceState, load_catalogue
from ..workloads import BUILDERS, Workload, load_calibration
from .estimator import Placement
from .policies import Decision, Policy, make_policy


@dataclass
class Report:
    workload: Workload
    decision: Decision
    result: ExecResult | None
    measured_latency_s: float
    error: str | None = None

    def explain(self) -> str:
        lines = [self.decision.explain()]
        if self.result:
            r = self.result
            where = r.backend or "local classical"
            if r.fallback_from:
                where += f" (fallback: {r.fallback_from} not configured)"
            lines.append(f"  executed {r.method} on {where}: {r.quality_metric} = {r.quality:.3f}, "
                         f"compute {r.compute_s:.3g}s, quantum {r.quantum_s:.3g}s, "
                         f"latency incl. modelled network {self.measured_latency_s:.3g}s")
        if self.error:
            lines.append(f"  ERROR: {self.error}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"workload": self.workload.to_dict(), "placement": self.decision.placement.label,
                "policy": self.decision.policy, "predicted": self.decision.chosen.row(),
                "alternatives": [e.row() for e in self.decision.estimates],
                "quantum_checks": self.decision.quantum_checks,
                "result": asdict(self.result) if self.result else None,
                "measured_latency_s": self.measured_latency_s, "error": self.error}


class Monitor:
    """Observes what can be observed for real: local CPU load and, when real
    hardware is configured, the provider's queue length."""

    AVG_JOB_S = 20.0  # assumed seconds per queued job ahead of us on a QPU

    def __init__(self, resources: dict[str, Resource], executor: Executor):
        self.resources = resources
        self.executor = executor

    def snapshot(self, w: Workload | None = None) -> ResourceState:
        st = ResourceState(now=time.time())
        cpu = psutil.cpu_percent(interval=0.05) / 100
        st.utilisation["edge_cpu"] = cpu
        # A busy local CPU delays edge work roughly in proportion to its load.
        st.queue_wait_s["edge_cpu"] = 0.0
        for name, r in self.resources.items():
            if r.is_quantum and name == "qpu":
                try:
                    be = self.executor.backend_for("qpu")
                    depth = be.queue_depth() if be.hardware else None
                except Exception:
                    depth = None
                if depth is not None:
                    st.qpu_queue_s[name] = depth * self.AVG_JOB_S
                self._calibrate(st, name, w)
        return st

    def _calibrate(self, st: ResourceState, name: str, w: Workload | None):
        """Calibration-aware routing: give the cost model the error figures of the best
        region of the best device for this workload, and that device's queue."""
        if w is None or not w.quantum_capable:
            return
        try:
            be = self.executor.backend_for(name)
        except Exception:
            return
        if not isinstance(be, CalibratedIBMBackend):
            return
        from ..quantum.calibration import best_device_figures
        fig = best_device_figures(w.quantum, be.devices)
        if fig:
            st.qpu_figures[name] = fig
            if fig.get("queue_s") is not None:
                st.qpu_queue_s[name] = fig["queue_s"]


class Orchestrator:
    def __init__(self, policy: str | Policy = "qedge_adaptive", sim_stack: str = "qiskit", qpu_stack: str = "ibm-auto",
                 config: str | None = None, log_path: str | None = "results/live_log.jsonl"):
        load_calibration()
        self.resources = load_catalogue(config)
        self.policy = make_policy(policy, self.resources) if isinstance(policy, str) else policy
        self.executor = Executor(sim_stack, qpu_stack)
        self.monitor = Monitor(self.resources, self.executor)
        self.log_path = log_path
        self._next_id = 0

    def plan(self, kind: str, **params) -> tuple[Workload, Decision]:
        import numpy as np
        w = BUILDERS[kind](self._next_id, np.random.default_rng(self._next_id), **params)
        self._next_id += 1
        return w, self.policy.decide(w, self.monitor.snapshot(w))

    def submit(self, kind: str, force: str | None = None, **params) -> Report:
        w, d = self.plan(kind, **params)
        if force:  # run a specific placement anyway, e.g. to compare against the decision
            classical, _, quantum = force.partition("+")
            p = Placement(classical, quantum or None)
            est = next((e for e in d.estimates if e.placement == p), d.chosen)
            d = Decision(w, p, est, d.estimates, d.policy + " (forced)", d.quantum_checks)
        result, err = None, None
        t0 = time.perf_counter()
        try:
            result = self.executor.run(w, d.placement.quantum)
        except Exception as e:  # report, don't crash the service
            err = f"{type(e).__name__}: {e}"
        wall = time.perf_counter() - t0
        b = d.chosen.breakdown
        modelled_network = b.get("transfer", 0.0)
        measured = (result.compute_s + result.quantum_s) if result else wall
        latency = measured + modelled_network
        if result is not None:
            self.policy.observe(d, {"latency_s": latency, "quality": result.quality})
        rep = Report(w, d, result, latency, err)
        self._log(rep)
        return rep

    def _log(self, rep: Report):
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rep.to_dict(), default=str) + "\n")
