"""Placement policies: the Q-Edge decision engine and the baselines it is compared to."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..resources import Resource, ResourceState
from ..workloads import Workload
from .estimator import CostModel, Estimate, Placement, candidate_placements


@dataclass
class Weights:
    latency: float = 1.0          # per unit of (latency / deadline)
    edge_energy: float = 0.3      # per edge_energy_ref joules of device battery
    total_energy: float = 0.05    # per total_energy_ref joules system-wide
    cost: float = 0.3             # per cost_ref USD
    quality: float = 10.0         # per unit of quality lost
    deadline_penalty: float = 5.0
    quality_penalty: float = 5.0
    edge_energy_ref: float = 10.0
    total_energy_ref: float = 1e4
    cost_ref: float = 1.0


@dataclass
class Decision:
    workload: Workload
    placement: Placement
    chosen: Estimate
    estimates: list[Estimate]
    policy: str
    quantum_checks: dict[str, list[tuple[str, bool, str]]] = field(default_factory=dict)
    explored: bool = False
    # Race mode (QEdgeHedgedPolicy): a classical placement started alongside the QPU job,
    # used if the QPU result misses the deadline.
    hedge: Placement | None = None
    hedge_estimate: Estimate | None = None
    hedge_p_miss: float | None = None   # estimated P(QPU queue > slack) behind the hedge verdict

    def explain(self) -> str:
        lines = [f"{self.policy}: workload {self.workload.id} ({self.workload.kind}) -> {self.placement.label}"]
        c = self.chosen
        lines.append(f"  predicted latency {c.latency_s:.3g}s (deadline {self.workload.deadline_s:.3g}s), "
                     f"edge energy {c.edge_energy_j:.3g} J, cost ${c.cost_usd:.3g}, quality {c.quality:.3f}")
        for label, checks in self.quantum_checks.items():
            verdict = "open" if all(ok for _, ok, _ in checks) else "closed"
            lines.append(f"  quantum gate for {label}: {verdict}")
            for name, ok, why in checks:
                lines.append(f"    [{'pass' if ok else 'FAIL'}] {name}: {why}")
        if self.hedge is not None:
            lines.append(f"  hedged: P(miss deadline in QPU queue) ~ {self.hedge_p_miss:.2f}, so {self.hedge.label} "
                         "runs in parallel as a fallback")
        return "\n".join(lines)


class Policy:
    name = "policy"

    def __init__(self, resources: dict[str, Resource], seed: int = 0):
        self.resources = resources
        self.model = CostModel(resources)
        self.rng = np.random.default_rng(seed)

    def estimates(self, w: Workload, st: ResourceState) -> list[Estimate]:
        return [self.model.estimate(w, p, st) for p in candidate_placements(w, self.resources)]

    def decide(self, w: Workload, st: ResourceState) -> Decision:
        raise NotImplementedError

    def observe(self, decision: Decision, outcome: dict) -> None:
        """Feedback hook; only the adaptive policy learns from it."""

    def _pick_by_preference(self, w, st, prefs: list[str]) -> Decision:
        ests = self.estimates(w, st)
        by = {e.placement.label: e for e in ests}
        for label in prefs + [e.placement.label for e in ests]:
            e = by.get(label)
            if e is not None and e.feasible:
                return Decision(w, e.placement, e, ests, self.name)
        e = ests[0]
        return Decision(w, e.placement, e, ests, self.name)


class EdgeOnly(Policy):
    name = "edge_only"

    def decide(self, w, st):
        return self._pick_by_preference(w, st, ["edge_gpu", "edge_cpu"] if w.parallel else ["edge_cpu", "edge_gpu"])


class CloudOnly(Policy):
    name = "cloud_only"

    def decide(self, w, st):
        prefs = ["cloud_gpu", "cloud_cpu"] if w.parallel else ["cloud_cpu", "cloud_gpu"]
        return self._pick_by_preference(w, st, prefs + ["edge_gpu", "edge_cpu"])


class QuantumFirst(Policy):
    """Naive 'quantum everywhere' baseline: offload whenever a QPU formulation exists."""
    name = "quantum_first"

    def decide(self, w, st):
        return self._pick_by_preference(w, st, ["cloud_cpu+qpu", "edge_cpu+qpu", "cloud_gpu", "cloud_cpu"])


class RandomPolicy(Policy):
    name = "random"

    def decide(self, w, st):
        ests = [e for e in self.estimates(w, st) if e.feasible] or self.estimates(w, st)
        e = ests[self.rng.integers(len(ests))]
        return Decision(w, e.placement, e, ests, self.name)


class LatencyGreedy(Policy):
    name = "latency_greedy"

    def decide(self, w, st):
        ests = self.estimates(w, st)
        feas = [e for e in ests if e.feasible] or ests
        e = min(feas, key=lambda e: e.latency_s)
        return Decision(w, e.placement, e, ests, self.name)


class QEdgeCostPolicy(Policy):
    """Q-Edge decision engine: hard constraints, a quantum-benefit gate, then a
    weighted multi-objective score over every remaining placement."""

    name = "qedge_static"
    min_fidelity = 0.05
    quality_tolerance = 0.01

    def __init__(self, resources, weights: Weights | None = None, seed: int = 0):
        super().__init__(resources, seed)
        self.w = weights or Weights()

    # Corrections hook for the adaptive subclass.
    def adjust(self, w: Workload, e: Estimate) -> Estimate:
        return e

    def score(self, w: Workload, e: Estimate) -> float:
        W = self.w
        s = (W.latency * e.latency_s / w.deadline_s
             + W.edge_energy * e.edge_energy_j / W.edge_energy_ref
             + W.total_energy * e.energy_j / W.total_energy_ref
             + W.cost * e.cost_usd / W.cost_ref
             + W.quality * (1 - e.quality))
        if e.latency_s > w.deadline_s:
            s += W.deadline_penalty
        if e.quality < w.min_quality:
            s += W.quality_penalty
        return s

    def quantum_gate(self, w: Workload, e: Estimate, best_classical: Estimate | None) -> list[tuple[str, bool, str]]:
        qname = e.placement.quantum
        q = self.resources[qname]
        checks = [("quantum formulation", w.quantum_capable, w.quantum.algorithm if w.quantum else "none")]
        if not w.quantum_capable:
            return checks
        qp = w.quantum
        if e.qpu_device:
            checks.append(("calibrated device", True, f"best qubit region of {e.qpu_device}"))
        else:
            checks.append(("fits device", qp.n_qubits <= q.max_qubits, f"{qp.n_qubits} of {q.max_qubits} qubits"))
        checks.append(("data may leave edge", not w.privacy, "private" if w.privacy else "ok"))
        if e.fidelity is not None and not q.sim_gflops:
            checks.append(("circuit fidelity", e.fidelity >= self.min_fidelity,
                           f"estimated {e.fidelity:.3f} (min {self.min_fidelity})"))
        bq = best_classical.quality if best_classical and best_classical.feasible else 0.0
        checks.append(("expected quality gain", e.quality >= bq - self.quality_tolerance,
                       f"quantum {e.quality:.3f} vs best classical {bq:.3f}"))
        checks.append(("meets deadline", e.latency_s <= w.deadline_s,
                       f"{e.latency_s:.3g}s vs {w.deadline_s:.3g}s"))
        return checks

    def decide(self, w, st):
        ests = [self.adjust(w, e) for e in self.estimates(w, st)]
        classical = [e for e in ests if not e.placement.is_hybrid and e.feasible]
        best_c = min(classical, key=lambda e: self.score(w, e)) if classical else None
        gate_log: dict[str, list] = {}
        for e in ests:
            if e.placement.is_hybrid:
                checks = self.quantum_gate(w, e, best_c)
                gate_log[e.placement.label] = checks
                if not all(ok for _, ok, _ in checks):
                    e.feasible = False
                    e.reasons.extend(f"gate: {n}" for n, ok, _ in checks if not ok)
            e.score = self.score(w, e) if e.feasible else float("inf")
        feas = [e for e in ests if e.feasible] or ests
        best = min(feas, key=lambda e: e.score if e.score is not None else float("inf"))
        best, explored = self.maybe_explore(w, feas, best)
        # One entry per quantum resource (the host choice does not change the gate verdict much).
        qlog: dict[str, list] = {}
        for label, checks in gate_log.items():
            q = label.split("+", 1)[1]
            if q not in qlog or label == best.placement.label or label.startswith("cloud_cpu"):
                qlog[q] = checks
        return Decision(w, best.placement, best, ests, self.name, qlog, explored)

    def maybe_explore(self, w, feas, best):
        return best, False


class QEdgeAdaptivePolicy(QEdgeCostPolicy):
    """Q-Edge with the feedback loop: learns how far reality deviates from the
    nominal model per placement and per workload kind, and explores a little."""

    name = "qedge_adaptive"

    def __init__(self, resources, weights=None, seed=0, alpha=0.25, epsilon=0.01):
        super().__init__(resources, weights, seed)
        self.alpha = alpha
        self.epsilon = epsilon
        self.lat_ratio: dict[tuple, float] = defaultdict(lambda: 1.0)
        self.res_ratio: dict[str, float] = defaultdict(lambda: 1.0)
        self.q_offset: dict[tuple, float] = defaultdict(float)
        self.seen: dict[tuple, int] = defaultdict(int)

    def adjust(self, w, e):
        key = (w.kind, e.placement.label)
        r = self.lat_ratio[key] if self.seen[key] >= 2 else self.res_ratio[e.placement.label]
        wait = e.breakdown.get("queue", 0.0)  # observed directly, so not corrected
        e.latency_s = wait + (e.latency_s - wait) * r
        e.quality = float(np.clip(e.quality + self.q_offset[key], 0, 1))
        e.breakdown["feedback_ratio"] = r
        return e

    def maybe_explore(self, w, feas, best):
        cands = [e for e in feas if e is not best and e.score is not None and e.score < best.score * 1.5 + 0.5
                 and e.latency_s <= w.deadline_s]
        if cands and self.rng.random() < self.epsilon:
            return cands[self.rng.integers(len(cands))], True
        return best, False

    def observe(self, decision, outcome):
        label = decision.placement.label
        key = (decision.workload.kind, label)
        b = decision.chosen.breakdown
        wait = b.get("queue", 0.0)
        raw = (decision.chosen.latency_s - wait) / max(b.get("feedback_ratio", 1.0), 1e-9)
        ratio = float(np.clip((outcome["latency_s"] - wait) / max(raw, 1e-6), 0.05, 50))
        a = self.alpha
        self.lat_ratio[key] = (1 - a) * self.lat_ratio[key] + a * ratio if self.seen[key] else ratio
        self.res_ratio[label] = (1 - a) * self.res_ratio[label] + a * ratio
        dq = outcome["quality"] - (decision.chosen.quality - self.q_offset[key])
        self.q_offset[key] = (1 - a) * self.q_offset[key] + a * dq
        self.seen[key] += 1


class QEdgeHedgedPolicy(QEdgeAdaptivePolicy):
    """Q-Edge adaptive plus *race mode*: when it sends work to a real QPU and the
    queue alone might blow the deadline, it also starts the best feasible
    classical placement and keeps whichever answer is usable in time.

    The miss probability is P(Q > slack), slack = deadline minus everything that
    is not provider queue (host share, slot wait, rounds, execution; raw model
    figures, not feedback-scaled). Q is modelled as exponential with mean = the
    provider's published estimate x a learned bias. The bias is the censored
    exponential MLE of (observed wait / published estimate): jobs cancelled in
    the queue only tell us the wait was *at least* that long. With one
    pseudo-observation at ratio 1 it starts at the provider's word.

    The fallback is the best-scoring classical placement predicted to meet the
    deadline; if there is none, the decision is not hedged. Hedging buys deadline
    safety with classical compute that is wasted whenever the QPU answer arrives
    in time; the simulator charges for both.
    """

    name = "qedge_hedged"
    hedges = True

    def __init__(self, resources, weights=None, seed=0, alpha=0.25, epsilon=0.01, threshold: float = 0.05,
                 learn_queue_bias: bool = True):
        super().__init__(resources, weights, seed, alpha, epsilon)
        self.threshold = threshold
        self.learn_queue_bias = learn_queue_bias
        self._ratio_sum = 1.0      # sum of observed wait / published mean, censored waits included
        self._uncensored = 1.0     # number of complete (uncensored) observations

    @property
    def queue_bias(self) -> float:
        return self._ratio_sum / self._uncensored if self.learn_queue_bias else 1.0

    def _queues(self, q: str) -> bool:
        r = self.resources[q]
        return r.is_quantum and not r.sim_gflops   # managed simulators have a short, steady queue

    def miss_probability(self, w: Workload, e: Estimate, st: ResourceState) -> float:
        q = e.placement.quantum
        b = e.breakdown
        non_queue = (b.get("queue", 0.0) + b.get("transfer", 0.0) + b.get("compute", 0.0)
                     + b.get("q_rounds", 0.0) + b.get("q_exec", 0.0) + st.queue_wait_s.get(q, 0.0))
        slack = w.deadline_s - non_queue
        mean = st.qpu_queue_s.get(q, self.resources[q].queue_mean_s) * self.queue_bias
        if slack <= 0:
            return 1.0
        return math.exp(-slack / mean) if mean > 0 else 0.0

    def decide(self, w, st):
        d = super().decide(w, st)
        q = d.placement.quantum
        if q is None or not self._queues(q):
            return d
        d.chosen.breakdown["q_published"] = st.qpu_queue_s.get(q, self.resources[q].queue_mean_s)
        d.hedge_p_miss = self.miss_probability(w, d.chosen, st)
        if d.hedge_p_miss > self.threshold:
            # A fallback that is itself expected to be late would only add cost.
            fallback = [e for e in d.estimates if not e.placement.is_hybrid and e.feasible
                        and e.latency_s <= w.deadline_s]
            if fallback:
                e = min(fallback, key=lambda e: e.score)
                d.hedge, d.hedge_estimate = e.placement, e
        return d

    def observe(self, decision, outcome):
        super().observe(decision, outcome)
        wait, published = outcome.get("qpu_queue_s"), decision.chosen.breakdown.get("q_published")
        if wait is None or not published:
            return
        self._ratio_sum += wait / published
        if not outcome.get("qpu_queue_censored", False):
            self._uncensored += 1


POLICIES = {cls.name: cls for cls in
            [EdgeOnly, CloudOnly, QuantumFirst, RandomPolicy, LatencyGreedy, QEdgeCostPolicy, QEdgeAdaptivePolicy]}
# Kept out of POLICIES so the E3 policy matrix (and everything iterating POLICIES) is unchanged.
EXTRA_POLICIES = {cls.name: cls for cls in [QEdgeHedgedPolicy]}


def make_policy(name: str, resources, seed: int = 0, **kw) -> Policy:
    return {**POLICIES, **EXTRA_POLICIES}[name](resources, seed=seed, **kw)
