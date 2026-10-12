"""Discrete-event simulation of an edge-cloud-quantum deployment.

The scheduler only sees the *nominal* resource catalogue plus what a monitor
can observe (slot queues, ping RTT, provider-reported QPU queue). The
simulator executes every decision against a hidden *true* catalogue, with
per-task randomness, so a policy is rewarded for decisions that hold up in
reality rather than on paper.

Two optional mechanisms (experiment E7) sit on top of the per-task loop:

* QPU job batching (``BatchConfig``): QPU-bound tasks wait in an open batch
  that closes when it is full or its window expires. A closed batch is one
  provider job: one queue wait, one job overhead per round, one per-task fee;
  with spatial packing, members that fit side by side on the chip share shots
  and run at the same time.
* Race mode: when a policy hedges a QPU decision (``Decision.hedge``), the
  classical fallback runs in parallel and the QPU answer is kept only if it
  arrives by the deadline.

With neither in use, the loop draws random numbers in exactly the same order as
before they existed, so earlier results reproduce bit for bit. Batching reuses
each task's own draws (the batch takes its first member's queue wait and
jitter), and fallback runs draw from a separate stream, so every variant sees
the same random numbers task by task.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..orchestrator.estimator import CostModel, Estimate
from ..orchestrator.policies import Decision, Policy, make_policy
from ..resources import Resource, ResourceState, load_catalogue, scale
from ..workloads import Workload, generate

DEFAULT_QPU_QUEUE_S = 120.0


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
    queue_trace: list[float] | None = None               # measured QPU waits (s) to sample from
    queue_dist: str = "exponential"                      # true QPU wait: "exponential" | "lognormal" (same mean)
    queue_sigma: float = 1.5                             # lognormal shape when queue_dist == "lognormal"


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

# Mix for the batching experiment: more of the kinds that have a quantum formulation.
QUANTUM_HEAVY_MIX = {"inference": 0.15, "training": 0.05, "optimization": 0.30,
                     "kernel_classification": 0.10, "quantum_simulation": 0.40}

# E7 scenarios. Kept out of SCENARIOS so the E3 matrix (which iterates SCENARIOS) is unchanged.
EXTRA_SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario("future_qpu_heavy_tail", "Hypothetical future QPU with future_qpu's mean queue, but the real wait is "
             "lognormal (sigma 1.5, same mean): most jobs start sooner, about 1% wait 10x the mean or more. "
             "The provider publishes the true mean.",
             belief_scale=dict(SCENARIOS["future_qpu"].belief_scale), queue_dist="lognormal", queue_sigma=1.5),
    Scenario("quantum_heavy_future", "Hypothetical future QPU with a workload mix weighted toward quantum simulation "
             "and optimisation and 3x the arrival rate, so enough work reaches the QPU to batch.",
             belief_scale=dict(SCENARIOS["future_qpu"].belief_scale), rate_per_s=0.6, mix=QUANTUM_HEAVY_MIX),
]}


def trace_scenario(path: str | None = None) -> Scenario:
    """Scenario whose QPU queue is sampled from a logged IBM trace
    (``python -m qedge calibration log``) instead of an assumed distribution."""
    from ..quantum.calibration import TRACE_PATH, queue_trace_seconds
    waits = queue_trace_seconds(path or TRACE_PATH)
    if not waits:
        raise FileNotFoundError(f"no logged queue data in {path or TRACE_PATH}; run `python -m qedge calibration log` "
                                "with QISKIT_IBM_TOKEN set")
    mean = float(np.mean(waits))
    return Scenario("ibm_trace", f"QPU queue drawn from {len(waits)} logged IBM observations (mean {mean:.0f}s); "
                    "the scheduler is told the trace mean.",
                    belief_scale={"qpu__queue_mean_s": mean / DEFAULT_QPU_QUEUE_S}, queue_trace=waits)


def get_scenario(name: str) -> Scenario:
    if name == "ibm_trace":
        return trace_scenario()
    return SCENARIOS[name] if name in SCENARIOS else EXTRA_SCENARIOS[name]


@dataclass
class BatchConfig:
    """QPU job batching (E7). Only tasks placed on a real QPU are batched.

    A batch closes when it holds ``max_tasks`` tasks or ``window_s`` after its
    first task arrived; expiry is checked in arrival order, so an expired batch
    closes, at its expiry time, when the next arrival is processed (open batches
    are flushed at the end). A closed batch is one job: one queue wait, one job
    overhead and one per-task fee per round. Multi-round members (QAOA) run in
    lockstep: one queue wait, then rounds back to back, each round a single job
    holding every member that still has rounds left.

    The policy gets feedback on batched tasks when the batch closes, not when
    they arrive, so it learns later and decides tasks arriving inside the window
    on older information. (Unbatched tasks give feedback at decision time, the
    simulator's existing simplification.)
    """

    window_s: float = 30.0
    max_tasks: int = 8
    spatial: bool = True             # members that fit run side by side on disjoint qubit regions
    buffer_overhead: float = 0.25    # idle qubits fencing each program off, as a fraction of its width (assumption)
    partial_packing: bool = True     # if the whole batch does not fit, pack subsets first-fit decreasing
                                     # (an extension); False = the strict rule: all side by side or all back to back
    pack_fidelity_factor: float | Callable[[int], float] = 1.0   # x fidelity when k programs share the chip
    inform_policy: bool = True       # the policy sees the expected wait for the batch to close as QPU slot wait

    def footprint(self, n_qubits: int) -> int:
        return n_qubits + math.ceil(self.buffer_overhead * n_qubits)

    def fidelity_factor(self, k: int) -> float:
        if k < 2:
            return 1.0
        f = self.pack_fidelity_factor
        return float(f(k)) if callable(f) else float(f)


@dataclass
class _Fallback:
    """Outcome of the classical run started next to a hedged QPU job."""
    latency_s: float
    quality: float
    cost_usd: float
    energy_j: float
    edge_energy_j: float


@dataclass
class _Member:
    """A QPU-bound task waiting for (or running in) a job, with its own draws."""
    i: int                      # row position
    w: Workload
    d: Decision
    truth: Estimate
    noise: float
    qnoise: float
    queue_draw: float           # this task's provider-queue draw; a batch uses its first member's
    fallback: _Fallback | None = None

    @property
    def t(self) -> float:
        return self.w.arrival_s

    @property
    def rounds(self) -> int:
        return self.w.quantum.n_rounds

    def b(self, key: str) -> float:
        return self.truth.breakdown.get(key, 0.0)


@dataclass
class _Batch:
    resource: str
    opened_s: float
    members: list[_Member] = field(default_factory=list)


@dataclass
class _Plan:
    """How a job's members execute once it leaves the queue (times before jitter)."""
    done_s: dict[int, float]    # member -> completion time after execution starts
    money: dict[int, float]     # member -> share of the job's QPU charges
    energy: dict[int, float]    # member -> share of the QPU's execution energy
    group: dict[int, int]       # member -> programs sharing the chip with it at once (1 = alone)
    busy_s: float               # device time the job holds


@dataclass
class _RunState:
    policy: Policy
    free: dict[str, list[float]]
    rows: list
    extras: bool
    jobs: int = 0

    def job_id(self) -> int:
        self.jobs += 1
        return self.jobs - 1


def _qpu_charge(w: Workload, q: Resource) -> float:
    """QPU money for one task run alone (mirrors CostModel._quantum for real QPUs)."""
    qp = w.quantum
    return qp.n_rounds * q.cost_per_task + qp.n_circuits * qp.shots * q.cost_per_shot


class Simulator:
    def __init__(self, scenario: Scenario | str = "baseline", catalogue: dict[str, Resource] | None = None,
                 seed: int = 0, batching: BatchConfig | None = None, record_extras: bool = False):
        self.sc = get_scenario(scenario) if isinstance(scenario, str) else scenario
        base = catalogue or load_catalogue()
        self.nominal = scale(base, **self.sc.belief_scale)
        self.true = scale(self.nominal, **self.sc.true_scale)
        self.true_model = CostModel(self.true)
        self.seed = seed
        self.batching = batching
        # Batch / hedge columns are added whenever batching or a hedging policy is used, or on request.
        self.record_extras = record_extras
        self.qpus = {n for n, r in self.true.items() if r.is_quantum and not r.sim_gflops}

    def workloads(self, n: int) -> list[Workload]:
        return generate(n, self.sc.rate_per_s, self.sc.mix, seed=self.seed)

    def run(self, policy: Policy | str, workloads: list[Workload]) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + 1)
        if isinstance(policy, str):
            policy = make_policy(policy, copy.deepcopy(self.nominal), seed=self.seed)
        bc = self.batching
        hedging = bool(getattr(policy, "hedges", False))   # policies that may hedge declare hedges = True
        # Fallback runs get their own stream so the main one stays aligned with unhedged runs.
        aux = np.random.default_rng([self.seed, 0xE7])
        free = {n: [0.0] * r.slots for n, r in self.true.items()}
        rs = _RunState(policy, free, [None] * len(workloads), bc is not None or hedging or self.record_extras)
        open_b: dict[str, _Batch] = {}
        for i, w in enumerate(workloads):
            t = w.arrival_s
            if bc is not None:
                for q, b in sorted(open_b.items(), key=lambda kv: kv[1].opened_s):
                    if b.opened_s + bc.window_s <= t:
                        self._close(rs, open_b.pop(q), b.opened_s + bc.window_s)
            qpu_true_mean = {n: r.queue_mean_s for n, r in self.true.items() if r.is_quantum}
            observed = ResourceState(
                now=t,
                queue_wait_s={n: max(0.0, min(s) - t) for n, s in free.items()},
                rtt_s={n: r.rtt_s * float(rng.lognormal(0, 0.1)) for n, r in self.true.items() if not r.is_edge},
                qpu_queue_s={n: m * self.sc.qpu_report_bias for n, m in qpu_true_mean.items()},
            )
            slot_wait = observed.queue_wait_s
            if bc is not None and bc.inform_policy:
                observed.queue_wait_s = {n: v + (self._close_wait(open_b.get(n), t) if n in self.qpus else 0.0)
                                         for n, v in slot_wait.items()}
            d = policy.decide(w, observed)
            # --- ground truth ---------------------------------------------------
            truth_state = copy.copy(observed)
            truth_state.queue_wait_s = slot_wait
            truth_state.rtt_s = {n: r.rtt_s for n, r in self.true.items() if not r.is_edge}
            if self.sc.queue_trace:
                truth_state.qpu_queue_s = {n: float(rng.choice(self.sc.queue_trace)) for n in qpu_true_mean}
            elif self.sc.queue_dist == "lognormal":
                truth_state.qpu_queue_s = {n: self._heavy_tail_wait(rng, n, m) for n, m in qpu_true_mean.items()}
            else:
                truth_state.qpu_queue_s = {n: float(rng.exponential(m)) if m > 0 else 0.0
                                           for n, m in qpu_true_mean.items()}
            truth = self.true_model.estimate(w, d.placement, truth_state)
            noise = float(rng.lognormal(0, self.sc.noise_sigma))
            qnoise = float(rng.normal(0, 0.01))
            q = d.placement.quantum
            if q in self.qpus and (bc is not None or d.hedge is not None):
                m = _Member(i, w, d, truth, noise, qnoise, truth_state.qpu_queue_s[q])
                if d.hedge is not None:
                    m.fallback = self._run_fallback(rs, w, d, truth_state, aux)
                if bc is None:
                    self._close(rs, _Batch(q, t, [m]), t)
                else:
                    b = open_b.setdefault(q, _Batch(q, t))
                    b.members.append(m)
                    if len(b.members) >= bc.max_tasks:
                        self._close(rs, open_b.pop(q), t)
                continue
            # --- one task, one job (the original model) --------------------------
            wait = truth.breakdown.get("queue", 0.0)
            latency = wait + (truth.latency_s - wait) * noise
            quality = float(np.clip(truth.quality + qnoise, 0, 1))
            if d.placement.back is not None:
                _occupy_split(free, t, d.placement, truth.breakdown, noise)
            else:
                busy_res = d.placement.quantum or d.placement.classical
                busy_for = (truth.breakdown.get("q_rounds", 0) + truth.breakdown.get("q_exec", 0)) \
                    if d.placement.quantum else truth.breakdown.get("compute", 0.0) * noise
                slots = free[busy_res]
                k = int(np.argmin(slots))
                start = max(t, slots[k]) + (truth.breakdown.get("transfer", 0.0) if not d.placement.quantum else 0.0)
                slots[k] = start + busy_for
            outcome = {"latency_s": latency, "quality": quality}
            if q in self.qpus:
                outcome["qpu_queue_s"] = truth_state.qpu_queue_s[q] * noise
            policy.observe(d, outcome)
            row = self._row(policy, w, d, truth.feasible, latency, quality, truth.edge_energy_j * noise,
                            truth.energy_j * noise, truth.cost_usd)
            if rs.extras:
                on_qpu = q in self.qpus
                row.update(batch_id=rs.job_id() if on_qpu else -1, batch_size=int(on_qpu), pack_group=int(on_qpu),
                           close_wait_s=0.0, qpu_queue_s=outcome.get("qpu_queue_s", np.nan),
                           qpu_cost_usd=_qpu_charge(w, self.true[q]) if on_qpu else 0.0, qpu_cancelled=False,
                           hedged=False, hedge_placement="", winner="", fallback_latency_s=np.nan,
                           wasted_cost_usd=0.0, wasted_energy_j=0.0,
                           hedge_p_miss=d.hedge_p_miss if d.hedge_p_miss is not None else np.nan)
            rs.rows[i] = row
        for q, b in sorted(open_b.items(), key=lambda kv: kv[1].opened_s):
            self._close(rs, b, b.opened_s + bc.window_s)
        return pd.DataFrame(rs.rows)

    # ---- helpers --------------------------------------------------------------
    def _row(self, policy, w, d, feasible, latency, quality, edge_energy, energy, cost) -> dict:
        return {
            "policy": policy.name, "scenario": self.sc.name, "id": w.id, "kind": w.kind, "arrival_s": w.arrival_s,
            "placement": d.placement.label, "hybrid": d.placement.is_hybrid,
            "quantum_resource": d.placement.quantum or "", "explored": d.explored,
            "feasible": feasible, "deadline_s": w.deadline_s, "latency_s": latency,
            "predicted_latency_s": d.chosen.latency_s, "deadline_met": latency <= w.deadline_s,
            "edge_energy_j": edge_energy, "energy_j": energy,
            "cost_usd": cost, "quality": quality, "min_quality": w.min_quality,
            "quality_met": quality >= w.min_quality, "privacy": w.privacy,
            "classical_quality": w.classical_quality,
        }

    def _heavy_tail_wait(self, rng, name: str, mean: float) -> float:
        """Lognormal wait with the given mean for real QPUs (median = mean * exp(-sigma^2 / 2));
        simulators keep their short exponential queue."""
        if mean <= 0:
            return 0.0
        if name not in self.qpus:
            return float(rng.exponential(mean))
        s = self.sc.queue_sigma
        return float(rng.lognormal(math.log(mean) - s * s / 2, s))

    def _close_wait(self, b: _Batch | None, t: float) -> float:
        """Wait a task arriving now should expect before its batch closes (worst case: full window)."""
        bc = self.batching
        if bc.max_tasks <= 1:
            return 0.0
        if b is None:
            return bc.window_s
        if len(b.members) + 1 >= bc.max_tasks:
            return 0.0
        return max(0.0, b.opened_s + bc.window_s - t)

    def _run_fallback(self, rs: _RunState, w: Workload, d: Decision, truth_state: ResourceState, aux) -> _Fallback:
        """Start the hedge's classical placement now; it always runs to completion and is always paid for."""
        tc = self.true_model.estimate(w, d.hedge, truth_state)
        noise = float(aux.lognormal(0, self.sc.noise_sigma))
        qnoise = float(aux.normal(0, 0.01))
        wait = tc.breakdown.get("queue", 0.0)
        latency = wait + (tc.latency_s - wait) * noise
        if d.hedge.back is not None:
            _occupy_split(rs.free, w.arrival_s, d.hedge, tc.breakdown, noise)
        else:
            slots = rs.free[d.hedge.classical]
            k = int(np.argmin(slots))
            start = max(w.arrival_s, slots[k]) + tc.breakdown.get("transfer", 0.0)
            slots[k] = start + tc.breakdown.get("compute", 0.0) * noise
        quality = float(np.clip(tc.quality + qnoise, 0, 1))
        # The fallback's outcome is observed like any classical run.
        rs.policy.observe(Decision(w, d.hedge, d.hedge_estimate, d.estimates, d.policy),
                          {"latency_s": latency, "quality": quality})
        return _Fallback(latency, quality, tc.cost_usd, tc.energy_j * noise, tc.edge_energy_j * noise)

    def _pack(self, run: list[_Member], q: Resource) -> list[list[_Member]]:
        """Groups of members that execute side by side; groups run back to back."""
        bc = self.batching
        if bc is None or not bc.spatial or len(run) == 1:
            return [[m] for m in run]
        foot = {m.i: bc.footprint(m.w.quantum.n_qubits) for m in run}
        if sum(foot.values()) <= q.max_qubits:
            return [list(run)]
        if not bc.partial_packing:
            return [[m] for m in run]
        groups: list[list[_Member]] = []
        room: list[int] = []
        for m in sorted(run, key=lambda m: -foot[m.i]):
            for g, r in enumerate(room):
                if foot[m.i] <= r:
                    groups[g].append(m)
                    room[g] -= foot[m.i]
                    break
            else:
                groups.append([m])
                room.append(q.max_qubits - foot[m.i])
        return groups

    @staticmethod
    def _group_round(act: list[_Member]) -> tuple[float, float]:
        """(shots, execution time) of one round of programs running side by side.

        Circuit j of every member runs inside one combined circuit, which needs the most shots
        and the slowest per-shot time among the members that have a circuit j; a member with
        fewer circuits drops out of the later combined circuits. Members of the same shape
        (circuits, rounds, shots) reduce to the slowest member's time and one member's shots."""
        if len({(m.w.quantum.n_circuits, m.rounds, m.w.quantum.shots) for m in act}) == 1:
            m0 = act[0]
            return m0.w.quantum.n_circuits * m0.w.quantum.shots / m0.rounds, max(m.b("q_exec") / m.rounds for m in act)
        circ = {m.i: m.w.quantum.n_circuits / m.rounds for m in act}           # circuits per round
        order = sorted(act, key=lambda m: -circ[m.i])
        shots = t = s_max = tau_max = 0.0
        for k, m in enumerate(order):
            qp = m.w.quantum
            s_max = max(s_max, qp.shots)
            tau_max = max(tau_max, m.b("q_exec") / (qp.n_circuits * qp.shots))  # seconds per shot
            step = circ[m.i] - (circ[order[k + 1].i] if k + 1 < len(order) else 0.0)
            shots += step * s_max
            t += step * s_max * tau_max
        return shots, t

    def _plan(self, run: list[_Member], q: Resource) -> _Plan:
        """Lockstep rounds: each round is one job with the members that still have rounds left.
        Within a group the members' circuits are run in combined circuits (``_group_round``), so
        shots are shared; groups, and members that are not packed, add up."""
        groups = self._pack(run, q)
        hop = {m.i: m.b("q_rounds") / m.rounds - q.job_overhead_s for m in run}  # host<->QPU hop per round
        n_rounds = max(m.rounds for m in run)
        t = exec_total = shots_total = 0.0
        done: dict[int, float] = {}
        for r in range(1, n_rounds + 1):
            exec_r = shots_r = hop_r = 0.0
            for g in groups:
                act = [m for m in g if m.rounds >= r]
                if act:
                    s, e = self._group_round(act)
                    exec_r += e
                    shots_r += s
                    hop_r = max(hop_r, max(hop[m.i] for m in act))
            t += q.job_overhead_s + hop_r + exec_r
            exec_total += exec_r
            shots_total += shots_r
            done.update({m.i: t for m in run if m.rounds == r})
        money = n_rounds * q.cost_per_task + shots_total * q.cost_per_shot
        alone = {m.i: _qpu_charge(m.w, q) for m in run}
        execs = {m.i: m.b("q_exec") for m in run}
        sa, se = sum(alone.values()), sum(execs.values())
        return _Plan(done_s=done,
                     money={i: money * a / sa if sa > 0 else money / len(run) for i, a in alone.items()},
                     energy={i: exec_total * q.power_w * e / se if se > 0 else 0.0 for i, e in execs.items()},
                     group={m.i: len(g) for g in groups for m in g},
                     busy_s=t)

    def _close(self, rs: _RunState, b: _Batch, t_close: float) -> None:
        """Submit a batch as one job and settle every member's outcome."""
        q = self.true[b.resource]
        ms = b.members
        nb = ms[0].noise           # one job, one jitter factor
        queue = ms[0].queue_draw   # one provider queue wait per job
        slots = rs.free[b.resource]
        k = int(np.argmin(slots))
        slot_wait = max(0.0, slots[k] - t_close)
        # The job leaves once every member's host share (upload, optimiser set-up) is done.
        submit = max([t_close] + [m.t + m.b("queue") + (m.b("transfer") + m.b("compute")) * m.noise for m in ms])
        exec_start = submit + (slot_wait + queue) * nb
        # A hedged member whose deadline passes while the job is still queued is withdrawn: none of its circuits run.
        withdrawn = {m.i for m in ms if m.fallback is not None and exec_start >= m.t + m.w.deadline_s}
        run = [m for m in ms if m.i not in withdrawn]
        plan = self._plan(run, q) if run else None
        if run:
            start = max(t_close, slots[k])
            slots[k] = start + plan.busy_s
        lat_q = {m.i: exec_start + plan.done_s[m.i] * nb - m.t for m in run}
        # Cancelled outright once no member still waits for it (all hedged and late); started jobs are still paid.
        cancelled = bool(run) and all(m.fallback is not None and lat_q[m.i] > m.w.deadline_s for m in run)
        job = rs.job_id()
        for m in ms:
            w, d, ran = m.w, m.d, m.i not in withdrawn
            group = plan.group[m.i] if ran else 0
            quality_q = self._packed_quality(m, group)
            quality_q = float(np.clip(quality_q + m.qnoise, 0, 1))
            share_money = plan.money[m.i] if ran else 0.0
            share_energy = plan.energy[m.i] if ran else 0.0
            host_cost = m.truth.cost_usd - _qpu_charge(w, q)
            host_energy = m.truth.energy_j - m.b("q_exec") * q.power_w
            q_cost = host_cost + share_money
            q_energy = (host_energy + share_energy) * m.noise
            q_edge = m.truth.edge_energy_j * m.noise
            latency_q = lat_q[m.i] if ran else math.inf
            fb = m.fallback
            if fb is None:
                latency, quality, winner, wasted_c, wasted_e = latency_q, quality_q, "", 0.0, 0.0
            elif latency_q <= w.deadline_s:
                latency, quality, winner, wasted_c, wasted_e = latency_q, quality_q, "quantum", fb.cost_usd, fb.energy_j
            else:
                # Progressive delivery: the classical answer is handed over when ready and stands
                # because the QPU answer did not arrive by the deadline.
                latency, quality, winner = fb.latency_s, fb.quality, "classical"
                wasted_c, wasted_e = q_cost, q_energy
            fbc, fbe, fbx = (fb.cost_usd, fb.energy_j, fb.edge_energy_j) if fb else (0.0, 0.0, 0.0)
            # Feedback at batch close. A cancelled job's latency is censored at the deadline and its quality
            # is unseen (the prediction is fed back, which leaves the learned quality offset unchanged).
            censored = not ran or cancelled
            if ran:
                wait_seen, wait_censored = queue * nb, False
            else:
                wait_seen, wait_censored = max(0.0, m.t + w.deadline_s - submit - slot_wait * nb), True
            rs.policy.observe(d, {"latency_s": w.deadline_s if censored else latency_q,
                                  "quality": d.chosen.quality if censored else quality_q,
                                  "qpu_queue_s": wait_seen, "qpu_queue_censored": wait_censored})
            row = self._row(rs.policy, w, d, m.truth.feasible, latency, quality, q_edge + fbx, q_energy + fbe,
                            q_cost + fbc)
            row.update(batch_id=job, batch_size=len(ms), pack_group=group, close_wait_s=t_close - m.t,
                       qpu_queue_s=queue * nb, qpu_cost_usd=share_money, qpu_cancelled=not ran,
                       hedged=fb is not None, hedge_placement=d.hedge.label if fb else "", winner=winner,
                       fallback_latency_s=fb.latency_s if fb else np.nan,
                       wasted_cost_usd=wasted_c, wasted_energy_j=wasted_e,
                       hedge_p_miss=d.hedge_p_miss if d.hedge_p_miss is not None else np.nan)
            rs.rows[m.i] = row

    def _packed_quality(self, m: _Member, group: int) -> float:
        f = self.batching.fidelity_factor(group) if self.batching is not None else 1.0
        if f == 1.0 or m.truth.fidelity is None:
            return m.truth.quality
        return CostModel.quantum_quality(m.w, m.truth.fidelity * f)


def _occupy_split(free: dict[str, list[float]], t: float, p, b: dict, noise: float) -> None:
    """Reserve the slots of a classical split placement (split_inference): the edge front is
    busy only for its share of the network, the upload then runs on the radio, and the cloud
    back is busy for the rest of the network once the activation has arrived. (Every other
    placement offsets its single slot by the transfer instead, which for an edge resource is 0.)"""
    front = free[p.classical]
    k = int(np.argmin(front))
    front[k] = max(t, front[k]) + b.get("compute", 0.0) * noise
    back = free[p.back]
    kb = int(np.argmin(back))
    back[kb] = max(front[k] + b.get("transfer", 0.0), back[kb]) + b.get("back_compute", 0.0) * noise


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
