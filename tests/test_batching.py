"""QPU batching and hedged (race-mode) execution in the simulator (E7).

The first block guards reproducibility: with batching off and the existing
policies, Simulator.run must return exactly what it returned before batching
and hedging existed. It checks against a verbatim copy of the old loop and
against fingerprints recorded from the old code.
"""
import copy
import dataclasses
import hashlib

import numpy as np
import pandas as pd
import pytest

from qedge import workloads as W
from qedge.orchestrator.estimator import CostModel, Placement
from qedge.orchestrator.policies import EXTRA_POLICIES, POLICIES, Decision, Policy, QEdgeHedgedPolicy, make_policy
from qedge.resources import ResourceState
from qedge.simulation.simulator import EXTRA_SCENARIOS, SCENARIOS, BatchConfig, Simulator, _qpu_charge, get_scenario
from qedge.workloads import make_workload

# workloads.CALIBRATION defaults at the commit the fingerprints were recorded on.
DEFAULT_CALIBRATION = {
    "qaoa_gain": {"a": 1.3, "b": 0.9}, "opt_c": 0.55, "qaoa_override": None,
    "sa_ratio": {"slope": 0.0, "at": 8, "value": 1.0, "floor": 0.95},
    "kernel_acc": {"quantum": 0.86, "rbf": 0.90}, "mps_chi_max": 1024, "entanglement_bits_per_step": 0.5,
}


@pytest.fixture
def default_calibration():
    """Other tests may load results/calibration.json into the global; pin the defaults here."""
    saved = copy.deepcopy(W.CALIBRATION)
    W.CALIBRATION.clear()
    W.CALIBRATION.update(copy.deepcopy(DEFAULT_CALIBRATION))
    yield
    W.CALIBRATION.clear()
    W.CALIBRATION.update(saved)


# ---------------------------------------------------------------- reproducibility

def legacy_run(sim: Simulator, policy, workloads) -> pd.DataFrame:
    """Simulator.run as of commit e673fc9, copied verbatim (self -> sim)."""
    from qedge.simulation.simulator import make_policy as mk
    rng = np.random.default_rng(sim.seed + 1)
    if isinstance(policy, str):
        policy = mk(policy, copy.deepcopy(sim.nominal), seed=sim.seed)
    free = {n: [0.0] * r.slots for n, r in sim.true.items()}
    rows = []
    for w in workloads:
        t = w.arrival_s
        qpu_true_mean = {n: r.queue_mean_s for n, r in sim.true.items() if r.is_quantum}
        observed = ResourceState(
            now=t,
            queue_wait_s={n: max(0.0, min(s) - t) for n, s in free.items()},
            rtt_s={n: r.rtt_s * float(rng.lognormal(0, 0.1)) for n, r in sim.true.items() if not r.is_edge},
            qpu_queue_s={n: m * sim.sc.qpu_report_bias for n, m in qpu_true_mean.items()},
        )
        d = policy.decide(w, observed)
        truth_state = copy.copy(observed)
        truth_state.rtt_s = {n: r.rtt_s for n, r in sim.true.items() if not r.is_edge}
        if sim.sc.queue_trace:
            truth_state.qpu_queue_s = {n: float(rng.choice(sim.sc.queue_trace)) for n in qpu_true_mean}
        else:
            truth_state.qpu_queue_s = {n: float(rng.exponential(m)) if m > 0 else 0.0
                                       for n, m in qpu_true_mean.items()}
        truth = sim.true_model.estimate(w, d.placement, truth_state)
        noise = float(rng.lognormal(0, sim.sc.noise_sigma))
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
            "policy": policy.name, "scenario": sim.sc.name, "id": w.id, "kind": w.kind, "arrival_s": t,
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


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_default_run_matches_legacy_loop_exactly(scenario):
    sim = Simulator(scenario, seed=0)
    wl = sim.workloads(80)
    for p in ["qedge_adaptive", "qedge_static", "quantum_first", "latency_greedy", "random"]:
        pd.testing.assert_frame_equal(sim.run(p, wl), legacy_run(sim, p, wl), check_exact=True)


# Recorded from the code before batching/hedging existed (commit e673fc9), 150 tasks, seed 0:
# [sha256(placements)[:16], deadlines met, sum latency, sum cost, sum energy, sum quality]
FINGERPRINTS = {
    "baseline/qedge_adaptive": ["4ef52d2a284a0ae3", 144, 4069.685701086406, 0.8274833980663179, 280605.5758819673, 145.18226017904396],
    "baseline/quantum_first": ["e7ec5a36e7a884b0", 34, 437472.02090147743, 54564.811366538874, 253770191.33782607, 130.90148707941688],
    "baseline/latency_greedy": ["086c6ea0b457af01", 137, 2967.025322678185, 0.06684012579649992, 44504.603803357655, 144.53581356963724],
    "qpu_busy/qedge_adaptive": ["4ef52d2a284a0ae3", 144, 4069.685701086406, 0.8274833980663179, 280605.5758819673, 145.18226017904396],
    "qpu_busy/quantum_first": ["e7ec5a36e7a884b0", 30, 499236.62559347344, 54564.811366538874, 253770191.33782607, 130.90148707941688],
    "qpu_busy/latency_greedy": ["086c6ea0b457af01", 137, 2967.025322678185, 0.06684012579649992, 44504.603803357655, 144.53581356963724],
    "future_qpu/qedge_adaptive": ["2765c61105719586", 144, 3714.2255906741284, 9.579171846633143, 701741.0310698741, 146.48826782649735],
    "future_qpu/quantum_first": ["e7ec5a36e7a884b0", 35, 431295.5604322779, 5456.513866538873, 253770191.33782607, 144.25961965002665],
    "future_qpu/latency_greedy": ["086c6ea0b457af01", 137, 2967.025322678185, 0.06684012579649992, 44504.603803357655, 144.53581356963724],
    "future_qpu_busy/qedge_adaptive": ["2765c61105719586", 144, 5194.2349685984345, 9.579171846633143, 701741.0310698741, 146.48826782649735],
    "future_qpu_busy/quantum_first": ["e7ec5a36e7a884b0", 31, 464922.9563201424, 5456.513866538873, 253770191.33782607, 144.25961965002665],
    "future_qpu_busy/latency_greedy": ["086c6ea0b457af01", 137, 2967.025322678185, 0.06684012579649992, 44504.603803357655, 144.53581356963724],
}


@pytest.mark.parametrize("scenario", ["baseline", "qpu_busy", "future_qpu", "future_qpu_busy"])
def test_default_run_matches_recorded_fingerprints(scenario, default_calibration):
    sim = Simulator(scenario, seed=0)
    wl = sim.workloads(150)
    for p in ["qedge_adaptive", "quantum_first", "latency_greedy"]:
        df = sim.run(p, wl)
        h, met, *sums = FINGERPRINTS[f"{scenario}/{p}"]
        assert hashlib.sha256("|".join(df.placement).encode()).hexdigest()[:16] == h
        assert int(df.deadline_met.sum()) == met
        got = [df.latency_s.sum(), df.cost_usd.sum(), df.energy_j.sum(), df.quality.sum()]
        assert got == pytest.approx(sums, rel=1e-12)


def test_registries_unchanged_and_new_names_resolve():
    assert list(POLICIES) == ["edge_only", "cloud_only", "quantum_first", "random", "latency_greedy",
                              "qedge_static", "qedge_adaptive"]
    assert list(SCENARIOS) == ["baseline", "network_congested", "edge_overload", "qpu_busy", "free_qpu_tier",
                               "future_qpu", "future_qpu_busy"]
    assert "qedge_hedged" in EXTRA_POLICIES and "qedge_hedged" not in POLICIES
    sim = Simulator("future_qpu")
    assert isinstance(make_policy("qedge_hedged", sim.nominal), QEdgeHedgedPolicy)
    for name in EXTRA_SCENARIOS:
        assert get_scenario(name).name == name and name not in SCENARIOS


@pytest.mark.parametrize("scenario", ["future_qpu", "quantum_heavy_future", "future_qpu_heavy_tail"])
def test_degenerate_batching_equals_one_job_per_task(scenario):
    """A window of 0 s (or batches of one) is no batching, up to float rounding."""
    sim = Simulator(scenario, seed=1)
    wl = sim.workloads(300)
    base = sim.run("qedge_adaptive", wl)
    for bc in [BatchConfig(window_s=0, max_tasks=1), BatchConfig(window_s=0, max_tasks=8)]:
        df = Simulator(scenario, seed=1, batching=bc).run("qedge_adaptive", wl)
        pd.testing.assert_frame_equal(base, df[base.columns], check_exact=False, rtol=1e-9)
    assert (base.quantum_resource == "qpu").any()


def test_hedged_policy_that_never_hedges_equals_adaptive():
    sim = Simulator("future_qpu_busy", seed=2)
    wl = sim.workloads(300)
    base = sim.run("qedge_adaptive", wl)
    df = sim.run(QEdgeHedgedPolicy(copy.deepcopy(sim.nominal), seed=2, threshold=1.0), wl)
    assert not df.hedged.any()
    pd.testing.assert_frame_equal(base, df[base.columns].assign(policy="qedge_adaptive"), check_exact=True)


def test_batching_changes_nothing_when_no_task_uses_the_qpu():
    sim = Simulator("baseline", seed=0)
    wl = sim.workloads(300)
    base = sim.run("qedge_adaptive", wl)
    df = Simulator("baseline", seed=0, batching=BatchConfig(120)).run("qedge_adaptive", wl)
    assert not (base.quantum_resource == "qpu").any()
    pd.testing.assert_frame_equal(base, df[base.columns], check_exact=True)


# ---------------------------------------------------------------- controlled batches

QUIET = dataclasses.replace(SCENARIOS["future_qpu"], name="quiet_future", noise_sigma=0.0)  # no jitter


class AlwaysQPU(Policy):
    """Test policy: every task to cloud_cpu+qpu, optionally hedged with one classical placement."""
    name = "always_qpu"
    hedges = True

    def __init__(self, resources, seed=0, hedge_with: str | None = None):
        super().__init__(resources, seed)
        self.hedge_with = hedge_with

    def decide(self, w, st):
        e = self.model.estimate(w, Placement("cloud_cpu", "qpu"), st)
        d = Decision(w, e.placement, e, [e], self.name)
        if self.hedge_with:
            h = self.model.estimate(w, Placement(self.hedge_with), st)
            d.hedge, d.hedge_estimate = h.placement, h
        return d


def qsim_tasks(arrivals, n=40, steps=4, deadline=1e5):
    ns = n if isinstance(n, list) else [n] * len(arrivals)
    out = []
    for i, (t, q) in enumerate(zip(arrivals, ns)):
        w = make_workload("quantum_simulation", i, seed=i, n=q, steps=steps, deadline_s=deadline)
        w.arrival_s = float(t)
        out.append(w)
    return out


def run_always(scenario, wl, batching=None, hedge_with=None):
    sim = Simulator(scenario, seed=0, batching=batching, record_extras=True)
    return sim.run(AlwaysQPU(copy.deepcopy(sim.nominal), hedge_with=hedge_with), wl)


def test_batch_closes_when_full():
    df = run_always("future_qpu", qsim_tasks([0, 1, 2, 3, 4, 5]), BatchConfig(window_s=100, max_tasks=3))
    assert df.batch_id.nunique() == 2
    assert (df.batch_id.iloc[:3].nunique(), df.batch_id.iloc[3:].nunique()) == (1, 1)
    assert df.close_wait_s.tolist() == [2, 1, 0, 2, 1, 0]
    assert (df.batch_size == 3).all()


def test_batch_closes_at_window_expiry_and_flushes_at_end():
    df = run_always("future_qpu", qsim_tasks([0, 10, 50, 55, 200]), BatchConfig(window_s=30, max_tasks=8))
    groups = df.groupby("batch_id").id.apply(list).tolist()
    assert groups == [[0, 1], [2, 3], [4]]
    # closed at expiry (30, 80) when the next arrival is processed; the last one flushed at 230
    assert df.close_wait_s.tolist() == [30, 20, 30, 25, 30]


def test_one_queue_draw_per_batch():
    wl = qsim_tasks([0, 1, 2, 3, 4, 5])
    alone = run_always("future_qpu", wl)
    batched = run_always("future_qpu", wl, BatchConfig(window_s=100, max_tasks=3))
    assert alone.batch_id.nunique() == 6
    # every member waits the batch's single queue draw: its first member's own draw
    assert batched.qpu_queue_s.iloc[:3].tolist() == [alone.qpu_queue_s.iloc[0]] * 3
    assert batched.qpu_queue_s.iloc[3:].tolist() == [alone.qpu_queue_s.iloc[3]] * 3
    done = batched.arrival_s + batched.latency_s       # one job, one completion time
    assert np.allclose(done.iloc[:3], done.iloc[0]) and np.allclose(done.iloc[3:], done.iloc[3])


def test_spatial_packing_runs_side_by_side_and_shares_shots():
    wl = qsim_tasks([0, 1], n=20)
    spatial = run_always(QUIET, wl, BatchConfig(window_s=10, max_tasks=2, spatial=True))
    seq = run_always(QUIET, wl, BatchConfig(window_s=10, max_tasks=2, spatial=False))
    sim = Simulator(QUIET)
    est = CostModel(sim.true).estimate(wl[0], Placement("cloud_cpu", "qpu"), ResourceState())
    q, qp = sim.true["qpu"], wl[0].quantum
    assert spatial.pack_group.tolist() == [2, 2] and seq.pack_group.tolist() == [1, 1]
    # back to back takes one more execution time than side by side
    gap = (seq.arrival_s + seq.latency_s) - (spatial.arrival_s + spatial.latency_s)
    assert np.allclose(gap, est.breakdown["q_exec"])
    # one fee per job either way; side by side pays the shots once
    shots = qp.n_circuits * qp.shots * q.cost_per_shot
    assert seq.qpu_cost_usd.sum() == pytest.approx(q.cost_per_task + 2 * shots)
    assert spatial.qpu_cost_usd.sum() == pytest.approx(q.cost_per_task + shots)
    assert seq.qpu_cost_usd.sum() < 2 * _qpu_charge(wl[0], q)


def test_programs_that_do_not_fit_run_back_to_back():
    wl = qsim_tasks([0, 1, 2], n=[60, 60, 10])   # footprints 75 + 75 + 13 > 133 qubits
    part = run_always(QUIET, wl, BatchConfig(window_s=10, max_tasks=3))
    allfit = run_always(QUIET, wl, BatchConfig(window_s=10, max_tasks=3, partial_packing=False))
    assert part.pack_group.tolist() == [2, 1, 2]      # first-fit decreasing: {60, 10} and {60}
    assert allfit.pack_group.tolist() == [1, 1, 1]
    two = run_always(QUIET, wl[:2], BatchConfig(window_s=10, max_tasks=2))
    assert two.pack_group.tolist() == [1, 1]


def test_packed_programs_of_different_shapes_pay_circuit_aligned_shots():
    """Kernel (310 circuits x 500 shots) next to TFIM (5 x 4000): the five combined circuits need 4000 shots
    each and the kernel's other 305 circuits 500 each, which is more than the larger member's shots alone."""
    k = make_workload("kernel_classification", 0, seed=0, features=4, n_train=20, deadline_s=1e5, privacy=False)
    s = make_workload("quantum_simulation", 1, seed=1, n=20, steps=4, deadline_s=1e5)
    k.arrival_s, s.arrival_s = 0.0, 1.0
    q = Simulator(QUIET).true["qpu"]
    df = run_always(QUIET, [k, s], BatchConfig(window_s=10, max_tasks=2))
    assert df.pack_group.tolist() == [2, 2]
    kc, ks, sc, ss = k.quantum.n_circuits, k.quantum.shots, s.quantum.n_circuits, s.quantum.shots
    assert (kc, ss) == (310, 4000)
    shots = sc * max(ks, ss) + (kc - sc) * ks
    assert df.qpu_cost_usd.sum() == pytest.approx(q.cost_per_task + shots * q.cost_per_shot)
    assert kc * ks < shots < kc * ks + sc * ss


def test_pack_fidelity_factor_lowers_packed_quality_only():
    wl = qsim_tasks([0, 1], n=20, steps=12)
    alone = run_always(QUIET, wl)
    same = run_always(QUIET, wl, BatchConfig(window_s=10, max_tasks=2))
    worse = run_always(QUIET, wl, BatchConfig(window_s=10, max_tasks=2, pack_fidelity_factor=lambda k: 0.9 ** k))
    assert same.quality.tolist() == alone.quality.tolist()
    assert (worse.quality < same.quality).all()


def test_multi_round_members_run_in_lockstep():
    """QAOA (26 rounds) with a 1-round job: one fee per round, and the short member is done after round 1."""
    qaoa = make_workload("optimization", 0, seed=0, n=12, deadline_s=1e5)
    tfim = make_workload("quantum_simulation", 1, seed=1, n=20, steps=4, deadline_s=1e5)
    qaoa.arrival_s, tfim.arrival_s = 0.0, 1.0
    q = Simulator(QUIET).true["qpu"]
    rounds = qaoa.quantum.n_rounds
    s_qaoa, s_tfim = qaoa.quantum.n_circuits * qaoa.quantum.shots, tfim.quantum.n_circuits * tfim.quantum.shots
    seq = run_always(QUIET, [qaoa, tfim], BatchConfig(window_s=10, max_tasks=2, spatial=False))
    assert seq.qpu_cost_usd.sum() == pytest.approx(rounds * q.cost_per_task + (s_qaoa + s_tfim) * q.cost_per_shot)
    # side by side (15 + 25 qubits fit): round 1 pays the larger shot count, rounds 2..26 QAOA's alone
    packed = run_always(QUIET, [qaoa, tfim], BatchConfig(window_s=10, max_tasks=2))
    shots = max(s_qaoa / rounds, s_tfim) + (rounds - 1) * s_qaoa / rounds
    assert packed.qpu_cost_usd.sum() == pytest.approx(rounds * q.cost_per_task + shots * q.cost_per_shot)
    for df in (seq, packed):
        done = (df.arrival_s + df.latency_s).tolist()
        assert done[1] < done[0] - (rounds - 1) * q.job_overhead_s


# ---------------------------------------------------------------- race mode

def test_hedged_task_never_misses_when_its_fallback_alone_would_meet_the_deadline():
    wins = 0
    for seed in (0, 1):
        sim = Simulator("future_qpu_busy", seed=seed)
        df = sim.run(QEdgeHedgedPolicy(copy.deepcopy(sim.nominal), seed=seed, threshold=0.0), sim.workloads(800))
        h = df[df.hedged]
        assert len(h)
        assert h[h.fallback_latency_s <= h.deadline_s].deadline_met.all()
        assert (h[h.winner == "quantum"].latency_s <= h[h.winner == "quantum"].deadline_s).all()
        cl = h[h.winner == "classical"]
        assert (cl.latency_s == cl.fallback_latency_s).all()
        wins += len(cl)
    assert wins > 0   # the QPU did miss sometimes, so the fallback was exercised


def test_qpu_job_cancelled_in_queue_costs_nothing():
    stuck = dataclasses.replace(QUIET, name="stuck_queue", queue_trace=[1e6])   # every job waits 1e6 s
    wl = qsim_tasks([0], deadline=3600)
    df = run_always(stuck, wl, hedge_with="cloud_gpu")
    r = df.iloc[0]
    assert r.hedged and r.qpu_cancelled and r.winner == "classical"
    assert r.qpu_cost_usd == 0.0
    assert r.wasted_cost_usd < 1e-3          # only the host's small classical share was spent
    assert r.wasted_energy_j < 1e3           # no 25 kW QPU execution energy
    sim = Simulator(stuck)
    fb = CostModel(sim.true).estimate(wl[0], Placement("cloud_gpu"), ResourceState())
    assert r.cost_usd - r.wasted_cost_usd == pytest.approx(fb.cost_usd)
    # without the hedge the same job runs (late) and is paid in full
    plain = run_always(stuck, wl).iloc[0]
    assert not plain.deadline_met and plain.qpu_cost_usd == pytest.approx(_qpu_charge(wl[0], sim.true["qpu"]))


class HedgeSome(AlwaysQPU):
    """AlwaysQPU that hedges only the listed task ids (with cloud_gpu) and records its feedback."""

    def __init__(self, resources, seed=0, hedge_ids=()):
        super().__init__(resources, seed, hedge_with="cloud_gpu")
        self.hedge_ids, self.outcomes = set(hedge_ids), []

    def decide(self, w, st):
        d = super().decide(w, st)
        if w.id not in self.hedge_ids:
            d.hedge = d.hedge_estimate = None
        return d

    def observe(self, decision, outcome):
        self.outcomes.append((decision.workload.id, decision.placement.quantum, outcome))


def test_hedged_member_withdrawn_from_a_queued_batch_pays_nothing_and_others_share_without_it():
    queued = dataclasses.replace(QUIET, name="queue_200", queue_trace=[200.0])   # every job waits 200 s
    wl = qsim_tasks([0, 1, 2], n=30)
    wl[0].deadline_s, wl[2].deadline_s = 100, 250      # task 0 expires while queued, task 2 does not
    sim = Simulator(queued, batching=BatchConfig(window_s=10, max_tasks=3))
    pol = HedgeSome(copy.deepcopy(sim.nominal), hedge_ids=(0, 2))
    df = sim.run(pol, wl)
    q = sim.true["qpu"]
    assert df.qpu_cancelled.tolist() == [True, False, False]
    assert df.winner.tolist() == ["classical", "", "quantum"] and df.deadline_met.all()
    assert df.qpu_cost_usd.iloc[0] == 0.0 and df.pack_group.tolist() == [0, 2, 2]
    # the two remaining 30-qubit programs share one fee and one set of shots
    shots = wl[1].quantum.n_circuits * wl[1].quantum.shots
    assert df.qpu_cost_usd.iloc[1:].sum() == pytest.approx(q.cost_per_task + shots * q.cost_per_shot)
    qpu_feedback = {i: o for i, res, o in pol.outcomes if res == "qpu"}
    assert qpu_feedback[0]["qpu_queue_censored"] and qpu_feedback[0]["latency_s"] == 100
    assert not qpu_feedback[2]["qpu_queue_censored"] and qpu_feedback[2]["qpu_queue_s"] == 200


def test_qpu_job_that_started_before_the_deadline_is_paid_even_if_late():
    late = dataclasses.replace(QUIET, name="late_queue", queue_trace=[95.0])  # starts at ~95 s, done ~103 s
    wl = qsim_tasks([0], deadline=100)
    r = run_always(late, wl, hedge_with="cloud_gpu").iloc[0]
    assert r.winner == "classical" and not r.qpu_cancelled
    assert r.qpu_cost_usd == pytest.approx(_qpu_charge(wl[0], Simulator(late).true["qpu"]))
    assert r.wasted_cost_usd >= r.qpu_cost_usd


def test_miss_probability_uses_exponential_queue_and_learns_bias():
    sim = Simulator("future_qpu")
    pol = QEdgeHedgedPolicy(copy.deepcopy(sim.nominal))
    w = make_workload("quantum_simulation", 0, seed=0, n=40, steps=4, deadline_s=100)
    st = ResourceState(qpu_queue_s={"qpu": 12.0})
    e = pol.model.estimate(w, Placement("cloud_cpu", "qpu"), st)
    b = e.breakdown
    slack = 100 - (b["queue"] + b["transfer"] + b["compute"] + b["q_rounds"] + b["q_exec"])
    assert pol.miss_probability(w, e, st) == pytest.approx(np.exp(-slack / 12.0))
    d = Decision(w, e.placement, e, [e], pol.name)
    e.breakdown["q_published"] = 12.0
    for _ in range(9):
        pol.observe(d, {"latency_s": 100.0, "quality": e.quality, "qpu_queue_s": 120.0})
    assert pol.queue_bias == pytest.approx((1 + 9 * 10) / 10)
    assert pol.miss_probability(w, e, st) > np.exp(-slack / 12.0)


def test_heavy_tail_queue_keeps_the_mean():
    sim = Simulator("future_qpu_heavy_tail")
    rng = np.random.default_rng(0)
    x = np.array([sim._heavy_tail_wait(rng, "qpu", 12.0) for _ in range(200_000)])
    assert x.mean() == pytest.approx(12.0, rel=0.03)
    assert np.median(x) == pytest.approx(12.0 * np.exp(-1.5 ** 2 / 2), rel=0.03)
    assert (x > 120).mean() > 100 * np.exp(-10)    # far more 10x-mean waits than an exponential queue


def test_e7_smoke(tmp_path, monkeypatch, default_calibration):   # restores the global calibration after
    from qedge.experiments import e7_batching as E
    monkeypatch.setattr(E, "SCENARIOS", ["baseline", "quantum_heavy_future"])
    monkeypatch.setattr(E, "FUTURE", ["quantum_heavy_future"])
    monkeypatch.setattr(E, "SWEEP_SCENARIOS", ["quantum_heavy_future"])
    monkeypatch.setattr(E, "THRESHOLDS", (0.05, 1.0))
    res = E.run(quick=True)
    for name, df in res.items():
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    s = res["e7_batching_summary"]
    assert s[s.scenario == "baseline"].deadline_hit_rate.nunique() == 1   # nothing changes without QPU work
    assert any(v.startswith("hedged+batch") for v in s.variant)
    assert len(E.make_figures(str(tmp_path), str(tmp_path))) == 2
