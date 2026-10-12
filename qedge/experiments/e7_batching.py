"""E7: does batching QPU jobs, or racing a classical fallback against them, pay off?

Both ideas work around the QPU's overheads (fixed per-job costs, queue risk)
rather than its speed. Batching shares
one provider job (one queue wait, one job overhead, one per-task fee) among
several tasks and, when their programs fit side by side on the chip, shares
shots too; the price is the time a task waits for its batch to close. Race mode
(``qedge_hedged``) starts a classical placement next to a QPU job whose queue
might blow the deadline and keeps whichever answer is usable in time; the
price is classical compute that is thrown away whenever the QPU answer is on
time. Neither makes the QPU itself better: the quantum part still has to earn
its place through the cost model's quality gate.

Variants, all on the same workloads and random numbers per (scenario, seed):

* ``adaptive``: qedge_adaptive, one job per task (the E3 policy);
* ``adaptive+batch{5,30,120}``: windows of 5/30/120 s, up to 8 tasks, spatial packing
  of first-fit-decreasing subsets (an extension beyond the strict packing rule);
* ``adaptive+batch{5,30,120}-allfit``: the strict rule, all members side by side or
  all back to back. With 34-79 qubit programs on 133 qubits whole batches rarely fit,
  so most of the batching saving depends on subset packing; compare these rows;
* diagnostics at 30 s: ``-seq`` (no spatial packing: members run back to back),
  ``-blind`` (the policy is not told about the batch window);
* ``hedged``: qedge_hedged, P(miss) threshold 0.05;
* ``hedged+batchW``: hedging plus the best window W, chosen by ``best_window``.

``future_qpu`` scenarios are hypothetical hardware; ``baseline`` is a sanity
check (today's QPU is never chosen, so every variant must agree). A separate
sweep varies the hedging threshold. QPU-task rates rest on few tasks per run
(see the qpu_tasks column), so differences of a task or two are noise.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd

from ..orchestrator.policies import make_policy
from ..simulation.simulator import BatchConfig, Simulator
from ..workloads import load_calibration

SCENARIOS = ["baseline", "future_qpu", "future_qpu_busy", "future_qpu_heavy_tail", "quantum_heavy_future"]
FUTURE = SCENARIOS[1:]
WINDOWS = (5, 30, 120)
MAX_TASKS = 8
HIT_TOLERANCE = 0.005     # best_window: max loss of overall deadline hit rate vs no batching
THRESHOLDS = (0.0, 0.001, 0.01, 0.05, 0.2, 1.0)
SWEEP_SCENARIOS = ["future_qpu_busy", "future_qpu_heavy_tail", "quantum_heavy_future"]


def variants(best: float | None = None) -> dict[str, tuple[str, BatchConfig | None]]:
    v = {"adaptive": ("qedge_adaptive", None)}
    v.update({f"adaptive+batch{w}": ("qedge_adaptive", BatchConfig(w, MAX_TASKS)) for w in WINDOWS})
    v.update({f"adaptive+batch{w}-allfit": ("qedge_adaptive", BatchConfig(w, MAX_TASKS, partial_packing=False))
              for w in WINDOWS})
    v["adaptive+batch30-seq"] = ("qedge_adaptive", BatchConfig(30, MAX_TASKS, spatial=False))
    v["adaptive+batch30-blind"] = ("qedge_adaptive", BatchConfig(30, MAX_TASKS, inform_policy=False))
    v["hedged"] = ("qedge_hedged", None)
    if best is not None:
        v[f"hedged+batch{best:g}"] = ("qedge_hedged", BatchConfig(best, MAX_TASKS))
    return v


def metrics(df: pd.DataFrame) -> dict:
    """Per-run metrics. QPU tasks are those the policy routed to the QPU (hedged ones included)."""
    q = df[df.quantum_resource == "qpu"]
    h = q[q.hedged]
    nq, n = len(q), len(df)
    jobs = q.batch_id.nunique()
    return {
        "tasks": n, "qpu_tasks": nq,
        "deadline_hit_rate": df.deadline_met.mean(),
        "qpu_deadline_hit_rate": q.deadline_met.mean() if nq else np.nan,
        "mean_latency_s": df.latency_s.mean(), "p95_latency_s": df.latency_s.quantile(0.95),
        "qpu_mean_latency_s": q.latency_s.mean() if nq else np.nan,
        "qpu_mean_close_wait_s": q.close_wait_s.mean() if nq else np.nan,
        "cost_usd_per_task": df.cost_usd.mean(),
        "qpu_cost_usd": q.qpu_cost_usd.sum(),
        "qpu_cost_per_qpu_task": q.qpu_cost_usd.sum() / nq if nq else np.nan,
        "qpu_jobs": jobs, "mean_batch_size": nq / jobs if jobs else np.nan,
        "mean_programs_per_chip": q.loc[~q.qpu_cancelled, "pack_group"].mean() if nq else np.nan,
        "hedged_tasks": len(h), "hedge_rate": len(h) / nq if nq else np.nan,
        "fallback_wins": int((h.winner == "classical").sum()),
        "fallback_win_rate": (h.winner == "classical").mean() if len(h) else np.nan,
        "qpu_cancelled_in_queue": int(q.qpu_cancelled.sum()),
        "wasted_cost_usd_per_task": df.wasted_cost_usd.sum() / n,
        "wasted_energy_j_per_task": df.wasted_energy_j.sum() / n,
        "energy_kj_per_task": df.energy_j.mean() / 1e3,
        "mean_quality": df.quality.mean(), "quality_hit_rate": df.quality_met.mean(),
    }


def _run(sim: Simulator, wl, policy: str, bc: BatchConfig | None, **policy_kw) -> pd.DataFrame:
    sim.batching = bc
    pol = make_policy(policy, copy.deepcopy(sim.nominal), seed=sim.seed, **policy_kw)
    return sim.run(pol, wl)


def summarise(per_run: pd.DataFrame, keys=("scenario", "variant")) -> pd.DataFrame:
    """Pool seeds: counts are summed and rates recomputed from them; per-run means are averaged
    (equal task counts per run); *_sd is the spread across seeds."""
    keys = list(keys)
    g = per_run.groupby(keys, sort=False)
    out = g[["tasks", "qpu_tasks", "qpu_jobs", "hedged_tasks", "fallback_wins", "qpu_cancelled_in_queue",
             "qpu_cost_usd"]].sum()
    out.insert(0, "seeds", g.size())
    for c in ["deadline_hit_rate", "mean_latency_s", "p95_latency_s", "cost_usd_per_task", "wasted_cost_usd_per_task",
              "wasted_energy_j_per_task", "energy_kj_per_task", "mean_quality", "quality_hit_rate"]:
        out[c] = g[c].mean()
    out["deadline_hit_rate_sd"] = g.deadline_hit_rate.std()

    def pooled(col, weight):
        return g.apply(lambda d: np.average(d[col].fillna(0), weights=d[weight]) if d[weight].sum() else np.nan,
                       include_groups=False)

    out["qpu_deadline_hit_rate"] = pooled("qpu_deadline_hit_rate", "qpu_tasks")
    out["qpu_mean_latency_s"] = pooled("qpu_mean_latency_s", "qpu_tasks")
    out["qpu_mean_close_wait_s"] = pooled("qpu_mean_close_wait_s", "qpu_tasks")
    out["qpu_cost_per_qpu_task"] = out.qpu_cost_usd / out.qpu_tasks.replace(0, np.nan)
    out["mean_batch_size"] = out.qpu_tasks / out.qpu_jobs.replace(0, np.nan)
    out["qpu_jobs_per_run"] = out.qpu_jobs / out.seeds
    out["qpu_tasks_per_run"] = out.qpu_tasks / out.seeds
    out["hedge_rate"] = out.hedged_tasks / out.qpu_tasks.replace(0, np.nan)
    out["fallback_win_rate"] = out.fallback_wins / out.hedged_tasks.replace(0, np.nan)
    out = out.reset_index()
    if "variant" in keys:
        ref = out[out.variant == "adaptive"].set_index("scenario")
        out["d_deadline_hit_pp"] = 100 * (out.deadline_hit_rate - out.scenario.map(ref.deadline_hit_rate))
        out["d_qpu_cost_per_qpu_task_pct"] = 100 * (out.qpu_cost_per_qpu_task
                                                    / out.scenario.map(ref.qpu_cost_per_qpu_task) - 1)
        out["d_mean_latency_s"] = out.mean_latency_s - out.scenario.map(ref.mean_latency_s)
    return out


def best_window(summary: pd.DataFrame) -> float:
    """Window for hedged+batch: across the future scenarios, the lowest QPU cost per QPU task
    among windows whose mean overall deadline hit rate is within HIT_TOLERANCE of no batching;
    if none qualifies, the window with the highest hit rate."""
    s = summary[summary.scenario.isin(FUTURE)]
    ref = s[s.variant == "adaptive"].deadline_hit_rate.mean()
    rows = []
    for w in WINDOWS:
        d = s[s.variant == f"adaptive+batch{w}"]
        rows.append((w, d.deadline_hit_rate.mean(), d.qpu_cost_usd.sum() / d.qpu_tasks.sum()))
    ok = [r for r in rows if r[1] >= ref - HIT_TOLERANCE]
    return float(min(ok, key=lambda r: r[2])[0] if ok else max(rows, key=lambda r: r[1])[0])


def run(quick: bool = False) -> dict[str, pd.DataFrame]:
    load_calibration()
    seeds, n = ((0,), 300) if quick else ((0, 1, 2), 800)
    cache = {}
    rows = []

    def record(sc, seed, name, policy, bc, **kw):
        sim, wl = cache[(sc, seed)]
        df = _run(sim, wl, policy, bc, **kw)
        rows.append({"scenario": sc, "variant": name, "policy": policy, "window_s": bc.window_s if bc else 0.0,
                     "max_tasks": bc.max_tasks if bc else 1, "spatial": bool(bc and bc.spatial),
                     "seed": seed, **metrics(df)})

    for sc in SCENARIOS:
        for seed in seeds:
            sim = Simulator(sc, seed=seed, record_extras=True)
            cache[(sc, seed)] = (sim, sim.workloads(n))
            for name, (policy, bc) in variants().items():
                record(sc, seed, name, policy, bc)
        print(f"[e7] {sc}: {len(variants())} variants x {len(seeds)} seeds", flush=True)
    best = best_window(summarise(pd.DataFrame(rows)))
    print(f"[e7] best batching window for hedged+batch: {best:g}s", flush=True)
    for sc in SCENARIOS:
        for seed in seeds:
            record(sc, seed, f"hedged+batch{best:g}", "qedge_hedged", BatchConfig(best, MAX_TASKS))
    per_run = pd.DataFrame(rows)

    # Hedging threshold sweep, no batching. Threshold 1.0 never hedges (= qedge_adaptive).
    sweep = []
    for sc in SWEEP_SCENARIOS:
        for seed in seeds:
            sim, wl = cache[(sc, seed)]
            for thr in THRESHOLDS:
                df = _run(sim, wl, "qedge_hedged", None, threshold=thr)
                sweep.append({"scenario": sc, "threshold": thr, "seed": seed, **metrics(df)})
    thr = summarise(pd.DataFrame(sweep), keys=("scenario", "threshold"))
    return {"e7_batching": per_run, "e7_batching_summary": summarise(per_run), "e7_hedge_threshold": thr}


# ---------------------------------------------------------------- figures
def _style(best: str):
    from .figures import AQUA, GRAY, ORANGE
    return {"adaptive": ("no batching", GRAY), "adaptive+batch5": ("batch 5 s", "#86b6ef"),
            "adaptive+batch30": ("batch 30 s", "#2a78d6"), "adaptive+batch120": ("batch 120 s", "#184f95"),
            "hedged": ("hedged", ORANGE), best: (f"hedged + batch {best.split('batch')[1]} s", AQUA)}


SC_LABEL = {"baseline": "baseline\n(today's QPU)", "future_qpu": "future QPU", "future_qpu_busy": "future QPU,\nbusy",
            "future_qpu_heavy_tail": "future QPU,\nheavy-tail queue", "quantum_heavy_future": "future QPU,\nquantum-heavy mix"}


def make_figures(res_dir: str = "results", fig_dir: str = "figures") -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    from .figures import INK, MUTED, _save
    os.makedirs(fig_dir, exist_ok=True)
    s = pd.read_csv(os.path.join(res_dir, "e7_batching_summary.csv"))
    per_run = pd.read_csv(os.path.join(res_dir, "e7_batching.csv"))
    best = next(v for v in s.variant.unique() if v.startswith("hedged+batch"))
    style = _style(best)
    names = list(style)
    k = int(s.seeds.max())
    out = []

    def legend(fig, ax):
        h, lab = ax.get_legend_handles_labels()
        order = sorted(range(len(lab)), key=lambda i: lab[i] == "strict packing rule")   # markers last
        h, lab = [h[i] for i in order], [lab[i] for i in order]
        fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(h), fontsize=9)

    # 1. Deadline hit rate: all tasks (per-seed range) and QPU-routed tasks (pooled over seeds).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.8 / len(names)
    for ax, col, title in [(axes[0], "deadline_hit_rate", "All tasks"),
                           (axes[1], "qpu_deadline_hit_rate", "Tasks routed to the QPU")]:
        scs = SCENARIOS if col == "deadline_hit_rate" else FUTURE
        for j, v in enumerate(names):
            label, color = style[v]
            x = np.arange(len(scs)) + (j - (len(names) - 1) / 2) * width
            y = [s[(s.scenario == sc) & (s.variant == v)][col].iloc[0] for sc in scs]
            if col == "deadline_hit_rate":
                r = per_run[per_run.variant == v].groupby("scenario")[col]
                lo = [y[i] - r.min()[sc] for i, sc in enumerate(scs)]
                hi = [r.max()[sc] - y[i] for i, sc in enumerate(scs)]
                ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=1, capsize=0)
            ax.scatter(x, y, s=42, color=color, edgecolor="white", linewidth=1, zorder=3, label=label)
        ax.set_xticks(range(len(scs)))
        labels = [SC_LABEL[sc] for sc in scs]
        if col != "deadline_hit_rate":
            nq = s[s.variant == "adaptive"].set_index("scenario").qpu_tasks_per_run
            labels = [f"{lab}\n~{nq[sc]:.0f} QPU tasks/run" for lab, sc in zip(labels, scs)]
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, loc="left")
        ax.set_ylabel("deadline hit rate")
    axes[0].text(0, 1.0, f"dots: mean of {k} seeds; whiskers: seed range (axis does not start at 0)",
                 transform=axes[0].transAxes, fontsize=8, color=MUTED, va="bottom")
    axes[1].text(0, 1.0, f"pooled over {k} seeds; few tasks, so one miss moves a dot by several points",
                 transform=axes[1].transAxes, fontsize=8, color=MUTED, va="bottom")
    axes[0].set_title("All tasks\n", loc="left")
    axes[1].set_title("Tasks routed to the QPU\n", loc="left")
    legend(fig, axes[0])
    p = os.path.join(fig_dir, "e7_deadline_hit.png")
    _save(fig, p)
    out.append(p)

    # 2. QPU money per QPU task, and what batching costs in QPU-task latency.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x0 = np.arange(len(FUTURE))
    for j, v in enumerate(names):
        label, color = style[v]
        x = x0 + (j - (len(names) - 1) / 2) * width
        d = s[s.variant == v].set_index("scenario")
        axes[0].bar(x, d.qpu_cost_per_qpu_task[FUTURE], width * 0.9, color=color, label=label)
        strict = s[s.variant == f"{v}-allfit"].set_index("scenario")
        if len(strict):
            axes[0].scatter(x, strict.qpu_cost_per_qpu_task[FUTURE], marker="_", s=110, linewidth=2, color=INK,
                            zorder=3, label="strict packing rule" if v == "adaptive+batch5" else None)
        # Log scale has no zero baseline, so latency is drawn as dots, not bars.
        axes[1].scatter(x, d.qpu_mean_latency_s[FUTURE], s=42, color=color, edgecolor="white", linewidth=1,
                        zorder=3, label=label)
    for ax in axes:
        ax.set_xticks(x0)
        ax.set_xticklabels([SC_LABEL[sc] for sc in FUTURE], fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_title("QPU charges per QPU-routed task\n\n", loc="left")
    axes[0].set_ylabel("USD per task (hypothetical future prices)")
    axes[1].set_title("Mean latency of QPU-routed tasks\n\n", loc="left")
    axes[1].set_ylabel("seconds (log scale)")
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    lo, hi = axes[1].get_ylim()
    axes[1].set_yticks([t for t in (10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000) if lo <= t <= hi])
    axes[0].text(0, 1.0, "bars: subsets of a batch packed side by side; black ticks: strict rule\n(whole batch "
                 "side by side, or none). Baseline omitted: no task goes to today's QPU", transform=axes[0].transAxes,
                 fontsize=8, color=MUTED, va="bottom")
    axes[1].text(0, 1.0, "includes the wait for a batch to close", transform=axes[1].transAxes,
                 fontsize=8, color=MUTED, va="bottom")
    legend(fig, axes[0])
    p = os.path.join(fig_dir, "e7_qpu_cost.png")
    _save(fig, p)
    out.append(p)
    return out


if __name__ == "__main__":
    import sys
    res = run(quick="--quick" in sys.argv)
    os.makedirs("results", exist_ok=True)
    for name, df in res.items():
        df.to_csv(os.path.join("results", f"{name}.csv"), index=False)
    print("\n".join(make_figures()))
