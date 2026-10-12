"""E7 (live part): race mode in the real orchestrator.

Each run forces a quantum-capable workload onto the QPU path (``ibm-auto``:
the best IBM device from calibration, emulated locally without an account)
with ``race=True``, so the best classical placement runs at the same time.
Deadlines are chosen on both sides of the emulator's run time, so both
outcomes appear: the QPU result delivered in time, and the classical fallback
delivered at the deadline with the QPU work cancelled.

It also measures how late a waiting thread wakes while the emulator runs in
another thread (the GIL), which bounds how precisely a deadline can be kept
when both paths share one machine. On real hardware the QPU path is a network
wait and this lag disappears.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd

from ..orchestrator.engine import Orchestrator

# (scenario, kind, params, deadline_s, repeats)
CASES = [
    ("qsim_short", "quantum_simulation", {"n": 8, "steps": 3}, 3.0, 1),
    ("qsim_long", "quantum_simulation", {"n": 8, "steps": 3}, 60.0, 1),
    ("qsim_long_warm", "quantum_simulation", {"n": 8, "steps": 3}, 5.0, 1),
    ("qaoa_mid", "optimization", {"n": 6}, 15.0, 1),
    ("qaoa_repeat", "optimization", {"n": 6}, 6.0, 6),
]


def race_runs(quick: bool = False, log_path: str | None = None) -> pd.DataFrame:
    orch = Orchestrator(log_path=log_path)
    rows = []
    warmed = set()
    for scenario, kind, params, deadline, repeats in CASES:
        if kind not in warmed:  # load calibration snapshots before the clock starts
            orch.plan(kind, **params)
            warmed.add(kind)
        for i in range(1 if quick else repeats):
            rep = orch.submit(kind, force="cloud_cpu+qpu", race=True, deadline_s=deadline, **params)
            status_on_return = rep.race.quantum_status if rep.race else None
            if rep.race is not None:
                rep.race.wait_settled()  # let an abandoned emulator thread finish before the next run
            row = {"scenario": scenario, "run": i, "kind": kind, "params": str(params),
                   "quantum_status_on_return": status_on_return,
                   "delivered_quality": rep.result.quality if rep.result else None,
                   "qpu_backend": rep.race.quantum_result.backend
                   if rep.race and rep.race.quantum_result else None}
            row.update(rep.race.row() if rep.race else {"raced": False})
            rows.append(row)
            print(f"[e7-race] {scenario} #{i}: {rep.race.summary() if rep.race else 'not raced'}", flush=True)
    return pd.DataFrame(rows)


def gil_lag(quick: bool = False, wait_s: float = 0.05) -> pd.DataFrame:
    """Lag of a timed 50 ms wait, idle vs while QAOA runs on the emulator in another thread."""
    from ..quantum.algorithms import random_graph, run_qaoa
    from ..quantum.backends import get_backend

    def lags(stop: threading.Event | None, n: int | None, seconds: float | None) -> list[float]:
        out, t_end = [], time.perf_counter() + (seconds or 1e9)
        while (stop is None or not stop.is_set()) and time.perf_counter() < t_end and (n is None or len(out) < n):
            t0 = time.perf_counter()
            time.sleep(wait_s)
            out.append((time.perf_counter() - t0 - wait_s) * 1000)
        return out

    rows = [{"condition": "idle", "lag_ms": x} for x in lags(None, None, 1.0 if quick else 3.0)]
    be = get_backend("ibm-auto")
    stop = threading.Event()

    def work():
        try:
            run_qaoa(be, 6, random_graph(6, 0.3, seed=1), shots=1000, maxiter=3 if quick else 8, seed=1)
        finally:
            stop.set()

    th = threading.Thread(target=work)
    th.start()
    rows += [{"condition": "emulator running", "lag_ms": x} for x in lags(stop, None, None)]
    th.join()
    return pd.DataFrame(rows)


def run(quick: bool = False) -> dict[str, pd.DataFrame]:
    runs = race_runs(quick, log_path="results/race_demo_log.jsonl")
    lag = gil_lag(quick)
    summary = (lag.groupby("condition").lag_ms
               .agg(samples="count", median="median", p95=lambda s: float(np.percentile(s, 95)), max="max")
               .reset_index())
    return {"race_demo": runs, "race_gil_lag": summary}


def make_figures(res_dir: str = "results", fig_dir: str = "figures"):
    """Delivered time vs deadline for every race, coloured by which path won."""
    import os

    import matplotlib.pyplot as plt

    from .figures import BLUE, GRAY, INK, MUTED, _save
    p = os.path.join(res_dir, "race_demo.csv")
    if not os.path.exists(p):
        return []
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for winner, color, label in [("quantum", BLUE, "QPU result delivered"),
                                 ("classical", GRAY, "classical fallback delivered")]:
        d = df[df.winner == winner]
        ax.scatter(d.deadline_s, d.delivered_s, color=color, s=40, label=label, zorder=3)
    lim = max(df.deadline_s.max(), df.delivered_s.max()) * 1.1
    ax.plot([0, lim], [0, lim], color=INK, lw=1, ls="--", label="deadline")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ticks = [1, 2, 5, 10, 20, 50, 100]
    for axis, scale in ((ax.xaxis, ax.set_xscale), (ax.yaxis, ax.set_yscale)):
        scale("log")
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlim(1, lim)
    ax.set_ylim(1, lim)
    ax.set_xlabel("deadline (s)")
    ax.set_ylabel("result delivered at (s)")
    ax.set_title("Race mode: when each result was delivered")
    ax.legend(fontsize=8)
    ax.tick_params(colors=MUTED)
    out = os.path.join(fig_dir, "e7_race.png")
    _save(fig, out)
    return [out]


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    for name, df in run(a.quick).items():
        df.to_csv(f"results/{name}.csv", index=False)
    make_figures()
