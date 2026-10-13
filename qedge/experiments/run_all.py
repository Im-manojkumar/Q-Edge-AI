"""Run every experiment, save CSVs to results/ and figures to figures/."""
from __future__ import annotations

import json
import os
import platform
import time

import pandas as pd

from . import (e1_stacks, e2_quantum_vs_classical, e3_e4_e5, e6_calibration, e7_batching, e7_packing, e7_race,
               e8_new_algorithms, e9_split, figures)


def save(frames: dict[str, pd.DataFrame], out_dir: str):
    for name, df in frames.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


def main(quick: bool = False, only: list[str] | None = None, out_dir: str = "results", fig_dir: str = "figures",
         qpu_stack: str = "ibm-auto"):
    steps = {
        "e1": lambda: {"e1_stacks": e1_stacks.run(quick)},
        "e2": lambda: e2_quantum_vs_classical.run(quick, out_dir),
        "e3": lambda: e3_e4_e5.e3(quick),
        "e4": e3_e4_e5.e4,
        "e5": lambda: e3_e4_e5.e5(quick, qpu_stack),
        "e6": lambda: e6_calibration.run(quick),
        "e7": lambda: {**e7_packing.run(quick), **e7_batching.run(quick), **e7_race.run(quick)},
        "e8": lambda: e8_new_algorithms.run(quick, out_dir),
        "e9": lambda: e9_split.run(quick, out_dir),
    }
    # An unknown name would otherwise run nothing yet still rewrite run_meta.json and every
    # figure, which looks like success; stop before touching anything.
    unknown = sorted(set(only or ()) - set(steps))
    if unknown:
        raise SystemExit(f"unknown experiment step(s) {', '.join(unknown)}; this run_all has {', '.join(steps)}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    meta = {"quick": quick, "python": platform.python_version(), "machine": platform.machine(),
            "processor": platform.processor(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import cirq, pennylane, qiskit, qiskit_aer, braket._sdk as bk  # noqa: E401
        meta["versions"] = {"qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
                            "pennylane": pennylane.__version__, "cirq": cirq.__version__,
                            "amazon-braket-sdk": bk.__version__}
    except Exception:
        pass
    meta_path = os.path.join(out_dir, "run_meta.json")
    timings = {}
    if only and os.path.exists(meta_path):  # partial rerun: keep timings of the steps not rerun
        timings = json.load(open(meta_path)).get("timings_s", {})
    for name, fn in steps.items():
        if only and name not in only:
            continue
        t0 = time.time()
        print(f"[qedge] running {name} ...", flush=True)
        save(fn(), out_dir)
        timings[name] = round(time.time() - t0, 1)
        print(f"[qedge] {name} done in {timings[name]}s", flush=True)
    meta["timings_s"] = timings
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    figures.make_all(out_dir, fig_dir)
    for mod in (e7_packing, e7_batching, e7_race, e8_new_algorithms):
        mod.make_figures(out_dir, fig_dir)
    # E9's quick results carry a .quick suffix (they never replace the full run's), and so do
    # their figures.
    e9_split.make_figures(out_dir, fig_dir, quick=quick)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    main(a.quick, a.only)
