"""Command line: python -m qedge <command>."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def _params(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qedge", description="Q-Edge AI: hybrid quantum-classical workload orchestration")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="show which quantum stacks and real devices are usable")

    s = sub.add_parser("submit", help="plan and run one workload for real")
    s.add_argument("kind", choices=["inference", "training", "optimization", "kernel_classification", "quantum_simulation"])
    s.add_argument("params", nargs="*", help="key=value, e.g. n=8 steps=4")
    s.add_argument("--policy", default="qedge_adaptive")
    s.add_argument("--sim-stack", default="qiskit", choices=["qiskit", "pennylane", "cirq", "braket-local"])
    s.add_argument("--qpu-stack", default="ibm", choices=["ibm", "braket", "ibm-emulator"])
    s.add_argument("--force", help="run this placement instead, e.g. cloud_cpu+qpu")
    s.add_argument("--plan-only", action="store_true", help="decide but do not execute")

    m = sub.add_parser("simulate", help="discrete-event comparison of policies")
    m.add_argument("--scenario", default="baseline")
    m.add_argument("--tasks", type=int, default=600)
    m.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("experiments", help="run E1-E5 and regenerate figures")
    e.add_argument("--quick", action="store_true")
    e.add_argument("--only", nargs="*", choices=["e1", "e2", "e3", "e4", "e5"])
    e.add_argument("--qpu-stack", default="ibm")

    sub.add_parser("report", help="rebuild REPORT.md from results/")
    sub.add_parser("dashboard", help="launch the Streamlit dashboard")

    a = ap.parse_args(argv)

    if a.cmd == "status":
        from .quantum.backends import provider_status
        for r in provider_status():
            mark = "OK " if r["available"] else "-- "
            fb = f"  (falls back to {r['fallback']})" if r.get("fallback") else ""
            print(f"{mark}{r['backend']:<20} {r['detail']}{fb}")
    elif a.cmd == "submit":
        from .orchestrator.engine import Orchestrator
        orch = Orchestrator(a.policy, sim_stack=a.sim_stack, qpu_stack=a.qpu_stack)
        if a.plan_only:
            _, d = orch.plan(a.kind, **_params(a.params))
            print(d.explain())
        else:
            print(orch.submit(a.kind, force=a.force, **_params(a.params)).explain())
    elif a.cmd == "simulate":
        import pandas as pd
        from .orchestrator.policies import POLICIES
        from .simulation.simulator import Simulator, summarise
        sim = Simulator(a.scenario, seed=a.seed)
        wl = sim.workloads(a.tasks)
        df = pd.concat([sim.run(p, wl) for p in POLICIES])
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(summarise(df).drop(columns=["scenario"]).round(3).to_string(index=False))
    elif a.cmd == "experiments":
        from .experiments.run_all import main as run_all
        run_all(a.quick, a.only, qpu_stack=a.qpu_stack)
        from .experiments.report import build
        build()
    elif a.cmd == "report":
        from .experiments.report import build
        build()
    elif a.cmd == "dashboard":
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]))


if __name__ == "__main__":
    main()
