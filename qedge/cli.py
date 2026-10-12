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
    s.add_argument("--qpu-stack", default="ibm-auto", choices=["ibm-auto", "ibm", "braket", "ibm-emulator"],
                   help="ibm-auto picks device and qubits from calibration (default)")
    s.add_argument("--force", help="run this placement instead, e.g. cloud_cpu+qpu")
    s.add_argument("--plan-only", action="store_true", help="decide but do not execute")

    m = sub.add_parser("simulate", help="discrete-event comparison of policies")
    m.add_argument("--scenario", default="baseline")
    m.add_argument("--tasks", type=int, default=600)
    m.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("experiments", help="run E1-E6 and regenerate figures")
    e.add_argument("--quick", action="store_true")
    e.add_argument("--only", nargs="*", choices=["e1", "e2", "e3", "e4", "e5", "e6"])
    e.add_argument("--qpu-stack", default="ibm-auto")

    c = sub.add_parser("calibration", help="calibration-aware routing: devices, qubit choice, live logging, drift")
    c.add_argument("action", choices=["show", "pick", "log", "drift"])
    c.add_argument("params", nargs="*", help="for pick: a workload kind and key=value, e.g. quantum_simulation n=8 steps=3")
    c.add_argument("--every", type=float, default=600, help="log: seconds between polls")
    c.add_argument("--count", type=int, default=1, help="log: number of polls (144 at 600 s = one day)")
    c.add_argument("--snapshots", action="store_true", help="show/pick: ignore live IBM data")

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
    elif a.cmd == "calibration":
        _calibration(a)
    elif a.cmd == "report":
        from .experiments.report import build
        build()
    elif a.cmd == "dashboard":
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]))


def _calibration(a):
    import time

    import pandas as pd

    from .quantum import calibration as cal
    if a.action == "show":
        devs = cal.load_devices(prefer_live=not a.snapshots)
        df = pd.DataFrame([d.summary() for d in devs]).sort_values("median_2q_error")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(df.round(5).to_string(index=False))
        if devs[0].source != "live":
            print("\n(snapshots of real IBM devices; set QISKIT_IBM_TOKEN for today's live calibration)")
    elif a.action == "pick":
        import numpy as np

        from .executor import representative_circuit
        from .quantum.backends.qiskit_backend import to_qiskit
        from .workloads import BUILDERS
        if not a.params:
            sys.exit("usage: qedge calibration pick <kind> key=value ...")
        kind, params = a.params[0], _params(a.params[1:])
        w = BUILDERS[kind](0, np.random.default_rng(0), **params)
        if not w.quantum_capable:
            sys.exit(f"{kind} has no quantum formulation")
        circ = representative_circuit(w)
        devs = cal.load_devices(prefer_live=not a.snapshots)
        best, all_ = cal.select_device(to_qiskit(circ), devs)
        rows = sorted((c.row() for c in all_), key=lambda r: -r["esp"])
        df = pd.DataFrame(rows)[["device", "esp", "default_esp", "two_q_count", "depth", "queue_s", "layout"]]
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(df.round(4).to_string(index=False))
        print(f"\nchosen: {best.device} on qubits {best.layout} (estimated success probability {best.esp:.3f})")
    elif a.action == "log":
        from .quantum.backends.ibm_backend import IBMQuantumBackend
        if not IBMQuantumBackend.credentials_present():
            sys.exit("logging needs live data: set QISKIT_IBM_TOKEN (reading calibration and queues uses no QPU time)")
        for i in range(a.count):
            rows = cal.log_once()
            print(f"[{time.strftime('%H:%M:%S')}] logged {len(rows)} devices: "
                  + ", ".join(f"{r['name']}={r['pending_jobs']} jobs" for r in rows))
            if i + 1 < a.count:
                time.sleep(a.every)
    elif a.action == "drift":
        rows = cal.drift_from_trace()
        if not rows:
            sys.exit(f"need at least two logged calibrations per device in {cal.TRACE_PATH}; run `qedge calibration log`")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
