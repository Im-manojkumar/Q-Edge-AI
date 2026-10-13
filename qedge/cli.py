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
    s.add_argument("kind", choices=["inference", "training", "optimization", "kernel_classification", "quantum_simulation",
                                    "ground_state", "scheduling"])
    s.add_argument("params", nargs="*", help="key=value, e.g. n=8 steps=4, or model=h2 for ground_state")
    s.add_argument("--policy", default="qedge_adaptive")
    s.add_argument("--sim-stack", default="qiskit", choices=["qiskit", "pennylane", "cirq", "braket-local"])
    s.add_argument("--qpu-stack", default="ibm-auto", choices=["ibm-auto", "ibm", "braket", "ibm-emulator"],
                   help="ibm-auto picks device and qubits from calibration (default)")
    s.add_argument("--force", help="run this placement instead, e.g. cloud_cpu+qpu")
    s.add_argument("--plan-only", action="store_true", help="decide but do not execute")
    s.add_argument("--race", action="store_true",
                   help="if the QPU is used, run the best classical placement at the same time and deliver the QPU "
                        "result only if it arrives before the deadline")
    s.add_argument("--deadline", type=float, help="deadline in seconds (also the race deadline)")

    m = sub.add_parser("simulate", help="discrete-event comparison of policies")
    m.add_argument("--scenario", default="baseline")
    m.add_argument("--tasks", type=int, default=600)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--batch-window", type=float, help="group QPU tasks into shared jobs within this many seconds")
    m.add_argument("--max-batch", type=int, default=8, help="most tasks per shared QPU job")
    m.add_argument("--hedge", action="store_true", help="also run qedge_hedged (race mode with classical fallback)")

    k = sub.add_parser("pack", help="run several copies of a small circuit workload side by side in one QPU job")
    k.add_argument("kind", choices=["quantum_simulation"])
    k.add_argument("params", nargs="*", help="key=value, e.g. n=4 steps=2")
    k.add_argument("--copies", type=int, default=4)
    k.add_argument("--buffer", type=int, default=1, choices=[0, 1], help="1 leaves a free qubit ring around each program")
    k.add_argument("--shots", type=int, default=4000)

    e = sub.add_parser("experiments", help="run E1-E7 and regenerate figures")
    e.add_argument("--quick", action="store_true")
    e.add_argument("--only", nargs="*", choices=["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"])
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
            params = _params(a.params)
            rep = orch.submit(a.kind, force=a.force, race=a.race, deadline_s=a.deadline, **params)
            print(rep.explain())
            if rep.race is not None and rep.raced:
                rep.race.wait_settled()  # an abandoned emulator thread finishes before we exit
                print("  final: " + rep.race.summary())
    elif a.cmd == "simulate":
        import pandas as pd
        from .orchestrator.policies import POLICIES
        from .simulation.simulator import Simulator, summarise
        from .simulation.simulator import BatchConfig
        batching = BatchConfig(a.batch_window, a.max_batch) if a.batch_window else None
        sim = Simulator(a.scenario, seed=a.seed, batching=batching)
        wl = sim.workloads(a.tasks)
        names = list(POLICIES) + (["qedge_hedged"] if a.hedge else [])
        df = pd.concat([sim.run(p, wl) for p in names])
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(summarise(df).drop(columns=["scenario"]).round(3).to_string(index=False))
    elif a.cmd == "experiments":
        from .experiments.run_all import main as run_all
        run_all(a.quick, a.only, qpu_stack=a.qpu_stack)
        from .experiments.report import build
        build()
    elif a.cmd == "pack":
        import numpy as np
        import pandas as pd

        from .quantum.backends import get_backend
        from .quantum.packing import run_packed_workloads
        from .workloads import BUILDERS
        ws = [BUILDERS[a.kind](i, np.random.default_rng(i), **_params(a.params)) for i in range(a.copies)]
        run = run_packed_workloads(get_backend("ibm-auto"), ws, shots=a.shots, buffer=a.buffer)
        with pd.option_context("display.width", 200, "display.max_colwidth", 40):
            print(pd.DataFrame(run.results).round(4).to_string(index=False))
        c = run.cost
        print(f"\n{len(run.results)} programs in one {'emulated ' if run.emulated else ''}job on {run.device}: "
              f"{c.qpu_s_per_program:.3g} QPU-s and ${c.price_per_program:.3g} per program "
              f"(alone: {c.qpu_s_per_program_solo:.3g} QPU-s, ${c.price_per_program_solo:.3g})")
        if run.unplaced or run.skipped:
            print(f"not packed: {run.unplaced or ''} {run.skipped or ''}")
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
