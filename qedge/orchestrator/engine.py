"""The live Q-Edge orchestrator: profile -> monitor -> decide -> execute -> evaluate -> feedback.

    from qedge.orchestrator.engine import Orchestrator
    orch = Orchestrator(qpu_stack="ibm-auto")     # best real IBM device if QISKIT_IBM_TOKEN is set
    report = orch.submit("optimization", n=8)
    print(report.explain())

Race mode hedges a QPU placement against its queue, which is seconds to hours
and only partly observable before submitting:

    report = orch.submit("quantum_simulation", n=8, steps=3, force="cloud_cpu+qpu", race=True, deadline_s=5)

The quantum path and the best feasible classical placement start together. The
QPU result is delivered if it arrives before the deadline; otherwise the
classical one is, and the QPU work is cancelled. That caps what a wrong guess
about the queue can cost at the deadline, in exchange for always doing the
classical work too.

Caveat: here both paths run on this one machine (the "QPU" is a local emulator
of an IBM device), so they compete for CPU and for Python's GIL. The classical
side can run slower than it would alone, and the thread waiting for the
deadline can wake late while the emulator holds the GIL, so a fallback can be
delivered somewhat after the deadline (``RaceInfo.overshoot_s``). Race timings
are not the timings either path would have on its own hardware; on a real QPU
the quantum thread mostly waits on the network, which releases the GIL.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field

import psutil

from ..executor import ExecResult, Executor
from ..quantum.backends import CalibratedIBMBackend
from ..quantum.backends.base import Cancelled, CancelToken
from ..resources import Resource, ResourceState, load_catalogue
from ..workloads import BUILDERS, Workload, load_calibration
from . import split
from .estimator import Estimate, Placement
from .policies import Decision, Policy, make_policy

log = logging.getLogger(__name__)


def _fmt(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


@dataclass
class RaceInfo:
    """What happened when a QPU placement was raced against a classical fallback.

    Times are wall-clock seconds from submission (planning included), the clock
    the deadline is measured on. ``quantum_status``:

      completed  the QPU result arrived before the deadline and was delivered
      failed     the QPU path raised (before the deadline, or after it but before it
                 could be cancelled); the classical result was delivered
      cancelled  the deadline passed, the QPU work was cancelled and stopped
                 (``Cancelled`` at its next job, or the provider cancelled it)
      late       the QPU work finished after the deadline (it could not be stopped, or
                 it ended between the deadline and the waiting thread waking up); its
                 result is kept in ``quantum_result`` but was not delivered (or was,
                 marked late, if the classical fallback had failed)

    "Before the deadline" is judged by when the QPU thread finished, not by when the
    waiting thread noticed, which can be late under GIL contention.

    A local emulator cannot be interrupted inside a job, so after a cancel its
    thread keeps running in the background until the job ends. Until then
    ``quantum_settled`` is False and the status is provisional; ``wait_settled()``
    blocks until it settles.
    """

    raced: bool = False
    reason: str = ""                          # why there was no race
    deadline_s: float | None = None
    fallback: str | None = None               # classical placement raced against the QPU
    winner: str | None = None                 # "quantum" | "classical" | None (nothing to deliver)
    quantum_status: str | None = None
    quantum_settled: bool = True
    quantum_wall_s: float | None = None       # when the QPU path ended
    classical_wall_s: float | None = None     # when the classical path ended
    delivered_s: float | None = None          # when the delivered result was handed back
    qpu_ran_before_cancel_s: float | None = None
    quantum_error: str | None = None
    classical_error: str | None = None
    jobs_cancelled: list[str] = field(default_factory=list)   # provider job ids (live hardware only)
    cancel_errors: list[str] = field(default_factory=list)
    classical_result: ExecResult | None = None
    quantum_result: ExecResult | None = None

    def wait_settled(self, timeout: float | None = None) -> bool:
        """Block until both race threads have ended and been recorded (an abandoned
        QPU thread keeps running after delivery). True once settled."""
        futs: tuple[Future, ...] = getattr(self, "_futures", ())
        if not futs:
            return self.quantum_settled
        wait(futs, timeout=timeout)
        end = time.perf_counter() + 1.0  # done-callbacks run just after a future completes
        while not self._settled() and all(f.done() for f in futs) and time.perf_counter() < end:
            time.sleep(0.005)
        return self._settled()

    def _settled(self) -> bool:
        return self.quantum_settled and (self.classical_wall_s is not None or not self.raced)

    @property
    def overshoot_s(self) -> float | None:
        """How far past the deadline the result was delivered (measured wall clock, no
        modelled network). A fallback handed over at the deadline is always a few ms
        late; much more means the waiting thread could not get the GIL in time."""
        if not self.raced or self.delivered_s is None or self.winner is None:
            return None
        return max(0.0, self.delivered_s - self.deadline_s)

    def row(self) -> dict:
        """Flat summary for tables and CSVs."""
        q, c = self.quantum_result, self.classical_result
        return {"raced": self.raced, "reason": self.reason, "deadline_s": self.deadline_s,
                "fallback": self.fallback, "winner": self.winner, "quantum_status": self.quantum_status,
                "quantum_settled": self.quantum_settled, "quantum_wall_s": self.quantum_wall_s,
                "classical_wall_s": self.classical_wall_s, "delivered_s": self.delivered_s,
                "overshoot_s": self.overshoot_s, "qpu_ran_before_cancel_s": self.qpu_ran_before_cancel_s,
                "quantum_quality": q.quality if q else None, "classical_quality": c.quality if c else None,
                "quantum_error": self.quantum_error, "classical_error": self.classical_error}

    def summary(self) -> str:
        if not self.raced:
            return f"race: {self.reason}"
        head = f"race vs {self.fallback} (deadline {self.deadline_s:.3g}s): "
        cw = f"ready at {self.classical_wall_s:.3g}s" if self.classical_wall_s is not None else "not finished"
        if self.winner == "quantum" and self.quantum_status == "completed":
            return head + f"QPU result delivered at {self.quantum_wall_s:.3g}s; classical fallback {cw}"
        if self.winner == "quantum":
            return head + (f"classical fallback failed ({self.classical_error}); QPU result delivered late "
                           f"at {self.quantum_wall_s:.3g}s")
        if self.winner == "classical":
            out = head + f"classical result ({cw}) delivered at {self.delivered_s:.3g}s; QPU {self.quantum_status}"
            if self.quantum_status == "failed":
                out += f" ({self.quantum_error})"
            if self.qpu_ran_before_cancel_s is not None:
                out += f" after running {self.qpu_ran_before_cancel_s:.3g}s"
            if not self.quantum_settled:
                out += ("; its thread is still finishing in the background (a local emulator cannot be "
                        "interrupted mid-job) and its result will be discarded")
            elif self.quantum_status == "late":
                out += f"; it finished anyway at {self.quantum_wall_s:.3g}s and was discarded"
            return out
        return head + f"no result (QPU {self.quantum_status}: {self.quantum_error}; classical: {self.classical_error})"


@dataclass
class Report:
    workload: Workload
    decision: Decision
    result: ExecResult | None    # in a race: the delivered result (QPU or classical fallback)
    measured_latency_s: float    # in a race: wall clock to delivery + modelled network
    error: str | None = None
    race: RaceInfo | None = None  # set only when submit(race=True)

    @property
    def raced(self) -> bool:
        return bool(self.race and self.race.raced)

    @property
    def winner(self) -> str | None:
        return self.race.winner if self.race else None

    @property
    def quantum_status(self) -> str | None:
        return self.race.quantum_status if self.race else None

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
            if r.kind == "split_inference":
                lines.append("  " + split_summary(r.details))
        if self.race is not None:
            lines.append("  " + self.race.summary())
        if self.error:
            lines.append(f"  ERROR: {self.error}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {"workload": self.workload.to_dict(), "placement": self.decision.placement.label,
             "policy": self.decision.policy, "predicted": self.decision.chosen.row(),
             "alternatives": [e.row() for e in self.decision.estimates],
             "quantum_checks": self.decision.quantum_checks,
             "result": asdict(self.result) if self.result else None,
             "measured_latency_s": self.measured_latency_s, "error": self.error}
        if self.race is not None:  # absent unless a race was asked for, so old logs keep their shape
            d["race"] = asdict(self.race)
        return d


def split_summary(d: dict) -> str:
    """One line on what a split_inference run cut where and what left the edge."""
    if not d.get("left_edge"):
        where = "whole model on the edge; nothing left it"
    else:
        names = ", ".join("the raw images" if t == "input" else f"the {t} activation" if t != "embedding"
                          else f"the {d['k']}-number embedding" for t in d["left_edge"])
        if "input" in d["left_edge"]:
            cut = (f"{d['front']} ran the backbone on the uploaded images, the quantum head ran on {d['remote']}"
                   if d.get("path") else f"whole model on {d['remote']}")
        elif d.get("path"):
            cut = f"edge ({d['front']}) ran up to the embedding, the quantum head ran on {d['remote']}"
        else:
            cut = f"edge ({d['front']}) ran up to {d['cut']}, {d['remote']} ran the rest"
        where = (f"{cut}; left the edge: {names}, {d['bytes_up_per_image']:,} B per image, "
                 f"{d['bytes_up_total']:,} B for {d['batch']} images (raw data left the edge: "
                 f"{'yes' if d['raw_data_left_edge'] else 'no'})")
    out = f"split: {where}"
    if d.get("path"):
        qpu = f", {d['qpu_s']:.3g} QPU-s modelled" if d.get("qpu_s") is not None else ""
        out += (f"\n  quantum head: {d['jobs']} job(s), {d['circuits']} circuits ({d['path']}) on {d['device']}{qpu}; "
                f"agrees with the exact quantum head on {d['agree_exact_quantum']:.0%} of images")
    return out


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
        try:   # the measured split profile (E8) when it exists, as for calibration.json
            split.load_profiles(keep=True)
        except ValueError as e:   # a malformed file: keep the built-in profiles, say so
            log.warning("ignoring results/split_profile.json: %s", e)
            split.load_profiles("", keep=True)
        self.resources = load_catalogue(config)
        self.policy = make_policy(policy, self.resources) if isinstance(policy, str) else policy
        self.executor = Executor(sim_stack, qpu_stack, self.resources)
        self.monitor = Monitor(self.resources, self.executor)
        self.log_path = log_path
        self._next_id = 0

    def plan(self, kind: str, **params) -> tuple[Workload, Decision]:
        import numpy as np
        w = BUILDERS[kind](self._next_id, np.random.default_rng(self._next_id), **params)
        self._next_id += 1
        return w, self.policy.decide(w, self.monitor.snapshot(w))

    def submit(self, kind: str, force: str | None = None, race: bool = False, deadline_s: float | None = None,
               **params) -> Report:
        """Plan one workload, execute it for real, and report.

        force       run this placement label (e.g. "cloud_cpu+qpu", "edge_gpu>conv2>cloud_gpu")
                    instead of the decision; parsed by ``Placement.parse``.
        race        if the executed placement sends circuits to the QPU, also run the best
                    feasible classical placement at the same time and deliver per ``_race``.
                    No effect on any other placement (the report says why).
        deadline_s  the workload's deadline, passed to the workload builder exactly as
                    before; in a race it is also the wall-clock deadline from this call.
                    Defaults to the deadline the builder draws.
        """
        t_submit = time.perf_counter()
        if deadline_s is not None:
            params["deadline_s"] = deadline_s
        w, d = self.plan(kind, **params)
        if force:  # run a specific placement anyway, e.g. to compare against the decision
            p = Placement.parse(force, self.resources)
            est = next((e for e in d.estimates if e.placement == p), None)
            if est is None and w.kind == "split_inference":  # not a candidate (e.g. "cloud_cpu+qpu"): cost it
                est = self.policy.model.estimate(w, p, self.monitor.snapshot(w))
            d = Decision(w, p, est or d.chosen, d.estimates, d.policy + " (forced)", d.quantum_checks)
        race_info = None
        if race:
            fallback, why = self.race_fallback(d)
            if fallback is not None:
                return self._race(w, d, fallback, t_submit, float(w.deadline_s))
            race_info = RaceInfo(raced=False, reason=why, deadline_s=float(w.deadline_s))
        result, err = None, None
        t0 = time.perf_counter()
        try:
            result = self._execute(w, d.placement)
        except Exception as e:  # report, don't crash the service
            err = f"{type(e).__name__}: {e}"
        wall = time.perf_counter() - t0
        b = d.chosen.breakdown
        modelled_network = b.get("transfer", 0.0)
        measured = (result.compute_s + result.quantum_s) if result else wall
        latency = measured + modelled_network
        if result is not None:
            self.policy.observe(d, {"latency_s": latency, "quality": result.quality})
        rep = Report(w, d, result, latency, err, race_info)
        self._log(rep)
        return rep

    def _execute(self, w: Workload, p: Placement, cancel: CancelToken | None = None, quantum: bool = True):
        """Run ``w`` as placement ``p``. Only split_inference reads the placement (where the
        network is cut); every other kind runs exactly as before, with the quantum
        resource or classically (``quantum=False``: a race's classical path)."""
        q = p.quantum if quantum else None
        if w.kind == "split_inference":
            return self.executor.run(w, q, cancel=cancel, placement=p)
        if cancel is None and quantum:
            return self.executor.run(w, q)
        return self.executor.run(w, q, cancel=cancel)

    # ------------------------------------------------------------------ race mode
    def uses_qpu(self, p: Placement) -> bool:
        """True when ``p`` sends circuits to a QPU (not to a state-vector simulator)."""
        if not p.is_hybrid:
            return False
        r = self.resources.get(p.quantum)
        return not r.sim_gflops if r is not None else p.quantum != "quantum_sim"

    def race_fallback(self, d: Decision) -> tuple[Estimate | None, str]:
        """The classical placement to race against ``d``: lowest policy score among the
        feasible non-hybrid estimates (lowest predicted latency for policies that do
        not score). Returns (None, reason) when there is nothing to race."""
        if not self.uses_qpu(d.placement):
            return None, f"not raced: {d.placement.label} does not use a QPU"
        classical = [e for e in d.estimates if not e.placement.is_hybrid]
        feas = [e for e in classical if e.feasible]
        if not feas:
            why = "; ".join(f"{e.placement.label}: {', '.join(e.reasons) or 'infeasible'}" for e in classical)
            return None, f"not raced: no feasible classical placement to fall back to ({why or 'none offered'})"

        def key(e: Estimate):
            s = e.score if e.score is not None and math.isfinite(e.score) else math.inf
            return (s, e.latency_s)
        return min(feas, key=key), ""

    def _race(self, w: Workload, d: Decision, fallback: Estimate, t_submit: float, deadline: float) -> Report:
        """Run the QPU path and ``fallback`` concurrently and pick what to deliver.

        Rules: the QPU result is delivered if it arrives before the deadline. If the
        QPU path raises first, the classical result is delivered. Otherwise, at the
        deadline the QPU work is cancelled and the classical result is delivered,
        waited for if it is not ready. One exception: if the classical path has
        already failed by the deadline, cancelling would leave nothing, so the QPU
        is left to finish and its result is delivered marked late.
        """
        token = CancelToken()
        t_deadline = t_submit + deadline
        marks: dict[str, float] = {}

        def path(tag: str, p: Placement, cancel: CancelToken | None) -> ExecResult:
            marks[f"{tag}_start"] = time.perf_counter()
            try:
                return self._execute(w, p, cancel, quantum=tag == "q")
            finally:
                marks[f"{tag}_end"] = time.perf_counter()

        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"qedge-race-{w.id}")
        fq = pool.submit(path, "q", d.placement, token)
        fc = pool.submit(path, "c", fallback.placement, None)
        pool.shutdown(wait=False)  # never block on an abandoned QPU thread

        info = RaceInfo(raced=True, deadline_s=deadline, fallback=fallback.placement.label)
        wait([fq], timeout=max(0.0, t_deadline - time.perf_counter()))
        # Judge the QPU by when its thread finished, not by when this thread woke up: while
        # the emulator holds the GIL the wake-up can be late, and a result that landed in
        # that gap missed the deadline. (q_end is written before the future completes.)
        q_on_time = fq.done() and marks.get("q_end", math.inf) <= t_deadline
        result: ExecResult | None = None
        if q_on_time and fq.exception() is None:
            result, info.winner, info.quantum_status = fq.result(), "quantum", "completed"
        elif q_on_time:  # the QPU path raised before the deadline
            exc = fq.exception()
            info.quantum_status = "cancelled" if isinstance(exc, Cancelled) else "failed"
            info.quantum_error = _fmt(exc)
        elif fc.done() and fc.exception() is not None:  # no fallback left: let the QPU finish
            wait([fq])
            if fq.exception() is None:
                result, info.winner, info.quantum_status = fq.result(), "quantum", "late"
            else:
                info.quantum_status, info.quantum_error = "failed", _fmt(fq.exception())
        elif fq.done():  # finished just after the deadline, before this thread woke: nothing to cancel
            exc = fq.exception()
            info.quantum_status = "late" if exc is None else "failed"
            info.quantum_error = _fmt(exc) if exc is not None else None
        else:
            token.cancel(f"race deadline {deadline:.3g}s reached")
            info.quantum_status = "cancelled"
            info.qpu_ran_before_cancel_s = token.cancelled_at - marks.get("q_start", token.cancelled_at)
        if info.winner is None:
            try:
                result, info.winner = fc.result(), "classical"  # the fallback: wait for it if needed
            except Exception as e:
                info.classical_error = _fmt(e)
        t_delivered = time.perf_counter()

        info.delivered_s = t_delivered - t_submit
        info._futures = (fq, fc)  # not a field: kept out of asdict / the log
        c_done, q_done = fc.done(), fq.done()  # read once: callbacks below cover the rest
        if c_done:
            self._settle_classical(info, fc, marks, t_submit)
        if q_done:
            self._settle_quantum(info, fq, marks, t_submit, token)
        else:
            info.quantum_settled = False

        # Delivered latency uses the same clock as the deadline; the winner's modelled
        # network transfer is added as in the non-race path.
        won = d.chosen if info.winner == "quantum" else fallback
        latency = info.delivered_s + (won.breakdown.get("transfer", 0.0) if result else 0.0)
        err = None
        if result is None:
            err = (f"race: QPU {info.quantum_status} ({info.quantum_error}); "
                   f"classical fallback failed ({info.classical_error})")
        else:
            # Feedback on the delivered placement, measured as outside a race. A cancelled QPU
            # run only tells us it took longer than the deadline (censored), so it is not fed back.
            fd = d if info.winner == "quantum" else Decision(w, fallback.placement, fallback, d.estimates,
                                                             d.policy + " (race fallback)", d.quantum_checks)
            self.policy.observe(fd, {"latency_s": result.compute_s + result.quantum_s
                                     + won.breakdown.get("transfer", 0.0), "quality": result.quality})
        rep = Report(w, d, result, latency, err, info)
        log.info("workload %s: %s", w.id, info.summary())
        self._log(rep)
        if not q_done:
            fq.add_done_callback(lambda f: self._settle_quantum(info, f, marks, t_submit, token))
        if not c_done:  # the QPU won; the local classical path is short and left to finish
            fc.add_done_callback(lambda f: self._settle_classical(info, f, marks, t_submit))
        return rep

    @staticmethod
    def _settle_classical(info: RaceInfo, fc: Future, marks: dict, t_submit: float):
        info.classical_wall_s = marks.get("c_end", time.perf_counter()) - t_submit
        exc = fc.exception()
        if exc is None:
            info.classical_result = fc.result()
        elif info.classical_error is None:
            info.classical_error = _fmt(exc)

    @staticmethod
    def _settle_quantum(info: RaceInfo, fq: Future, marks: dict, t_submit: float, token: CancelToken):
        """Record how the QPU path ended (possibly after the report was delivered)."""
        info.quantum_wall_s = marks.get("q_end", time.perf_counter()) - t_submit
        exc = fq.exception()
        if exc is None:
            info.quantum_result = fq.result()
            if info.winner != "quantum":
                info.quantum_status = "late"  # ran to the end after the deadline; discarded
        elif info.quantum_error is None:
            info.quantum_error = _fmt(exc)
        info.jobs_cancelled = list(token.jobs_cancelled)
        info.cancel_errors = list(token.job_cancel_errors)
        info.quantum_settled = True

    def _log(self, rep: Report):
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rep.to_dict(), default=str) + "\n")
