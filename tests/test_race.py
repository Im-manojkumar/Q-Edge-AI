"""Race mode: cooperative cancellation and the QPU-vs-classical race in the orchestrator.

Slow and failing "QPUs" are fake backends that sleep, so these tests take a few
seconds and need no account. The live IBM path is tested with Qiskit Runtime
mocked out; the ibm-auto angle regression test uses a 5-qubit device emulator.
"""
import dataclasses
import json
import threading
import time

import pytest

from qedge.orchestrator.engine import Orchestrator, Report
from qedge.quantum.algorithms import random_graph, run_qaoa
from qedge.quantum.backends import QuantumBackend
from qedge.quantum.backends.base import CancellableBackend, Cancelled, CancelToken, await_job, cancel_scope
from qedge.quantum.circuit import Circuit

OLD_REPORT_KEYS = {"workload", "placement", "policy", "predicted", "alternatives", "quantum_checks", "result",
                   "measured_latency_s", "error"}


class FakeQPU(QuantumBackend):
    """Sleeps ``delay`` per run(); can fail, or cancel a token on its k-th call."""

    name = "fake-qpu"
    framework = "fake"

    def __init__(self, delay=0.0, fail=False, cancel_on_call=None, token=None):
        super().__init__()
        self.delay, self.fail = delay, fail
        self.cancel_on_call, self.token = cancel_on_call, token
        self.calls = 0

    def _run(self, circuits, shots):
        self.calls += 1
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("device exploded")
        if self.cancel_on_call == self.calls:
            self.token.cancel("test")
        return [{"0" * c.n_qubits: shots} for c in circuits], {}


def orch_with(tmp_path, qpu: QuantumBackend) -> Orchestrator:
    o = Orchestrator(log_path=str(tmp_path / "log.jsonl"))
    o.executor._backends[o.executor.qpu_stack] = qpu  # stands in for ibm-auto
    return o


QSIM = dict(kind="quantum_simulation", n=5, steps=2, force="cloud_cpu+qpu")


# ---------------------------------------------------------------- cancellation primitives
def test_token_cancels_pending_run():
    tok = CancelToken()
    inner = FakeQPU()
    be = CancellableBackend(inner, tok)
    assert be.run([Circuit(2)], 10).counts == [{"00": 10}]
    tok.cancel("deadline")
    with pytest.raises(Cancelled, match="deadline"):
        be.run([Circuit(2)], 10)
    assert inner.calls == 1 and tok.cancelled and tok.cancelled_at is not None
    assert be.name == "fake-qpu" and not be.hardware  # delegated


def test_cancel_lands_at_next_run_boundary():
    tok = CancelToken()
    inner = FakeQPU(delay=0.3)
    be = CancellableBackend(inner, tok)
    out = {}

    def two_jobs():
        try:
            be.run([Circuit(1)], 1)
            out["first"] = True
            be.run([Circuit(1)], 1)
        except Cancelled:
            out["cancelled"] = True

    t = threading.Thread(target=two_jobs)
    t.start()
    time.sleep(0.1)
    tok.cancel()
    t.join(2)
    # the in-flight job finishes, the next one never starts
    assert out == {"first": True, "cancelled": True} and inner.calls == 1


def test_scope_governs_plain_backends():
    tok = CancelToken()
    tok.cancel()
    with cancel_scope(tok), pytest.raises(Cancelled):
        FakeQPU().run([Circuit(1)], 1)
    assert FakeQPU().run([Circuit(1)], 1).counts  # outside the scope nothing changes


def test_qaoa_loop_stops_after_cancellation():
    tok = CancelToken()
    inner = FakeQPU(cancel_on_call=3, token=tok)
    with pytest.raises(Cancelled):
        run_qaoa(CancellableBackend(inner, tok), 4, random_graph(4, 0.6, seed=1), shots=32, maxiter=25)
    assert inner.calls == 3  # an uncancelled run makes 1 grid + 25 COBYLA + 1 final = 27 calls


class FakeRuntimeJob:
    """Mimics RuntimeJobV2: result() blocks until done or cancelled."""

    def __init__(self, refuse_cancel=False):
        self.refuse_cancel = refuse_cancel
        self.cancel_calls = 0
        self._done = threading.Event()
        self._cancelled = False

    def job_id(self):
        return "job-123"

    def cancel(self):
        self.cancel_calls += 1
        if self.refuse_cancel:
            raise RuntimeError("409 job cannot be cancelled")
        self._cancelled = True
        self._done.set()

    def finish(self):
        self._done.set()

    def result(self):
        self._done.wait(5)
        if self._cancelled:
            raise RuntimeError("Unable to retrieve result for job job-123. Job was cancelled.")
        return "counts"


def test_await_job_cancels_provider_job():
    tok, job = CancelToken(), FakeRuntimeJob()
    threading.Timer(0.1, tok.cancel, args=("deadline",)).start()
    with pytest.raises(Cancelled, match="job-123"):
        await_job(job, tok)
    assert job.cancel_calls == 1 and tok.jobs_cancelled == ["job-123"]


def test_await_job_when_provider_refuses_cancel():
    tok, job = CancelToken(), FakeRuntimeJob(refuse_cancel=True)
    threading.Timer(0.05, tok.cancel).start()
    threading.Timer(0.2, job.finish).start()
    assert await_job(job, tok) == "counts"  # result arrives anyway; the caller discards it
    assert tok.job_cancel_errors and "409" in tok.job_cancel_errors[0]


def test_register_after_cancel_cancels_immediately():
    tok, job = CancelToken(), FakeRuntimeJob()
    tok.cancel()
    tok.register(job)
    assert job.cancel_calls == 1


def test_ibm_backend_registers_runtime_job(monkeypatch):
    """IBMQuantumBackend._run with Runtime mocked out: the job is registered and cancelled."""
    import qiskit_ibm_runtime
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    from qedge.quantum.backends.ibm_backend import IBMQuantumBackend

    job = FakeRuntimeJob()

    class FakeSampler:
        def __init__(self, mode):
            pass

        def run(self, isa, shots):
            return job

    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", FakeSampler)
    be = IBMQuantumBackend("fake")
    be._backend = FakeManilaV2()
    tok = CancelToken()
    threading.Timer(0.3, tok.cancel).start()
    c = Circuit(2)
    c.h(0).cx(0, 1)
    with pytest.raises(Cancelled):
        CancellableBackend(be, tok).run([c], 100)
    assert job.cancel_calls == 1


# ---------------------------------------------------------------- the race
def test_slow_qpu_short_deadline_classical_wins(tmp_path):
    o = orch_with(tmp_path, FakeQPU(delay=1.5))
    rep = o.submit(QSIM["kind"], n=5, steps=2, force=QSIM["force"], race=True, deadline_s=0.4)
    r = rep.race
    assert rep.raced and rep.winner == "classical" and rep.error is None
    assert rep.quantum_status in ("cancelled", "late")
    assert rep.result.mode == "classical" and r.classical_result is rep.result
    assert r.fallback in {"edge_cpu", "edge_gpu", "cloud_cpu", "cloud_gpu"}
    assert 0.4 <= r.delivered_s < 1.4  # delivered at the deadline, not when the QPU finished
    assert r.overshoot_s < 0.5  # the fake QPU sleeps, so the waiting thread wakes on time
    assert 0.1 < r.qpu_ran_before_cancel_s <= 0.4
    assert not r.quantum_settled  # the sleeping job cannot be interrupted
    assert r.wait_settled(5)
    assert r.quantum_status == "late" and r.quantum_wall_s > 1.4 and r.quantum_result.mode == "hybrid"
    assert "race vs" in rep.explain() and "QPU" in rep.explain()
    json.dumps(rep.to_dict(), default=str)


def test_iterative_qpu_work_is_stopped_by_the_deadline(tmp_path):
    qpu = FakeQPU(delay=0.25)
    o = orch_with(tmp_path, qpu)
    rep = o.submit("optimization", n=4, force="cloud_cpu+qpu", race=True, deadline_s=0.5)
    assert rep.winner == "classical"
    assert rep.race.wait_settled(5)
    assert rep.quantum_status == "cancelled" and "Cancelled" in rep.race.quantum_error
    assert qpu.calls <= 3  # stopped at the next run() instead of making all 27


def test_fast_qpu_long_deadline_quantum_wins(tmp_path):
    o = orch_with(tmp_path, FakeQPU(delay=0.05))
    rep = o.submit(QSIM["kind"], n=5, steps=2, force=QSIM["force"], race=True, deadline_s=30)
    r = rep.race
    assert rep.winner == "quantum" and rep.quantum_status == "completed" and rep.error is None
    assert rep.result.mode == "hybrid" and rep.result.backend == "fake-qpu"
    assert r.quantum_wall_s < 5 and r.delivered_s < 5 and r.qpu_ran_before_cancel_s is None
    assert r.overshoot_s == 0.0 and r.row()["winner"] == "quantum"
    assert r.wait_settled(5) and r.classical_result.mode == "classical"


def test_qpu_failure_classical_wins(tmp_path):
    o = orch_with(tmp_path, FakeQPU(fail=True))
    rep = o.submit(QSIM["kind"], n=5, steps=2, force=QSIM["force"], race=True, deadline_s=30)
    assert rep.winner == "classical" and rep.quantum_status == "failed"
    assert "device exploded" in rep.race.quantum_error
    assert rep.race.delivered_s < 5  # no need to wait for the deadline once the QPU has failed


def test_failed_fallback_lets_the_qpu_finish(tmp_path, monkeypatch):
    o = orch_with(tmp_path, FakeQPU(delay=0.6))
    real = o.executor.run

    def run(w, q=None, cancel=None):
        if q is None:
            raise MemoryError("classical path broke")
        return real(w, q, cancel)

    monkeypatch.setattr(o.executor, "run", run)
    rep = o.submit(QSIM["kind"], n=5, steps=2, force=QSIM["force"], race=True, deadline_s=0.2)
    assert rep.winner == "quantum" and rep.quantum_status == "late" and rep.result.mode == "hybrid"
    assert "classical path broke" in rep.race.classical_error


def test_race_not_run_without_qpu(tmp_path):
    o = orch_with(tmp_path, FakeQPU())
    rep = o.submit("optimization", force="cloud_cpu+quantum_sim", race=True, n=5)
    assert rep.race is not None and not rep.raced and "does not use a QPU" in rep.race.reason
    assert rep.result.mode == "hybrid" and "race: not raced" in rep.explain()


def test_race_not_run_without_feasible_classical(tmp_path):
    o = orch_with(tmp_path, FakeQPU())
    for r in o.resources.values():
        if not r.is_quantum:
            r.memory_gb = 0.0
    rep = o.submit(QSIM["kind"], n=5, steps=2, force=QSIM["force"], race=True, deadline_s=30)
    assert not rep.raced and "no feasible classical placement" in rep.race.reason
    assert rep.result.mode == "hybrid" and rep.result.backend == "fake-qpu"


def test_race_false_unchanged(tmp_path):
    o = orch_with(tmp_path, FakeQPU())
    rep = o.submit("optimization", force="cloud_cpu+quantum_sim", n=5)
    assert rep.race is None and not rep.raced and rep.winner is None
    assert set(rep.to_dict()) == OLD_REPORT_KEYS
    old_fields = ["workload", "decision", "result", "measured_latency_s", "error"]
    assert [f.name for f in dataclasses.fields(Report)][:5] == old_fields
    assert "race" not in rep.explain()
    # deadline_s still reaches the workload builder exactly as a plain keyword did before
    assert o.submit("quantum_simulation", n=5, steps=2, deadline_s=7.0).workload.deadline_s == 7.0
    line = json.loads(open(tmp_path / "log.jsonl").readline())
    assert "race" not in line


def test_calibrated_backend_runs_each_circuits_own_angles():
    """ibm-auto shares a layout across circuits that differ only in angles; it must still
    run each circuit's angles (it used to replay the first circuit's transpiled copy)."""
    import math

    from qedge.quantum.backends import get_backend
    be = get_backend("ibm-auto")
    be._device_names = ["FakeManilaV2"]  # 5 qubits: a cheap emulator
    be.refresh()

    def c(theta):
        x = Circuit(2)
        x.rx(0, theta).cx(0, 1)
        return x

    zero, one = be.run([c(0.0), c(math.pi)], 500).counts   # one layout for both
    assert zero.get("00", 0) > 450 and one.get("11", 0) > 450
    assert be.run([c(math.pi)], 500).counts[0].get("11", 0) > 450  # cached layout, its own angle
