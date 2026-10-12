"""The single interface every quantum stack implements.

Also home to cooperative cancellation, which race mode needs: a QPU job that
misses its deadline must stop so it does not keep burning queue slots, QPU
seconds or (for emulators) local CPU. Nothing here can interrupt a call that is
already inside an SDK. Instead every ``QuantumBackend.run()`` checks a
``CancelToken`` first, so an iterative algorithm (QAOA makes ~27 sequential
``run()`` calls) stops at its next call, and remote jobs registered on the token
are cancelled with the provider when the token fires.
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field

from ..circuit import Circuit

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoiseSpec:
    """Uniform depolarising noise, defined the same way for every SDK.

    ``p1`` / ``p2`` are the total Pauli-error probability applied to *each*
    qubit a 1-qubit / 2-qubit gate touches (X, Y, Z each with p/3). Backends
    convert this to their own parametrisation so results stay comparable.
    """

    p1: float = 0.0
    p2: float = 0.0

    @property
    def is_noiseless(self) -> bool:
        return self.p1 <= 0 and self.p2 <= 0

    @classmethod
    def from_level(cls, p: float) -> "NoiseSpec":
        return cls(p1=p, p2=10 * p)


@dataclass
class RunResult:
    counts: list[dict[str, int]]
    backend: str
    framework: str
    shots: int
    wall_time_s: float
    queue_time_s: float = 0.0
    hardware: bool = False
    fallback_from: str | None = None
    job_ids: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class Cancelled(Exception):
    """Raised by ``run()`` once the token governing it has been cancelled."""


class CancelToken:
    """One-shot cancellation flag shared by everything a single race submits.

    ``cancel()`` sets the flag (a ``threading.Event``) and best-effort cancels
    every provider job registered on the token. It is safe to call from any
    thread, more than once.
    """

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._jobs: list = []
        self.reason = ""
        self.cancelled_at: float | None = None   # time.perf_counter() when cancel() fired
        self.job_cancel_errors: list[str] = []
        self.jobs_cancelled: list[str] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise Cancelled(self.reason or "cancelled")

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self._event.is_set():
                return
            self.reason = reason
            self.cancelled_at = time.perf_counter()
            self._event.set()
            jobs = list(self._jobs)
        for job in jobs:
            self._cancel_job(job)

    def register(self, job) -> None:
        """Track an in-flight provider job (anything with ``cancel()``). A job
        registered after the token fired is cancelled straight away."""
        with self._lock:
            self._jobs.append(job)
            late = self._event.is_set()
        if late:
            self._cancel_job(job)

    def unregister(self, job) -> None:
        with self._lock:
            if job in self._jobs:
                self._jobs.remove(job)

    def _cancel_job(self, job) -> None:
        jid = _job_id(job)
        try:
            job.cancel()
            self.jobs_cancelled.append(jid)
        except Exception as e:  # already finished, or the provider refused: keep going
            self.job_cancel_errors.append(f"{jid}: {type(e).__name__}: {e}")
            log.warning("could not cancel job %s: %s", jid, e)


def _job_id(job) -> str:
    try:
        return str(job.job_id())
    except Exception:
        return repr(job)


_scope = threading.local()


@contextmanager
def cancel_scope(token: CancelToken | None):
    """Make ``token`` govern every ``QuantumBackend.run()`` on this thread."""
    stack = getattr(_scope, "stack", None)
    if stack is None:
        stack = _scope.stack = []
    stack.append(token)
    try:
        yield token
    finally:
        stack.pop()


def current_token() -> CancelToken | None:
    stack = getattr(_scope, "stack", None)
    return stack[-1] if stack else None


def await_job(job, token: CancelToken | None = None):
    """Block on a provider job's ``result()`` with ``token`` able to cancel it.

    The token cancels the job with the provider; ``result()`` then raises (Qiskit
    Runtime: ``RuntimeInvalidStateError``), which becomes ``Cancelled``. If the
    provider refuses the cancel (job already running past the point of no
    return) the result arrives normally and the caller decides what to do with it.
    """
    if token is None:
        return job.result()
    token.register(job)
    try:
        return job.result()
    except Exception as e:
        if token.cancelled:
            raise Cancelled(f"job {_job_id(job)} cancelled: {token.reason}") from e
        raise
    finally:
        token.unregister(job)


class QuantumBackend(ABC):
    """Run a batch of framework-neutral circuits and return counts."""

    name: str = "abstract"
    framework: str = "none"
    hardware: bool = False
    max_qubits: int = 28

    def __init__(self, noise: NoiseSpec | None = None):
        self.noise = noise or NoiseSpec()

    # Whether this backend can actually be used right now (SDK installed,
    # credentials present, device online).
    def available(self) -> tuple[bool, str]:
        return True, "local simulator"

    def status(self) -> dict:
        ok, why = self.available()
        return {"name": self.name, "framework": self.framework, "hardware": self.hardware,
                "available": ok, "detail": why, "max_qubits": self.max_qubits,
                "queue_depth": self.queue_depth() if ok else None}

    def queue_depth(self) -> int | None:
        return 0

    def run(self, circuits: list[Circuit], shots: int = 1024) -> RunResult:
        tok = current_token()
        if tok is not None:
            tok.raise_if_cancelled()
        for c in circuits:
            if c.n_qubits > self.max_qubits:
                raise ValueError(f"{self.name}: {c.n_qubits} qubits exceeds limit {self.max_qubits}")
        t0 = time.perf_counter()
        counts, extra = self._run(circuits, shots)
        wall = time.perf_counter() - t0
        return RunResult(counts=counts, backend=self.name, framework=self.framework, shots=shots,
                         wall_time_s=wall, hardware=self.hardware,
                         queue_time_s=float(extra.pop("queue_time_s", 0.0)),
                         job_ids=extra.pop("job_ids", []), meta=extra)

    @abstractmethod
    def _run(self, circuits: list[Circuit], shots: int) -> tuple[list[dict[str, int]], dict]:
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class CancellableBackend(QuantumBackend):
    """Binds a ``CancelToken`` to a shared backend for one piece of work.

    ``run()`` raises ``Cancelled`` before touching the backend once the token has
    fired, and runs the inner backend inside ``cancel_scope(token)`` so live
    hardware adapters can register their provider job on it. Local simulators
    and emulators cannot be interrupted mid-call: a ``run()`` already in progress
    finishes, and the token stops the work at the next ``run()`` boundary.
    Everything else (name, hardware, devices, fallback_from, ...) is delegated.
    """

    def __init__(self, inner: QuantumBackend, token: CancelToken):
        self.inner = inner
        self.token = token

    # Class attributes on QuantumBackend would shadow __getattr__, so delegate them explicitly.
    name = property(lambda self: self.inner.name)              # type: ignore[assignment]
    framework = property(lambda self: self.inner.framework)    # type: ignore[assignment]
    hardware = property(lambda self: self.inner.hardware)      # type: ignore[assignment]
    max_qubits = property(lambda self: self.inner.max_qubits)  # type: ignore[assignment]
    noise = property(lambda self: self.inner.noise)

    def __getattr__(self, item):
        if item == "inner":  # not yet set (copy / unpickle): avoid recursion
            raise AttributeError(item)
        return getattr(self.inner, item)

    def available(self):
        return self.inner.available()

    def queue_depth(self):
        return self.inner.queue_depth()

    def run(self, circuits: list[Circuit], shots: int = 1024) -> RunResult:
        self.token.raise_if_cancelled()
        with cancel_scope(self.token):
            return self.inner.run(circuits, shots)

    def _run(self, circuits, shots):  # pragma: no cover - run() is overridden
        return self.inner._run(circuits, shots)

    def __repr__(self) -> str:
        return f"<CancellableBackend {self.inner!r}>"
