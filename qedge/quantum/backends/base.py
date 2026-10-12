"""The single interface every quantum stack implements."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..circuit import Circuit


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
