"""Framework-neutral circuit description.

Every quantum algorithm in Q-Edge AI is written once against this small gate
list and then translated by each backend (Qiskit, PennyLane, Cirq, Braket).
That is what lets the orchestrator run the *same* workload on any stack and
compare them fairly.

Conventions (identical across all backends):
  * rotation gates use the standard exp(-i * theta/2 * P) definition
  * ``rzz(theta)`` is exp(-i * theta/2 * Z(x)Z)
  * measured bitstrings are written with qubit 0 as the LEFTMOST character
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

ONE_QUBIT = {"h", "x", "y", "z", "s", "sdg", "rx", "ry", "rz"}
TWO_QUBIT = {"cx", "cz", "rzz", "swap"}
PARAMETRIC = {"rx", "ry", "rz", "rzz"}
SUPPORTED = ONE_QUBIT | TWO_QUBIT


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


@dataclass
class Circuit:
    n_qubits: int
    gates: list[Gate] = field(default_factory=list)
    name: str = "circuit"

    # -- builders -----------------------------------------------------------
    def add(self, name: str, *qubits: int, params: Iterable[float] = ()) -> "Circuit":
        name = name.lower()
        if name not in SUPPORTED:
            raise ValueError(f"unsupported gate {name!r}")
        for q in qubits:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"qubit {q} out of range for {self.n_qubits}-qubit circuit")
        self.gates.append(Gate(name, tuple(int(q) for q in qubits), tuple(float(p) for p in params)))
        return self

    def h(self, q): return self.add("h", q)
    def x(self, q): return self.add("x", q)
    def y(self, q): return self.add("y", q)
    def z(self, q): return self.add("z", q)
    def s(self, q): return self.add("s", q)
    def sdg(self, q): return self.add("sdg", q)
    def rx(self, q, t): return self.add("rx", q, params=[t])
    def ry(self, q, t): return self.add("ry", q, params=[t])
    def rz(self, q, t): return self.add("rz", q, params=[t])
    def cx(self, a, b): return self.add("cx", a, b)
    def cz(self, a, b): return self.add("cz", a, b)
    def rzz(self, a, b, t): return self.add("rzz", a, b, params=[t])
    def swap(self, a, b): return self.add("swap", a, b)

    def compose(self, other: "Circuit") -> "Circuit":
        if other.n_qubits != self.n_qubits:
            raise ValueError("qubit count mismatch")
        self.gates.extend(other.gates)
        return self

    def inverse(self) -> "Circuit":
        inv = Circuit(self.n_qubits, name=f"{self.name}_dg")
        swap_name = {"s": "sdg", "sdg": "s"}
        for g in reversed(self.gates):
            if g.name in PARAMETRIC:
                inv.gates.append(Gate(g.name, g.qubits, tuple(-p for p in g.params)))
            else:
                inv.gates.append(Gate(swap_name.get(g.name, g.name), g.qubits, g.params))
        return inv

    # -- profile used by the cost model -------------------------------------
    @property
    def n_1q(self) -> int:
        return sum(1 for g in self.gates if len(g.qubits) == 1)

    @property
    def n_2q(self) -> int:
        return sum(1 for g in self.gates if len(g.qubits) == 2)

    def depth(self) -> int:
        level = [0] * self.n_qubits
        for g in self.gates:
            d = max(level[q] for q in g.qubits) + 1
            for q in g.qubits:
                level[q] = d
        return max(level, default=0)

    def profile(self) -> dict:
        return {"n_qubits": self.n_qubits, "depth": self.depth(), "n_1q": self.n_1q, "n_2q": self.n_2q}


def ghz(n: int) -> Circuit:
    c = Circuit(n, name=f"ghz{n}")
    c.h(0)
    for i in range(n - 1):
        c.cx(i, i + 1)
    return c


def normalise_counts(counts: dict, n_qubits: int) -> dict[str, int]:
    """Coerce counts from any SDK into {bitstring(q0 first): int}."""
    out: dict[str, int] = {}
    for k, v in counts.items():
        key = str(k).replace(" ", "")
        if key.isdigit() and len(key) != n_qubits:  # integer-encoded outcome
            key = format(int(key), f"0{n_qubits}b")
        out[key] = out.get(key, 0) + int(v)
    return out


def probabilities(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
