"""PennyLane translation and simulators (default.qubit / default.mixed / lightning)."""
from __future__ import annotations

from ..circuit import Circuit, normalise_counts
from .base import QuantumBackend

_ONE = {"h": "Hadamard", "x": "PauliX", "y": "PauliY", "z": "PauliZ", "s": "S"}


def apply_pennylane(c: Circuit, noise=None):
    import pennylane as qml

    for g in c.gates:
        q, p, n = g.qubits, g.params, g.name
        if n in _ONE:
            getattr(qml, _ONE[n])(wires=q[0])
        elif n == "sdg":
            qml.adjoint(qml.S(wires=q[0]))
        elif n == "rx":
            qml.RX(p[0], wires=q[0])
        elif n == "ry":
            qml.RY(p[0], wires=q[0])
        elif n == "rz":
            qml.RZ(p[0], wires=q[0])
        elif n == "cx":
            qml.CNOT(wires=list(q))
        elif n == "cz":
            qml.CZ(wires=list(q))
        elif n == "rzz":
            qml.IsingZZ(p[0], wires=list(q))
        elif n == "swap":
            qml.SWAP(wires=list(q))
        if noise is not None and not noise.is_noiseless:
            prob = noise.p1 if len(q) == 1 else noise.p2
            if prob > 0:
                for w in q:
                    qml.DepolarizingChannel(prob, wires=w)


class PennyLaneBackend(QuantumBackend):
    framework = "pennylane"
    max_qubits = 26

    def __init__(self, device: str = "default.qubit", noise=None):
        super().__init__(noise)
        if noise is not None and not noise.is_noiseless:
            device = "default.mixed"  # channels need a density-matrix simulator
            self.max_qubits = 12
        self.device = device
        self.name = f"pennylane:{device}"

    def available(self):
        try:
            import pennylane as qml
            qml.device(self.device, wires=1)
            return True, f"PennyLane {self.device}"
        except Exception as e:  # pragma: no cover
            return False, str(e)

    def _run(self, circuits, shots):
        import pennylane as qml

        out = []
        for c in circuits:
            dev = qml.device(self.device, wires=c.n_qubits)

            def fn(c=c):
                apply_pennylane(c, self.noise)
                return qml.counts(wires=list(range(c.n_qubits)))

            raw = qml.set_shots(qml.QNode(fn, dev), shots=shots)()
            out.append(normalise_counts({str(k): int(v) for k, v in raw.items()}, c.n_qubits))
        return out, {}
