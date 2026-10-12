"""Cirq translation and simulators (state-vector, or density-matrix with noise)."""
from __future__ import annotations

import math

from ..circuit import Circuit
from .base import QuantumBackend


def to_cirq(c: Circuit, noise=None):
    import cirq

    qs = cirq.LineQubit.range(c.n_qubits)
    ops = []
    for g in c.gates:
        q = [qs[i] for i in g.qubits]
        p, n = g.params, g.name
        if n == "h":
            ops.append(cirq.H(q[0]))
        elif n in ("x", "y", "z"):
            ops.append(getattr(cirq, n.upper())(q[0]))
        elif n == "s":
            ops.append(cirq.S(q[0]))
        elif n == "sdg":
            ops.append(cirq.S(q[0]) ** -1)
        elif n == "rx":
            ops.append(cirq.rx(p[0])(q[0]))
        elif n == "ry":
            ops.append(cirq.ry(p[0])(q[0]))
        elif n == "rz":
            ops.append(cirq.rz(p[0])(q[0]))
        elif n == "cx":
            ops.append(cirq.CNOT(q[0], q[1]))
        elif n == "cz":
            ops.append(cirq.CZ(q[0], q[1]))
        elif n == "rzz":
            # ZZPowGate(t, global_shift=-1/2) == exp(-i*pi*t/2 * ZZ)
            ops.append(cirq.ZZPowGate(exponent=p[0] / math.pi, global_shift=-0.5)(q[0], q[1]))
        elif n == "swap":
            ops.append(cirq.SWAP(q[0], q[1]))
        if noise is not None and not noise.is_noiseless:
            prob = noise.p1 if len(q) == 1 else noise.p2
            if prob > 0:
                ops.extend(cirq.depolarize(prob)(x) for x in q)
    circ = cirq.Circuit(ops)
    circ.append(cirq.measure(*qs, key="m"))
    return circ


class CirqBackend(QuantumBackend):
    framework = "cirq"
    max_qubits = 28

    def __init__(self, noise=None):
        super().__init__(noise)
        self.noisy = noise is not None and not noise.is_noiseless
        self.name = "cirq:density-matrix" if self.noisy else "cirq:simulator"
        if self.noisy:
            self.max_qubits = 12

    def available(self):
        try:
            import cirq  # noqa: F401
            return True, "Cirq local simulator"
        except ImportError as e:  # pragma: no cover
            return False, str(e)

    def _run(self, circuits, shots):
        import cirq

        sim = cirq.DensityMatrixSimulator(seed=None) if self.noisy else cirq.Simulator()
        out = []
        for c in circuits:
            hist = sim.run(to_cirq(c, self.noise), repetitions=shots).histogram(key="m")
            # Cirq packs measurement into an int with qubit 0 as the most significant bit.
            out.append({format(k, f"0{c.n_qubits}b"): int(v) for k, v in hist.items()})
        return out, {}
