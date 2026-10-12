"""Qiskit translation plus the Aer simulator and an IBM-device emulator."""
from __future__ import annotations

import math

from ..circuit import Circuit, normalise_counts
from .base import NoiseSpec, QuantumBackend


def to_qiskit(c: Circuit, measure: bool = True):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(c.n_qubits, name=c.name)
    for g in c.gates:
        q, p = g.qubits, g.params
        n = g.name
        if n in ("h", "x", "y", "z", "s", "sdg"):
            getattr(qc, n)(q[0])
        elif n in ("rx", "ry", "rz"):
            getattr(qc, n)(p[0], q[0])
        elif n == "cx":
            qc.cx(q[0], q[1])
        elif n == "cz":
            qc.cz(q[0], q[1])
        elif n == "rzz":
            qc.rzz(p[0], q[0], q[1])
        elif n == "swap":
            qc.swap(q[0], q[1])
    if measure:
        qc.measure_all()
    return qc


def qiskit_counts(raw: dict, n: int) -> dict[str, int]:
    # Qiskit prints qubit 0 as the rightmost bit; flip to our convention.
    return normalise_counts({k.replace(" ", "")[::-1]: v for k, v in raw.items()}, n)


def _aer_noise_model(noise: NoiseSpec):
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    model = NoiseModel()
    # Qiskit's depolarizing_error(lam) = (1-lam) rho + lam I/2, i.e. Pauli prob 3*lam/4.
    if noise.p1 > 0:
        e1 = depolarizing_error(min(1.0, 4 * noise.p1 / 3), 1)
        model.add_all_qubit_quantum_error(e1, ["h", "x", "y", "z", "s", "sdg", "rx", "ry", "rz"])
    if noise.p2 > 0:
        e = depolarizing_error(min(1.0, 4 * noise.p2 / 3), 1)
        model.add_all_qubit_quantum_error(e.tensor(e), ["cx", "cz", "rzz", "swap"])
    return model


class QiskitAerBackend(QuantumBackend):
    name = "qiskit-aer"
    framework = "qiskit"
    max_qubits = 30

    def available(self):
        try:
            import qiskit_aer  # noqa: F401
            return True, "Qiskit Aer local simulator"
        except ImportError as e:  # pragma: no cover
            return False, f"qiskit-aer not installed ({e})"

    def _run(self, circuits, shots):
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        kwargs = {} if self.noise.is_noiseless else {"noise_model": _aer_noise_model(self.noise)}
        sim = AerSimulator(**kwargs)
        qcs = [to_qiskit(c) for c in circuits]
        # Basis gates are kept as-is so the injected noise sees the same gate list
        # as the other frameworks.
        qcs = transpile(qcs, sim, optimization_level=0, num_processes=1)
        res = sim.run(qcs, shots=shots).result()
        return [qiskit_counts(res.get_counts(i), c.n_qubits) for i, c in enumerate(circuits)], {}


class QiskitFakeIBMBackend(QuantumBackend):
    """Realistic IBM device emulator: real topology, basis gates and calibrated
    noise from a snapshot of an IBM QPU, run locally on Aer. No account needed.
    Used as the stand-in when real IBM hardware is not configured."""

    framework = "qiskit"
    hardware = False

    def __init__(self, device: str = "FakeTorino", noise: NoiseSpec | None = None):
        super().__init__(noise)
        self.device = device
        self.name = f"ibm-emulator:{device}"
        self._backend = None
        self._sim = None
        self._pm = None

    def _get(self):
        if self._backend is None:
            from qiskit_ibm_runtime import fake_provider
            self._backend = getattr(fake_provider, self.device)()
        return self._backend

    def available(self):
        try:
            self._get()
            return True, f"Local emulation of {self.device} noise and topology"
        except Exception as e:  # pragma: no cover
            return False, str(e)

    @property
    def max_qubits(self):  # type: ignore[override]
        # Aer has to simulate the qubits a circuit touches, so cap well below device size.
        return min(self._get().num_qubits, 24)

    def _run(self, circuits, shots):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer import AerSimulator

        be = self._get()
        if self._sim is None:
            self._sim = AerSimulator.from_backend(be)
            self._pm = generate_preset_pass_manager(optimization_level=1, backend=be, seed_transpiler=7)
        sim, pm = self._sim, self._pm
        isa = pm.run([to_qiskit(c) for c in circuits], num_processes=1)
        res = sim.run(isa, shots=shots).result()
        depths = [qc.depth() for qc in isa]
        return ([qiskit_counts(res.get_counts(i), c.n_qubits) for i, c in enumerate(circuits)],
                {"transpiled_depths": depths})


def estimate_fidelity(c: Circuit, noise: NoiseSpec) -> float:
    """Crude success-probability estimate used by the cost model."""
    return math.prod([(1 - noise.p1)] * c.n_1q + [(1 - noise.p2) ** 2] * c.n_2q)
