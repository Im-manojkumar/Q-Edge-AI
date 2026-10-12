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


class CalibratedIBMBackend(QuantumBackend):
    """Calibration-aware IBM routing: for each circuit, pick the device and the
    physical qubits with the highest estimated success probability from today's
    calibration, then run there.

    With IBM credentials it considers every operational device on the account
    and runs on the real one it picks. Without credentials it does the same over
    calibration snapshots of real IBM devices and runs on a local emulator of
    the chosen one (that device's own noise model), so the whole decision path
    can be exercised for free.
    """

    framework = "qiskit"

    def __init__(self, devices: list[str] | None = None, prefer_live: bool = True, queue_weight: float = 0.0,
                 noise: NoiseSpec | None = None):
        super().__init__(noise)
        self._device_names = devices
        self.prefer_live = prefer_live
        self.queue_weight = queue_weight
        self._devices = None
        self._layouts: dict = {}
        self._sims: dict = {}
        self.name = "ibm-auto"
        self.last_choice = None

    @property
    def devices(self):
        if self._devices is None:
            from ..calibration import load_devices
            self._devices = load_devices(self.prefer_live, self._device_names)
        return self._devices

    @property
    def live(self) -> bool:
        return bool(self.devices) and self.devices[0].source == "live"

    @property
    def hardware(self):  # type: ignore[override]
        return self.live

    @property
    def max_qubits(self):  # type: ignore[override]
        # Emulation has to simulate the active qubits, so stay well below device size.
        return max(d.num_qubits for d in self.devices) if self.live else 24

    def available(self):
        try:
            devs = self.devices
        except Exception as e:  # pragma: no cover
            return False, f"no calibration data: {e}"
        kind = "live IBM devices" if self.live else "IBM calibration snapshots (local emulation)"
        return True, f"Calibration-aware routing over {len(devs)} {kind}"

    def queue_depth(self):
        known = [d.pending_jobs for d in self.devices if d.pending_jobs is not None]
        return min(known) if known else None

    def refresh(self):
        """Reload calibrations (live data changes through the day) and drop cached layouts."""
        self._devices = None
        self._layouts.clear()

    @staticmethod
    def _structure(c: Circuit) -> tuple:
        # Circuits that differ only in angles share a layout (e.g. every QAOA iteration).
        return (c.n_qubits, tuple((g.name, g.qubits) for g in c.gates))

    def choose(self, c: Circuit):
        from ..calibration import select_device
        key = self._structure(c)
        if key not in self._layouts:
            best, _ = select_device(to_qiskit(c), self.devices, queue_weight=self.queue_weight)
            self._layouts[key] = best
        return self._layouts[key]

    def _isa(self, c: Circuit, choice):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        if self._structure(c) == self._structure(self._chosen_for):
            return choice.circuit
        cal = next(d for d in self.devices if d.name == choice.device)
        pm = generate_preset_pass_manager(optimization_level=1, backend=cal.backend,
                                          initial_layout=choice.layout[: c.n_qubits], seed_transpiler=11)
        return pm.run(to_qiskit(c))

    def _run(self, circuits, shots):
        big = max(circuits, key=lambda c: (c.n_qubits, len(c.gates)))
        choice = self.choose(big)
        self._chosen_for = big
        self.last_choice = choice
        cal = next(d for d in self.devices if d.name == choice.device)
        isa = [self._isa(c, choice) for c in circuits]
        meta = {"device": choice.device, "calibration": cal.source, "layout": choice.layout,
                "predicted_esp": choice.esp, "qiskit_default_esp": choice.default_esp,
                "transpiled_depths": [qc.depth() for qc in isa]}
        if self.live:
            from qiskit_ibm_runtime import SamplerV2
            import time as _t
            t0 = _t.perf_counter()
            job = SamplerV2(mode=cal.backend).run(isa, shots=shots)
            result = job.result()
            total = _t.perf_counter() - t0
            try:
                usage = job.usage()
            except Exception:
                usage = None
            counts = [qiskit_counts(r.data.meas.get_counts(), c.n_qubits) for r, c in zip(result, circuits)]
            meta.update(job_ids=[job.job_id()], qpu_seconds=usage, queue_time_s=max(0.0, total - (usage or 0.0)))
            return counts, meta
        from qiskit_aer import AerSimulator
        if cal.name not in self._sims:
            self._sims[cal.name] = AerSimulator.from_backend(cal.backend)
        res = self._sims[cal.name].run(isa, shots=shots).result()
        return [qiskit_counts(res.get_counts(i), c.n_qubits) for i, c in enumerate(circuits)], meta
