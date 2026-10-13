"""Qiskit translation plus the Aer simulator and an IBM-device emulator."""
from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

from ..circuit import Circuit, normalise_counts
from .base import NoiseSpec, QuantumBackend, await_job, current_token


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


# ---- opt-in ISA hook (used by qedge.quantum.mitigation) ----------------------
# Zero-noise extrapolation needs folded circuits that the transpiler has not had a
# chance to simplify (optimisation level >= 1 cancels U U^dagger pairs), and readout
# correction needs to know which physical qubit each classical bit was measured on.
# Both only exist after layout and routing, inside ``_run``. ``isa_scope`` lets a
# caller see and replace the final circuits on this thread without changing
# ``QuantumBackend.run``; outside a scope every backend behaves exactly as before.

@dataclass
class ISAScope:
    """``transform(isa_circuits, backend) -> circuits to execute`` (any length; each
    must already be executable on the target, it is not transpiled again).
    ``run_options`` go to ``AerSimulator.run`` (e.g. ``seed_simulator``); live devices
    ignore them. Counts come back one per *executed* circuit, read with each
    circuit's own classical-bit count."""

    transform: Callable | None = None
    run_options: dict = field(default_factory=dict)


_isa_local = threading.local()


@contextmanager
def isa_scope(transform: Callable | None = None, **run_options):
    stack = getattr(_isa_local, "stack", None)
    if stack is None:
        stack = _isa_local.stack = []
    scope = ISAScope(transform, run_options)
    stack.append(scope)
    try:
        yield scope
    finally:
        stack.pop()


def current_isa_scope() -> ISAScope | None:
    stack = getattr(_isa_local, "stack", None)
    return stack[-1] if stack else None


def _apply_scope(scope: ISAScope | None, isa: list, backend) -> list:
    if scope is None or scope.transform is None:
        return isa
    return list(scope.transform(list(isa), backend))


def _scoped_counts(res, isa: list) -> list[dict[str, int]]:
    return [qiskit_counts(res.get_counts(i), qc.num_clbits) for i, qc in enumerate(isa)]


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
        scope = current_isa_scope()
        if scope is not None:
            qcs = _apply_scope(scope, qcs, self)
            res = sim.run(qcs, shots=shots, **scope.run_options).result()
            return _scoped_counts(res, qcs), {"transpiled_depths": [qc.depth() for qc in qcs]}
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
        scope = current_isa_scope()
        if scope is not None:
            isa = _apply_scope(scope, isa, self)
            res = sim.run(isa, shots=shots, **scope.run_options).result()
            return _scoped_counts(res, isa), {"transpiled_depths": [qc.depth() for qc in isa]}
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

    Under a ``CancelToken`` (race mode) it checks the token once the layout
    search is done, before submitting or emulating. On a live device the
    Runtime job is registered on the token so a cancel reaches IBM (untested:
    no account was available); the local emulation, once started, runs to the end.
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

    @staticmethod
    def _signature(c: Circuit) -> tuple:
        return (c.n_qubits, tuple((g.name, g.qubits, g.params) for g in c.gates))

    def choose(self, c: Circuit):
        return self._choose(c)[0]

    def _choose(self, c: Circuit):
        """(layout choice, signature of the exact circuit it was searched on). Stored as one
        tuple so concurrent callers never pair one circuit's choice with another's signature."""
        from ..calibration import select_device
        key = self._structure(c)
        hit = self._layouts.get(key)
        if hit is None:
            best, _ = select_device(to_qiskit(c), self.devices, queue_weight=self.queue_weight)
            hit = self._layouts[key] = (best, self._signature(c))
        return hit

    def _isa(self, c: Circuit, choice, source: tuple):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        # choice.circuit is the transpiled copy of the circuit the layout was searched on,
        # angles included. Reusing it for a circuit that only shares the structure would
        # run the old angles (every QAOA iteration would repeat the first one), so any
        # other circuit is transpiled onto the same layout.
        if self._signature(c) == source:
            return choice.circuit
        cal = next(d for d in self.devices if d.name == choice.device)
        pm = generate_preset_pass_manager(optimization_level=1, backend=cal.backend,
                                          initial_layout=choice.layout[: c.n_qubits], seed_transpiler=11)
        return pm.run(to_qiskit(c))

    def _run(self, circuits, shots):
        # Layout and ISA are local to this call: a race can leave an abandoned emulator
        # thread still inside _run while the next job uses the same backend.
        big = max(circuits, key=lambda c: (c.n_qubits, len(c.gates)))
        choice, source = self._choose(big)
        self.last_choice = choice
        cal = next(d for d in self.devices if d.name == choice.device)
        isa = [self._isa(c, choice, source) for c in circuits]
        scope = current_isa_scope()
        isa = _apply_scope(scope, isa, self)
        meta = {"device": choice.device, "calibration": cal.source, "layout": choice.layout,
                "predicted_esp": choice.esp, "qiskit_default_esp": choice.default_esp,
                "transpiled_depths": [qc.depth() for qc in isa]}
        tok = current_token()
        if tok is not None:
            tok.raise_if_cancelled()  # the layout search may have outlived the deadline: don't submit
        if self.live:
            from qiskit_ibm_runtime import SamplerV2
            import time as _t
            t0 = _t.perf_counter()
            job = SamplerV2(mode=cal.backend).run(isa, shots=shots)
            result = await_job(job, tok)  # untested live: cancel() -> RuntimeJobV2.cancel()
            total = _t.perf_counter() - t0
            try:
                usage = job.usage()
            except Exception:
                usage = None
            if scope is not None:  # executed circuits may differ from the submitted ones
                counts = [qiskit_counts(_creg(r).get_counts(), qc.num_clbits) for r, qc in zip(result, isa)]
            else:
                counts = [qiskit_counts(r.data.meas.get_counts(), c.n_qubits) for r, c in zip(result, circuits)]
            meta.update(job_ids=[job.job_id()], qpu_seconds=usage, queue_time_s=max(0.0, total - (usage or 0.0)))
            return counts, meta
        from qiskit_aer import AerSimulator
        if cal.name not in self._sims:
            self._sims[cal.name] = AerSimulator.from_backend(cal.backend)
        if scope is not None:
            res = self._sims[cal.name].run(isa, shots=shots, **scope.run_options).result()
            return _scoped_counts(res, isa), meta
        res = self._sims[cal.name].run(isa, shots=shots).result()
        return [qiskit_counts(res.get_counts(i), c.n_qubits) for i, c in enumerate(circuits)], meta


def _creg(pub_result):
    # SamplerV2 names the data field after the circuit's classical register ("meas"
    # for measure_all, "c" for circuits rebuilt by calibration.best_layout).
    data = pub_result.data
    bits = getattr(data, "meas", None)
    return bits if bits is not None else next(iter(data.values()))
