"""Backend registry: one name per quantum stack, with hardware fallback."""
from __future__ import annotations

import logging

from .base import NoiseSpec, QuantumBackend, RunResult
from .braket_backend import BraketCloudBackend, BraketLocalBackend
from .cirq_backend import CirqBackend
from .ibm_backend import IBMQuantumBackend
from .pennylane_backend import PennyLaneBackend
from .qiskit_backend import QiskitAerBackend, QiskitFakeIBMBackend

log = logging.getLogger(__name__)

# The four simulator stacks compared head-to-head.
SIMULATOR_STACKS = ["qiskit", "pennylane", "cirq", "braket-local"]
HARDWARE_STACKS = ["ibm", "braket"]

_FACTORIES = {
    "qiskit": lambda noise: QiskitAerBackend(noise),
    "pennylane": lambda noise: PennyLaneBackend("default.qubit", noise),
    "pennylane-lightning": lambda noise: PennyLaneBackend("lightning.qubit", noise),
    "cirq": lambda noise: CirqBackend(noise),
    "braket-local": lambda noise: BraketLocalBackend(noise),
    "ibm-emulator": lambda noise: QiskitFakeIBMBackend("FakeTorino"),
    "ibm": lambda noise: IBMQuantumBackend(),
    "braket": lambda noise: BraketCloudBackend(),
}

# What a hardware name degrades to when the user has no credentials.
_FALLBACK = {"ibm": "ibm-emulator", "braket": "braket-local"}
_FALLBACK_NOISE = {"braket": NoiseSpec.from_level(1e-3)}


def available_backends() -> list[str]:
    return list(_FACTORIES)


def get_backend(name: str, noise: NoiseSpec | None = None, allow_fallback: bool = True) -> QuantumBackend:
    if name not in _FACTORIES:
        raise KeyError(f"unknown backend {name!r}; choose from {list(_FACTORIES)}")
    be = _FACTORIES[name](noise)
    ok, why = be.available()
    if ok:
        return be
    if allow_fallback and name in _FALLBACK:
        fb = _FACTORIES[_FALLBACK[name]](_FALLBACK_NOISE.get(name, noise))
        log.warning("%s unavailable (%s); falling back to %s", name, why, fb.name)
        fb.fallback_from = name  # type: ignore[attr-defined]
        fb.fallback_reason = why  # type: ignore[attr-defined]
        return fb
    raise RuntimeError(f"backend {name} unavailable: {why}")


def run_on(name: str, circuits, shots: int = 1024, noise: NoiseSpec | None = None) -> RunResult:
    be = get_backend(name, noise)
    res = be.run(circuits, shots)
    res.fallback_from = getattr(be, "fallback_from", None)
    return res


def provider_status() -> list[dict]:
    """Which stacks and real devices are usable right now (shown on the dashboard)."""
    rows = []
    for name, fac in _FACTORIES.items():
        try:
            be = fac(None)
            ok, why = be.available()
            rows.append({"backend": name, "framework": be.framework, "real_hardware": be.hardware,
                         "available": ok, "detail": why,
                         "fallback": None if ok else _FALLBACK.get(name)})
        except Exception as e:  # pragma: no cover
            rows.append({"backend": name, "available": False, "detail": str(e)})
    return rows


__all__ = ["NoiseSpec", "QuantumBackend", "RunResult", "get_backend", "run_on", "provider_status",
           "available_backends", "SIMULATOR_STACKS", "HARDWARE_STACKS"]
