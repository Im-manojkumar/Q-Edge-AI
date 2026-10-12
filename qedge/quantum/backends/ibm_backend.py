"""Real IBM Quantum hardware through Qiskit Runtime (SamplerV2).

Uses the user's own IBM Quantum Platform account (the free Open Plan gives
10 minutes of QPU time per 28 days). Set:

    QISKIT_IBM_TOKEN     API key from quantum.cloud.ibm.com
    QISKIT_IBM_INSTANCE  optional instance CRN / name
    IBM_QUANTUM_BACKEND  optional device name, e.g. ibm_torino (default: least busy)

or save the account once with QiskitRuntimeService.save_account(...).
"""
from __future__ import annotations

import os
import time

from .base import QuantumBackend
from .qiskit_backend import qiskit_counts, to_qiskit


class IBMQuantumBackend(QuantumBackend):
    framework = "qiskit"
    hardware = True

    def __init__(self, backend_name: str | None = None):
        super().__init__(None)
        self.backend_name = backend_name or os.environ.get("IBM_QUANTUM_BACKEND")
        self.name = f"ibm:{self.backend_name or 'least-busy'}"
        self._service = None
        self._backend = None
        self.max_qubits = 127

    def _service_obj(self):
        if self._service is None:
            from qiskit_ibm_runtime import QiskitRuntimeService
            token = os.environ.get("QISKIT_IBM_TOKEN")
            kwargs = {"channel": os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum_platform")}
            if token:
                kwargs["token"] = token
            if os.environ.get("QISKIT_IBM_INSTANCE"):
                kwargs["instance"] = os.environ["QISKIT_IBM_INSTANCE"]
            self._service = QiskitRuntimeService(**kwargs)
        return self._service

    def _get(self, min_qubits: int = 1):
        if self._backend is None:
            svc = self._service_obj()
            if self.backend_name:
                self._backend = svc.backend(self.backend_name)
            else:
                self._backend = svc.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)
            self.name = f"ibm:{self._backend.name}"
            self.max_qubits = self._backend.num_qubits
        return self._backend

    @staticmethod
    def credentials_present() -> bool:
        if os.environ.get("QISKIT_IBM_TOKEN"):
            return True
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            return bool(QiskitRuntimeService.saved_accounts())
        except Exception:
            return False

    def available(self):
        if not self.credentials_present():
            return False, "No IBM Quantum credentials (set QISKIT_IBM_TOKEN)"
        try:
            be = self._get()
            st = be.status()
            if not st.operational:
                return False, f"{be.name} not operational: {st.status_msg}"
            return True, f"IBM Quantum {be.name} ({be.num_qubits} qubits)"
        except Exception as e:
            return False, f"IBM Quantum unavailable: {e}"

    def queue_depth(self):
        try:
            return int(self._get().status().pending_jobs)
        except Exception:
            return None

    def _run(self, circuits, shots):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2

        be = self._get(max(c.n_qubits for c in circuits))
        pm = generate_preset_pass_manager(optimization_level=2, backend=be)
        isa = pm.run([to_qiskit(c) for c in circuits], num_processes=1)
        sampler = SamplerV2(mode=be)
        t0 = time.perf_counter()
        job = sampler.run(isa, shots=shots)
        result = job.result()
        total = time.perf_counter() - t0
        usage = None
        try:
            usage = job.usage()  # QPU seconds billed
        except Exception:
            pass
        counts = [qiskit_counts(r.data.meas.get_counts(), c.n_qubits) for r, c in zip(result, circuits)]
        return counts, {"job_ids": [job.job_id()], "submit_to_result_s": total, "qpu_seconds": usage,
                        "queue_time_s": max(0.0, total - (usage or 0.0)),
                        "transpiled_depths": [qc.depth() for qc in isa]}
