"""Amazon Braket translation: local simulators and real cloud devices.

Real devices (IonQ, IQM, Rigetti, ... or the managed SV1/DM1 simulators) are
reached with the user's own AWS credentials. Set:

    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or AWS_PROFILE)
    AWS_DEFAULT_REGION        e.g. us-east-1
    BRAKET_DEVICE_ARN         required opt-in, e.g. arn:aws:braket:::device/qpu/iqm/Garnet
                              (or .../quantum-simulator/amazon/sv1 for the managed simulator)
    BRAKET_S3_BUCKET          optional; default Braket bucket is used otherwise
"""
from __future__ import annotations

import os
import time

from ..circuit import Circuit, normalise_counts
from .base import QuantumBackend

DEFAULT_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"


def to_braket(c: Circuit, noise=None):
    from braket.circuits import Circuit as BC

    bc = BC()
    for q in range(c.n_qubits):  # make every qubit appear so it is measured
        bc.i(q)
    for g in c.gates:
        q, p, n = g.qubits, g.params, g.name
        if n in ("h", "x", "y", "z", "s"):
            getattr(bc, n)(q[0])
        elif n == "sdg":
            bc.si(q[0])
        elif n in ("rx", "ry", "rz"):
            getattr(bc, n)(q[0], p[0])
        elif n == "cx":
            bc.cnot(q[0], q[1])
        elif n == "cz":
            bc.cz(q[0], q[1])
        elif n == "rzz":
            bc.zz(q[0], q[1], p[0])
        elif n == "swap":
            bc.swap(q[0], q[1])
        if noise is not None and not noise.is_noiseless:
            prob = noise.p1 if len(q) == 1 else noise.p2
            if prob > 0:
                for w in q:
                    bc.depolarizing(w, probability=min(prob, 0.75))
    return bc


class BraketLocalBackend(QuantumBackend):
    framework = "braket"
    max_qubits = 25

    def __init__(self, noise=None):
        super().__init__(noise)
        self.noisy = noise is not None and not noise.is_noiseless
        self.sim = "braket_dm" if self.noisy else "braket_sv"
        self.name = f"braket-local:{self.sim}"
        if self.noisy:
            self.max_qubits = 12

    def available(self):
        try:
            from braket.devices import LocalSimulator  # noqa: F401
            return True, "Amazon Braket local simulator"
        except ImportError as e:  # pragma: no cover
            return False, str(e)

    def _run(self, circuits, shots):
        from braket.devices import LocalSimulator

        dev = LocalSimulator(self.sim)
        out = []
        for c in circuits:
            res = dev.run(to_braket(c, self.noise), shots=shots).result()
            out.append(normalise_counts(dict(res.measurement_counts), c.n_qubits))
        return out, {}


class BraketCloudBackend(QuantumBackend):
    """Real Amazon Braket device or managed simulator, billed to the user's AWS account."""

    framework = "braket"
    hardware = True

    def __init__(self, device_arn: str | None = None):
        super().__init__(None)
        self.device_arn = device_arn or os.environ.get("BRAKET_DEVICE_ARN", "")
        self.hardware = bool(self.device_arn) and "quantum-simulator" not in self.device_arn
        self.name = "braket:" + (self.device_arn.rsplit("/", 1)[-1] or "unconfigured")
        self._device = None
        self.max_qubits = 25

    @staticmethod
    def credentials_present() -> bool:
        return bool(os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"))

    def _get(self):
        if self._device is None:
            from braket.aws import AwsDevice
            self._device = AwsDevice(self.device_arn)
            try:
                self.max_qubits = int(self._device.properties.paradigm.qubitCount)
            except Exception:
                pass
        return self._device

    def available(self):
        if not self.device_arn:
            return False, "Set BRAKET_DEVICE_ARN to opt in to a (billed) Braket device"
        if not self.credentials_present():
            return False, "No AWS credentials (set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)"
        try:
            dev = self._get()
            if str(dev.status) != "ONLINE":
                return False, f"{self.device_arn} status {dev.status}"
            return True, f"Amazon Braket device {dev.name}"
        except Exception as e:
            return False, f"Braket unavailable: {e}"

    def queue_depth(self):
        try:
            from braket.aws.queue_information import QueueType
            return int(self._get().queue_depth().quantum_tasks[QueueType.NORMAL])
        except Exception:
            return None

    def _run(self, circuits, shots):
        dev = self._get()
        kwargs = {}
        bucket = os.environ.get("BRAKET_S3_BUCKET")
        if bucket:
            kwargs["s3_destination_folder"] = (bucket, os.environ.get("BRAKET_S3_PREFIX", "qedge"))
        t0 = time.perf_counter()
        batch = dev.run_batch([to_braket(c) for c in circuits], shots=shots, **kwargs)
        results = batch.results(fail_unsuccessful=True)
        out = [normalise_counts(dict(r.measurement_counts), c.n_qubits) for r, c in zip(results, circuits)]
        return out, {"job_ids": [t.id for t in batch.tasks], "queue_time_s": 0.0,
                     "submit_to_result_s": time.perf_counter() - t0}
