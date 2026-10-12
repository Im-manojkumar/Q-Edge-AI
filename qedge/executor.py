"""Runs real payloads for each workload kind, classically or hybrid.

Classical work runs on this machine (it plays the role of whichever classical
tier was chosen; network transfer for cloud tiers is added from the model,
not slept). Quantum work runs on a real quantum stack: a local simulator from
any of the four SDKs, the IBM device emulator, or real IBM / Braket hardware
when credentials are configured.

``run(w, q, cancel=token)`` hands the quantum path a backend bound to a
``CancelToken``, so race mode can stop it: every circuit job checks the token
first, which ends a QAOA loop or a chunked kernel at its next job.

``split_inference`` (idea 3) also takes the ``placement``, because where the
network is cut is part of what runs: the trained digits model from
``qedge.ml.zoo`` runs its edge part up to the cut and its remote part from the
transmitted tensor (both on this machine; the transfer is modelled, as for the
other kinds), or hands the k-number embedding to the quantum head on a real
quantum stack. Its result says what left the edge, in bytes.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .classical.solvers import maxcut_annealing
from .quantum.algorithms import (magnetisation, maxcut_optimum, quantum_kernel, random_graph, run_qaoa,
                                 tfim_exact_magnetisation, tfim_trotter)
from .quantum.backends import QuantumBackend, get_backend
from .quantum.backends.base import CancellableBackend, CancelToken

HW_MAX_CIRCUITS = int(os.environ.get("QEDGE_HW_MAX_CIRCUITS", "60"))
HW_MAX_SHOTS = int(os.environ.get("QEDGE_HW_MAX_SHOTS", "2000"))


def KERNEL_BANDWIDTH(d: int) -> float:
    """Feature-map bandwidth 1/d (Shaydulin & Wild, 2022): keeps kernel values away from 0."""
    return 1.0 / d


@dataclass
class ExecResult:
    kind: str
    mode: str                  # "classical" | "hybrid"
    method: str
    quality: float
    quality_metric: str
    compute_s: float
    quantum_s: float = 0.0
    backend: str | None = None
    fallback_from: str | None = None
    details: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _digits():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    return train_test_split(X / 16.0, y, test_size=0.3, random_state=0)


@lru_cache(maxsize=1)
def _pretrained_mlp():
    from sklearn.neural_network import MLPClassifier
    Xtr, Xte, ytr, yte = _digits()
    return MLPClassifier((64,), max_iter=300, random_state=0).fit(Xtr, ytr)


def kernel_dataset(d: int, n_train: int, n_test: int, seed: int = 0):
    """Breast-cancer data reduced to d features (PCA) and scaled to [0, pi]."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[: n_train + n_test]
    Z = PCA(d, random_state=0).fit_transform((X - X.mean(0)) / X.std(0))[idx]
    Z = MinMaxScaler((0, np.pi)).fit_transform(Z)
    return Z[:n_train], Z[n_train:], y[idx][:n_train], y[idx][n_train:]


def representative_circuit(w):
    """One circuit with the structure this workload runs on a QPU. Angles do not
    change qubit placement, so this is what layout and device choice are made on."""
    from .quantum.algorithms import kernel_circuit, qaoa_circuit
    p = w.params
    if w.kind == "optimization":
        n, reps = int(p["n"]), int(p.get("p", 1))
        return qaoa_circuit(n, random_graph(n, p.get("density", 0.3), seed=w.id), [0.4] * reps, [0.3] * reps)
    if w.kind == "kernel_classification":
        d = int(p["features"])
        x = np.linspace(0.1, 1.0, d) * KERNEL_BANDWIDTH(d) * np.pi
        return kernel_circuit(x, x[::-1])
    if w.kind == "quantum_simulation":
        return tfim_trotter(int(p["n"]), int(p["steps"]))
    raise ValueError(f"{w.kind} has no quantum circuit")


# Samples per packed QPU job for the quantum head on ibm-auto (idea 2's packing): the setting
# of the first idea-3 measurement (40 k=4 test images on FakeTorino, 8 per job: 5 jobs, 1.45
# modelled QPU-s against 11.6 one per job). Samples that do not fit go into the next job.
PACK_PER_JOB = 8


class Executor:
    # Where split_inference finds its trained models (``qedge.ml.zoo``); quick models are
    # for tests and demos only (few epochs, not the reported accuracies).
    model_cache = "results/models"
    quick_models = False

    def __init__(self, sim_stack: str = "qiskit", qpu_stack: str = "ibm", resources: dict | None = None):
        self.sim_stack = sim_stack
        self.qpu_stack = qpu_stack
        self.resources = resources   # catalogue: which resources are on the edge (split_inference)
        self._backends: dict[str, QuantumBackend] = {}
        self._lock = threading.Lock()  # race mode calls backend_for from worker threads

    def backend_for(self, quantum_resource: str, cancel: CancelToken | None = None) -> QuantumBackend:
        name = self.sim_stack if quantum_resource == "quantum_sim" else self.qpu_stack
        with self._lock:
            if name not in self._backends:
                self._backends[name] = get_backend(name)
            be = self._backends[name]
        # The shared backend stays unbound; the token only governs this call's work.
        return CancellableBackend(be, cancel) if cancel is not None else be

    # ------------------------------------------------------------------
    def run(self, w, quantum_resource: str | None = None, cancel: CancelToken | None = None,
            placement=None) -> ExecResult:
        """Execute ``w`` classically (``quantum_resource=None``) or hybrid. With
        ``cancel``, the quantum path raises ``Cancelled`` at its next circuit job
        once the token fires. ``placement`` (an ``estimator.Placement``) is read by
        split_inference only; every other kind ignores it."""
        if w.kind == "split_inference":
            return self._split_inference(w, quantum_resource, cancel, placement)
        fn = getattr(self, f"_{w.kind}")
        return fn(w, quantum_resource, cancel)

    def _guard(self, be: QuantumBackend, n_circ: int, shots: int):
        if be.hardware and (n_circ > HW_MAX_CIRCUITS or shots > HW_MAX_SHOTS):
            raise RuntimeError(f"refusing {n_circ} circuits x {shots} shots on real hardware "
                               f"(limits QEDGE_HW_MAX_CIRCUITS={HW_MAX_CIRCUITS}, QEDGE_HW_MAX_SHOTS={HW_MAX_SHOTS})")

    def _inference(self, w, q, cancel=None):
        Xtr, Xte, ytr, yte = _digits()
        batch = int(w.params.get("batch", 8))
        rng = np.random.default_rng(w.id)
        idx = rng.integers(0, len(Xte), batch)
        model = _pretrained_mlp()
        t0 = time.perf_counter()
        pred = model.predict(Xte[idx])
        dt = time.perf_counter() - t0
        return ExecResult("inference", "classical", "mlp", float((pred == yte[idx]).mean()), "accuracy", dt,
                          details={"batch": batch})

    def _training(self, w, q, cancel=None):
        from sklearn.neural_network import MLPClassifier
        Xtr, Xte, ytr, yte = _digits()
        n = min(int(w.params.get("samples", 800)), len(Xtr))
        t0 = time.perf_counter()
        m = MLPClassifier((64,), max_iter=max(20, 20 * int(w.params.get("epochs", 1))), random_state=w.id)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xtr[:n], ytr[:n])
        dt = time.perf_counter() - t0
        return ExecResult("training", "classical", "mlp-sgd", float(m.score(Xte, yte)), "test accuracy", dt,
                          details={"samples": n})

    def _optimization(self, w, q, cancel=None):
        n = int(w.params["n"])
        edges = random_graph(n, w.params.get("density", 0.3), seed=w.id)
        ref = maxcut_optimum(n, edges) if n <= 20 else maxcut_annealing(n, edges, sweeps=3000, seed=99)[1]
        if q is None:
            t0 = time.perf_counter()
            if n <= 20:
                val, method = maxcut_optimum(n, edges), "exact enumeration"
            else:
                val, method = maxcut_annealing(n, edges)[1], "simulated annealing"
            dt = time.perf_counter() - t0
            return ExecResult("optimization", "classical", method, val / ref, "approximation ratio", dt,
                              details={"n": n, "edges": len(edges), "cut": val, "reference": ref})
        be = self.backend_for(q, cancel)
        iters = 8 if be.hardware else 25
        shots = 512 if be.hardware else 1000
        self._guard(be, 12 + iters + 1, shots)
        t0 = time.perf_counter()
        r = run_qaoa(be, n, edges, p=int(w.params.get("p", 1)), shots=shots, maxiter=iters, seed=w.id)
        dt = time.perf_counter() - t0
        return ExecResult("optimization", "hybrid", "QAOA", r.best_cut / ref, "approximation ratio",
                          r.classical_time_s, r.quantum_time_s, be.name, getattr(be, "fallback_from", None),
                          {"n": n, "edges": len(edges), "cut": r.best_cut, "expected_cut": r.expected_cut,
                           "reference": ref, "circuit_evals": r.n_circuit_evals, "jobs": r.n_jobs, "wall_s": dt})

    def _kernel_classification(self, w, q, cancel=None):
        from sklearn.svm import SVC
        d = int(w.params["features"])
        n_train, n_test = int(w.params["n_train"]), int(w.params["n_test"])
        Xtr, Xte, ytr, yte = kernel_dataset(d, n_train, n_test, seed=w.id)
        if q is None:
            t0 = time.perf_counter()
            acc = SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xtr, ytr).score(Xte, yte)
            return ExecResult("kernel_classification", "classical", "RBF-SVM", float(acc), "test accuracy",
                              time.perf_counter() - t0, details={"features": d, "n_train": n_train})
        be = self.backend_for(q, cancel)
        n_circ = n_train * (n_train - 1) // 2 + n_test * n_train
        self._guard(be, n_circ, 500)
        bw = KERNEL_BANDWIDTH(d)  # narrower feature map avoids an all-diagonal kernel
        Ktr, m1 = quantum_kernel(be, Xtr * bw, shots=500)
        Kte, m2 = quantum_kernel(be, Xte * bw, Xtr * bw, shots=500)
        t0 = time.perf_counter()
        acc = SVC(kernel="precomputed", C=1.0).fit(Ktr, ytr).score(Kte, yte)
        return ExecResult("kernel_classification", "hybrid", "quantum-kernel SVM", float(acc), "test accuracy",
                          time.perf_counter() - t0, m1["quantum_time_s"] + m2["quantum_time_s"], be.name,
                          getattr(be, "fallback_from", None),
                          {"features": d, "n_train": n_train, "circuits": m1["n_circuits"] + m2["n_circuits"]})

    def _quantum_simulation(self, w, q, cancel=None):
        n, steps = int(w.params["n"]), int(w.params["steps"])
        if n > 20:
            raise ValueError("live execution keeps quantum_simulation at <= 20 qubits so it can be verified exactly")
        t0 = time.perf_counter()
        exact = tfim_exact_magnetisation(n, steps)
        t_exact = time.perf_counter() - t0
        if q is None:
            return ExecResult("quantum_simulation", "classical", "state vector", 1.0, "1 - |error| in <Z>",
                              t_exact, details={"n": n, "steps": steps, "magnetisation": exact})
        be = self.backend_for(q, cancel)
        self._guard(be, 1, 4000)
        r = be.run([tfim_trotter(n, steps)], shots=4000)
        m = magnetisation(r.counts[0])
        return ExecResult("quantum_simulation", "hybrid", "Trotter circuit", float(max(0.0, 1 - abs(m - exact))),
                          "1 - |error| in <Z>", 0.0, r.wall_time_s, be.name, getattr(be, "fallback_from", None),
                          {"n": n, "steps": steps, "magnetisation": m, "exact": exact})

    def prepare_split(self, k: int):
        """Load the trained split model for embedding size ``k`` (training it on a cold cache),
        so that a later run, or a race, does not spend its time on it."""
        from .ml import zoo
        return zoo.get_model(int(k), cache_dir=self.model_cache, quick=self.quick_models)

    def _split_inference(self, w, q, cancel=None, placement=None):
        """A batch of test images through the trained split model, cut as ``placement`` says.

        * whole model (``edge_gpu``, ``cloud_gpu``): the network with the workload's
          classical head; for a cloud resource the raw images are what is sent;
        * classical split (``edge_gpu>conv2>cloud_gpu``): the front up to the cut, then
          the back from the float32 activation that would be transmitted;
        * quantum head (``edge_gpu>embedding>qpu``): the front up to the embedding, then
          the head's circuits on the quantum stack for that resource, its linear readout
          on the edge. On ``ibm-auto`` a batch is packed several samples per job
          (``QuantumHead.predict_packed``, idea 2); a single image, a simulator or any
          other stack runs one circuit per image (``predict_on_backend``).

        Without a placement: the whole model on the edge, or with ``q`` the quantum head
        behind an edge front (named edge_cpu; everything runs on this machine either way).
        Only the "digits" profile has a trained model; "camera" is an assumption profile.
        Images: ``batch`` test-split images drawn with the workload id as seed (without
        replacement), so the quantum path and its classical fallback see the same batch."""
        from .ml import zoo
        from .orchestrator import split
        from .orchestrator.estimator import Placement
        from .resources import load_catalogue

        p = w.params
        if p.get("profile", "digits") != "digits":
            raise ValueError(f"profile {p.get('profile')!r} has no trained model to run (an assumption profile); "
                             "only 'digits' runs for real")
        k, batch, shots = int(p["k"]), int(p["batch"]), int(p.get("shots", 1000))
        if placement is not None and q is not None and placement.quantum != q:
            raise ValueError(f"placement {placement.label} does not send the head to {q}")
        q = q or (placement.quantum if placement is not None else None)
        if placement is None:
            placement = Placement("edge_cpu", q, split="embedding") if q else Placement("edge_cpu")
        res = self.resources or load_catalogue()
        sent = split.transmitted(placement, res)
        cut = placement.split or ("embedding" if q else None)
        bad = split.shape_problem(placement, res)   # the cost model marks these infeasible; force must not run them
        if bad is not None:
            raise ValueError(f"cannot run {placement.label}: {bad}")

        tr = zoo.get_model(k, cache_dir=self.model_cache, quick=self.quick_models)
        Xte, yte = zoo.data()[4], zoo.data()[5]
        idx = np.random.default_rng(w.id).choice(len(Xte), batch, replace=batch > len(Xte))
        X, y = Xte[idx], yte[idx]
        head = p.get("qhead", "quantum") if q else p.get("head", "linear")
        net = tr.with_head("linear" if q else head)          # with q only its backbone is used
        front_on_edge = res[placement.classical].is_edge

        # Edge part: up to the cut. The tensor handed to the remote part is the float32 array
        # the edge would transmit (with a cloud front: the raw image); its size is what is sent.
        start = cut if (cut is not None and front_on_edge) else "input"
        t0 = time.perf_counter()
        H = net.forward_to(start, X) if sent else None
        t_front = time.perf_counter() - t0
        up = H.nbytes // batch if sent else 0
        details = {"placement": placement.label, "profile": "digits", "k": k, "batch": batch, "cut": cut,
                   "head": head, "front": placement.classical,
                   "remote": placement.back or q or (None if front_on_edge else placement.classical),
                   "left_edge": sent, "raw_data_left_edge": "input" in sent, "bytes_up_per_image": int(up),
                   "bytes_up_total": int(up * batch), "images": [int(i) for i in idx],
                   "model": {"source": tr.model.meta.get("zoo"), "quick": bool(self.quick_models),
                             "path": tr.model.meta.get("zoo_path")}}

        if q is None:
            t1 = time.perf_counter()
            logits = net.forward_from(start, H) if sent else net.forward_from("input", X)
            t_rest = time.perf_counter() - t1
            pred = np.argmax(logits, axis=1)
            unsplit = net.predict(X)               # float64 end to end, no float32 hand-over
            down = split.result_bytes(split.get_profile("digits")) if sent else 0
            details.update(edge_s=(t_front if front_on_edge else 0.0) + (0.0 if sent else t_rest),
                           remote_s=t_rest if sent else 0.0, bytes_down_per_image=down,
                           bytes_down_total=down * batch, agree_unsplit=float(np.mean(pred == unsplit)))
            method = f"split CNN ({head} head)" if placement.split else f"CNN ({head} head)"
            return ExecResult("split_inference", "classical", f"{method}, {placement.label}",
                              float(np.mean(pred == y)), "batch accuracy", t_front + t_rest, details=details)

        # Quantum head behind the front. A cloud host (``cloud_cpu+qpu``) received the raw
        # images and computes the embedding there; an edge front sends only the embedding.
        from .quantum.backends import CalibratedIBMBackend
        from .quantum.packing import packing_cost
        qh = tr.heads[head]
        t0 = time.perf_counter()
        Z = (H if start == "embedding" else net.forward_to("embedding", X)).astype(np.float64)
        t_host = time.perf_counter() - t0
        be = self.backend_for(q, cancel)
        if cancel is not None:
            cancel.raise_if_cancelled()
        inner = getattr(be, "inner", be)
        packed = isinstance(inner, CalibratedIBMBackend) and batch > 1
        if packed:
            run = qh.predict_packed(inner, Z, shots=shots, per_job=PACK_PER_JOB, cancel=cancel)
            device = ",".join(run.meta["devices"])
            qpu_s, qpu_s_solo, price = run.meta["qpu_s"], run.meta["qpu_s_solo"], run.meta["price"]
        else:
            self._guard(be, batch, shots)
            run = qh.predict_on_backend(be, Z, shots)
            device = run.meta.get("device") or run.backend
            qpu_s = qpu_s_solo = price = None
            if q != "quantum_sim":                 # a QPU route (emulated or real): modelled QPU time
                depths = run.meta.get("transpiled_depths") or [qh.circuit_profile()["depth"]] * batch
                c = packing_cost(list(depths), shots)
                qpu_s = qpu_s_solo = c.qpu_s_per_program_solo * batch
                price = c.price_per_program_solo * batch
        t1 = time.perf_counter()
        exact = qh.predict_exact(Z)                # reference: the same head without shot noise or device errors
        t_ref = time.perf_counter() - t1
        pred = run.predictions
        details.update(edge_s=t_front + t_host if front_on_edge else 0.0,
                       remote_s=0.0 if front_on_edge else t_front + t_host,
                       bytes_down_per_image=4 * k, bytes_down_total=4 * k * batch, shots=shots,
                       path="packed" if packed else "one circuit per image", device=device, jobs=run.n_jobs,
                       circuits=run.n_circuits, qpu_s=qpu_s, qpu_s_unpacked=qpu_s_solo, price_usd=price,
                       qpu_s_note=None if qpu_s is None else "modelled: packing_cost with catalogue QPU figures",
                       agree_exact_quantum=float(np.mean(pred == exact)),
                       exact_quantum_accuracy=float(np.mean(exact == y)), reference_s=t_ref)
        if packed:
            details["packing"] = {"per_job": PACK_PER_JOB, "jobs": [{kk: j[kk] for kk in ("device", "k", "depth",
                                  "unplaced", "emulated", "job_id")} for j in run.meta["jobs"]]}
        layers = qh.layers
        return ExecResult("split_inference", "hybrid",
                          f"split CNN + quantum head ({k} qubits, {layers} layers){', packed' if packed else ''}",
                          float(np.mean(pred == y)), "batch accuracy", t_front + t_host, run.wall_s, be.name,
                          getattr(be, "fallback_from", None) or run.fallback_from, details)
