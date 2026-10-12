"""Calibration-aware QPU routing.

IBM publishes, for every device, per-qubit readout error, T1/T2 and 1-qubit gate
error, and per-coupler 2-qubit gate error, plus the number of jobs waiting.
Reading them costs no QPU time. This module turns that data into routing
decisions:

* ``load_devices()`` returns live calibrations when IBM credentials are set,
  otherwise calibration *snapshots* of real IBM devices that ship with
  qiskit-ibm-runtime (real numbers from a past day, not made up).
* ``best_layout()`` picks the physical qubits a circuit should run on by
  enumerating every placement of its interaction graph on the device coupling
  map and scoring each by estimated success probability (ESP), in the spirit
  of mapomatic (Nation & Treinish, PRX Quantum 2023).
* ``select_device()`` does the same across devices and adds the queue.
* ``region_figures()`` gives the cost model per-workload error figures for the
  best n-qubit region of each device, instead of one static number.
* ``drift()`` compares two calibrations of the same device.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache

# Snapshot devices used when no IBM account is configured: three Eagle (ECR)
# and five Heron (CZ) generations, so device choice is a real trade-off.
SNAPSHOT_DEVICES = ["FakeTorino", "FakeFez", "FakeMarrakesh", "FakeKingston", "FakeAachen",
                    "FakeBrisbane", "FakeSherbrooke", "FakeKyiv"]
TWO_Q_GATES = ("cz", "ecr", "cx")
ONE_Q_GATES = ("sx", "x")
AVG_JOB_S = 20.0           # assumed seconds per job ahead of us in a public queue
BROKEN = 0.5               # gate or readout error at or above this is treated as unusable
TRACE_PATH = "results/ibm_trace.jsonl"


@dataclass
class DeviceCalibration:
    name: str
    num_qubits: int
    source: str                                  # "live" | "snapshot"
    two_q_gate: str
    taken_at: str = ""                           # provider's calibration timestamp, if known
    readout: dict[int, float] = field(default_factory=dict)
    one_q: dict[int, float] = field(default_factory=dict)
    two_q: dict[tuple[int, int], float] = field(default_factory=dict)
    t1_us: dict[int, float] = field(default_factory=dict)
    t2_us: dict[int, float] = field(default_factory=dict)
    pending_jobs: int | None = None
    backend: object = field(default=None, repr=False, compare=False)

    # ---- summaries ---------------------------------------------------------
    def _good(self, d: dict) -> list[float]:
        return [v for v in d.values() if v < BROKEN]

    @property
    def median_readout(self) -> float:
        return statistics.median(self._good(self.readout))

    @property
    def median_1q(self) -> float:
        return statistics.median(self._good(self.one_q))

    @property
    def median_2q(self) -> float:
        return statistics.median(self._good(self.two_q))

    @property
    def broken_couplers(self) -> int:
        return sum(1 for v in self.two_q.values() if v >= BROKEN) // 2

    @property
    def queue_s(self) -> float | None:
        return None if self.pending_jobs is None else self.pending_jobs * AVG_JOB_S

    def edge_error(self, a: int, b: int) -> float:
        return self.two_q.get((a, b), self.two_q.get((b, a), 1.0))

    def summary(self) -> dict:
        return {"device": self.name, "source": self.source, "qubits": self.num_qubits,
                "two_q_gate": self.two_q_gate, "median_2q_error": self.median_2q,
                "median_readout_error": self.median_readout, "median_1q_error": self.median_1q,
                "median_t1_us": statistics.median(self.t1_us.values()) if self.t1_us else None,
                "broken_couplers": self.broken_couplers, "pending_jobs": self.pending_jobs,
                "taken_at": self.taken_at}

    def to_json(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "backend"}
        d["two_q"] = {f"{a}-{b}": v for (a, b), v in self.two_q.items()}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "DeviceCalibration":
        d = dict(d)
        d["two_q"] = {tuple(int(x) for x in k.split("-")): v for k, v in d["two_q"].items()}
        for k in ("readout", "one_q", "t1_us", "t2_us"):
            d[k] = {int(q): v for q, v in d.get(k, {}).items()}
        return cls(**d)


def from_backend(be, source: str, pending_jobs: int | None = None) -> DeviceCalibration:
    """Read calibration data from any Qiskit BackendV2 (real device or snapshot)."""
    t = be.target
    two = next(g for g in TWO_Q_GATES if g in t.operation_names)
    one = next(g for g in ONE_Q_GATES if g in t.operation_names)

    def err(props):
        return 1.0 if props is None or props.error is None else float(props.error)

    cal = DeviceCalibration(name=be.name, num_qubits=be.num_qubits, source=source, two_q_gate=two,
                            pending_jobs=pending_jobs, backend=be)
    cal.readout = {qs[0]: err(p) for qs, p in t["measure"].items()}
    cal.one_q = {qs[0]: err(p) for qs, p in t[one].items()}
    cal.two_q = {(qs[0], qs[1]): err(p) for qs, p in t[two].items()}
    for q in range(be.num_qubits):
        try:
            qp = be.qubit_properties(q)
        except Exception:
            qp = None
        if qp is not None and qp.t1:
            cal.t1_us[q] = qp.t1 * 1e6
        if qp is not None and qp.t2:
            cal.t2_us[q] = qp.t2 * 1e6
    try:
        props = be.properties()
        cal.taken_at = str(getattr(props, "last_update_date", "") or "")
    except Exception:
        pass
    return cal


@lru_cache(maxsize=None)
def snapshot(device: str) -> DeviceCalibration:
    from qiskit_ibm_runtime import fake_provider
    return from_backend(getattr(fake_provider, device)(), "snapshot")


def live_devices(min_qubits: int = 1) -> list[DeviceCalibration]:
    """Every operational IBM device this account can use, with today's calibration."""
    from .backends.ibm_backend import IBMQuantumBackend
    svc = IBMQuantumBackend()._service_obj()
    out = []
    for be in svc.backends(operational=True, simulator=False, min_num_qubits=min_qubits):
        try:
            out.append(from_backend(be, "live", int(be.status().pending_jobs)))
        except Exception:  # one bad device should not stop routing
            continue
    return out


def load_devices(prefer_live: bool = True, devices: list[str] | None = None) -> list[DeviceCalibration]:
    """Live calibrations if an IBM account is configured, otherwise real snapshots."""
    from .backends.ibm_backend import IBMQuantumBackend
    if prefer_live and IBMQuantumBackend.credentials_present():
        try:
            live = live_devices()
            if live:
                return live
        except Exception:
            pass
    return [snapshot(d) for d in (devices or SNAPSHOT_DEVICES)]


# ---- estimated success probability -----------------------------------------
def esp(cal: DeviceCalibration, ops: list[tuple[str, tuple[int, ...]]], measured: list[int]) -> float:
    """Probability that no gate or readout error occurs, from per-qubit data."""
    log_p = 0.0
    for name, qs in ops:
        if len(qs) == 2:
            e = cal.edge_error(*qs)
        elif name in ("rz", "delay", "barrier", "id", "reset"):
            continue  # virtual or free on IBM devices
        else:
            e = cal.one_q.get(qs[0], 1.0)
        if e >= 1.0:
            return 0.0
        log_p += math.log1p(-e)
    for q in measured:
        e = cal.readout.get(q, 1.0)
        if e >= 1.0:
            return 0.0
        log_p += math.log1p(-e)
    return math.exp(log_p)


def _ops(qc) -> tuple[list[tuple[str, tuple[int, ...]]], list[int]]:
    ops, meas = [], []
    for ins in qc.data:
        qs = tuple(qc.find_bit(q).index for q in ins.qubits)
        if ins.operation.name == "measure":
            meas.append(qs[0])
        elif ins.operation.name != "barrier":
            ops.append((ins.operation.name, qs))
    return ops, meas


@dataclass
class LayoutChoice:
    device: str
    layout: list[int]                 # physical qubit of each active virtual qubit
    esp: float
    default_esp: float                # ESP of the layout Qiskit picks on its own
    two_q_count: int
    depth: int
    n_layouts_scored: int
    queue_s: float | None = None
    circuit: object = field(default=None, repr=False)

    def row(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "circuit"}


def best_layout(cal: DeviceCalibration, qc, max_layouts: int = 4000, restarts: int = 8,
                seed: int = 11) -> LayoutChoice:
    """Transpile once, then try every placement of the routed circuit on the device
    and keep the one with the highest ESP."""
    import rustworkx as rx
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    be = cal.backend
    if be is None:
        be = snapshot(_class_name(cal.name)).backend
    pm = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=seed)
    tqc = pm.run(qc)
    ops, meas = _ops(tqc)
    default = esp(cal, ops, meas)

    active = sorted({q for _, qs in ops for q in qs} | set(meas))
    idx = {q: i for i, q in enumerate(active)}
    g = rx.PyGraph()
    g.add_nodes_from(active)
    for _, qs in ops:
        if len(qs) == 2:
            a, b = idx[qs[0]], idx[qs[1]]
            if not g.has_edge(a, b):
                g.add_edge(a, b, None)
    cm = rx.PyGraph()
    cm.add_nodes_from(range(cal.num_qubits))
    for (a, b), e in cal.two_q.items():
        if e < BROKEN and a < b and not cm.has_edge(a, b):
            cm.add_edge(a, b, None)
        elif e < BROKEN and a > b and not cm.has_edge(b, a):
            cm.add_edge(b, a, None)

    best_map, best_esp, n = {q: q for q in active}, default, 0
    if len(active) <= cal.num_qubits:
        # VF2 enumerates placements in node order, so every run would stay in one
        # corner of the chip. Restart on shuffled copies of the coupling map to
        # cover the whole device.
        rng = random.Random(seed)
        per_restart = max(1, max_layouts // restarts)
        for r in range(restarts):
            order = list(range(cal.num_qubits))
            if r:
                rng.shuffle(order)
            pos = {q: i for i, q in enumerate(order)}
            cm_r = rx.PyGraph()
            cm_r.add_nodes_from(order)
            for a, b in cm.edge_list():
                cm_r.add_edge(pos[a], pos[b], None)
            k = 0
            for mapping in rx.vf2_mapping(cm_r, g, subgraph=True, induced=False, call_limit=per_restart * 50):
                n += 1
                k += 1
                inv = {active[c]: order[d] for d, c in mapping.items()}
                p = esp(cal, [(nm, tuple(inv[q] for q in qs)) for nm, qs in ops], [inv[q] for q in meas])
                if p > best_esp:
                    best_esp, best_map = p, inv
                if k >= per_restart:
                    break
    # Rebuild the executable circuit on the chosen qubits.
    from qiskit import QuantumCircuit
    out = QuantumCircuit(tqc.num_qubits, tqc.num_clbits)
    for ins in tqc.data:
        qs = [out.qubits[best_map.get(tqc.find_bit(q).index, tqc.find_bit(q).index)] for q in ins.qubits]
        cs = [out.clbits[tqc.find_bit(c).index] for c in ins.clbits]
        out.append(ins.operation, qs, cs)
    if any(k != v for k, v in best_map.items()):
        # Moving qubits can flip a directional gate (ECR); re-translate in place.
        fix = generate_preset_pass_manager(optimization_level=0, backend=be, routing_method="none",
                                           initial_layout=list(range(out.num_qubits)), seed_transpiler=seed)
        fixed = fix.run(out)
        ops2, meas2 = _ops(fixed)
        if esp(cal, ops2, meas2) >= default:
            out, ops, best_esp = fixed, ops2, esp(cal, ops2, meas2)
        else:  # the extra gates cost more than the move gained; keep Qiskit's choice
            out, best_esp, best_map = tqc, default, {q: q for q in active}
    n_2q = sum(1 for _, qs in ops if len(qs) == 2)
    # layout for each virtual qubit of the original circuit
    init = tqc.layout.initial_index_layout(filter_ancillas=True) if tqc.layout is not None else list(range(qc.num_qubits))
    layout = [best_map.get(p, p) for p in init]
    return LayoutChoice(cal.name, layout, best_esp, default, n_2q, tqc.depth(), n, cal.queue_s, out)


def default_layout_circuit(cal: DeviceCalibration, qc, optimization_level: int = 1, seed: int = 7):
    """What v1 did: let the preset pass manager choose the qubits."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=cal.backend,
                                      seed_transpiler=seed)
    tqc = pm.run(qc)
    ops, meas = _ops(tqc)
    return tqc, esp(cal, ops, meas)


def select_device(qc, devices: list[DeviceCalibration], deadline_s: float | None = None,
                  queue_weight: float = 0.0) -> tuple[LayoutChoice, list[LayoutChoice]]:
    """Best device for a circuit: highest ESP, optionally penalising queue time.

    ``queue_weight`` is the ESP a user would trade for one hour less waiting.
    Devices whose known queue already exceeds the deadline are skipped.
    """
    choices = []
    for cal in devices:
        if qc.num_qubits > cal.num_qubits:
            continue
        if deadline_s is not None and cal.queue_s is not None and cal.queue_s > deadline_s:
            continue
        choices.append(best_layout(cal, qc))
    if not choices:
        raise RuntimeError("no device fits this circuit and deadline")

    def score(c: LayoutChoice) -> float:
        return c.esp - queue_weight * ((c.queue_s or 0.0) / 3600)

    return max(choices, key=score), choices


# ---- figures for the cost model --------------------------------------------
def best_region(cal: DeviceCalibration, n: int) -> list[int]:
    """Greedy best connected n-qubit region: start at the best coupler, grow along
    the lowest-error coupler, scoring qubits by readout error as well."""
    if n <= 1:
        return [min(cal.readout, key=cal.readout.get)]
    edges = sorted((e, a, b) for (a, b), e in cal.two_q.items() if a < b and e < BROKEN)
    if not edges:
        return list(range(n))
    best, best_cost = None, math.inf
    for _, a0, b0 in edges[:12]:  # a few seeds keep it fast and robust
        region = {a0, b0}
        cost = 0.0
        while len(region) < n:
            cand = [(e + cal.readout.get(b if a in region else a, 1.0), b if a in region else a)
                    for e, a, b in edges if (a in region) != (b in region)]
            if not cand:
                break
            c, q = min(cand)
            region.add(q)
            cost += c
        if len(region) == n and cost < best_cost:
            best, best_cost = sorted(region), cost
    return best or list(range(n))


def region_figures(cal: DeviceCalibration, n: int) -> dict:
    """p1 / p2 / readout for the best n-qubit region, in the cost model's convention
    (p2 is per qubit: a 2-qubit gate succeeds with probability (1 - p2)^2)."""
    region = best_region(cal, n)
    rs = set(region)
    e2 = [e for (a, b), e in cal.two_q.items() if a in rs and b in rs and a < b and e < BROKEN] or [cal.median_2q]
    e2m = statistics.fmean(e2)
    return {"device": cal.name, "source": cal.source, "region": region,
            "p1": statistics.fmean(cal.one_q[q] for q in region),
            "p2": 1 - math.sqrt(1 - e2m),
            "readout_err": statistics.fmean(cal.readout[q] for q in region),
            "max_qubits": cal.num_qubits, "queue_s": cal.queue_s}


def best_device_figures(profile, devices: list[DeviceCalibration]) -> dict:
    """Pick the device whose best region gives this circuit profile the highest
    fidelity, and return its figures for the cost model."""
    best, best_f = None, -1.0
    for cal in devices:
        if profile.n_qubits > cal.num_qubits:
            continue
        f = region_figures(cal, profile.n_qubits)
        fid = profile.fidelity(f["p1"], f["p2"], f["readout_err"])
        if fid > best_f:
            best, best_f = {**f, "fidelity": fid}, fid
    return best or {}


# ---- drift -----------------------------------------------------------------
def drift(old: DeviceCalibration, new: DeviceCalibration, factor: float = 1.5) -> dict:
    """What got worse between two calibrations of the same device."""
    worse_q = [q for q, e in new.readout.items() if e > factor * old.readout.get(q, e) and e > 0.02]
    worse_e = [k for k, e in new.two_q.items() if k[0] < k[1] and e > factor * old.two_q.get(k, e) and e > 0.005]
    newly_broken = [k for k, e in new.two_q.items() if k[0] < k[1] and e >= BROKEN and old.two_q.get(k, 0) < BROKEN]
    rel = new.median_2q / old.median_2q - 1
    return {"device": new.name, "median_2q_change": rel,
            "median_readout_change": new.median_readout / old.median_readout - 1,
            "worse_qubits": worse_q, "worse_couplers": worse_e, "newly_broken_couplers": newly_broken,
            "degraded": rel > factor - 1 or bool(newly_broken)}


def layout_still_good(cal: DeviceCalibration, choice: LayoutChoice, tolerance: float = 0.8) -> bool:
    """Re-score a cached layout on a fresh calibration; False means re-route."""
    ops, meas = _ops(choice.circuit)
    return esp(cal, ops, meas) >= tolerance * choice.esp


# ---- logging live data -------------------------------------------------------
def log_once(path: str = TRACE_PATH) -> list[dict]:
    """Append one record per live device (queue and full calibration) to a JSONL trace."""
    devs = live_devices()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    now = time.time()
    rows = []
    with open(path, "a") as f:
        for cal in devs:
            row = {"t": now, **cal.to_json()}
            f.write(json.dumps(row) + "\n")
            rows.append(row)
    return rows


def read_trace(path: str = TRACE_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def queue_trace_seconds(path: str = TRACE_PATH, device: str | None = None) -> list[float]:
    """Observed queue waits (pending jobs x AVG_JOB_S) from a logged trace."""
    return [r["pending_jobs"] * AVG_JOB_S for r in read_trace(path)
            if r.get("pending_jobs") is not None and (device is None or r["name"] == device)]


def drift_from_trace(path: str = TRACE_PATH) -> list[dict]:
    """Compare the first and last logged calibration of every device in a trace."""
    by_dev: dict[str, list[dict]] = {}
    for r in read_trace(path):
        by_dev.setdefault(r["name"], []).append(r)
    out = []
    for name, rows in by_dev.items():
        if len(rows) >= 2:
            a = DeviceCalibration.from_json({k: v for k, v in rows[0].items() if k != "t"})
            b = DeviceCalibration.from_json({k: v for k, v in rows[-1].items() if k != "t"})
            out.append({**drift(a, b), "hours": (rows[-1]["t"] - rows[0]["t"]) / 3600})
    return out


def _class_name(device_name: str) -> str:
    # "fake_torino" -> "FakeTorino"
    return "".join(p.capitalize() for p in device_name.split("_"))
