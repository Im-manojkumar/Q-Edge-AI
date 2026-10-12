"""Spatial multi-programming ("job packing"): several small circuits from
different edge clients share ONE QPU job.

Every QPU job pays a queue wait, a fixed job overhead and, on Braket-style
pricing, a per-task fee plus a per-shot fee; IBM's free Open Plan meters QPU
seconds instead. The circuits Q-Edge sends (4-8 qubits) use a few percent of a
133-156 qubit chip. Running k of them side by side in one wide circuit shares
one queue wait, one task fee and one set of shots. The price is paid in
quality: later programs get worse qubits than they would alone, and programs
on neighbouring qubits can disturb each other (crosstalk). This idea is not
ours; it follows Das et al., "A Case for Multi-Programming Quantum Computers"
(MICRO 2019), Liu & Dou, "QuCloud" (2021) and Ohkura et al. (2022). Murali et
al., "Software Mitigation of Crosstalk on NISQ Computers" (ASPLOS 2020) is why
neighbouring programs are kept apart by default. Those papers are motivation
only; no number here is taken from them.

What this module does, on top of the calibration-aware routing in
``calibration.py``:

* ``allocate()`` places programs greedily, largest first, each on the
  highest-ESP qubits still free. ``buffer=1`` also blocks the coupling-map
  neighbours of every used qubit, so no two programs touch; ``buffer=0`` only
  blocks the used qubits. A program that no longer fits stops the allocation
  and is reported in ``unplaced``; nothing is dropped silently.
* ``pack()`` builds one device-wide circuit (each program on its own region,
  with its own classical register) and ``split_counts()`` turns the joint
  result back into per-program counts, qubit 0 first.
* ``emulate()`` runs a packed job on the device's own noise model, one program
  at a time (why that is exact is explained in its docstring), optionally with
  an explicit, labelled crosstalk penalty for buffer-0 packing.
* ``run_packed_workloads()`` is the live-path helper; ``packing_cost()`` does
  the QPU-seconds and price accounting from the catalogue's QPU figures.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from . import calibration as C
from .circuit import Circuit

FREE_PLAN_S = 600.0          # IBM Open Plan: 10 QPU minutes per 28 days
MAX_JOINT_QUBITS = 22        # emulate_joint() refuses wider jobs (Aer state grows as 2^n)
MAX_LAYOUTS = 20000          # placements scored per connected component of a program


class CapacityError(RuntimeError):
    """Raised when programs do not fit on the device and partial packing was not allowed."""


# ---- one program's routed template -------------------------------------------
@dataclass
class _Template:
    """A program transpiled once for the device, reduced to what placement needs."""

    tqc: object                               # level-3 ISA circuit on Qiskit's own qubits
    active: list[int]                         # physical qubits it touches (incl. routing ancillas)
    components: list[list[int]]               # connected components of its interaction graph
    edges: list[tuple[int, int]]              # interaction edges (a < b) between active qubits
    directed: list[tuple[int, int]]           # 2q gate directions used (control, target) as routed
    n1: dict[int, int]                        # non-virtual 1q gates per active qubit
    n2: dict[tuple[int, int], int]            # 2q gates per interaction edge
    meas: dict[int, int]                      # measurements per active qubit
    n_clbits: int


_TEMPLATES: dict[tuple, _Template] = {}


def _as_qiskit(c):
    from .backends.qiskit_backend import to_qiskit
    if isinstance(c, Circuit):
        return to_qiskit(c)
    if not any(ins.operation.name == "measure" for ins in c.data):
        c = c.copy()
        c.measure_all()
    return c


def _signature(qc) -> tuple:
    return tuple((ins.operation.name, tuple(qc.find_bit(q).index for q in ins.qubits),
                  tuple(round(float(p), 12) for p in ins.operation.params)) for ins in qc.data)


def _template(cal: C.DeviceCalibration, qc, seed: int = 11) -> _Template:
    """Transpile at optimisation level 3 (as ``best_layout`` does), then record the
    routed interaction graph and gate counts so any placement can be scored fast."""
    key = (cal.name, qc.num_qubits, _signature(qc), seed)
    if key in _TEMPLATES:
        return _TEMPLATES[key]
    import rustworkx as rx
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    be = cal.backend if cal.backend is not None else C.snapshot(C._class_name(cal.name)).backend
    tqc = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=seed).run(qc)
    ops, meas = C._ops(tqc)
    active = sorted({q for _, qs in ops for q in qs} | set(meas))
    n1 = {q: 0 for q in active}
    n2: dict[tuple[int, int], int] = {}
    directed: set[tuple[int, int]] = set()
    for name, qs in ops:
        if len(qs) == 2:
            e = (min(qs), max(qs))
            n2[e] = n2.get(e, 0) + 1
            directed.add(qs)
        elif name not in ("rz", "delay", "barrier", "id", "reset"):  # same rule as calibration.esp
            n1[qs[0]] += 1
    m = {q: 0 for q in active}
    for q in meas:
        m[q] += 1
    g = rx.PyGraph()
    g.add_nodes_from(active)
    idx = {q: i for i, q in enumerate(active)}
    for a, b in n2:
        g.add_edge(idx[a], idx[b], None)
    comps = sorted((sorted(active[i] for i in comp) for comp in rx.connected_components(g)),
                   key=lambda c: (-len(c), c[0]))
    t = _Template(tqc, active, comps, sorted(n2), sorted(directed), n1, n2, m, tqc.num_clbits)
    _TEMPLATES[key] = t
    return t


# ---- allocation --------------------------------------------------------------
@dataclass
class Placement:
    index: int                    # position in the input list
    name: str
    n_clbits: int                 # logical qubits measured (size of its classical register)
    layout: list[int]             # physical qubit of each logical qubit at the start
    qubits: list[int]             # every physical qubit the program touches (incl. routing ancillas)
    couplers: list[tuple[int, int]]  # physical couplers its 2q gates use (a < b)
    esp: float
    two_q_count: int
    depth: int
    order: int                    # placement order (0 = placed first, the largest program)
    n_layouts_scored: int
    exhaustive: bool              # every placement of every component was scored
    circuit: object = field(default=None, repr=False)   # device-wide ISA circuit, own 'meas' register

    def row(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "circuit"}


@dataclass
class Allocation:
    device: str
    buffer: int
    placements: list[Placement]   # placed programs, in input order
    unplaced: list[int]           # input indices that did not fit (allocation stops at the first)
    n_programs: int
    free_qubits_left: int

    @property
    def capacity_reached(self) -> bool:
        return bool(self.unplaced)

    @property
    def complete(self) -> bool:
        return not self.unplaced

    def __iter__(self):
        return iter(self.placements)

    def __len__(self) -> int:
        return len(self.placements)

    def __getitem__(self, i) -> Placement:
        return self.placements[i]


def neighbours(cal: C.DeviceCalibration) -> dict[int, set[int]]:
    """Physical coupling-map adjacency, broken couplers included: a coupler that is
    too noisy to run gates on is still a physical link that can carry crosstalk."""
    nb: dict[int, set[int]] = {q: set() for q in range(cal.num_qubits)}
    for a, b in cal.two_q:
        nb[a].add(b)
        nb[b].add(a)
    return nb


def unusable_qubits(cal: C.DeviceCalibration) -> set[int]:
    """Qubits whose readout or 1q gate error is at or above ``BROKEN``."""
    return {q for q in range(cal.num_qubits)
            if cal.readout.get(q, 1.0) >= C.BROKEN or cal.one_q.get(q, 1.0) >= C.BROKEN}


def _grow(qubits: set[int], nb: dict[int, set[int]], radius: int) -> set[int]:
    out, frontier = set(qubits), set(qubits)
    for _ in range(radius):
        frontier = {n for q in frontier for n in nb[q]} - out
        out |= frontier
    return out


def _log1m(e: float) -> float:
    return math.log1p(-e) if e < 1.0 else -math.inf


def _place_template(cal: C.DeviceCalibration, t: _Template, free: set[int],
                    max_layouts: int = MAX_LAYOUTS) -> tuple[dict[int, int] | None, int, bool]:
    """Highest-ESP placement of a routed template on the free qubits.

    ESP factorises over the connected components of the program's interaction
    graph, so each component is placed on its own (largest first) by exhaustive
    VF2 enumeration over the free part of the coupling map, scored with the same
    terms as ``calibration.esp``. Couplers with error >= BROKEN are not in the
    graph, so they can never be used. Placing components one at a time avoids
    enumerating the product of independent placements (millions for a QAOA graph
    with isolated vertices)."""
    import rustworkx as rx

    mapping: dict[int, int] = {}
    scored, exhaustive = 0, True
    taken: set[int] = set()
    for comp in t.components:
        avail = sorted(free - taken)
        if len(avail) < len(comp):
            return None, scored, exhaustive
        pos = {q: i for i, q in enumerate(avail)}
        # Directed graphs: a gate may only land on a coupler that supports it in that
        # direction (matters for ECR devices; CZ couplers are listed both ways).
        big = rx.PyDiGraph()
        big.add_nodes_from(avail)
        for (a, b), e in cal.two_q.items():
            if e < C.BROKEN and a in pos and b in pos:
                big.add_edge(pos[a], pos[b], None)
        cidx = {q: i for i, q in enumerate(comp)}
        small = rx.PyDiGraph()
        small.add_nodes_from(comp)
        cedges = [(a, b) for (a, b) in t.edges if a in cidx]
        for a, b in t.directed:
            if a in cidx:
                small.add_edge(cidx[a], cidx[b], None)
        # Per-qubit and per-edge log-success terms of this component.
        q_terms = [(q, t.n1[q], t.meas[q]) for q in comp]
        best, best_lp, k = None, -math.inf, 0
        for m in rx.vf2_mapping(big, small, subgraph=True, induced=False):
            inv = {comp[c]: avail[d] for d, c in m.items()}
            lp = 0.0
            for q, n1, nm in q_terms:
                p = inv[q]
                lp += n1 * _log1m(cal.one_q.get(p, 1.0)) + nm * _log1m(cal.readout.get(p, 1.0))
            for a, b in cedges:
                lp += t.n2[(a, b)] * _log1m(cal.edge_error(inv[a], inv[b]))
            if lp > best_lp:
                best, best_lp = inv, lp
            k += 1
            if k >= max_layouts:
                exhaustive = False
                break
        scored += k
        if best is None or best_lp == -math.inf:
            return None, scored, exhaustive
        mapping.update(best)
        taken |= set(best.values())
    return mapping, scored, exhaustive


def _rebuild(cal: C.DeviceCalibration, t: _Template, mapping: dict[int, int], seed: int = 11):
    """The template's ISA circuit moved onto the chosen physical qubits."""
    from qiskit import QuantumCircuit

    tqc = t.tqc
    out = QuantumCircuit(tqc.num_qubits, tqc.num_clbits)
    for ins in tqc.data:
        src = [tqc.find_bit(q).index for q in ins.qubits]
        if ins.operation.name == "barrier":
            qs = [mapping[q] for q in src if q in mapping]
            if qs:
                out.barrier(*qs)
            continue
        qs = [out.qubits[mapping[q]] for q in src]
        cs = [out.clbits[tqc.find_bit(c).index] for c in ins.clbits]
        out.append(ins.operation, qs, cs)
    target = cal.backend.target if cal.backend is not None else None
    misaligned = target is not None and any(
        len(ins.qubits) == 2 and not target.instruction_supported(
            ins.operation.name, tuple(out.find_bit(q).index for q in ins.qubits))
        for ins in out.data if ins.operation.name != "barrier")
    if misaligned:
        # Moving a directional gate (ECR) can flip it; re-translate in place, no routing.
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        fix = generate_preset_pass_manager(optimization_level=0, backend=cal.backend, routing_method="none",
                                           initial_layout=list(range(out.num_qubits)), seed_transpiler=seed)
        out = fix.run(out)
    return out


def _touched(qc) -> tuple[set[int], list[tuple[int, int]]]:
    qubits, couplers = set(), []
    for ins in qc.data:
        if ins.operation.name == "barrier":
            continue
        qs = tuple(qc.find_bit(q).index for q in ins.qubits)
        qubits |= set(qs)
        if len(qs) == 2:
            couplers.append((min(qs), max(qs)))
    return qubits, couplers


def allocate(cal: C.DeviceCalibration, circuits: list, buffer: int = 1, seed: int = 11,
             max_layouts: int = MAX_LAYOUTS, strict: bool = False) -> Allocation:
    """Greedy spatial allocation of several programs on one device.

    Programs are placed largest first (active qubits after routing, then 2q
    gates); each gets the highest-ESP placement among qubits that are neither
    used nor blocked. After a program is placed its qubits are blocked, plus
    every qubit within ``buffer`` coupling-map hops of them (``buffer=1``: no
    two programs are adjacent; ``buffer=0``: only the used qubits). Qubits with
    readout or 1q error >= BROKEN are never used.

    The first program that does not fit stops the allocation; it and every
    program after it in placement order are returned in ``unplaced`` (or
    ``CapacityError`` is raised with ``strict=True``). Greedy placement is not
    an optimal packing, so the number placed is a lower bound on what the chip
    could hold."""
    nb = neighbours(cal)
    free = set(range(cal.num_qubits)) - unusable_qubits(cal)
    qcs = [_as_qiskit(c) for c in circuits]
    temps = [_template(cal, qc, seed) for qc in qcs]
    order = sorted(range(len(qcs)), key=lambda i: (-len(temps[i].active), -sum(temps[i].n2.values()), i))
    placed: dict[int, Placement] = {}
    unplaced: list[int] = []
    for rank, i in enumerate(order):
        t = temps[i]
        mapping, scored, exhaustive = _place_template(cal, t, free, max_layouts)
        if mapping is None:
            unplaced = order[rank:]
            break
        qc = _rebuild(cal, t, mapping, seed)
        used, couplers = _touched(qc)
        # Placement must stay on free qubits and good couplers, whatever the search did.
        if not used <= free or any(cal.edge_error(a, b) >= C.BROKEN for a, b in couplers):
            raise RuntimeError(f"placement of program {i} touches a blocked qubit or broken coupler")
        ops, meas = C._ops(qc)
        init = (t.tqc.layout.initial_index_layout(filter_ancillas=True) if t.tqc.layout is not None
                else list(range(qcs[i].num_qubits)))
        placed[i] = Placement(index=i, name=qcs[i].name, n_clbits=t.n_clbits,
                              layout=[mapping.get(p, p) for p in init], qubits=sorted(used),
                              couplers=sorted(set(couplers)), esp=C.esp(cal, ops, meas),
                              two_q_count=len(couplers), depth=qc.depth(), order=rank,
                              n_layouts_scored=scored, exhaustive=exhaustive, circuit=qc)
        free -= _grow(used, nb, buffer)
    alloc = Allocation(cal.name, buffer, [placed[i] for i in sorted(placed)], sorted(unplaced),
                       len(qcs), len(free))
    if strict and unplaced:
        raise CapacityError(f"{cal.name}: {len(placed)} of {len(qcs)} programs fit with buffer={buffer}; "
                            f"unplaced inputs {sorted(unplaced)}")
    return alloc


def capacity(cal: C.DeviceCalibration, make_programs, buffer: int = 1, start: int = 8,
             limit: int = 256, seed: int = 11) -> int:
    """Greedy capacity: a k such that ``make_programs(k)`` (a list of k circuits)
    packs completely and ``make_programs(k + 1)`` does not.

    Doubling then bisection. Greedy allocation is not strictly monotone in k (a
    different set changes the placement order), so a larger k could
    occasionally fit as well; the value is what this allocator achieves, a lower
    bound on what the chip could hold."""
    def fits(k: int) -> bool:
        return k == 0 or allocate(cal, make_programs(k), buffer, seed).complete

    lo, hi = 0, max(1, start)
    while hi <= limit and fits(hi):
        lo, hi = hi, hi * 2
    hi = min(hi, limit + 1)
    while hi - lo > 1:   # invariant: lo fits, hi does not (or is past the limit)
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


# ---- packing -----------------------------------------------------------------
@dataclass
class PackedJob:
    device: str
    buffer: int
    allocation: Allocation
    circuit: object = field(repr=False)          # one device-wide circuit, one register per program
    registers: list[str] = field(default_factory=list)   # register of each placed program, input order
    exposed: dict[int, list[tuple[int, int]]] = field(default_factory=dict)  # program -> couplers next to others
    cal: C.DeviceCalibration | None = field(default=None, repr=False)

    @property
    def placements(self) -> list[Placement]:
        return self.allocation.placements

    @property
    def unplaced(self) -> list[int]:
        return self.allocation.unplaced

    @property
    def k(self) -> int:
        return len(self.placements)

    @property
    def depth(self) -> int:
        return max(p.depth for p in self.placements)

    @property
    def active_qubits(self) -> int:
        return sum(len(p.qubits) for p in self.placements)


def exposed_couplers(cal: C.DeviceCalibration, placements: list[Placement]) -> dict[int, list[tuple[int, int]]]:
    """For each program, the couplers its 2q gates use that touch a qubit adjacent to
    another program's qubits. Empty for every program when buffer >= 1."""
    nb = neighbours(cal)
    owner = {q: p.index for p in placements for q in p.qubits}
    out = {}
    for p in placements:
        hit = [(a, b) for a, b in p.couplers
               if any(owner.get(n, p.index) != p.index for q in (a, b) for n in nb[q])]
        out[p.index] = hit
    return out


def pack(cal: C.DeviceCalibration, circuits: list, buffer: int = 1, allow_partial: bool = False,
         allocation: Allocation | None = None, seed: int = 11) -> PackedJob:
    """One device-wide circuit holding every program on its own region.

    Program i keeps its own ISA gates on its allocated qubits and writes its
    measurements into its own classical register ``p<i>`` (i = input index).
    Raises ``CapacityError`` if some program does not fit, unless
    ``allow_partial`` (then ``job.unplaced`` lists them)."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    alloc = allocation or allocate(cal, circuits, buffer, seed)
    if alloc.unplaced and not allow_partial:
        raise CapacityError(f"{cal.name}: only {len(alloc)} of {alloc.n_programs} programs fit "
                            f"with buffer={alloc.buffer}; unplaced inputs {alloc.unplaced}")
    if not alloc.placements:
        raise CapacityError(f"{cal.name}: no program fits")
    wide = QuantumCircuit(QuantumRegister(cal.num_qubits, "q"), name=f"packed_{len(alloc)}")
    regs = []
    for p in alloc.placements:
        reg = ClassicalRegister(p.n_clbits, f"p{p.index}")
        wide.add_register(reg)
        regs.append(reg.name)
        src = p.circuit
        for ins in src.data:
            qs = [wide.qubits[src.find_bit(q).index] for q in ins.qubits]
            cs = [reg[src.find_bit(c).index] for c in ins.clbits]
            wide.append(ins.operation, qs, cs)
    return PackedJob(cal.name, alloc.buffer, alloc, wide, regs, exposed_couplers(cal, alloc.placements), cal)


def split_counts(job: PackedJob, raw_counts: dict) -> list[dict[str, int]]:
    """Per-program counts, qubit 0 first, in the order of ``job.placements``.

    Accepts Aer-style joint counts, whose keys hold one space-separated field
    per classical register with the LAST register leftmost and, inside each
    field, bit 0 rightmost; keys without spaces are cut by register size. Also
    accepts ``{register_name: counts}``, which is what SamplerV2 returns per
    register."""
    from .backends.qiskit_backend import qiskit_counts

    sizes = [p.n_clbits for p in job.placements]
    if raw_counts and set(raw_counts) == set(job.registers):
        return [qiskit_counts(raw_counts[r], n) for r, n in zip(job.registers, sizes)]
    out: list[dict[str, int]] = [{} for _ in sizes]
    for key, v in raw_counts.items():
        if " " in key:
            fields = key.split()[::-1]        # register 0 first
        else:
            fields, pos = [], len(key)
            for n in sizes:                   # register 0 is the rightmost slice
                fields.append(key[pos - n:pos])
                pos -= n
        if len(fields) != len(sizes) or any(len(f) != n for f, n in zip(fields, sizes)):
            raise ValueError(f"count key {key!r} does not match registers {job.registers} of sizes {sizes}")
        for i, f in enumerate(fields):
            b = f[::-1]                       # Qiskit: bit 0 rightmost -> ours: qubit 0 first
            out[i][b] = out[i].get(b, 0) + int(v)
    return out


# ---- emulation -----------------------------------------------------------------
_BASE_NOISE: dict[str, object] = {}


def base_noise_model(cal: C.DeviceCalibration):
    """The device's own Aer noise model (what ``AerSimulator.from_backend`` uses), cached."""
    if cal.name not in _BASE_NOISE:
        from qiskit_aer.noise import NoiseModel
        _BASE_NOISE[cal.name] = NoiseModel.from_backend(cal.backend)
    return _BASE_NOISE[cal.name]


def extra_depolarising(base_error: float, factor: float, d: int = 4) -> float:
    """Depolarising parameter lam such that composing ``depolarizing_error(lam)``
    with a channel of average gate error ``base_error`` gives average gate error
    ``factor * base_error``.

    Uses process fidelity, which composes exactly with a depolarising channel:
    F(D_lam . E) = (1 - lam) F(E) + lam / d^2, and r = (1 - F) d / (d + 1)."""
    if factor <= 1.0 or base_error <= 0.0:
        return 0.0
    f_base = 1 - base_error * (d + 1) / d
    f_target = max(1 / d ** 2, 1 - factor * base_error * (d + 1) / d)
    return min(1.0, (f_base - f_target) / (f_base - 1 / d ** 2))


def crosstalk_noise_model(cal: C.DeviceCalibration, couplers, factor: float):
    """Base device noise plus extra depolarising error on the given couplers so
    their 2q gate error becomes ``factor`` times the calibrated error.

    ASSUMPTION (labelled): crosstalk is modelled as a multiplicative increase of
    the 2q gate error on couplers that touch a qubit adjacent to another
    program, applied to every such gate whether or not a neighbouring gate runs
    at the same moment (conservative: no timing model). ``factor == 1`` returns
    the unmodified base model.

    Checked empirically on qiskit-aer 0.17: ``add_quantum_error`` on an
    (instruction, qubits) pair that already carries the backend's error APPENDS
    (composes) the new error rather than replacing it, so the result is base
    error plus the extra; ``test_crosstalk_composes_with_base_error`` would fail
    if a future Aer replaced it instead."""
    base = base_noise_model(cal)
    couplers = sorted({(min(a, b), max(a, b)) for a, b in couplers})
    if factor == 1.0 or not couplers:
        return base
    from qiskit_aer.noise import depolarizing_error

    nm = copy.deepcopy(base)
    for a, b in couplers:
        lam = extra_depolarising(cal.edge_error(a, b), factor)
        if lam <= 0:
            continue
        err = depolarizing_error(lam, 2)
        for qs in ((a, b), (b, a)):
            if qs in cal.two_q:
                nm.add_quantum_error(err, cal.two_q_gate, list(qs), warnings=False)
    return nm


def _simulator(cal: C.DeviceCalibration, noise_model, seed: int):
    from qiskit_aer import AerSimulator
    return AerSimulator.from_backend(cal.backend, noise_model=noise_model, seed_simulator=seed)


def emulate(job: PackedJob, shots: int = 4000, crosstalk_factor: float = 1.0, seed: int = 17) -> list[dict[str, int]]:
    """Run a packed job on an emulator of its device; per-program counts, qubit 0 first.

    Why one program at a time is exact: Aer cannot hold 40+ active qubits in
    one state, but the device noise model from ``NoiseModel.from_backend`` only
    contains errors local to the qubits each instruction acts on (gate errors
    on the gate's qubits, readout error per measured qubit; idle-time noise
    only for explicit delays, and these circuits are not scheduled). Programs
    on disjoint qubits share no gate, so the noisy joint state stays a product
    state and the joint output distribution factorises into the per-program
    distributions. Each shot of the joint job is then an independent draw from
    every program at once, so the per-program counts of a joint run are
    independent multinomials with the per-program probabilities: exactly what
    simulating each program alone, with the same shots, produces.
    ``test_per_program_emulation_matches_joint`` checks this on a job small
    enough to simulate jointly.

    ``crosstalk_factor`` > 1 adds the labelled crosstalk penalty of
    ``crosstalk_noise_model`` to couplers next to another program; with
    buffer >= 1 there are none, so it has no effect."""
    from .backends.qiskit_backend import qiskit_counts

    cal = job.cal
    exposed = [e for p in job.placements for e in job.exposed.get(p.index, [])]
    nm = crosstalk_noise_model(cal, exposed, crosstalk_factor)
    res = _simulator(cal, nm, seed).run([p.circuit for p in job.placements], shots=shots).result()
    return [qiskit_counts(res.get_counts(i), p.n_clbits) for i, p in enumerate(job.placements)]


def emulate_joint(job: PackedJob, shots: int = 4000, crosstalk_factor: float = 1.0,
                  seed: int = 17) -> list[dict[str, int]]:
    """Simulate the whole packed circuit at once (only feasible for small jobs);
    used to check that ``emulate`` is equivalent."""
    if job.active_qubits > MAX_JOINT_QUBITS:
        raise ValueError(f"{job.active_qubits} active qubits is too many to simulate jointly "
                         f"(limit {MAX_JOINT_QUBITS}); use emulate()")
    cal = job.cal
    exposed = [e for p in job.placements for e in job.exposed.get(p.index, [])]
    nm = crosstalk_noise_model(cal, exposed, crosstalk_factor)
    res = _simulator(cal, nm, seed).run(job.circuit, shots=shots).result()
    return split_counts(job, res.get_counts())


# ---- throughput and cost -------------------------------------------------------
@dataclass
class PackingCost:
    k: int
    shots: int
    depth: int                         # deepest packed program sets the shot length
    qpu_s_job: float
    qpu_s_per_program: float
    qpu_s_per_program_solo: float      # mean over programs of running each alone
    price_job: float
    price_per_program: float
    price_per_program_solo: float
    programs_per_free_plan: float      # programs per 10 free IBM minutes, packed
    programs_per_free_plan_solo: float
    qpu_s_per_program_with_overhead: float       # sensitivity: if the per-job overhead is metered too
    qpu_s_per_program_solo_with_overhead: float

    def row(self) -> dict:
        return dict(self.__dict__)


def job_qpu_seconds(shots: int, depth: int, qpu=None) -> float:
    """QPU seconds for one job: shots x (repetition delay + depth x layer time)."""
    from ..resources import load_catalogue
    q = qpu or load_catalogue()["qpu"]
    return shots * (q.shot_overhead_s + depth * q.layer_time_s)


def packing_cost(depths: list[int], shots: int = 4000, qpu=None) -> PackingCost:
    """Accounting for k programs (ISA depths ``depths``) packed into one job, against
    running each as its own job. Uses the catalogue QPU figures: per-shot
    repetition delay and layer time for QPU seconds, Braket-style per-task and
    per-shot prices for money (an assumption applied to every device; Braket
    does not sell IBM devices), and IBM's 10-minute free plan for throughput."""
    from ..resources import load_catalogue
    q = qpu or load_catalogue()["qpu"]
    k = len(depths)
    if k == 0:
        raise ValueError("no programs")
    t_job = job_qpu_seconds(shots, max(depths), q)
    t_solo = sum(job_qpu_seconds(shots, d, q) for d in depths) / k
    price_job = q.cost_per_task + shots * q.cost_per_shot
    return PackingCost(k=k, shots=shots, depth=max(depths), qpu_s_job=t_job, qpu_s_per_program=t_job / k,
                       qpu_s_per_program_solo=t_solo, price_job=price_job, price_per_program=price_job / k,
                       price_per_program_solo=price_job, programs_per_free_plan=FREE_PLAN_S / (t_job / k),
                       programs_per_free_plan_solo=FREE_PLAN_S / t_solo,
                       qpu_s_per_program_with_overhead=(t_job + q.job_overhead_s) / k,
                       qpu_s_per_program_solo_with_overhead=t_solo + q.job_overhead_s)


# ---- live-path helper ------------------------------------------------------------
PACKABLE_KINDS = {"quantum_simulation"}
_SKIP_REASON = {
    "optimization": "QAOA is an optimisation loop of many sequential jobs, not one circuit",
    "kernel_classification": "a quantum kernel needs O(n_train^2) circuits, not one",
    "inference": "classical workload, no circuit",
    "training": "classical workload, no circuit",
}


@dataclass
class PackedRun:
    device: str
    calibration: str                  # "live" | "snapshot"
    emulated: bool
    buffer: int
    crosstalk_factor: float
    shots: int
    results: list[dict]               # one row per packed workload
    unplaced: list[int]               # workload ids that did not fit on the device
    skipped: list[dict]               # workloads not packable, with the reason
    cost: PackingCost
    qpu_seconds_measured: float | None = None
    job_ids: list[str] = field(default_factory=list)
    job: PackedJob | None = field(default=None, repr=False)


def _devices_from(backend_or_cal) -> tuple[list[C.DeviceCalibration], bool]:
    if backend_or_cal is None:
        return C.load_devices(), False
    if isinstance(backend_or_cal, C.DeviceCalibration):
        return [backend_or_cal], backend_or_cal.source == "live"
    if isinstance(backend_or_cal, (list, tuple)):
        return list(backend_or_cal), any(d.source == "live" for d in backend_or_cal)
    if hasattr(backend_or_cal, "devices"):            # CalibratedIBMBackend ("ibm-auto")
        return list(backend_or_cal.devices), bool(getattr(backend_or_cal, "live", False))
    raise TypeError(f"expected a CalibratedIBMBackend, DeviceCalibration or list of them, got {type(backend_or_cal)}")


def select_packing_device(devices: list[C.DeviceCalibration], circuits: list, buffer: int = 1,
                          queue_weight: float = 0.0) -> tuple[C.DeviceCalibration, Allocation]:
    """Device for a packed job: most programs placed, then the highest mean ESP,
    minus ``queue_weight`` per hour of known queue (as ``select_device`` scores one
    circuit)."""
    best, best_key = None, None
    for cal in devices:
        alloc = allocate(cal, circuits, buffer)
        if not alloc.placements:
            continue
        mean_esp = sum(p.esp for p in alloc) / len(alloc)
        key = (len(alloc), mean_esp - queue_weight * ((cal.queue_s or 0.0) / 3600))
        if best_key is None or key > best_key:
            best, best_key = (cal, alloc), key
    if best is None:
        raise CapacityError("no device fits any of these programs")
    return best


def run_packed_workloads(backend_or_cal, workloads: list, shots: int = 4000, buffer: int = 1,
                         crosstalk_factor: float = 1.0, seed: int = 17, queue_weight: float = 0.0) -> PackedRun:
    """Pack single-circuit workloads from several clients into one QPU job.

    Only ``quantum_simulation`` workloads (one Trotter circuit each, checked
    against the exact magnetisation as ``Executor._quantum_simulation`` does)
    are packed; every other kind is listed in ``skipped`` with the reason.
    Picks the device (``select_packing_device``), packs, runs, splits and
    returns per-workload quality ``1 - |<Z> - exact|``, the device region, and
    each workload's share of the job's QPU seconds and price. Workloads that do
    not fit on the chosen device are returned in ``unplaced``.

    With a live ``CalibratedIBMBackend`` the one joint circuit is submitted
    through SamplerV2 and each register is read back separately, subject to the
    executor's hardware shot limit (QEDGE_HW_MAX_SHOTS). That path follows the
    qiskit-ibm-runtime 0.50 API but has NOT been run here (no IBM account);
    everything else runs on the local emulator of the chosen device, where
    ``crosstalk_factor`` applies (on hardware the device's own crosstalk
    applies instead)."""
    from ..executor import HW_MAX_SHOTS, representative_circuit
    from .algorithms import magnetisation, tfim_exact_magnetisation

    devices, live = _devices_from(backend_or_cal)
    keep, skipped = [], []
    for w in workloads:
        if w.kind in PACKABLE_KINDS and int(w.params.get("n", 0)) <= 20:
            keep.append(w)
        else:
            why = _SKIP_REASON.get(w.kind, "kept to <= 20 qubits so the result can be checked exactly")
            skipped.append({"id": w.id, "kind": w.kind, "reason": why})
    if not keep:
        raise ValueError("no packable workload (only quantum_simulation with n <= 20 is packed)")
    circuits = [representative_circuit(w) for w in keep]
    cal, alloc = select_packing_device(devices, circuits, buffer, queue_weight)
    job = pack(cal, circuits, buffer, allow_partial=True, allocation=alloc)
    usage, job_ids = None, []
    if live and cal.source == "live":  # pragma: no cover - needs an IBM account
        if shots > HW_MAX_SHOTS:
            raise RuntimeError(f"refusing {shots} shots on real hardware (limit QEDGE_HW_MAX_SHOTS={HW_MAX_SHOTS})")
        from qiskit_ibm_runtime import SamplerV2
        rjob = SamplerV2(mode=cal.backend).run([job.circuit], shots=shots)
        pub = rjob.result()[0]
        counts = split_counts(job, {r: getattr(pub.data, r).get_counts() for r in job.registers})
        job_ids = [rjob.job_id()]
        try:
            usage = float(rjob.usage())
        except Exception:
            usage = None
        emulated = False
    else:
        counts = emulate(job, shots, crosstalk_factor, seed)
        emulated = True
    cost = packing_cost([p.depth for p in job.placements], shots)
    rows = []
    for p, cnt in zip(job.placements, counts):
        w = keep[p.index]
        n, steps = int(w.params["n"]), int(w.params["steps"])
        m, exact = magnetisation(cnt), tfim_exact_magnetisation(n, steps)
        rows.append({"workload": w.id, "n": n, "steps": steps, "device": cal.name, "region": p.qubits,
                     "esp": p.esp, "exposed_couplers": len(job.exposed.get(p.index, [])),
                     "magnetisation": m, "exact": exact, "quality": float(max(0.0, 1 - abs(m - exact))),
                     "qpu_s_share": cost.qpu_s_per_program, "price_share": cost.price_per_program})
    return PackedRun(cal.name, cal.source, emulated, buffer, crosstalk_factor, shots, rows,
                     [keep[i].id for i in job.unplaced], skipped, cost, usage, job_ids, job)
