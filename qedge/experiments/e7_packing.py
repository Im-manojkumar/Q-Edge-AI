"""E7: what does packing several clients' circuits into one QPU job cost in quality?

Every QPU job pays one queue wait, one job overhead, one task fee and, on
Braket-style pricing, a fee per shot. Packing k small programs side by side on
one chip (spatial multi-programming, see ``qedge.quantum.packing``) divides
those by k, but later programs get worse qubits and, with no buffer between
programs, can suffer crosstalk. This experiment measures that trade on
emulators of two IBM devices (calibration snapshots of fake_aachen and
fake_torino):

* program mixes: (a) k copies of TFIM n=4, 2 Trotter steps; (b) a mixed set
  cycling TFIM n=4, QAOA n=6 (p=1, density 0.4, fixed angles), quantum-kernel
  circuit with 4 features, TFIM n=6;
* k in 1, 2, 4, 8 and the device's greedy capacity for that mix and buffer;
* buffer 1 (no two programs adjacent) and buffer 0 (programs may touch);
* crosstalk factor 1, 2, 5 for buffer 0 only: it multiplies the 2q gate error
  on couplers next to another program, and with buffer 1 no such coupler
  exists, so buffer 1 runs at factor 1 only.

Per program, "measured" is the Hellinger fidelity of its emulated output to
its ideal distribution (as in E6), compared with the same program run alone
on its best-ESP placement on the same device. Crosstalk is an explicit,
labelled assumption, not a measurement: the emulator has no crosstalk of its
own, so factor 1 shows only the cost of worse qubits.
"""
from __future__ import annotations

import os
import time

import pandas as pd

from ..executor import representative_circuit
from ..quantum import calibration as C
from ..quantum import packing as P
from ..quantum.algorithms import tfim_trotter
from ..quantum.backends.qiskit_backend import qiskit_counts, to_qiskit
from ..workloads import make_workload
from .e6_calibration import _hellinger, _ideal_probs

DEVICES = ["FakeAachen", "FakeTorino"]
MIXES = ["tfim4", "mixed"]
KS = (1, 2, 4, 8)
CROSSTALK = (1.0, 2.0, 5.0)
SHOTS = 4000
SEED = 17
BUFFER1_NOTE = "buffer 1: no two programs adjacent, so the crosstalk factor has no coupler to act on (run at 1 only)"


def programs(mix: str, k: int) -> list[tuple[str, object]]:
    """The first k programs of a mix, as (label, Circuit). QAOA instances use the
    program's position as graph seed, so each client brings its own graph."""
    if mix == "tfim4":
        return [("tfim n=4", tfim_trotter(4, 2)) for _ in range(k)]
    out = []
    for i in range(k):
        j = i % 4
        if j == 0:
            out.append(("tfim n=4", tfim_trotter(4, 2)))
        elif j == 1:
            w = make_workload("optimization", i, seed=i, n=6, p=1, density=0.4)
            out.append(("qaoa n=6", representative_circuit(w)))
        elif j == 2:
            w = make_workload("kernel_classification", i, seed=i, features=4, n_train=20, n_test=10)
            out.append(("kernel d=4", representative_circuit(w)))
        else:
            out.append(("tfim n=6", tfim_trotter(6, 2)))
    return out


def _ideal(circ) -> dict[str, float]:
    # Qiskit orders bitstrings with qubit 0 rightmost; flip to qubit 0 first.
    return {k[::-1]: v for k, v in _ideal_probs(to_qiskit(circ)).items()}


def _key(circ) -> tuple:
    return (circ.n_qubits, tuple((g.name, g.qubits, g.params) for g in circ.gates))


def _solo(cal, circs: dict, shots: int, seed: int) -> dict[tuple, dict]:
    """Each distinct program alone on its best-ESP placement (nothing blocked), base
    device noise; run twice with different seeds to show the shot-noise floor."""
    from qiskit_aer import AerSimulator

    keys = list(circs)
    pl = [P.allocate(cal, [circs[k]], buffer=0)[0] for k in keys]
    sim = AerSimulator.from_backend(cal.backend, noise_model=P.base_noise_model(cal), seed_simulator=seed)
    out = {}
    runs = [sim.run([p.circuit for p in pl], shots=shots, seed_simulator=s).result() for s in (seed, seed + 1)]
    for i, (k, p) in enumerate(zip(keys, pl)):
        ideal = _ideal(circs[k])
        f = [_hellinger(ideal, qiskit_counts(r.get_counts(i), p.n_clbits)) for r in runs]
        out[k] = {"fidelity": f[0], "fidelity_repeat": f[1], "esp": p.esp, "depth": p.depth,
                  "region": p.qubits}
    return out


def run(quick: bool = False, shots: int = SHOTS, seed: int = SEED) -> dict[str, pd.DataFrame]:
    devices = DEVICES[:1] if quick else DEVICES
    ks = (1, 2, 4) if quick else KS
    rows = []
    for dev in devices:
        cal = C.snapshot(dev)
        t_dev = time.perf_counter()
        plan = {}
        for mix in MIXES:
            for buffer in (1, 0):
                cap = P.capacity(cal, lambda k, m=mix: [c for _, c in programs(m, k)], buffer)
                plan[(mix, buffer)] = (cap, sorted({k for k in ks if k <= cap} | ({cap} if not quick else set())))
        # Every distinct program that will run, for the solo baseline.
        distinct = {}
        for (mix, _), (cap, kk) in plan.items():
            for _, c in programs(mix, max(kk)):
                distinct.setdefault(_key(c), c)
        solo = _solo(cal, distinct, shots, seed)
        for (mix, buffer), (cap, kk) in plan.items():
            for k in kk:
                progs = programs(mix, k)
                job = P.pack(cal, [c for _, c in progs], buffer)
                cost = P.packing_cost([p.depth for p in job.placements], shots)
                factors = (1.0,) if buffer >= 1 else CROSSTALK
                for c in factors:
                    counts = P.emulate(job, shots, crosstalk_factor=c, seed=seed)
                    for p, cnt in zip(job.placements, counts):
                        label, circ = progs[p.index]
                        s = solo[_key(circ)]
                        f = _hellinger(_ideal(circ), cnt)
                        rows.append({"device": cal.name, "mix": mix, "buffer": buffer, "crosstalk_factor": c,
                                     "k": k, "capacity": cap, "program": p.index, "label": label,
                                     "n_qubits": circ.n_qubits, "placement_order": p.order,
                                     "region": " ".join(map(str, p.qubits)),
                                     "solo_region": " ".join(map(str, s["region"])),
                                     "esp_packed": p.esp, "esp_solo": s["esp"],
                                     "fidelity_packed": f, "fidelity_solo": s["fidelity"],
                                     "fidelity_solo_repeat": s["fidelity_repeat"],
                                     "fidelity_loss": s["fidelity"] - f,
                                     "exposed_couplers": len(job.exposed.get(p.index, [])),
                                     "two_q_count": p.two_q_count, "depth": p.depth, "solo_depth": s["depth"],
                                     "layouts_scored": p.n_layouts_scored, "exhaustive": p.exhaustive,
                                     "job_depth": job.depth, "active_qubits": job.active_qubits,
                                     **{f"cost_{kk_}": v for kk_, v in cost.row().items() if kk_ not in ("k", "shots")}})
                print(f"[e7] {cal.name} {mix} buffer={buffer} k={k}/{cap}", flush=True)
        print(f"[e7] {cal.name} done in {time.perf_counter() - t_dev:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    return {"e7_packing": df, "e7_packing_summary": summarise(df, shots)}


def summarise(df: pd.DataFrame, shots: int = SHOTS) -> pd.DataFrame:
    out = []
    keys = ["device", "mix", "buffer", "crosstalk_factor", "k"]
    for (dev, mix, buf, c, k), g in df.groupby(keys, sort=False):
        first = g.iloc[0]
        noise = float((g.fidelity_solo - g.fidelity_solo_repeat).abs().mean())
        out.append({"device": dev, "mix": mix, "buffer": int(buf), "crosstalk_factor": float(c), "k": int(k),
                    "capacity": int(first.capacity), "at_capacity": int(k) == int(first.capacity),
                    "mean_fidelity_packed": float(g.fidelity_packed.mean()),
                    "mean_fidelity_solo": float(g.fidelity_solo.mean()),
                    "fidelity_loss": float(g.fidelity_solo.mean() - g.fidelity_packed.mean()),
                    "worst_program_loss": float(g.fidelity_loss.max()),
                    "shot_noise_fidelity": noise,
                    "mean_esp_packed": float(g.esp_packed.mean()), "mean_esp_solo": float(g.esp_solo.mean()),
                    "programs_exposed": int((g.exposed_couplers > 0).sum()),
                    "exposed_couplers": int(g.exposed_couplers.sum()),
                    "active_qubits": int(first.active_qubits), "job_depth": int(first.job_depth),
                    "shots": shots,
                    "qpu_s_job": float(first.cost_qpu_s_job),
                    "qpu_s_per_program": float(first.cost_qpu_s_per_program),
                    "qpu_s_per_program_solo": float(first.cost_qpu_s_per_program_solo),
                    "price_per_program": float(first.cost_price_per_program),
                    "price_per_program_solo": float(first.cost_price_per_program_solo),
                    "programs_per_free_10min": float(first.cost_programs_per_free_plan),
                    "programs_per_free_10min_solo": float(first.cost_programs_per_free_plan_solo),
                    "qpu_s_per_program_with_overhead": float(first.cost_qpu_s_per_program_with_overhead),
                    "qpu_s_per_program_solo_with_overhead": float(first.cost_qpu_s_per_program_solo_with_overhead),
                    "all_placements_exhaustive": bool(g.exhaustive.all()),
                    "note": BUFFER1_NOTE if int(buf) >= 1 else
                    ("no program adjacent to another at this k" if int(g.exposed_couplers.sum()) == 0 else "")})
    return pd.DataFrame(out)


# ---- figures -------------------------------------------------------------------
SETTINGS = [(1, 1.0, "buffer 1", "BLUE", "-"), (0, 1.0, "buffer 0, no crosstalk", "ORANGE", "-"),
            (0, 2.0, "buffer 0, crosstalk x2", "YELLOW", "--"), (0, 5.0, "buffer 0, crosstalk x5", "MAGENTA", "--")]


def fidelity_figure(summ: pd.DataFrame, out: str):
    """Fidelity loss vs k, one line per buffer / crosstalk setting, panels per device and mix."""
    from . import figures as F
    plt = F.plt
    devs, mixes = list(summ.device.unique()), list(summ.mix.unique())
    fig, axes = plt.subplots(len(devs), len(mixes), figsize=(4.4 * len(mixes), 3.2 * len(devs)),
                             sharex=True, sharey=True, squeeze=False)
    for r, dev in enumerate(devs):
        for c, mix in enumerate(mixes):
            ax = axes[r][c]
            d = summ[(summ.device == dev) & (summ.mix == mix)]
            noise = d.shot_noise_fidelity.mean()
            ax.axhspan(-noise, noise, color=F.GRID, alpha=0.6, lw=0)
            ax.axhline(0, color=F.MUTED, lw=0.8)
            for buf, fac, label, col, ls in SETTINGS:
                g = d[(d.buffer == buf) & (d.crosstalk_factor == fac)].sort_values("k")
                if g.empty:
                    continue
                ax.plot(g.k, g.fidelity_loss, ls, color=getattr(F, col), marker="o", markersize=3.5, label=label)
                cap = g[g.at_capacity]
                ax.scatter(cap.k, cap.fidelity_loss, s=46, facecolors="none", edgecolors=getattr(F, col), zorder=3)
                if fac == 1.0:  # one capacity label per buffer
                    for _, rr in cap.iterrows():
                        ax.annotate(f"{int(rr.k)}", xy=(rr.k, rr.fidelity_loss), xytext=(5, -9 if buf else 3),
                                    textcoords="offset points", fontsize=7, color=getattr(F, col))
            ax.set_xscale("log", base=2)
            ax.set_title(f"{dev.replace('fake_', '')}: {'TFIM n=4 copies' if mix == 'tfim4' else 'mixed programs'}")
            if r == len(devs) - 1:
                ax.set_xlabel("programs packed in one job (k)")
            if c == 0:
                ax.set_ylabel("mean fidelity loss vs solo")
    ticks = [t for t in (1, 2, 4, 8, 16, 32) if t <= 2 * summ.k.max()]
    from matplotlib.ticker import NullFormatter
    for ax in axes.flat:
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0][-1].legend(fontsize=7, loc="upper left")
    axes[0][0].annotate("circle = greedy capacity (k shown)\nshaded = shot-noise level", xy=(0.03, 0.97),
                        xycoords="axes fraction", va="top", fontsize=7, color=F.MUTED)
    fig.suptitle("Job packing: quality cost of sharing one QPU job (emulated)", color=F.INK, x=0.01, ha="left")
    F._save(fig, out)


def cost_figure(summ: pd.DataFrame, out: str):
    """Braket-style price and IBM free-plan throughput per program vs k."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    from . import figures as F
    plt = F.plt
    d = summ[summ.crosstalk_factor == 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    # Price depends only on k and shots, not on the circuit or device: one line.
    pr = d.groupby("k").price_per_program.first().sort_index()
    axes[0].plot(pr.index, pr.values, color=F.BLUE, marker="o", markersize=3.5, label="packed")
    solo_price = d.price_per_program_solo.iloc[0]
    axes[0].axhline(solo_price, color=F.GRAY, ls="--", lw=1)
    axes[0].annotate(f"one job per program: ${solo_price:.2f}", xy=(1, solo_price), xytext=(0, 4),
                     textcoords="offset points", fontsize=8, color=F.MUTED)
    colors = {("fake_aachen", "tfim4"): F.BLUE, ("fake_aachen", "mixed"): F.AQUA,
              ("fake_torino", "tfim4"): F.ORANGE, ("fake_torino", "mixed"): F.MAGENTA}
    for (dev, mix, buf), g in d.groupby(["device", "mix", "buffer"]):
        g = g.sort_values("k")
        axes[1].plot(g.k, g.programs_per_free_10min, "-" if buf == 1 else ":", color=colors.get((dev, mix), F.GRAY),
                     marker="o", markersize=3, label=f"{dev.replace('fake_', '')}, {mix}, buffer {buf}")
    axes[1].axhline(d.programs_per_free_10min_solo.mean(), color=F.GRAY, ls="--", lw=1)
    axes[1].annotate("one job per program", xy=(1, d.programs_per_free_10min_solo.mean()), xytext=(0, 4),
                     textcoords="offset points", fontsize=8, color=F.MUTED)
    ks = [t for t in (1, 2, 4, 8, 16, 32) if t <= 2 * d.k.max()]
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("programs packed in one job (k)")
    axes[0].set_ylabel("price per program (USD, Braket-style)")
    axes[0].set_title(f"Price per program, {int(d.shots.iloc[0])} shots each")
    axes[1].set_ylabel("programs per 10 free IBM minutes")
    axes[1].set_title("Free-plan throughput (shot time only)")
    axes[1].legend(fontsize=6.5, loc="upper left")
    F._save(fig, out)


def make_figures(res_dir: str = "results", fig_dir: str = "figures"):
    p = os.path.join(res_dir, "e7_packing_summary.csv")
    if not os.path.exists(p):
        return
    os.makedirs(fig_dir, exist_ok=True)
    summ = pd.read_csv(p)
    fidelity_figure(summ, os.path.join(fig_dir, "e7_packing_fidelity.png"))
    cost_figure(summ, os.path.join(fig_dir, "e7_packing_cost.png"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    for name, frame in run(a.quick).items():
        frame.to_csv(os.path.join("results", f"{name}.csv"), index=False)
    make_figures()
