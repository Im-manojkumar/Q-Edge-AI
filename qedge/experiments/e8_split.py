"""E8: the split hybrid model, measured: heads, leakage, the quantum head on devices, the router.

Idea 3 cuts a small CNN between the edge and a remote resource and offers a
k-qubit quantum head on the k-number embedding (``qedge.ml``, ``orchestrator/
split.py``). Round 1 measured one seed; this experiment measures what the
router's profile and the report need, over seeds, and checks the cost model:

(a) Heads. Seeds 0, 1, 2; k in 2, 3, 4, 5, 6, 8. Transfer mode (backbone + linear
    head trained end to end, frozen, then every head on its embedding): linear,
    parameter-matched MLP, quantum (L=2) and quantum_l0 (the L=0 ablation). End-to-end
    mode (backbone and head trained jointly from scratch) at k = 4 and 8 for every head,
    the fairness check. Test and validation accuracy, parameters, training time; mean
    and spread over seeds, and the paired difference quantum minus linear / minus MLP
    per seed (same seed, same test images). The trained quantum heads are also read
    out with 100 and 1000 shots sampled from their exact output distributions.
(b) Leakage. Per (seed, k, cut) against ridge, MLP and kNN decoders
    (``qedge.ml.leakage``): re-identification, re-identification beyond the label
    (per image), r2, r2 beyond the label, the strongest attack and ``private_ok``. A few
    test images and their reconstructions from every cut (k=4, seed 0) are kept for a
    figure.
(c) Device. The seed-0 quantum head at k = 4 and 8 on all 360 test images: exact,
    100 / 1000 sampled shots, ``ibm-auto`` (calibration-chosen device and qubits,
    emulated with that device's noise model from its calibration snapshot) and
    ``predict_packed`` (8 or 16 samples per job). Accuracy, agreement with the exact
    prediction, device, jobs, QPU seconds and price (one job per sample vs packed) and
    wall time; and the cost model's quantum-head quality, f x acc + (1 - f) / 10 with the
    calibrated fidelity f, against the measured device accuracy (as E6 checks its
    predictors). A larger per_job is not run when the device never filled the smaller
    one (the capacity bounds both, so the jobs would be the same); the row says so.
(d) Profile. ``results/split_profile.json`` from (a) and (b): accuracies are the mean
    over seeds, leakage the worst case over seeds (and, for the shared cuts, over k).
    A ``--quick`` run writes ``split_profile.quick.json`` instead, so it never replaces
    a full-run profile.
(e) Router. With that profile loaded, the decisions of ``qedge_static`` and
    ``qedge_adaptive`` over: profile digits / camera; edge device; uplink 1-1000 Mbps;
    privacy off / on; batch 1 / 32; deadline 0.5 s (real time) / 3600 s (batch job);
    today's QPU (calibration snapshots) / a future QPU; measured head accuracies / what-ifs.
    Each row also scores the best quantum-head option as if its gate were open
    (``qpu_head_score``, ``sim_head_score`` against ``chosen_score``; lower is better), to
    show how far it is from being picked.

Choices and assumptions (labelled; the module docstrings of ``qedge.ml`` hold the rest):

* One data split (``DATA_SEED`` = 0) for every seed: the seeds change initialisation and
  minibatch order (and the attacker's MLP seed), not the 360 test images, so paired
  differences compare heads on the same images. The spread over seeds is therefore
  training noise only, not data-split noise.
* Training settings are ``nets.BACKBONE_FIT`` / ``nets.HEAD_FIT`` (the round-1 settings).
* Seeds run in parallel worker processes, each pinned to one BLAS thread. Every job is
  fully seeded, so the result does not depend on scheduling; the training times in the
  CSVs are single-thread times measured while up to four jobs share the machine.
* Shot-sampled accuracy uses one sampling seed per (seed, k, shots) (1000 + seed).
* Device runs are emulations from calibration snapshots (no IBM account is used, even if
  one is configured: ``prefer_live=False``). QPU seconds and prices are ``packing_cost``
  figures: catalogue QPU timings and Braket-style per-task / per-shot prices (as E7).
  "One job per sample" counts every circuit as its own task, as Braket bills it.
* The ``ibm-auto`` emulator is not seeded by its backend (``CalibratedIBMBackend`` builds
  ``AerSimulator.from_backend`` without a seed), so that row moves with shot noise from run
  to run: two full runs gave 0.944 and 0.944 at k=4, 0.919 and 0.925 at k=8. Training,
  leakage, sampled shots and the packed runs (seed 17) repeated exactly in those two runs.
* Modelled device quality: ``modelled_quality`` uses the cost model's fidelity (the head
  circuit after ``split.ROUTING_2Q``, best region of the best snapshot device);
  ``modelled_quality_esp`` uses the ESP of the layouts the device paths actually ran. Both
  plug in the exact accuracy of the same head on the same images.
* The router's camera profile is ``split.camera_profile``: an assumption profile that
  inherits the digits head accuracies and embedding privacy verdict measured here.
* Router edge devices: exactly one edge resource is available per condition. "sensor"
  is the catalogue edge_cpu with its throughput divided by ``SENSOR_SLOWDOWN`` (an
  assumption: a microcontroller-class device; its power is left at the edge_cpu figure).
  The uplink is set on every cloud and quantum resource; the downlink stays at the
  catalogue's 100 Mbps.
* Today's QPU in the router: the cost model gets the calibrated figures of the best
  qubit region of the best snapshot device for the head circuit, as the Monitor does
  with ``ibm-auto``. The "future" QPU is a what-if, not a forecast: those error rates
  x 0.01, queue x 0.1, per-task and per-shot prices x 0.1 (E4's "100x better" QPU).
* Accuracy what-ifs (labelled in every row): ``qhead_equal`` sets the quantum head's
  accuracy (quantum key, every k, both profiles) to the best classical head's, and
  ``qhead_equal_private`` also marks the embedding private. Neither is measured.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

SEEDS = (0, 1, 2)
KS = (2, 3, 4, 5, 6, 8)
E2E_KS = (4, 8)
HEADS = ("linear", "mlp", "quantum", "quantum_l0")
SHOTS = (100, 1000)
DEVICE_KS = (4, 8)
PER_JOB = (8, 16)
DEVICE_SHOTS = 1000
DATA_SEED = 0
RECON_K, RECON_SEED = 4, 0
RECON_CLASSES = (0, 2, 3, 5, 7, 8)      # first test image of each, for the reconstruction figure
CUTS = ("input", "conv1", "conv2", "embedding")
QUICK = {"seeds": (0,), "ks": (2, 4), "e2e_ks": (4,), "e2e_heads": ("linear", "quantum"), "device_ks": (4,),
         "device_n": 48, "per_job": (8,), "fit": {"epochs": 12, "patience": 5}}

# Router sweep grid (part e).
SENSOR_SLOWDOWN = 50.0           # assumption: sensor-class device = edge_cpu / 50 (0.5 GFLOPS)
UPLINKS = (1, 5, 20, 100, 1000)
DEADLINES = {"real-time": 0.5, "batch job": 3600.0}
ROUTER_GRID = {"profile": ("digits", "camera"), "edge": ("edge_gpu", "edge_cpu", "sensor"), "uplink": UPLINKS,
               "privacy": (False, True), "batch": (1, 32), "deadline": tuple(DEADLINES),
               "qpu": ("today", "future"), "whatif": ("measured", "qhead_equal", "qhead_equal_private")}
ROUTER_POLICIES = ("qedge_static", "qedge_adaptive")
ROUTER_K = 4
WHATIF_NOTE = {"measured": "measured head accuracies and leakage",
               "qhead_equal": "assumption: quantum head as accurate as the best classical head",
               "qhead_equal_private": "assumption: quantum head as accurate as the best classical head "
                                      "and the embedding private"}


# ---- (a) + (b): one training job per (mode, seed), run in worker processes --------------
def _limit_threads():
    """One BLAS thread per worker: parallel jobs do not oversubscribe the cores, and every
    job computes the same way however the jobs are scheduled."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:     # pragma: no cover - threadpoolctl ships with scikit-learn
        pass


def _fit_kw(fit: dict | None, defaults: dict) -> dict:
    return {**defaults, **(fit or {})}


def _leak_row(r: dict, seed: int) -> dict:
    """One flat CSV row from a ``leakage_report`` row."""
    n = r["n_test"]
    row = {"seed": seed, "k": r["k"], "cut": r["cut"], "values": r["values"], "n_test": n, "chance": r["chance"],
           "reid": r["reid"], "reid_images": round(r["reid"] * n), "reid_top5": r["reid_top5"],
           "reid_label": r["reid_label"], "reid_beyond_label": r["reid_beyond_label"],
           "reid_beyond_label_images": round(r["reid_beyond_label"] * n), "reid_with_label": r["reid_with_label"],
           "r2": r["r2"], "r2_label": r["r2_label"], "r2_beyond_label": r["r2_beyond_label"],
           "private_ok": bool(r["private_ok"]), "private_ok_absolute": bool(r["private_ok_absolute"]),
           "strongest_reid": r["strongest"]["reid"], "strongest_beyond_label": r["strongest"]["reid_beyond_label"],
           "strongest_r2": r["strongest"]["r2"]}
    for a, v in r["attacks"].items():
        row.update({f"reid_{a}": v["reid"], f"reid_beyond_label_{a}": v["reid_beyond_label"], f"r2_{a}": v["r2"]})
    return row


def reconstruction_examples(model, data, leak_rows: list[dict], seed: int = 0,
                            classes=RECON_CLASSES) -> list[dict]:
    """Originals, the label-only baseline and the strongest re-identification decoder's
    reconstruction from every cut, for the first test image of each class in ``classes``.
    ``rank`` is how many of the N test originals are closer to the reconstruction than the
    image's own (0 = re-identified)."""
    from ..ml import leakage as L
    Xtr, ytr, Xval, yval, Xte, yte = data
    idx = [int(np.flatnonzero(yte == c)[0]) for c in classes]
    flat = lambda A: np.asarray(A, np.float64).reshape(len(A), -1)          # noqa: E731
    rows = []

    def add(kind, cut, attack, rec_all):
        rank = L.reid_ranks(Xte, rec_all)
        for i in idx:
            rows.append({"kind": kind, "cut": cut, "attack": attack, "image": i, "label": int(yte[i]),
                         "rank": int(rank[i]), **{f"p{j}": float(v) for j, v in enumerate(flat(rec_all)[i])}})

    add("original", "", "", flat(Xte))
    add("label-only", "", "class mean", L.label_baseline(Xtr, ytr, yte))
    by_cut = {r["cut"]: r for r in leak_rows}
    for cut in CUTS:
        attack = by_cut[cut]["strongest"]["reid"]
        Htr, Hval, Hte = (model.forward_to(cut, X) for X in (Xtr, Xval, Xte))
        if attack == "ridge":
            dec = L.ridge_decoder(Htr, Xtr, Hval, Xval)[0]
        elif attack == "mlp":
            dec = L.mlp_decoder(Htr, Xtr, Hval, Xval, seed=seed)[0]
        else:
            dec = L.knn_decoder(Htr, Xtr, Hval, Xval)[0]["reid"]
        add("reconstruction", cut, attack, dec(Hte))
    return rows


def transfer_job(seed: int, ks=KS, fit: dict | None = None, recon: bool = True, keep_ks=DEVICE_KS) -> dict:
    """Transfer mode for one seed and every k: head accuracies, shot-sampled accuracy,
    leakage at every cut, the ``profile.summarise`` entry per k, and (seed 0) the trained
    quantum heads for the device runs and the reconstruction examples."""
    from ..ml import leakage as L
    from ..ml import nets
    from ..ml import profile as PR
    from ..ml.data import load_digits_split
    _limit_threads()
    data = load_digits_split(DATA_SEED)
    Xtr, ytr, Xval, yval, Xte, yte = data
    heads, shots, leak, per_k, keep, recon_rows = [], [], [], {}, {}, []
    input_row = None
    for k in ks:
        t0 = time.perf_counter()
        tr = nets.train_transfer(data, k, seed=seed, backbone_kw=_fit_kw(fit, nets.BACKBONE_FIT),
                                 head_kw=_fit_kw(fit, nets.HEAD_FIT))
        bb = tr.backbone_fit
        for h in HEADS:
            f, head = tr.fits[h], tr.heads[h]
            heads.append({"mode": "transfer", "k": k, "head": h, "seed": seed, "test_acc": tr.accuracy(h, Xte, yte),
                          "val_acc": float(f.best_val_acc), "n_params": int(head.n_params),
                          "hidden": head.config().get("hidden"), "train_s": float(f.seconds),
                          "epochs_run": len(f.history), "best_epoch": int(f.best_epoch),
                          "backbone_train_s": float(bb["seconds"]), "backbone_val_acc": float(bb["val_acc"])})
        q = tr.heads["quantum"]
        Zte = tr.model.embed(Xte).astype(np.float64)
        exact = q.predict_exact(Zte)
        for s in SHOTS:
            p = q.predict_shots(Zte, s, seed=1000 + seed)
            shots.append({"k": k, "seed": seed, "shots": s, "accuracy": float(np.mean(p == yte)),
                          "agreement_with_exact": float(np.mean(p == exact)),
                          "exact_accuracy": float(np.mean(exact == yte))})
        # The input cut does not depend on the model: attack it once per seed.
        if input_row is None:
            input_row = L.leakage_report(tr.model, data, cuts=("input",), seed=seed)[0]
        rows = [{**input_row, "k": k}] + L.leakage_report(tr.model, data, cuts=CUTS[1:], seed=seed)
        leak += [_leak_row(r, seed) for r in rows]
        per_k[k] = PR.summarise(tr, Xte, yte, rows)
        if seed == RECON_SEED and k in keep_ks:
            keep[k] = {"backbone": tr.model.backbone, "quantum": q.params, "layers": q.layers}
        if recon and seed == RECON_SEED and k == RECON_K:
            recon_rows = reconstruction_examples(tr.model, data, rows, seed=seed)
        print(f"[e8] transfer seed={seed} k={k}: " + " ".join(f"{r['head']} {r['test_acc']:.3f}" for r in heads[-4:])
              + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
    return {"heads": heads, "shots": shots, "leakage": leak, "per_k": per_k, "models": keep, "recon": recon_rows}


def e2e_job(seed: int, ks=E2E_KS, heads=HEADS, fit: dict | None = None) -> dict:
    """End-to-end mode for one seed: backbone and head trained jointly from scratch."""
    from ..ml import nets
    from ..ml.data import load_digits_split
    _limit_threads()
    data = load_digits_split(DATA_SEED)
    Xte, yte = data[4], data[5]
    rows = []
    for k in ks:
        for h in heads:
            m = nets.train_end_to_end(data, k, h, seed=seed, **_fit_kw(fit, nets.BACKBONE_FIT))
            rows.append({"mode": "end-to-end", "k": k, "head": h, "seed": seed, "test_acc": m.accuracy(Xte, yte),
                         "val_acc": float(m.meta["val_acc"]), "n_params": int(m.head.n_params),
                         "hidden": m.head.config().get("hidden"), "train_s": float(m.meta["seconds"]),
                         "epochs_run": int(m.meta["epochs_run"]), "best_epoch": int(m.meta["best_epoch"]),
                         "backbone_train_s": float("nan"), "backbone_val_acc": float("nan")})
            print(f"[e8] end-to-end seed={seed} k={k} {h}: {rows[-1]['test_acc']:.3f} ({m.meta['seconds']:.0f}s)",
                  flush=True)
    return {"heads": rows}


def _run_jobs(jobs: list[tuple], workers: int) -> list[dict]:
    """Run (function, kwargs) jobs in worker processes (spawned, so nothing is inherited
    half-initialised); results come back in job order."""
    if workers <= 1:
        return [fn(**kw) for fn, kw in jobs]
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(fn, **kw) for fn, kw in jobs]
        return [f.result() for f in futs]


# ---- (a) aggregation ---------------------------------------------------------------------
def summarise_heads(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std (ddof 1), min and max of test accuracy over seeds per (mode, k, head)."""
    g = df.groupby(["mode", "k", "head"], sort=False)
    out = g.agg(seeds=("seed", "nunique"), test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
                test_acc_min=("test_acc", "min"), test_acc_max=("test_acc", "max"),
                val_acc_mean=("val_acc", "mean"), n_params=("n_params", "first"), hidden=("hidden", "first"),
                train_s_mean=("train_s", "mean")).reset_index()
    return out


def paired_differences(df: pd.DataFrame, pairs=(("quantum", "linear"), ("quantum", "mlp"),
                                                ("quantum_l0", "linear"), ("quantum", "quantum_l0"))) -> pd.DataFrame:
    """Per (mode, k) and pair (a, b): test accuracy of head a minus head b with the same
    seed (same backbone in transfer mode), one column per seed, and mean / min / max."""
    rows = []
    for (mode, k), g in df.groupby(["mode", "k"], sort=False):
        acc = g.pivot_table(index="seed", columns="head", values="test_acc")
        for a, b in pairs:
            if a not in acc or b not in acc:
                continue
            d = (acc[a] - acc[b]).dropna()
            if d.empty:
                continue
            rows.append({"mode": mode, "k": int(k), "difference": f"{a} - {b}", "seeds": len(d),
                         "mean": float(d.mean()), "min": float(d.min()), "max": float(d.max()),
                         "seeds_a_better": int((d > 0).sum()), "seeds_equal": int((d == 0).sum()),
                         **{f"seed{int(s)}": float(v) for s, v in d.items()}})
    return pd.DataFrame(rows)


def summarise_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Per (k, cut): worst case over seeds (what the profile uses) and the mean."""
    out = []
    for (k, cut), g in df.groupby(["k", "cut"], sort=False):
        out.append({"k": int(k), "cut": cut, "seeds": g.seed.nunique(), "n_test": int(g.n_test.iloc[0]),
                    "chance": float(g.chance.iloc[0]),
                    "reid_mean": float(g.reid.mean()), "reid_min": float(g.reid.min()), "reid_max": float(g.reid.max()),
                    "reid_label": float(g.reid_label.max()),
                    "reid_beyond_label_mean": float(g.reid_beyond_label.mean()),
                    "reid_beyond_label_max": float(g.reid_beyond_label.max()),
                    "reid_beyond_label_images_max": int(g.reid_beyond_label_images.max()),
                    "r2_mean": float(g.r2.mean()), "r2_beyond_label_mean": float(g.r2_beyond_label.mean()),
                    "r2_beyond_label_max": float(g.r2_beyond_label.max()),
                    "private_ok_all_seeds": bool(g.private_ok.all()),
                    "private_ok_any_seed": bool(g.private_ok.any()),
                    "strongest_reid": "/".join(sorted(set(g.strongest_reid)))})
    return pd.DataFrame(out)


# ---- (d) profile ------------------------------------------------------------------------
def worst_leak(rows: list[dict]) -> dict:
    """The worst case over seeds of one cut's leakage rows (largest reid, reid beyond the
    label, r2 beyond the label), in the shape ``leakage.private_ok`` reads."""
    r0 = rows[0]
    out = {"cut": r0["cut"], "k": r0["k"], "n_test": r0["n_test"], "chance": r0["chance"],
           "attacks": {a: {} for r in rows for a in r.get("attacks", {})}}
    for key in ("reid", "reid_top5", "reid_label", "reid_beyond_label", "r2", "r2_beyond_label", "r2_label"):
        out[key] = float(max(r[key] for r in rows))
    return out


def combine_seeds(per_seed: dict[int, dict]) -> dict:
    """``measured_profile`` input from per-seed ``profile.summarise`` entries ({seed: {k: entry}}):
    test and validation accuracies are means over seeds, leakage rows the worst case over
    seeds; sizes, FLOPs, parameters and the circuit are the same for every seed (checked)."""
    seeds = sorted(per_seed)
    ks = sorted(set.intersection(*(set(per_seed[s]) for s in seeds)))
    out = {}
    for k in ks:
        es = [per_seed[s][k] for s in seeds]
        e0 = es[0]
        for e in es[1:]:
            if e["params"] != e0["params"] or e["quantum_circuit"] != e0["quantum_circuit"]:
                raise ValueError(f"k={k}: model shape differs between seeds")
        leak = [worst_leak([next(r for r in e["leakage"] if r["cut"] == cut) for e in es]) for cut in CUTS]
        mean = lambda key: {h: float(np.mean([e[key][h] for e in es])) for h in e0[key]}   # noqa: E731
        out[k] = {**e0, "accuracy": mean("accuracy"), "val_accuracy": mean("val_accuracy"), "leakage": leak,
                  "accuracy_std": {h: float(np.std([e["accuracy"][h] for e in es], ddof=1)) if len(es) > 1 else 0.0
                                   for h in e0["accuracy"]}}
    return out


def build_profile(per_seed: dict[int, dict]) -> dict:
    """The measured digits split profile (see the module docstring, part d)."""
    from ..ml import profile as PR
    from ..orchestrator import split
    combined = combine_seeds(per_seed)
    seeds = sorted(per_seed)
    prof = PR.measured_profile(combined, source=f"measured (E8): test accuracy mean over seeds "
                                                f"{', '.join(map(str, seeds))}; leakage worst case over seeds")
    for k, e in prof["embedding"].items():
        e["accuracy_std"] = combined[int(k)]["accuracy_std"]
        e["seeds"] = seeds
    prof["seeds"] = seeds
    prof["data_split_seed"] = DATA_SEED
    problems = split.check_profile(prof)
    if problems:
        raise ValueError("measured profile fails the schema: " + "; ".join(problems))
    PR.validate_profile(prof)
    return prof


# ---- (c) device -------------------------------------------------------------------------
def _qhead(model: dict, k: int):
    from ..ml.qhead import QuantumHead
    q = QuantumHead(k, layers=model["layers"])
    q.params = model["quantum"]
    return q


def device_runs(models: dict, n: int | None = None, per_job=PER_JOB, shots: int = DEVICE_SHOTS) -> pd.DataFrame:
    """The seed-0 quantum heads on the test split through every inference path (part c)."""
    from ..ml import nets
    from ..ml.data import load_digits_split
    from ..orchestrator import split
    from ..quantum import calibration as C
    from ..quantum import packing as P
    from ..quantum.backends.qiskit_backend import CalibratedIBMBackend
    from ..workloads import make_workload
    data = load_digits_split(DATA_SEED)
    Xte, yte = data[4][:n], data[5][:n]
    be = CalibratedIBMBackend(prefer_live=False)     # snapshots + emulation, never a real account
    rows = []
    for k in sorted(models):
        m = models[k]
        q = _qhead(m, k)
        # The float32 embedding the edge transmits, as in training and in transfer_job.
        Z = nets.SplitCNN(k, m["backbone"], q).embed(Xte).astype(np.float64)
        exact = q.predict_exact(Z)
        acc_exact = float(np.mean(exact == yte))
        ez = q.expvals_exact(Z)
        # What the router's cost model predicts: the routed head circuit's fidelity in the best
        # region of the best snapshot device, then f x acc + (1 - f) / classes.
        w = make_workload("split_inference", seed=0, profile="digits", k=k, batch=len(Z), shots=shots,
                          privacy=False, deadline_s=3600.0)
        fig = C.best_device_figures(w.quantum, be.devices)
        f_model = float(fig["fidelity"])
        model_q = lambda f: f * acc_exact + (1 - f) / split.PROFILES["digits"]["classes"]   # noqa: E731
        base = {"k": k, "n_samples": len(Z), "exact_accuracy": acc_exact, "cost_model_device": fig["device"],
                "cost_model_fidelity": f_model, "modelled_quality": model_q(f_model)}

        def row(path, pred, E=None, **kw):
            rows.append({**base, "path": path, "accuracy": float(np.mean(pred == yte)),
                         "agreement_with_exact": float(np.mean(pred == exact)),
                         "mean_abs_err_expz": float(np.abs(E - ez).mean()) if E is not None else 0.0, **kw})
            print(f"[e8] device k={k} {path}: accuracy {rows[-1]['accuracy']:.3f}", flush=True)

        row("exact", exact, ez, shots=0)
        for s in SHOTS:
            E = q.expvals_shots(Z, s, seed=2000 + s)
            row(f"sampled {s} shots", q.readout(E), E, shots=s)
        t0 = time.perf_counter()
        r = q.predict_on_backend(be, Z, shots=shots)
        wall = time.perf_counter() - t0
        depths = r.meta["transpiled_depths"]
        qpu_s = sum(P.job_qpu_seconds(shots, d) for d in depths)
        price = sum(P.packing_cost([d], shots).price_job for d in depths)   # one task per circuit
        esp = float(r.meta["predicted_esp"])
        row("ibm-auto (emulated)", r.predictions, r.expvals, shots=shots, device=r.meta["device"],
            jobs=len(depths), circuits=len(depths), qpu_s=qpu_s, qpu_s_one_job_per_sample=qpu_s, price_usd=price,
            price_one_job_per_sample_usd=price, wall_s=wall, mean_transpiled_depth=float(np.mean(depths)),
            predicted_esp=esp, modelled_quality_esp=model_q(esp))
        fill = None                     # (per_job, most samples any job held) of the previous packed run
        for pj in sorted(per_job):
            if fill is not None and fill[1] < fill[0]:
                # The device never took even fill[0] samples per job, so a larger per_job is
                # bound by the same capacity: the same jobs again. Said in the row instead of run.
                rows[-1]["note"] = (f"at most {fill[1]} samples fit in one job on {rows[-1]['device']}, so "
                                    f"per_job {pj} would give the same jobs; not run")
                print(f"[e8] device k={k} packed {pj} per job: skipped ({rows[-1]['note']})", flush=True)
                continue
            t0 = time.perf_counter()
            r = q.predict_packed(be, Z, shots=shots, per_job=pj)
            wall = time.perf_counter() - t0
            jobs = r.meta["jobs"]
            fill = (pj, max(j["k"] for j in jobs))
            esp = float(np.mean([e for j in jobs for e in j["esp"]]))
            row(f"packed {pj} per job", r.predictions, r.expvals, shots=shots, device=",".join(r.meta["devices"]),
                jobs=r.n_jobs, circuits=r.n_jobs, qpu_s=r.meta["qpu_s"], qpu_s_one_job_per_sample=r.meta["qpu_s_solo"],
                price_usd=r.meta["price"], price_one_job_per_sample_usd=r.meta["price_solo"], wall_s=wall,
                mean_transpiled_depth=float(np.mean([j["depth"] for j in jobs])), predicted_esp=esp,
                modelled_quality_esp=model_q(esp), samples_per_job_mean=len(Z) / r.n_jobs,
                samples_per_job_max=fill[1], note="")
    return pd.DataFrame(rows)


# ---- (e) router -------------------------------------------------------------------------
@contextlib.contextmanager
def whatif_profiles(name: str):
    """Temporarily apply an accuracy / privacy what-if to every loaded split profile."""
    from ..orchestrator import split
    saved = copy.deepcopy(split.PROFILES)
    try:
        if name != "measured":
            for prof in split.PROFILES.values():
                for e in prof["embedding"].values():
                    acc = e["accuracy"]
                    e["accuracy"] = {**acc, "quantum": max(acc["linear"], acc["mlp"])}
                    if name == "qhead_equal_private":
                        e["private_ok"] = True
                        e["whatif"] = WHATIF_NOTE[name]
        yield
    finally:
        split.PROFILES.clear()
        split.PROFILES.update(saved)


def router_catalogue(edge: str, uplink: float, qpu: str):
    """The catalogue for one router condition (see the module docstring for what is assumed)."""
    from ..resources import load_catalogue
    res = load_catalogue()
    front = "edge_cpu" if edge == "sensor" else edge
    for n, r in res.items():
        if r.is_edge:
            r.available = n == front
        else:
            r.uplink_mbps = float(uplink)
    if edge == "sensor":
        res["edge_cpu"].gflops /= SENSOR_SLOWDOWN
        res["edge_cpu"].serial_gflops /= SENSOR_SLOWDOWN
    if qpu == "future":
        q = res["qpu"]
        q.queue_mean_s *= 0.1
        q.cost_per_task *= 0.1
        q.cost_per_shot *= 0.1
    return res


def qpu_figures(w, devices, qpu: str) -> dict:
    """Calibrated QPU figures for the workload's head circuit (as the Monitor gives the router),
    error rates x 0.01 for the future what-if."""
    from ..quantum import calibration as C
    fig = dict(C.best_device_figures(w.quantum, devices))
    if qpu == "future" and fig:
        for key in ("p1", "p2", "readout_err"):
            fig[key] *= 0.01
    return fig


def _category(p) -> str:
    if p.quantum is not None:
        return "quantum head (QPU)" if p.quantum == "qpu" else "quantum head (simulator)"
    if p.split is not None:
        return "split to cloud"
    return "edge only" if p.classical.startswith("edge") else "cloud (raw input)"


def router_sweep(grid: dict | None = None, policies=ROUTER_POLICIES, k: int = ROUTER_K, devices=None) -> pd.DataFrame:
    """Decisions of each policy for every combination in ``grid`` (keys as ``ROUTER_GRID``;
    missing keys take the first ``ROUTER_GRID`` value), against the split profiles currently
    loaded in ``split.PROFILES``."""
    import itertools

    from ..orchestrator.policies import make_policy
    from ..quantum import calibration as C
    from ..resources import ResourceState
    from ..workloads import make_workload
    g = {key: tuple((grid or {}).get(key, ROUTER_GRID[key] if grid is None else ROUTER_GRID[key][:1]))
         for key in ROUTER_GRID}
    devices = devices if devices is not None else C.load_devices(prefer_live=False)
    rows = []
    for whatif in g["whatif"]:
        with whatif_profiles(whatif):
            fig_cache: dict = {}
            for profile, edge, uplink, privacy, batch, dl, qpu in itertools.product(
                    g["profile"], g["edge"], g["uplink"], g["privacy"], g["batch"], g["deadline"], g["qpu"]):
                res = router_catalogue(edge, uplink, qpu)
                w = make_workload("split_inference", seed=0, profile=profile, k=k, batch=batch, privacy=privacy,
                                  deadline_s=DEADLINES[dl])
                key = (w.quantum.n_qubits, w.quantum.depth, w.quantum.n_1q, w.quantum.n_2q, qpu)
                if key not in fig_cache:
                    fig_cache[key] = qpu_figures(w, devices, qpu)
                for pol_name in policies:
                    st = ResourceState()
                    if fig_cache[key]:
                        st.qpu_figures["qpu"] = dict(fig_cache[key])
                    pol = make_policy(pol_name, copy.deepcopy(res))
                    d = pol.decide(w, st)
                    c, ests = d.chosen, {e.placement.label: e for e in d.estimates}
                    # The best-scoring quantum-head option on each quantum resource, scored as if
                    # its gate were open: how far it is from being picked.
                    qhead = {q: min((e for e in d.estimates if e.placement.quantum == q),
                                    key=lambda e: pol.score(w, e), default=None) for q in ("qpu", "quantum_sim")}
                    qbest = qhead["qpu"]
                    row = {"profile": profile, "edge": edge, "uplink_mbps": uplink, "privacy": privacy,
                           "batch": batch, "deadline": dl, "deadline_s": DEADLINES[dl], "qpu": qpu, "whatif": whatif,
                           "whatif_note": WHATIF_NOTE[whatif], "k": k, "policy": pol_name,
                           "placement": d.placement.label, "category": _category(d.placement),
                           "cut": d.placement.split or "", "feasible": c.feasible,
                           "meets_deadline": c.latency_s <= w.deadline_s, "latency_s": c.latency_s,
                           "edge_energy_j": c.edge_energy_j, "cost_usd": c.cost_usd, "quality": c.quality,
                           "classical_quality": w.classical_quality, "candidates": len(ests),
                           "chosen_score": pol.score(w, c),
                           "qpu_head_score": pol.score(w, qbest) if qbest else float("nan"),
                           "sim_head_score": pol.score(w, qhead["quantum_sim"]) if qhead["quantum_sim"] else float("nan"),
                           "qpu_head_cost_usd": qbest.cost_usd if qbest else float("nan"),
                           "qpu_head_latency_s": qbest.latency_s if qbest else float("nan"),
                           "qpu_head_quality": qbest.quality if qbest else float("nan"),
                           "qpu_head_fidelity": qbest.fidelity if qbest else float("nan"),
                           "qpu_device": qbest.qpu_device if qbest else ""}
                    for q, tag in (("qpu", "qpu"), ("quantum_sim", "sim")):
                        checks = d.quantum_checks.get(q, [])
                        failed = [(n, why) for n, ok, why in checks if not ok]
                        row[f"gate_{tag}"] = ("open" if not failed else "closed") if checks else "not evaluated"
                        row[f"gate_{tag}_failed"] = "; ".join(n for n, _ in failed)
                        row[f"gate_{tag}_why"] = " | ".join(f"{n}: {why}" for n, why in failed)
                    rows.append(row)
    return pd.DataFrame(rows)


def summarise_router(df: pd.DataFrame) -> pd.DataFrame:
    """Per (policy, qpu, what-if, privacy, deadline): how often each kind of placement is chosen,
    and how often the QPU's quantum gate is open."""
    keys = ["policy", "qpu", "whatif", "privacy", "deadline"]
    out = []
    for vals, g in df.groupby(keys, sort=False):
        r = dict(zip(keys, vals))
        r["conditions"] = len(g)
        for cat in ("edge only", "split to cloud", "cloud (raw input)", "quantum head (QPU)",
                    "quantum head (simulator)"):
            r[cat] = int((g.category == cat).sum())
        r["gate_qpu_open"] = int((g.gate_qpu == "open").sum())
        r["gate_sim_open"] = int((g.gate_sim == "open").sum())
        r["meets_deadline"] = int(g.meets_deadline.sum())
        r["most_common_qpu_gate_failure"] = (g.loc[g.gate_qpu_failed != "", "gate_qpu_failed"].mode().iloc[0]
                                             if (g.gate_qpu_failed != "").any() else "")
        out.append(r)
    return pd.DataFrame(out)


# ---- run --------------------------------------------------------------------------------
def profile_path(out_dir: str, quick: bool) -> str:
    """Where this run writes its profile: a quick run never replaces the full-run profile."""
    return os.path.join(out_dir, "split_profile.quick.json" if quick else "split_profile.json")


def run(quick: bool = False, out_dir: str = "results", workers: int | None = None) -> dict[str, pd.DataFrame]:
    from ..orchestrator import split
    seeds = QUICK["seeds"] if quick else SEEDS
    ks = QUICK["ks"] if quick else KS
    e2e_ks = QUICK["e2e_ks"] if quick else E2E_KS
    e2e_heads = QUICK["e2e_heads"] if quick else HEADS
    fit = QUICK["fit"] if quick else None
    dev_ks = QUICK["device_ks"] if quick else DEVICE_KS
    workers = workers or min(4, os.cpu_count() or 1)
    t0 = time.perf_counter()
    # Transfer jobs first: they are the longest.
    jobs = [(transfer_job, {"seed": s, "ks": ks, "fit": fit, "keep_ks": dev_ks}) for s in seeds]
    jobs += [(e2e_job, {"seed": s, "ks": e2e_ks, "heads": e2e_heads, "fit": fit}) for s in seeds]
    out = _run_jobs(jobs, workers)
    tr, e2e = out[:len(seeds)], out[len(seeds):]
    print(f"[e8] training and leakage done in {time.perf_counter() - t0:.0f}s", flush=True)

    heads = pd.DataFrame([r for o in tr + e2e for r in o["heads"]])
    leak = pd.DataFrame([r for o in tr for r in o["leakage"]])
    per_seed = {s: o["per_k"] for s, o in zip(seeds, tr)}
    models = next(o["models"] for s, o in zip(seeds, tr) if s == RECON_SEED)

    prof = build_profile(per_seed)
    os.makedirs(out_dir, exist_ok=True)
    path = profile_path(out_dir, quick)
    with open(path, "w") as f:
        json.dump(prof, f, indent=1)
    print(f"[e8] profile written to {path}", flush=True)

    t1 = time.perf_counter()
    saved = copy.deepcopy(split.PROFILES)
    try:
        split.load_profiles(path)
        device = device_runs(models, n=QUICK["device_n"] if quick else None,
                             per_job=QUICK["per_job"] if quick else PER_JOB)
        print(f"[e8] device runs done in {time.perf_counter() - t1:.0f}s", flush=True)
        router = router_sweep()
    finally:
        split.PROFILES.clear()
        split.PROFILES.update(saved)
    print(f"[e8] done in {time.perf_counter() - t0:.0f}s", flush=True)
    return {"e8_heads": heads, "e8_heads_summary": summarise_heads(heads), "e8_heads_paired": paired_differences(heads),
            "e8_shots": pd.DataFrame([r for o in tr for r in o["shots"]]),
            "e8_leakage": leak, "e8_leakage_summary": summarise_leakage(leak),
            "e8_reconstructions": pd.DataFrame(next(o["recon"] for s, o in zip(seeds, tr) if s == RECON_SEED)),
            "e8_device": device, "e8_router": router, "e8_router_summary": summarise_router(router)}


# ---- figures ----------------------------------------------------------------------------
HEAD_STYLE = {"linear": ("linear", "INK", "o"), "mlp": ("MLP (parameter-matched)", "AQUA", "s"),
              "quantum": ("quantum, L=2", "BLUE", "D"), "quantum_l0": ("quantum, L=0 (ablation)", "YELLOW", "^")}


def heads_figure(summ: pd.DataFrame, paired: pd.DataFrame, out: str):
    """Accuracy vs k per head (mean ± std over seeds), transfer and end to end, and the paired
    difference quantum minus linear per seed."""
    from . import figures as F
    plt = F.plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1.5, 0.8, 1.2]})
    off = {"linear": -0.15, "mlp": -0.05, "quantum": 0.05, "quantum_l0": 0.15}
    for ax, mode, title in ((axes[0], "transfer", "Transfer: heads on a frozen backbone"),
                            (axes[1], "end-to-end", "End to end: trained jointly")):
        d = summ[summ["mode"] == mode]
        for h, (label, col, mk) in HEAD_STYLE.items():
            g = d[d["head"] == h].sort_values("k")
            if g.empty:
                continue
            ax.errorbar(g.k + off[h], g.test_acc_mean, yerr=g.test_acc_std.fillna(0), color=getattr(F, col),
                        marker=mk, markersize=4, capsize=2, lw=1.5 if mode == "transfer" else 0,
                        elinewidth=1, label=label)
        ax.set_xticks(sorted(d.k.unique()))
        ax.set_xlabel("embedding size k (= qubits)")
        ax.set_title(title)
        if mode == "end-to-end":
            ax.set_xlim(min(d.k) - 1.5, max(d.k) + 1.5)
    lo = min(0.5, summ.test_acc_mean.min() - 0.05)
    for ax in axes[:2]:
        ax.set_ylim(lo, 1.0)
    axes[0].set_ylabel("test accuracy (mean ± std over seeds)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    ax = axes[2]
    ax.axhline(0, color=F.MUTED, lw=0.8)
    for mode, col, ls in (("transfer", F.BLUE, "-"), ("end-to-end", F.ORANGE, "--")):
        g = paired[(paired["mode"] == mode) & (paired.difference == "quantum - linear")].sort_values("k")
        if g.empty:
            continue
        seed_cols = [c for c in g.columns if re.fullmatch(r"seed\d+", c)]
        for c in seed_cols:
            ax.scatter(g.k + (0.1 if mode == "end-to-end" else 0), g[c], s=12, color=col, alpha=0.6, lw=0)
        ax.plot(g.k + (0.1 if mode == "end-to-end" else 0), g["mean"], ls, color=col, marker="o", markersize=3.5,
                label=f"{mode}: mean (dots: seeds)")
    ax.set_xticks(sorted(paired.k.unique()))
    ax.set_xlabel("embedding size k (= qubits)")
    ax.set_ylabel("accuracy: quantum minus linear")
    ax.set_title("Paired difference, same seed")
    ax.legend(fontsize=7.5, loc="lower right")
    F._save(fig, out)


CUT_STYLE = {"input": ("input (raw image)", "GRAY"), "conv1": ("conv1", "ORANGE"), "conv2": ("conv2", "MAGENTA"),
             "embedding": ("embedding (k numbers)", "BLUE")}


def leakage_figure(summ: pd.DataFrame, out: str):
    """Re-identification per cut vs k (log scale): absolute with chance and label-only lines,
    and beyond the label (per image) with the privacy limit. Range bars span the seeds."""
    from matplotlib.ticker import FuncFormatter

    from . import figures as F
    plt = F.plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    n = int(summ.n_test.iloc[0])
    chance = float(summ.chance.iloc[0])
    floor = 0.5 / n
    for ax, col_mean, col_lo, col_hi, title in (
            (axes[0], "reid_mean", "reid_min", "reid_max", "Re-identified test images"),
            (axes[1], "reid_beyond_label_mean", None, "reid_beyond_label_max", "Re-identified beyond the label")):
        for cut, (label, c) in CUT_STYLE.items():
            g = summ[summ.cut == cut].sort_values("k")
            if g.empty:
                continue
            y = g[col_mean].clip(lower=floor)
            lo = g[col_lo].clip(lower=floor) if col_lo else None
            hi = g[col_hi].clip(lower=floor)
            ax.plot(g.k, y, color=getattr(F, c), marker="o", markersize=4, label=label)
            if lo is not None:
                ax.fill_between(g.k, lo, hi, color=getattr(F, c), alpha=0.15, lw=0)
            else:
                ax.plot(g.k, hi, color=getattr(F, c), ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_ylim(floor * 0.8, 1.6)
        ax.set_xticks(sorted(summ.k.unique()))
        ax.set_xlabel("embedding size k")
        ax.set_title(title)
    kmax = summ.k.max()
    axes[0].axhline(chance, color=F.MUTED, ls="--", lw=1)
    axes[0].annotate(f"chance (1/{n})", xy=(kmax, chance), xytext=(0, 3), textcoords="offset points",
                     ha="right", fontsize=7.5, color=F.MUTED)
    lab = float(summ.reid_label.max())
    axes[0].axhline(lab, color=F.YELLOW, ls="--", lw=1)
    axes[0].annotate(f"label only ({lab:.1%})", xy=(kmax, lab), xytext=(0, -10), textcoords="offset points",
                     ha="right", fontsize=7.5, color=F.MUTED)
    axes[1].axhline(5 * chance, color=F.INK, ls="--", lw=1)
    axes[1].annotate("privacy limit: 5 x chance (policy assumption)", xy=(kmax, 5 * chance), xytext=(0, -10),
                     textcoords="offset points", ha="right", fontsize=7.5, color=F.MUTED)
    pct = FuncFormatter(lambda v, _: f"{v:.1%}" if v < 0.01 else f"{v:.0%}")
    axes[0].yaxis.set_major_formatter(pct)
    axes[0].set_ylabel(f"fraction of the {n} test images")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, bbox_to_anchor=(0.5, -0.06))
    axes[0].annotate("band: min-max over seeds; input and conv1 overlap", xy=(0.02, 0.03), xycoords="axes fraction",
                     fontsize=7, color=F.MUTED)
    axes[1].annotate("solid: mean, dotted: worst seed", xy=(0.02, 0.03), xycoords="axes fraction", fontsize=7,
                     color=F.MUTED)
    fig.suptitle("Leakage against ridge, MLP and kNN decoders (strongest per metric); not a guarantee",
                 color=F.INK, x=0.01, ha="left")
    F._save(fig, out)


def reconstructions_figure(rec: pd.DataFrame, out: str):
    """Original, label-only baseline and the strongest decoder's reconstruction from each cut."""
    from . import figures as F
    plt = F.plt
    cols = [("original", "", "original"), ("label-only", "", "label only")] + \
           [("reconstruction", c, CUT_STYLE[c][0].split(" (")[0]) for c in CUTS]
    images = list(dict.fromkeys(rec.image))
    pix = [f"p{j}" for j in range(64)]
    fig, axes = plt.subplots(len(images), len(cols), figsize=(1.25 * len(cols), 1.3 * len(images) + 0.4))
    for r, img in enumerate(images):
        for c, (kind, cut, title) in enumerate(cols):
            ax = axes[r][c]
            row = rec[(rec.image == img) & (rec.kind == kind) & (rec.cut.fillna("") == cut)].iloc[0]
            ax.imshow(row[pix].to_numpy(float).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title + (f"\n({row.attack})" if kind == "reconstruction" else ""), fontsize=8)
            if kind != "original":
                ok = int(row["rank"]) == 0
                ax.text(0.5, -0.08, "re-identified" if ok else f"rank {int(row['rank'])}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=6.5, color=F.ORANGE if ok else F.MUTED)
    fig.suptitle("What each cut gives away (k=4, seed 0; strongest re-identification decoder)",
                 color=F.INK, x=0.01, ha="left", fontsize=10)
    F._save(fig, out)


CATEGORY_COLORS = {"edge only": "GRID", "split to cloud": "AQUA", "cloud (raw input)": "ORANGE",
                   "quantum head (QPU)": "MAGENTA", "quantum head (simulator)": "BLUE"}
EDGE_LABELS = {"edge_gpu": "edge GPU", "edge_cpu": "edge CPU", "sensor": "sensor\n(CPU / 50)"}


def _short(label: str, edge: str) -> str:
    if edge == "sensor":
        label = label.replace("edge_cpu", "sensor")
    parts = label.split(">")
    if len(parts) == 3:
        return f"{parts[1]}\n> {parts[2].replace('cloud_', '').replace('quantum_sim', 'sim')}"
    return label.replace("_", " ")


def split_map_figure(router: pd.DataFrame, out: str, batch: int = 32):
    """Chosen placement over uplink x edge device for the camera profile: panels privacy off / on,
    real-time and batch-job deadline (qedge_static, today's QPU, measured accuracies). The digits
    panels are drawn only if some digits batch leaves the edge; otherwise a note says so."""
    import matplotlib.patches as mpatches

    from . import figures as F
    plt = F.plt
    d = router[(router.policy == "qedge_static") & (router.qpu == "today") & (router.whatif == "measured")
               & (router.batch == batch)]
    panels = [(p, dl) for p in ("camera", "digits") for dl in DEADLINES if not d[(d.profile == p) & (d.deadline == dl)].empty]
    digits_edge = (d.profile == "digits").any() and (d[d.profile == "digits"].category == "edge only").all()
    if digits_edge:
        panels = [x for x in panels if x[0] != "digits"]
    edges = [e for e in ("sensor", "edge_cpu", "edge_gpu") if e in set(d.edge)]
    ups = sorted(d.uplink_mbps.unique())
    fig, axes = plt.subplots(len(panels), 2, figsize=(10.5, 1.0 + 1.25 * len(edges) * len(panels)), squeeze=False)
    for r, (prof, dl) in enumerate(panels):
        for c, priv in enumerate((False, True)):
            ax = axes[r][c]
            g = d[(d.profile == prof) & (d.deadline == dl) & (d.privacy == priv)]
            for i, e in enumerate(edges):
                for j, u in enumerate(ups):
                    cell = g[(g.edge == e) & (g.uplink_mbps == u)]
                    if cell.empty:
                        continue
                    x = cell.iloc[0]
                    ax.add_patch(mpatches.Rectangle((j, i), 0.96, 0.92, color=getattr(F, CATEGORY_COLORS[x.category])))
                    miss = "" if x.meets_deadline else "\nmisses deadline"
                    ax.text(j + 0.48, i + 0.46, _short(x.placement, e) + miss, ha="center", va="center", fontsize=6.5,
                            color=F.INK)
            ax.set_xlim(0, len(ups))
            ax.set_ylim(0, len(edges))
            ax.set_xticks([j + 0.48 for j in range(len(ups))], [f"{u:g}" for u in ups])
            ax.set_yticks([i + 0.46 for i in range(len(edges))], [EDGE_LABELS[e] for e in edges])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            ax.set_title(f"{prof}, {dl} deadline ({DEADLINES[dl]:g} s), privacy {'on' if priv else 'off'}", fontsize=10)
            if r == len(panels) - 1:
                ax.set_xlabel("uplink (Mbps)")
    handles = [mpatches.Patch(color=getattr(F, col), label=cat) for cat, col in CATEGORY_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    note = "; digits: edge only in every condition" if digits_edge else ""
    fig.suptitle(f"Router choice (qedge_static), batch of {batch}, measured profile, today's QPU{note}",
                 color=F.INK, x=0.01, ha="left")
    F._save(fig, out)


DEVICE_PATHS = ["exact", "sampled 100 shots", "sampled 1000 shots", "ibm-auto (emulated)", "packed 8 per job",
                "packed 16 per job"]


def device_figure(dev: pd.DataFrame, out: str):
    """Quantum-head accuracy per inference path (with the cost model's prediction for the
    device paths), and QPU seconds per path."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    from . import figures as F
    plt = F.plt
    ks = sorted(dev.k.unique())
    paths = [p for p in DEVICE_PATHS if p in set(dev.path)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    wdt = 0.8 / len(ks)
    kcol = [F.BLUE, F.ORANGE, F.AQUA]
    x = np.arange(len(paths))
    for i, k in enumerate(ks):
        g = dev[dev.k == k].set_index("path").reindex(paths)
        xs = x + (i - (len(ks) - 1) / 2) * wdt
        ax.bar(xs, g.accuracy, wdt * 0.9, color=kcol[i % 3], label=f"k={k} measured")
        m = g.modelled_quality.where(g.qpu_s.notna())
        ax.scatter(xs, m, marker="_", s=180, color=F.INK, zorder=3, label="cost model (region figures)" if i == 0 else None)
        if "modelled_quality_esp" in g:
            ax.scatter(xs, g.modelled_quality_esp, marker="x", s=22, color=F.MAGENTA, zorder=3,
                       label="cost model with circuit ESP" if i == 0 else None)
    lo = max(0.0, min(dev.accuracy.min(), dev.modelled_quality.min()) - 0.1)
    ax.set_ylim(lo, 1.0)
    ax.set_xticks(x, [p.replace(" (emulated)", "\n(emulated)").replace(" per job", "\nper job")
                      .replace("sampled ", "sampled\n") for p in paths], fontsize=8)
    ax.set_ylabel(f"test accuracy ({int(dev.n_samples.iloc[0])} images)")
    ax.set_title("Quantum head (seed 0): exact, sampled, emulated device, packed")
    ax.legend(fontsize=7, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.17))
    ax = axes[1]
    q = dev[dev.qpu_s.notna()]
    qp = [p for p in paths if p in set(q.path)]
    x = np.arange(len(qp))
    for i, k in enumerate(ks):
        g = q[q.k == k].set_index("path").reindex(qp)
        xs = x + (i - (len(ks) - 1) / 2) * wdt
        ax.bar(xs, g.qpu_s, wdt * 0.9, color=kcol[i % 3], label=f"k={k}")
        for xx, v, pr in zip(xs, g.qpu_s, g.price_usd):
            ax.annotate(f"${pr:,.0f}", xy=(xx, v), xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=6.5, color=F.MUTED)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x, [p.replace(" (emulated)", "\n(one job per sample)").replace(" per job", "\nper job")
                      for p in qp], fontsize=8)
    ax.set_ylabel("QPU seconds (catalogue timings)")
    ax.set_title("QPU time and Braket-style price for the test set")
    notes = [f"k={int(r.k)}: {r.note}" for r in dev.itertuples() if isinstance(getattr(r, "note", None), str) and r.note]
    if notes:
        ax.annotate("\n".join(notes), xy=(0.5, -0.3), xycoords="axes fraction", ha="center", va="top", fontsize=7,
                    color=F.MUTED)
    for a in axes:
        a.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper right")
    F._save(fig, out)


def make_figures(res_dir: str = "results", fig_dir: str = "figures"):
    p = lambda name: os.path.join(res_dir, f"{name}.csv")          # noqa: E731
    if not os.path.exists(p("e8_heads_summary")):
        return
    os.makedirs(fig_dir, exist_ok=True)
    heads_figure(pd.read_csv(p("e8_heads_summary")), pd.read_csv(p("e8_heads_paired")),
                 os.path.join(fig_dir, "e8_heads.png"))
    if os.path.exists(p("e8_leakage_summary")):
        leakage_figure(pd.read_csv(p("e8_leakage_summary")), os.path.join(fig_dir, "e8_leakage.png"))
    if os.path.exists(p("e8_reconstructions")):
        reconstructions_figure(pd.read_csv(p("e8_reconstructions")), os.path.join(fig_dir, "e8_reconstructions.png"))
    if os.path.exists(p("e8_router")):
        split_map_figure(pd.read_csv(p("e8_router")), os.path.join(fig_dir, "e8_split_map.png"))
    if os.path.exists(p("e8_device")):
        device_figure(pd.read_csv(p("e8_device")), os.path.join(fig_dir, "e8_device.png"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    for name, frame in run(a.quick).items():
        frame.to_csv(os.path.join("results", f"{name}.csv"), index=False)
    make_figures()
