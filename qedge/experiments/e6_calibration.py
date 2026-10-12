"""E6: does calibration-aware routing pay off?

For TFIM and QAOA circuits of several sizes, on eight IBM devices (live
calibration when an account is configured, otherwise real calibration
snapshots), compare:

* v1 path: one fixed device (Torino) with the preset pass manager's qubits;
* calibration-aware path: the device and qubits with the highest estimated
  success probability (ESP);
* oracle: the device that actually gave the best result.

"Measured" is the Hellinger fidelity between the device's noisy output and the
ideal distribution, run on an emulator built from that same device's
calibration. It also checks how well three predictors track the measurements:
the static catalogue figures v1 used, the best-region figures the cost model
now gets, and the circuit-level ESP.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..executor import representative_circuit
from ..quantum import calibration as C
from ..quantum.backends.qiskit_backend import qiskit_counts, to_qiskit
from ..resources import load_catalogue
from ..workloads import BUILDERS

V1_DEVICE = "fake_torino"


def _ideal_probs(qc):
    from qiskit.quantum_info import Statevector
    sv = Statevector(qc.remove_final_measurements(inplace=False))
    return sv.probabilities_dict()


def _hellinger(p: dict, counts: dict) -> float:
    shots = sum(counts.values())
    keys = set(p) | set(counts)
    bc = sum(np.sqrt(p.get(k, 0.0) * counts.get(k, 0) / shots) for k in keys)
    return float(bc ** 2)


def _cases(quick: bool):
    if quick:
        return [("quantum_simulation", {"n": n, "steps": 2}) for n in (4, 8)] + \
               [("optimization", {"n": n, "p": 1, "density": 0.4}) for n in (6, 8)]
    return [("quantum_simulation", {"n": n, "steps": s}) for n in (4, 6, 8, 10, 12) for s in (2, 4)] + \
           [("optimization", {"n": n, "p": 1, "density": 0.4}) for n in (6, 8, 10, 12)] + \
           [("kernel_classification", {"features": d, "n_train": 20, "n_test": 10}) for d in (4, 8)]


def run(quick: bool = False, shots: int = 4000) -> dict[str, pd.DataFrame]:
    from qiskit_aer import AerSimulator

    devices = C.load_devices()
    names = [d.name for d in devices]
    v1 = V1_DEVICE if V1_DEVICE in names else names[0]
    q = load_catalogue()["qpu"]
    sims = {}
    rows = []
    for ci, (kind, params) in enumerate(_cases(quick)):
        w = BUILDERS[kind](ci, np.random.default_rng(ci), **params)
        circ = representative_circuit(w)
        qc = to_qiskit(circ)
        ideal = _ideal_probs(qc)
        # Qiskit orders bitstrings with qubit 0 rightmost; convert ideal to our convention.
        ideal = {k[::-1]: v for k, v in ideal.items()}
        static_pred = w.quantum.fidelity(q.p1, q.p2, q.readout_err)
        case = f"{kind.split('_')[0]} " + " ".join(f"{k}={v}" for k, v in params.items() if k in ("n", "steps", "features"))
        for cal in devices:
            if circ.n_qubits > cal.num_qubits:
                continue
            if cal.name not in sims:
                sims[cal.name] = AerSimulator.from_backend(cal.backend, seed_simulator=17)
            sim = sims[cal.name]
            t0 = time.perf_counter()
            l1_qc, l1_esp = C.default_layout_circuit(cal, qc)
            choice = C.best_layout(cal, qc)
            t_route = time.perf_counter() - t0
            region = C.region_figures(cal, circ.n_qubits)
            region_pred = w.quantum.fidelity(region["p1"], region["p2"], region["readout_err"])
            res = sim.run([l1_qc, choice.circuit], shots=shots).result()
            f_l1 = _hellinger(ideal, qiskit_counts(res.get_counts(0), circ.n_qubits))
            f_best = _hellinger(ideal, qiskit_counts(res.get_counts(1), circ.n_qubits))
            rows.append({"case": case, "kind": kind, "n_qubits": circ.n_qubits, "device": cal.name,
                         "source": cal.source, "median_2q_error": cal.median_2q,
                         "static_pred": static_pred, "region_pred": region_pred,
                         "esp_default": l1_esp, "esp_best": choice.esp, "esp_qiskit_l3": choice.default_esp,
                         "fidelity_default": f_l1, "fidelity_best": f_best,
                         "two_q_default": sum(1 for i in l1_qc.data if len(i.qubits) == 2),
                         "two_q_best": choice.two_q_count, "route_time_s": t_route,
                         "layouts_scored": choice.n_layouts_scored})
        print(f"[e6] {case}: {len(devices)} devices", flush=True)
    df = pd.DataFrame(rows)
    return {"e6_calibration": df, "e6_summary": summarise(df, v1), "e6_prediction": prediction_stats(df)}


def summarise(df: pd.DataFrame, v1: str = V1_DEVICE) -> pd.DataFrame:
    out = []
    for case, g in df.groupby("case", sort=False):
        base = g[g.device == v1]
        if base.empty:
            continue
        pick = g.loc[g.esp_best.idxmax()]
        oracle = g.loc[g.fidelity_best.idxmax()]
        out.append({"case": case, "n_qubits": int(g.n_qubits.iloc[0]),
                    "v1_fidelity": float(base.fidelity_default.iloc[0]),
                    "same_device_best_layout": float(base.fidelity_best.iloc[0]),
                    "calibrated_device": pick.device, "calibrated_fidelity": float(pick.fidelity_best),
                    "oracle_device": oracle.device, "oracle_fidelity": float(oracle.fidelity_best),
                    "picked_oracle": pick.device == oracle.device,
                    "v1_error": 1 - float(base.fidelity_default.iloc[0]),
                    "calibrated_error": 1 - float(pick.fidelity_best),
                    "static_pred": float(g.static_pred.iloc[0]),
                    "calibrated_esp": float(pick.esp_best)})
    return pd.DataFrame(out)


def prediction_stats(df: pd.DataFrame) -> pd.DataFrame:
    """How well each predictor tracks measured fidelity across all (circuit, device) pairs."""
    from scipy.stats import spearmanr
    rows = []
    meas = df.fidelity_best
    for name, col in [("static catalogue (v1)", "static_pred"), ("best-region figures (cost model)", "region_pred"),
                      ("circuit ESP (router)", "esp_best")]:
        rho = spearmanr(df[col], meas).statistic if df[col].nunique() > 1 else float("nan")
        # Within one circuit, can the predictor rank devices correctly?
        within = [spearmanr(g[col], g.fidelity_best).statistic for _, g in df.groupby("case")
                  if g[col].nunique() > 1]
        picks = [g.loc[g[col].idxmax(), "device"] == g.loc[g.fidelity_best.idxmax(), "device"]
                 for _, g in df.groupby("case") if g[col].nunique() > 1]
        rows.append({"predictor": name, "spearman_all": rho,
                     "spearman_within_circuit": float(np.nanmean(within)) if within else float("nan"),
                     "picks_best_device": float(np.mean(picks)) if picks else float("nan")})
    return pd.DataFrame(rows)
