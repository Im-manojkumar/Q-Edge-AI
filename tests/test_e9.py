"""E9 helpers: aggregation over seeds, the profile built from them, and the router sweep.

No training here: the inputs are hand-made rows (fixture values, not measurements) and
the built-in placeholder profiles.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from qedge.experiments import e9_split as E
from qedge.ml import nets
from qedge.ml.qhead import QuantumHead, quantum_params
from qedge.orchestrator import split
from qedge.quantum import calibration as C


@pytest.fixture
def default_profiles(tmp_path):
    """Built-in profiles only (whatever results/ holds), restored afterwards."""
    saved = copy.deepcopy(split.PROFILES)
    split.load_profiles(str(tmp_path / "none.json"))
    yield split.PROFILES
    split.PROFILES.clear()
    split.PROFILES.update(saved)


def head_rows():
    acc = {(0, "linear"): 0.95, (0, "mlp"): 0.94, (0, "quantum"): 0.93,
           (1, "linear"): 0.96, (1, "mlp"): 0.96, (1, "quantum"): 0.97}
    return pd.DataFrame([{"mode": "transfer", "k": 4, "head": h, "seed": s, "test_acc": a, "val_acc": a,
                          "n_params": 50, "hidden": None, "train_s": 1.0} for (s, h), a in acc.items()])


def test_head_summary_and_paired_differences():
    df = head_rows()
    summ = E.summarise_heads(df).set_index("head")
    assert summ.loc["linear", "test_acc_mean"] == pytest.approx(0.955)
    assert summ.loc["quantum", "test_acc_std"] == pytest.approx(np.std([0.93, 0.97], ddof=1))
    assert summ.loc["quantum", "seeds"] == 2
    paired = E.paired_differences(df).set_index("difference")
    q_lin = paired.loc["quantum - linear"]
    assert q_lin.seed0 == pytest.approx(-0.02) and q_lin.seed1 == pytest.approx(0.01)
    assert q_lin["mean"] == pytest.approx(-0.005) and q_lin["min"] == pytest.approx(-0.02)
    assert q_lin.seeds_a_better == 1
    assert "quantum_l0 - linear" not in paired.index          # no quantum_l0 rows: pair skipped


def fake_entry(k: int, acc: dict, leak: dict) -> dict:
    """A ``profile.summarise``-shaped entry with hand-set numbers (no training)."""
    m = nets.SplitCNN(k, nets.init_backbone(k, np.random.default_rng(0)), nets.LinearHead(k))
    h = nets.mlp_hidden_for(k, quantum_params(k, 2))
    rows = []
    for cut in E.CUTS:
        reid, beyond, r2b = leak[cut]
        rows.append({"cut": cut, "k": k, "n_test": 360, "chance": 1 / 360, "reid": reid, "reid_top5": reid,
                     "reid_label": 10 / 360, "reid_beyond_label": beyond, "r2": 0.5, "r2_label": 0.4,
                     "r2_beyond_label": r2b, "attacks": {"ridge": {}, "knn": {}}})
    return {"cut_stats": m.cut_stats(), "leakage": rows, "accuracy": acc, "val_accuracy": acc,
            "params": {"linear": 10 * k + 10, "mlp": nets.mlp_params(k, h), "quantum": quantum_params(k, 2)},
            "head_flop": {"linear": 20 * k, "mlp": 2 * (k * h + h * 10)},
            "quantum_circuit": QuantumHead(k, 2).circuit_profile()}


def test_profile_takes_mean_accuracy_and_worst_case_leakage():
    acc0 = {"linear": 0.90, "mlp": 0.92, "quantum": 0.88, "quantum_l0": 0.80}
    acc1 = {"linear": 0.94, "mlp": 0.90, "quantum": 0.90, "quantum_l0": 0.84}
    private = {"input": (1.0, 0.97, 0.6), "conv1": (1.0, 0.97, 0.5), "conv2": (0.8, 0.78, 0.3),
               "embedding": (0.02, 0.001, 0.0)}
    leaky = {**private, "embedding": (0.10, 0.05, 0.08)}
    per_seed = {0: {4: fake_entry(4, acc0, private)}, 1: {4: fake_entry(4, acc1, leaky)}}
    prof = E.build_profile(per_seed)
    assert split.check_profile(prof) == []
    e = prof["embedding"]["4"]
    assert e["accuracy"]["linear"] == pytest.approx(0.92) and e["accuracy"]["quantum"] == pytest.approx(0.89)
    assert e["accuracy_std"]["linear"] == pytest.approx(np.std([0.90, 0.94], ddof=1))
    # Classical head chosen on mean validation accuracy: linear 0.92 > mlp 0.91.
    assert e["classical_head"] == "linear"
    # Seed 0 alone would pass the privacy rule at the embedding; the worst seed does not.
    assert e["reid"] == pytest.approx(0.10) and e["reid_beyond_label"] == pytest.approx(0.05)
    assert not e["private_ok"]
    assert prof["seeds"] == [0, 1] and "worst case over seeds" in prof["source"]
    only0 = E.build_profile({0: per_seed[0]})
    assert only0["embedding"]["4"]["private_ok"]
    # Only k=4 measured: the other k the router offers are listed as not measured.
    assert prof["k_not_measured"] == [k for k in split.K_RANGE if k != 4]


def test_full_run_measures_every_k_the_router_offers():
    # Otherwise the loaded profile would mix measured entries with built-in placeholders.
    assert set(E.KS) == set(split.K_RANGE)
    assert set(E.DEVICE_KS) <= set(E.KS) and set(E.E2E_KS) <= set(E.KS) and E.RECON_K in E.KS


def test_leakage_summary_is_worst_over_seeds():
    rows = []
    for seed, beyond in ((0, 0.01), (1, 0.03)):
        rows.append({"seed": seed, "k": 4, "cut": "embedding", "n_test": 360, "chance": 1 / 360, "reid": beyond + 0.01,
                     "reid_label": 10 / 360, "reid_beyond_label": beyond, "reid_beyond_label_images": round(beyond * 360),
                     "r2": 0.4, "r2_beyond_label": 0.05, "private_ok": seed == 0, "strongest_reid": "knn"})
    s = E.summarise_leakage(pd.DataFrame(rows)).iloc[0]
    assert s.reid_beyond_label_max == pytest.approx(0.03) and s.reid_beyond_label_images_max == 11
    assert not s.private_ok_all_seeds and s.private_ok_any_seed


def test_quick_run_never_writes_the_full_profile():
    assert E.profile_path("results", quick=False).endswith("split_profile.json")
    assert E.profile_path("results", quick=True).endswith("split_profile.quick.json")
    assert E.result_name("e9_router", quick=False) == "e9_router"
    assert E.result_name("e9_router", quick=True) == "e9_router.quick"


def test_quick_figures_read_and_write_only_quick_files(tmp_path):
    res, figs = tmp_path / "results", tmp_path / "figures"
    res.mkdir()
    summ = E.summarise_heads(head_rows())
    paired = E.paired_differences(head_rows())
    summ.to_csv(res / "e9_heads_summary.quick.csv", index=False)
    paired.to_csv(res / "e9_heads_paired.quick.csv", index=False)
    E.make_figures(str(res), str(figs))                     # full run: no full CSVs, nothing drawn
    assert not figs.exists() or not any(figs.iterdir())
    E.make_figures(str(res), str(figs), quick=True)
    assert sorted(p.name for p in figs.iterdir()) == ["e9_heads.quick.png"]


def test_router_catalogue_conditions():
    res = E.router_catalogue("sensor", 5, "future")
    assert [n for n, r in res.items() if r.is_edge and r.available] == ["edge_cpu"]
    assert res["edge_cpu"].gflops == pytest.approx(25 / E.SENSOR_SLOWDOWN)
    assert all(r.uplink_mbps == 5 for r in res.values() if not r.is_edge)
    assert res["qpu"].queue_mean_s == pytest.approx(12.0)
    assert E.router_catalogue("edge_gpu", 20, "today")["qpu"].queue_mean_s == pytest.approx(120.0)


def test_router_sweep_on_a_tiny_grid(default_profiles):
    before = copy.deepcopy(split.PROFILES)
    grid = {"profile": ("camera",), "edge": ("sensor", "edge_gpu"), "uplink": (1000,), "privacy": (False, True),
            "batch": (32,), "deadline": ("real-time",), "qpu": ("today",), "whatif": ("measured", "qhead_equal")}
    df = E.router_sweep(grid, devices=[C.snapshot("FakeTorino")])
    assert len(df) == 2 * 2 * 2 * 2                      # edges x privacy x what-ifs x policies
    assert split.PROFILES == before                      # the what-if is undone
    # Built-in placeholders: nothing transmitted is private, so privacy keeps everything on the edge.
    assert set(df[df.privacy].category) == {"edge only"}
    # A sensor-class edge with a fast uplink sends the camera batch to the cloud.
    x = df[(df.edge == "sensor") & ~df.privacy & (df.policy == "qedge_static") & (df.whatif == "measured")].iloc[0]
    assert x.category == "cloud (raw input)" and x.meets_deadline
    # Placeholder quantum head is less accurate: the QPU gate closes on quality; the what-if
    # makes it as accurate as the best classical head and that check passes on the simulator.
    m = df[(df.whatif == "measured") & ~df.privacy]
    assert m.gate_qpu.eq("closed").all() and m.gate_qpu_failed.str.contains("expected quality gain").all()
    eq = df[(df.whatif == "qhead_equal") & ~df.privacy]
    assert not eq.gate_sim_failed.str.contains("expected quality gain").any()
    s = E.summarise_router(df)
    assert s.conditions.sum() == len(df)


def test_router_picks_a_split_when_a_cut_pays(default_profiles):
    """The zero splits in the E9 sweep comes from the profiles (camera cuts are either larger
    than the input or leave almost no work to offload). Give the camera stem an activation
    30x smaller than the input (a what-if, not a measured model) and a slow edge on a slow
    uplink splits there."""
    grid = {"profile": ("camera",), "edge": ("sensor",), "uplink": (1,), "privacy": (False,), "batch": (32,),
            "deadline": ("batch job",), "qpu": ("today",), "whatif": ("measured",)}
    devs = [C.snapshot("FakeTorino")]
    base = E.router_sweep(grid, devices=devs)
    assert set(base.category) == {"cloud (raw input)"}
    assert (base.best_split_score > base.chosen_score).all() and (base.best_split_any_ruled_out == "").all()
    stem = next(c for c in split.PROFILES["camera"]["cuts"] if c["name"] == "stem")
    assert stem["bytes"] > split.PROFILES["camera"]["input_bytes"]
    stem["bytes"] = split.PROFILES["camera"]["input_bytes"] // 30
    df = E.router_sweep(grid, devices=devs)
    assert set(df.category) == {"split to cloud"} and set(df.cut) == {"stem"}
    assert (df.chosen_score == df.best_split_score).all()
