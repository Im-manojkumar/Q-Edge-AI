"""The split profile: what the orchestrator's router needs to know about a model.

``measured_profile`` turns measurements of trained models (one per embedding
size k) into the SPLIT PROFILE SCHEMA that ``qedge/orchestrator/split.py``
reads from ``results/split_profile.json``::

    {"name", "label", "source", "classes", "input_bytes",
     "cuts": [{"name", "values", "bytes", "front_flop", "back_flop",
               "reid", "r2_beyond_label", "private_ok"}],          # input, conv1, conv2
     "embedding": {"<k>": {"bytes", "front_flop", "back_flop", "reid", "r2_beyond_label", "private_ok",
                           "accuracy": {"linear", "mlp", "quantum", "quantum_l0"},
                           "quantum_circuit": {"n_qubits", "layers", "depth", "n_1q", "n_2q"},
                           "params": {"linear", "mlp", "quantum"}}}}

It is a pure function: no training, no data. Conventions it applies (stated
here because they are choices):

* sizes and FLOPs are per sample; bytes are float32 activations;
* the shared cuts (input, conv1, conv2) come from one backbone per k. Their
  sizes and front FLOPs do not depend on k; their back FLOPs are taken from
  ``ref_k`` (the dense 64->k layer and a linear head change the back FLOPs by
  under 1000 across k = 2..8, out of about 46 000); their leakage is the WORST over all k (highest reid and
  r2_beyond_label) and they are private only if private for every k;
* an embedding's back FLOPs are those of the best classical head for that k
  (the head whose accuracy the router uses as classical quality);
* accuracies are test-split accuracies measured by the caller;
* ``private_ok`` is recomputed here with ``leakage.private_ok`` and the
  thresholds passed in, so the policy lives in one place.
"""
from __future__ import annotations

from . import leakage as L

SHARED_CUTS = ("input", "conv1", "conv2")
ACC_KEYS = ("linear", "mlp", "quantum", "quantum_l0")
PARAM_KEYS = ("linear", "mlp", "quantum")
CIRCUIT_KEYS = ("n_qubits", "layers", "depth", "n_1q", "n_2q")
CUT_KEYS = ("name", "values", "bytes", "front_flop", "back_flop", "reid", "r2_beyond_label", "private_ok")
EMB_KEYS = ("bytes", "front_flop", "back_flop", "reid", "r2_beyond_label", "private_ok", "accuracy",
            "quantum_circuit", "params")


def _by_name(rows: list[dict], key: str) -> dict:
    return {r[key]: r for r in rows}


def measured_profile(per_k: dict, name: str = "digits", label: str | None = None, source: str = "measured (E8)",
                     classes: int = 10, ref_k: int | None = None, max_reid: float | None = None,
                     max_r2_beyond_label: float = L.DEFAULT_MAX_R2_BEYOND_LABEL,
                     relative_to_label: bool = True) -> dict:
    """Build the split profile from measurements.

    ``per_k[k]`` holds, for the model with embedding size k:
    ``cut_stats`` (``SplitCNN.cut_stats()``), ``leakage`` (``leakage_report`` rows for
    input, conv1, conv2, embedding), ``accuracy`` (test accuracy per head in
    ``ACC_KEYS``), ``params`` (per head in ``PARAM_KEYS``), ``quantum_circuit``
    (``QuantumHead.circuit_profile()``) and optionally ``head_flop`` (classical
    FLOPs per sample per head; default: the head FLOPs in ``cut_stats``)."""
    if not per_k:
        raise ValueError("no measured model")
    ks = sorted(int(k) for k in per_k)
    get = lambda k: per_k[k] if k in per_k else per_k[str(k)]   # noqa: E731
    ref = ref_k if ref_k is not None else (4 if 4 in ks else ks[0])

    def ok(row: dict) -> bool:
        return L.private_ok(row, max_reid, max_r2_beyond_label, relative_to_label)

    ref_stats = _by_name(get(ref)["cut_stats"], "name")
    cuts = []
    for cut in SHARED_CUTS:
        leaks = [_by_name(get(k)["leakage"], "cut")[cut] for k in ks]
        s = ref_stats[cut]
        cuts.append({"name": cut, "values": int(s["values"]), "bytes": int(s["bytes"]),
                     "front_flop": float(s["front_flop"]), "back_flop": float(s["back_flop"]),
                     "reid": float(max(r["reid"] for r in leaks)),
                     "r2_beyond_label": float(max(r["r2_beyond_label"] for r in leaks)),
                     "reid_label": float(max(r["reid_label"] for r in leaks)),
                     "private_ok": all(ok(r) for r in leaks)})
    emb = {}
    for k in ks:
        m = get(k)
        s = _by_name(m["cut_stats"], "name")["embedding"]
        leak = _by_name(m["leakage"], "cut")["embedding"]
        acc = {h: float(m["accuracy"][h]) for h in ACC_KEYS}
        best = max(("linear", "mlp"), key=lambda h: acc[h])
        head_flop = (m.get("head_flop") or {}).get(best, s["head_flop"])
        emb[str(k)] = {"bytes": int(s["bytes"]), "front_flop": float(s["front_flop"]),
                       "back_flop": float(head_flop), "reid": float(leak["reid"]),
                       "r2_beyond_label": float(leak["r2_beyond_label"]), "reid_label": float(leak["reid_label"]),
                       "private_ok": ok(leak), "accuracy": acc,
                       "quantum_circuit": {c: int(m["quantum_circuit"][c]) for c in CIRCUIT_KEYS},
                       "params": {h: int(m["params"][h]) for h in PARAM_KEYS}}
    return {"name": name, "label": label or f"{name}: 8x8 CNN backbone, k-dim embedding, classical or quantum head",
            "source": source, "classes": int(classes), "input_bytes": int(ref_stats["input"]["bytes"]),
            "cuts": cuts, "embedding": emb}


def validate_profile(p: dict) -> None:
    """Raise ``ValueError`` if ``p`` does not follow the split profile schema."""
    top = ("name", "label", "source", "classes", "input_bytes", "cuts", "embedding")
    missing = [k for k in top if k not in p]
    if missing:
        raise ValueError(f"profile lacks {missing}")
    for c in p["cuts"]:
        if miss := [k for k in CUT_KEYS if k not in c]:
            raise ValueError(f"cut {c.get('name')} lacks {miss}")
    for k, e in p["embedding"].items():
        if not str(k).isdigit():
            raise ValueError(f"embedding key {k!r} is not an integer k")
        if miss := [x for x in EMB_KEYS if x not in e]:
            raise ValueError(f"embedding {k} lacks {miss}")
        for sub, keys in (("accuracy", ACC_KEYS), ("params", PARAM_KEYS), ("quantum_circuit", CIRCUIT_KEYS)):
            if miss := [x for x in keys if x not in e[sub]]:
                raise ValueError(f"embedding {k} {sub} lacks {miss}")


def summarise(transfer, Xte, yte, leakage_rows: list[dict]) -> dict:
    """One ``per_k`` entry from a ``nets.TransferResult``: evaluates every head on the
    test split once (no training) and collects sizes, parameters and the circuit."""
    heads = transfer.heads
    q = heads["quantum"]
    return {"cut_stats": transfer.model.cut_stats(), "leakage": leakage_rows,
            "accuracy": {h: transfer.accuracy(h, Xte, yte) for h in ACC_KEYS if h in heads},
            "params": {h: int(heads[h].n_params) for h in PARAM_KEYS if h in heads},
            "head_flop": {h: float(heads[h].flop) for h in ("linear", "mlp") if h in heads},
            "quantum_circuit": q.circuit_profile()}
