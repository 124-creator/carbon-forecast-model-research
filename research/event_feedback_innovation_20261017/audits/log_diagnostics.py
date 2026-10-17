"""Read public development receipts only; no prices, saved weights or fitting."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "carbon_workspace/research/cross_market_innovation_20261017"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all() and len(a)
    return dict(n=len(a), minimum=float(a.min()), median=float(np.median(a)),
                maximum=float(a.max()), mean=float(a.mean()))


def key(row):
    t, c = row["task"], row["config"]
    return (t["scope"], tuple(t["market_names"]), t["cutoff"], row["seed"],
            c.get("hidden"), c.get("shrink"), t.get("train_cap"))


def main():
    sources, rows = {}, []
    for stage in ("d1", "d2"):
        for path in sorted((ROOT / (stage + "_receipts")).glob("*.json")):
            row = json.loads(path.read_text())
            sources[str(path.relative_to(ROOT))] = sha(path)
            if "config" not in row or "task" not in row:
                continue
            row["_receipt"] = str(path.relative_to(ROOT))
            rows.append(row)
    public_path = ROOT / "DEVELOPMENT_AGGREGATE.json"
    aggregate = json.loads(public_path.read_text())
    sources[str(public_path.relative_to(ROOT))] = sha(public_path)
    d1path = ROOT / "D1_SUMMARY.json"
    d1_summary = json.loads(d1path.read_text())
    sources[str(d1path.relative_to(ROOT))] = sha(d1path)
    selected = [r for r in rows if r["config"]["family"] == "kcm"
                and r["config"]["hidden"] == 32 and r["config"]["shrink"] == 1
                and r["config"]["strength"] == 1 and r["task"].get("train_cap") is None]
    assert len(selected) == 9
    aug = {key(r): r for r in rows if r["config"]["family"] == "augment"}
    paired, all_epochs = [], []
    for row in sorted(selected, key=key):
        match = aug[key(row)]
        trace = row["trace"]
        per_view = {}
        for view in ("short", "sparse"):
            use = [e for e in trace if e["view"] == view]
            per_view[view] = {
                "raw_moment": summary([e["mechanism_penalty"] for e in use]),
                "moment_over_supervised": summary([e["mechanism_penalty"]/e["supervised"] for e in use]),
                "moment_over_prediction_shrink": summary([e["mechanism_penalty"]/e["prediction_penalty"] for e in use]),
                "positive_fraction": summary([e["positive_moment_fraction"] for e in use]),
                "total_preclip_gradient": summary([e["preclip_gradient"] for e in use]),
            }
        kernel_mean = [k["kernel_weight_sum"]/k["ordered_pairs"] for k in row["kernel_audit"]]
        paired.append(dict(
            cutoff=row["task"]["cutoff"], seed=row["seed"], receipt=row["_receipt"],
            matched_augmentation_receipt=match["_receipt"],
            initial_state_identical=row["initial_state_sha256"] == match["initial_state_sha256"],
            selected_state_identical=row["state_sha256"] == match["state_sha256"],
            initial_validation=row["initial_validation"],
            initial_validation_difference=row["initial_validation"] - match["initial_validation"],
            selected_epoch=row["selected_epoch"], augmentation_selected_epoch=match["selected_epoch"],
            total_epochs=row["epochs"], augmentation_total_epochs=match["epochs"],
            validation_difference=row["validation_score"] - match["validation_score"],
            kernel_mean_on_eligible_pairs=summary(kernel_mean),
            by_view=per_view,
        ))
        all_epochs.extend(dict(e, cutoff=row["task"]["cutoff"], seed=row["seed"]) for e in trace)
    a100 = [r for r in rows if r["config"]["family"] == "kcm"
            and r["config"]["hidden"] == 32 and r["config"]["shrink"] == 1
            and r["config"]["strength"] == 100 and r["task"].get("train_cap") is None]
    assert len(a100) == 9
    by_strength = {}
    for strength, subset in ((1, selected), (100, a100)):
        tr = [e for r in subset for e in r["trace"]]
        by_strength[str(strength)] = dict(
            fits=len(subset), epochs=len(tr),
            selected_epoch=summary([r["selected_epoch"] for r in subset]),
            selected_epoch_zero=sum(r["selected_epoch"] == 0 for r in subset),
            weighted_mechanism_over_supervised=summary([strength*e["mechanism_penalty"]/e["supervised"] for e in tr]),
            weighted_mechanism_over_prediction_shrink=summary([strength*e["mechanism_penalty"]/e["prediction_penalty"] for e in tr]),
            positive_moment_fraction=summary([e["positive_moment_fraction"] for e in tr]),
            zero_mechanism_epochs=sum(e["mechanism_penalty"] == 0 for e in tr),
            clipped_epochs=sum(e["preclip_gradient"] > 1 for e in tr),
        )
    d1_shrink = []
    for hidden in (16, 32):
        for shrink in (.1, 1):
            lookup = {r["config_id"]:r for r in d1_summary if r["scope"] == "pooled"}
            aid = f"augment_h{hidden}_s{shrink:g}_l0"
            kid = f"kcm_h{hidden}_s{shrink:g}_l1"
            fullid = f"gru_h{hidden}_s{shrink:g}_l0"
            d1_shrink.append(dict(hidden=hidden, shrink=shrink, seed=17,
                                  augmentation_score=lookup[aid]["score"],
                                  kcm1_score=lookup[kid]["score"],
                                  full_gru_score=lookup[fullid]["score"],
                                  kcm1_minus_augmentation=lookup[kid]["score"]-lookup[aid]["score"]))
    dev_scores = []
    for r in aggregate["development_summary"]:
        if r.get("scope") == "pooled" and r.get("train_cap") is None and r["config_id"] in (
                "kcm_h32_s1_l1", "kcm_h32_s1_l100", "augment_h32_s1_l0",
                "kcm_unblocked_h32_s1_l1", "pointwise_h32_s1_l1", "gru_h32_s1_l0"):
            dev_scores.append(dict(config_id=r["config_id"], score=r["score"]))
    output = dict(
        audit_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope="Post-hoc development diagnosis only; prior confirmation conclusion unchanged",
        source_sha256=sources, script_sha256=sha(Path(__file__)),
        paired_fits=paired, selected_lambda1_summary=by_strength["1"],
        strength_comparison=by_strength, d1_shrink_comparison=d1_shrink,
        selected_development_scores=dev_scores,
        selected_initial_states_all_identical=all(p["initial_state_identical"] for p in paired),
        selected_epoch_equal_pairs=sum(p["selected_epoch"] == p["augmentation_selected_epoch"] for p in paired),
        selected_state_identical_pairs=sum(p["selected_state_identical"] for p in paired),
        market_price_reads=0, weight_array_reads=0, confirmation_reads=0,
        market_fits=0, optimizer_steps=0,
        limitations=[
            "Loss magnitudes do not identify individual parameter-gradient norms.",
            "Preclip gradient is the total objective gradient, not a mechanism-only gradient.",
            "All analyses are observational post-hoc training-log descriptions, not randomized causal ablations.",
            "The full 9-pair criterion and existing D1 shrink variants are reported; no market refit or new configuration is attempted.",
        ],
    )
    path = HERE / "BCM_PUBLIC_LOG_DIAGNOSTIC.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k:output[k] for k in ("selected_lambda1_summary",
                                          "d1_shrink_comparison","selected_development_scores",
                                          "selected_initial_states_all_identical",
                                          "selected_epoch_equal_pairs","selected_state_identical_pairs")}, indent=2))


if __name__ == "__main__":
    main()
