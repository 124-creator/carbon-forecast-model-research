"""Run the unchanged example builder against corrected calendar inputs only.

This is a source correction receipt, not a fit or performance experiment.
The v1 files are checked before and after and are never overwritten.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    correction = Path(__file__).resolve().parent
    original = correction.parent
    rerun = correction / "rerun"
    freeze = json.loads((original / "PROTOCOL_FREEZE.json").read_text())
    for rel, expected in freeze["frozen_sha256"].items():
        assert sha(original / rel) == expected, rel
    assert (rerun / "local_only/calendar_vintages_development.csv").exists()
    if (rerun / "PROTOCOL_FREEZE.json").exists():
        raise RuntimeError("New freeze already exists; refuse to regenerate frozen input")
    shutil.copyfile(original / "local_only/auction_events_development.csv",
                    rerun / "local_only/auction_events_development.csv")
    result = subprocess.run([sys.executable, str(original / "prepare_examples.py"),
                             "--study-dir", str(rerun), "--horizons", "1", "5", "20"],
                            text=True, capture_output=True, check=True)
    (correction / "local_only/prepare_stdout.txt").write_text(result.stdout, encoding="utf-8")
    old_meta = pd.read_csv(original / "local_only/example_metadata.csv").fillna("")
    new_meta = pd.read_csv(rerun / "local_only/example_metadata.csv").fillna("")
    assert list(old_meta.columns) == list(new_meta.columns)
    assert (old_meta.example_id == new_meta.example_id).all()
    meta_diff = {c: int((old_meta[c] != new_meta[c]).sum()) for c in old_meta}
    assert all(n == 0 for c, n in meta_diff.items() if c != "supply_plan_matches_final")
    old = np.load(original / "local_only/examples_development.npz")
    new = np.load(rerun / "local_only/examples_development.npz")
    assert old.files == new.files
    changed_rows = np.zeros(len(old_meta), dtype=bool)
    array_diff = {}
    for name in old.files:
        a, b = old[name], new[name]
        assert a.shape == b.shape and a.dtype == b.dtype
        diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        rows = diff.reshape(len(a), -1).any(axis=1)
        changed_rows |= rows
        entry = {"shape": list(a.shape), "dtype": str(a.dtype), "changed_values": int(diff.sum()),
                 "changed_example_rows": int(rows.sum()),
                 "max_absolute_change": float(np.abs(a.astype(float) - b.astype(float))[diff].max()) if diff.any() else 0.0}
        if a.ndim == 2:
            entry["changed_column_indices"] = np.where(diff.any(axis=0))[0].tolist()
        array_diff[name] = entry
        if name not in ("base", "context"):
            assert not diff.any(), name
    assert array_diff["context"]["changed_column_indices"] == [0, 1]
    assert array_diff["base"]["changed_column_indices"] == [1]
    assert int(changed_rows.sum()) == 96
    affected = old_meta[changed_rows]
    assert set(affected.calendar_vintage_id) == {"2020_2019-12-12_22820d037bac"}
    assert set(affected.platform) == {"CAP"}
    assert not changed_rows[old_meta.issue_year.isin([2021, 2022, 2023]).to_numpy()].any()
    affected.to_csv(correction / "local_only/AFFECTED_EXAMPLE_IDENTITIES.csv", index=False)
    fold_effects = []
    target = pd.to_datetime(old_meta.target_date)
    mature = old_meta.completed.astype(bool).to_numpy()
    for year in (2021, 2022, 2023):
        masks = {
            "initial_training_mature_labels": ((target < pd.Timestamp(year=year-1, month=1, day=1)).to_numpy() & mature),
            "validation_mature_labels": ((old_meta.issue_year == year-1).to_numpy() & (target < pd.Timestamp(year=year, month=1, day=1)).to_numpy() & mature),
            "final_training_mature_labels": ((target < pd.Timestamp(year=year, month=1, day=1)).to_numpy() & mature),
            "evaluation_all_issued": (old_meta.issue_year == year).to_numpy(),
        }
        fold_effects.append({"issue_year": year, **{name: {"rows": int(mask.sum()), "changed_input_rows": int((mask & changed_rows).sum())} for name, mask in masks.items()}})
    old_audit = json.loads((original / "EXAMPLE_AUDIT.json").read_text())
    new_audit = json.loads((rerun / "EXAMPLE_AUDIT.json").read_text())
    unchanged_audit = ["n_examples", "n_origins", "counts_by_year_horizon", "skipped_origin_horizon_counts", "feature_names", "context_names", "price_only_columns", "missing_target_result_count"]
    assert all(old_audit[k] == new_audit[k] for k in unchanged_audit)
    frozen_after = {rel: sha(original / rel) == expected for rel, expected in freeze["frozen_sha256"].items()}
    assert all(frozen_after.values())
    receipt = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS", "purpose": "Source-certified calendar quantity correction; unchanged builder; no model fitting or scoring",
        "original_freeze_id": freeze["freeze_id"], "original_frozen_files_unchanged": frozen_after,
        "original_prepare_sha256": sha(original / "prepare_examples.py"),
        "n_examples": len(new_meta), "n_changed_example_inputs": int(changed_rows.sum()),
        "n_unique_affected_targets": int(affected[["target_date", "platform"]].drop_duplicates().shape[0]),
        "affected_issue_date_range": [affected.issue_date.min(), affected.issue_date.max()],
        "affected_target_date_range": [affected.target_date.min(), affected.target_date.max()],
        "affected_by_issue_year_horizon": affected.groupby(["issue_year", "horizon"]).size().reset_index(name="changed_inputs").to_dict("records"),
        "array_differences": array_diff, "metadata_differences": meta_diff,
        "unchanged_target_identities_labels_histories_anchors_scales": True,
        "unchanged_direct_evaluation_inputs_2021_2023": True,
        "fold_training_and_validation_impact": fold_effects,
        "planned_final_mismatch_count_before": old_audit["supply_plan_final_mismatch_count"],
        "planned_final_mismatch_count_after": new_audit["supply_plan_final_mismatch_count"],
        "mismatch_count_is_posthoc_diagnostic_not_rule_source": True,
        "new_input_sha256": {rel: sha(rerun / rel) for rel in ["local_only/auction_events_development.csv", "local_only/calendar_vintages_development.csv", "local_only/examples_development.npz", "local_only/example_metadata.csv", "EXAMPLE_AUDIT.json"]},
        "training_permitted_only_after_new_freeze": True,
    }
    dump(correction / "INPUT_CHANGE_SUMMARY.json", receipt)
    print(json.dumps({k: receipt[k] for k in ["status", "n_examples", "n_changed_example_inputs", "n_unique_affected_targets", "affected_by_issue_year_horizon", "planned_final_mismatch_count_before", "planned_final_mismatch_count_after", "new_input_sha256"]}))


if __name__ == "__main__":
    main()
