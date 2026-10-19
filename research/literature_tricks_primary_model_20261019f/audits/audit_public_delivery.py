"""Read-only comparison of stage-f private originals, public copies and replay.

Unlike the public-state replay entry point, this optional original-evidence
audit needs the retained private NPZ files. It does not fit or score a model.
Only aggregate comparison facts and file hashes are written to its new report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "source"))
import export_results as exporter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_count(value):
    if isinstance(value, dict):
        return sum(scalar_count(v) for v in value.values())
    if isinstance(value, list):
        return sum(scalar_count(v) for v in value)
    return int(isinstance(value, (int, float)) and not isinstance(value, bool))


def array_comparison(left, right, label):
    if left.shape != right.shape or left.dtype.kind != right.dtype.kind:
        raise AssertionError(f"Array identity differs: {label}")
    if left.dtype.kind in "biufc":
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise AssertionError(f"Nonfinite array: {label}")
        gap = float(np.max(np.abs(left.astype(float) - right.astype(float)))) if left.size else 0.
        if not np.allclose(left, right, rtol=1e-9, atol=1e-10):
            raise AssertionError(f"Original/replay values differ: {label}")
    else:
        gap = 0.
        if not np.array_equal(left, right):
            raise AssertionError(f"String array differs: {label}")
    return {"array": label, "absolute_gap": gap, "array_equal": bool(np.array_equal(left, right))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Use a fresh scope/audit receipt")
    receipt = json.loads((STUDY / "provenance/RESULT_EXPORT_RECEIPT.json").read_text())
    inventory, state_scalars, ridge_scalars, private_counts = [], 0, 0, {}
    for run in exporter.RUNS:
        records = receipt["runs"][run]["files"]
        if len(records) != len(exporter.expected_paths(run)):
            raise AssertionError("Missing public formal receipt")
        for record in records:
            public = STUDY / record["path"]
            relative = public.relative_to(STUDY / "results" / run)
            original = args.run_root / run / relative
            if public.read_bytes() != original.read_bytes() or digest(public) != record["sha256"]:
                raise AssertionError("Public result is not its original bytes")
            value = exporter.parse_json(public.read_bytes(), public)
            if isinstance(value, dict):
                if "state_dict" in value:
                    state_scalars += scalar_count(value["state_dict"])
                for field in ("coefficients", "intercept", "column_mean", "column_scale", "final_column_mean", "final_column_scale"):
                    if field in value:
                        ridge_scalars += scalar_count(value[field])
            inventory.append(record)
        for excluded in receipt["runs"][run]["excluded_private_files"]:
            private = args.run_root / run / excluded["path"]
            if digest(private) != excluded["sha256"]:
                raise AssertionError("An excluded original artifact changed")
            private_counts[private.suffix] = private_counts.get(private.suffix, 0) + 1
    if len(inventory) != 752:
        raise AssertionError("Wrong complete export count")

    comparisons, panels = [], 0
    standard_keys = ("predictions", "observed", "origins", "horizons", "deterministic")
    for market in exporter.MARKETS:
        market_folder = f"{market}_{exporter.FOLD}"
        for setting in ("raw", *exporter.SETTINGS):
            families = (*exporter.FAMILIES, "ridge_price", "rw") if setting == "raw" else (*exporter.FAMILIES, "ridge_price")
            for family in families:
                rel = Path("ewt") / market_folder / setting / family / "predictions.npz"
                with np.load(args.run_root / rel, allow_pickle=False) as original, np.load(args.replay_root / rel, allow_pickle=False) as replay:
                    for key in standard_keys:
                        comparisons.append(array_comparison(original[key], replay[key], str(rel) + "/" + key))
                panels += 1
        for family in exporter.FAMILIES:
            original_path = args.run_root / "dense_path" / market_folder / family / "predictions.npz"
            replay_path = args.replay_root / "dense_path" / market_folder / "raw_dense_path_supervision" / family / "predictions.npz"
            with np.load(original_path, allow_pickle=False) as original, np.load(replay_path, allow_pickle=False) as replay:
                for key in (*standard_keys, "dense_path_predictions"):
                    comparisons.append(array_comparison(original[key], replay[key], f"dense_path/{market}/{family}/{key}"))
            panels += 1
            rel = Path("matured_readout") / market_folder / family / "paths.npz"
            with np.load(args.run_root / rel, allow_pickle=False) as original, np.load(args.replay_root / rel, allow_pickle=False) as replay:
                # Replay omits only metadata duplicated in its separate receipts.
                omitted = set(original.files) - set(replay.files)
                if omitted != {"validation_losses", "initial_price_rms", "initial_target_second_moment"}:
                    raise AssertionError("Unexpected omissions in the regenerated online trace")
                for key in replay.files:
                    comparisons.append(array_comparison(original[key], replay[key], str(rel) + "/" + key))
                for index, strategy in enumerate(exporter.STRATEGIES):
                    panel_path = args.replay_root / "matured_readout" / market_folder / strategy / family / "predictions.npz"
                    with np.load(panel_path, allow_pickle=False) as panel:
                        comparisons.append(array_comparison(original["test_predictions"][index][None], panel["predictions"],
                                                            f"online_panel/{market}/{family}/{strategy}"))
                    panels += 1
    if panels != 90:
        raise AssertionError("Not all formal predictions were compared")

    manifest = json.loads((STUDY / "provenance/RUN_DATASETS.json").read_text())
    canonical_rows, canonical_hashes = [], set()
    for spec in manifest["datasets"]:
        path = Path(spec["file"])
        if digest(path) != spec["sha256"]:
            raise AssertionError("Original canonical bytes changed")
        canonical_hashes.add(spec["sha256"])
        canonical_rows.extend(line for line in path.read_bytes().splitlines()[1:] if line)
    rows_by_date = {}
    for row in canonical_rows:
        rows_by_date.setdefault(row[:10], []).append(row)
    date_comma = re.compile(rb"\d{4}-\d{2}-\d{2},")
    snapshot, raw_matches = [], []
    for path in sorted(STUDY.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.resolve() == args.report.resolve():
            continue
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        if path.suffix in {".npz", ".pt", ".pkl", ".pickle"} or sha in canonical_hashes:
            raise AssertionError(f"Private source/binary artifact found in public study: {path}")
        matches = sum(content.startswith(row, occurrence.start())
                      for occurrence in date_comma.finditer(content)
                      for row in rows_by_date.get(occurrence.group()[:10], ()))
        if matches:
            raw_matches.append({"path": str(path.relative_to(STUDY)), "canonical_row_string_matches": matches})
        snapshot.append({"path": str(path.relative_to(STUDY)), "sha256": sha, "bytes": len(content)})
    if raw_matches:
        raise AssertionError("Raw canonical row strings found in publication snapshot")
    if any(digest(STUDY / record["path"]) != record["sha256"] for record in snapshot):
        raise AssertionError("Public snapshot changed during the scope scan; retry with a fresh receipt")
    result = {"status": "PASS_EXACT_EXPORT_AND_PRIVATE_TO_PUBLIC_REPLAY", "created_utc": datetime.now(timezone.utc).isoformat(),
              "audit_source_sha256": digest(__file__), "export_receipt_sha256": digest(STUDY / "provenance/RESULT_EXPORT_RECEIPT.json"),
              "scope": "This timestamped study-file snapshot, excluding this not-yet-written audit receipt and Python bytecode. Files added or edited later require their own final delivery check.",
              "public_json_receipts_byte_identical": len(inventory),
              "authorized_stored_state_numeric_scalars_including_buffers": state_scalars,
              "authorized_ridge_coefficients_intercepts_and_scaler_scalars": ridge_scalars,
              "excluded_original_private_files_by_suffix": private_counts,
              "formal_prediction_panels_compared": panels, "array_comparisons": len(comparisons),
              "maximum_original_to_replay_array_absolute_gap": max(row["absolute_gap"] for row in comparisons),
              "all_compared_arrays_exactly_equal": all(row["array_equal"] for row in comparisons),
              "new_model_fits": 0, "new_model_forecasts": 0, "new_forecast_scores": 0,
              "original_canonical_rows_checked": len(canonical_rows), "canonical_row_strings_copied_publicly": 0,
              "public_npz_pt_or_canonical_blob_copies": 0,
              "replay_private_npz_files": sum(1 for _ in args.replay_root.rglob("*.npz")),
              "comparisons": comparisons, "snapshot_files": snapshot}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("comparisons", "snapshot_files")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
