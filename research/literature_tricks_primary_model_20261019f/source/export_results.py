"""Export every completed stage-f aggregate/fit receipt, byte for byte.

The exact path allowlist includes every declared arm and every selection fit.
Self-trained numeric model states and optimizer histories are authorized public
research artifacts. Source rows, per-origin predictions/online heads, binary
checkpoints and preprocessing arrays stay in the private run directory.

This program never trains, predicts, recomputes a forecast score or changes a
receipt's numbers. --check-only validates a completed candidate without copying.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPOSITORY = STUDY.parents[1]
MARKETS = ("TIANJIN_OFFICIAL_MEAN", "CEA_OFFICIAL_MEAN")
FOLD = "fixed602020"
FAMILIES = ("signed_context_response", "static_response", "amplitude_context", "signed_shock_only")
SETTINGS = ("window_past", "train_joint_test_window", "full_series_joint")
STRATEGIES = ("frozen", "expanding_equal", "expanding_half_life126", "rolling126", "rolling252")
MEAN_ALPHAS = (1e-5, 1e-4, 1e-3, .01, .1, 1., 10., 100.)
LRS = (.003, .01)
PRICE_ALPHAS = (.01, .1, 1., 10., 100., 1000., 10000., 100000.)
RUNS = ("ewt", "dense_path", "matured_readout")
# These exact keys carry row-level arrays in private output. Scalar metadata,
# range endpoints, horizon-sized metrics and self-trained states are retained.
FORBIDDEN_ARRAY_KEYS = frozenset({
    "predictions", "observed", "origins", "heads", "head_column_scale_path",
    "validation_observed", "test_observed", "validation_predictions", "test_predictions",
    "validation_heads", "test_heads", "dense_path_predictions", "x", "y", "price", "volume",
    "dates", "x_train", "x_val", "x_final", "x_test", "y_train", "y_val", "y_final", "y_test",
})


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_json(data, path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON token {value} in {path}")
    value = json.loads(data, parse_constant=reject)

    def inspect(item, location):
        if isinstance(item, dict):
            for key, val in item.items():
                volume_metadata = (key == "volume" and isinstance(val, dict)
                                   and set(val).issubset({"valid_rows", "usable_context_rows", "normalization",
                                                        "information", "missing", "model_usage"})
                                   and all(not isinstance(entry, (list, dict)) for entry in val.values()))
                if key in FORBIDDEN_ARRAY_KEYS and isinstance(val, (list, dict)) and not volume_metadata:
                    raise ValueError(f"Per-origin array key excluded: {path}:{location}/{key}")
                inspect(val, location + "/" + key)
        elif isinstance(item, list):
            for index, val in enumerate(item):
                inspect(val, location + f"/{index}")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"Nonfinite number in {path}:{location}")
    inspect(value, "")
    return value


def expected_paths(run):
    names = {Path(n) for n in ("PROTOCOL_BEFORE_RUN.json", "EXECUTION_COMPLETE.json", "ALL_SUMMARIES.json")}
    if run == "matured_readout":
        names.add(Path("PRIMARY_SELECTED_SUMMARIES.json"))
    for market in MARKETS:
        prefix = Path(f"{market}_{FOLD}")
        names.add(prefix / "INPUT_MANIFEST.json")
        if run == "ewt":
            for family in (*FAMILIES, "ridge_price", "rw"):
                names |= {prefix / "raw" / family / n for n in ("SUMMARY.json", "REUSED_BASELINE.json")}
            for setting in SETTINGS:
                names.add(prefix / setting / "PREPROCESS_MANIFEST.json")
                for family in (*FAMILIES, "ridge_price"):
                    folder = prefix / setting / family
                    names |= {folder / n for n in ("SELECTION.json", "REFIT.json", "SUMMARY.json")}
                    if family == "ridge_price":
                        names |= {folder / f"selection_alpha{alpha:g}.json" for alpha in PRICE_ALPHAS}
                    else:
                        names |= {folder / f"selection_alpha{alpha:g}_lr{lr}.json" for alpha in MEAN_ALPHAS for lr in LRS}
        elif run == "dense_path":
            for family in FAMILIES:
                folder = prefix / family
                names |= {folder / n for n in ("SELECTION.json", "REFIT.json", "SUMMARY.json")}
                names |= {folder / f"selection_alpha{alpha:g}_lr{lr}.json" for alpha in MEAN_ALPHAS for lr in LRS}
        else:
            for family in FAMILIES:
                folder = prefix / family
                names |= {folder / n for n in ("SELECTION.json", "EXECUTION_COMPLETE.json",
                                               "ALL_STRATEGY_SUMMARIES.json", "PRIMARY_SELECTED_SUMMARY.json")}
    return names


def require_complete(run, values, plan_sha):
    done = values[Path("EXECUTION_COMPLETE.json")]
    protocol = values[Path("PROTOCOL_BEFORE_RUN.json")]
    if done.get("status") != "completed" or protocol.get("plan_sha256") != plan_sha:
        raise AssertionError(f"Run is incomplete or uses another plan: {run}")
    required = {
        "ewt": ("source_hashes_unchanged", "plan_hash_unchanged", "all_new_scores_development"),
        "dense_path": ("source_and_plan_unchanged",),
        "matured_readout": ("source_hashes_unchanged", "plan_hash_unchanged", "data_hashes_unchanged",
                            "checkpoint_hashes_unchanged", "supplied_dataset_manifest_unchanged",
                            "upstream_protocol_and_completion_unchanged", "development_only",
                            "all_five_strategies_retained"),
    }[run]
    if any(done.get(key) is not True for key in required):
        raise AssertionError(f"Failed identity/development gate in {run}")
    counts = {"ewt": {"new_prediction_sets": 30, "reused_prediction_sets": 12, "new_fit_tasks": 462},
              "dense_path": {"summary_rows": 8, "selection_fits": 128, "final_refits": 8},
              "matured_readout": {"summary_rows": 40, "primary_selected_rows": 8, "ridge_solves": 13632,
                                   "new_nonlinear_fits": 0, "new_optimizer_steps": 0,
                                   "validation_strategy_paths": 40, "test_strategy_paths": 40}}[run]
    if any(done.get(key) != expected for key, expected in counts.items()):
        raise AssertionError(f"Declared execution count mismatch: {run}")

    rows = values[Path("ALL_SUMMARIES.json")]
    columns = ("market", "setting", "family") if run == "ewt" else (("market", "family", "strategy") if run == "matured_readout" else ("market", "family"))
    actual = [tuple(row[key] for key in columns) for row in rows]
    if run == "ewt":
        expected = {(market, setting, family) for market in MARKETS for setting in SETTINGS for family in (*FAMILIES, "ridge_price")}
        expected |= {(market, "raw", family) for market in MARKETS for family in (*FAMILIES, "ridge_price", "rw")}
    elif run == "dense_path":
        expected = {(market, family) for market in MARKETS for family in FAMILIES}
    else:
        expected = {(market, family, strategy) for market in MARKETS for family in FAMILIES for strategy in STRATEGIES}
    if len(actual) != len(expected) or set(actual) != expected:
        raise AssertionError(f"Missing, duplicate or unexpected arm: {run}")
    for row in rows:
        if row.get("role") != "development":
            raise AssertionError(f"Exposed score role changed in {run}")
        if row.get("horizons") != [1, 5, 20] or row.get("n_origins") != (196 if row["market"] == MARKETS[0] else 231):
            raise AssertionError(f"Evaluation panel identity changed in {run}")
        folder = Path(f"{row['market']}_{FOLD}")
        if run == "ewt":
            folder = folder / row["setting"]
        folder /= row["family"]
        if run != "matured_readout" and values[folder / "SUMMARY.json"] != row:
            raise AssertionError(f"Top-level and per-arm summary differ in {run}/{folder}")
    if run == "matured_readout":
        primary = [row for row in rows if row["selected_by_validation"]]
        if values[Path("PRIMARY_SELECTED_SUMMARIES.json")] != primary or len(primary) != 8:
            raise AssertionError("Primary readout summary omits a family or changes the validation choice")
        for market in MARKETS:
            for family in FAMILIES:
                folder = Path(f"{market}_{FOLD}") / family
                family_rows = [row for row in rows if row["market"] == market and row["family"] == family]
                selected = [row for row in family_rows if row["selected_by_validation"]]
                if values[folder / "ALL_STRATEGY_SUMMARIES.json"] != family_rows or len(selected) != 1:
                    raise AssertionError(f"All five readout strategies must be retained: {folder}")
                if values[folder / "PRIMARY_SELECTED_SUMMARY.json"] != selected[0]:
                    raise AssertionError(f"Selected readout summary differs: {folder}")
                if values[folder / "SELECTION.json"]["selected_strategy"] != selected[0]["strategy"]:
                    raise AssertionError(f"Readout summary does not match pre-test selection: {folder}")


def stage_candidate(run, run_root, plan_sha):
    source = run_root / run
    paths = expected_paths(run)
    actual_json = {p.relative_to(source) for p in source.rglob("*.json")}
    permitted_nonexported = {Path("PROGRESS_SUMMARIES.json")}
    missing = paths - actual_json
    unexpected = actual_json - paths - permitted_nonexported
    if missing or unexpected:
        raise AssertionError(f"{run}: missing={sorted(map(str, missing))}; unexpected JSON={sorted(map(str, unexpected))}")
    data, values = {}, {}
    for rel in sorted(paths):
        p = source / rel
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"Ordinary local evidence file required: {p}")
        content = p.read_bytes()
        values[rel] = parse_json(content, p)
        data[rel] = content
    require_complete(run, values, plan_sha)
    excluded = []
    for p in sorted(source.rglob("*")):
        if not p.is_file() or p.relative_to(source) in paths:
            continue
        excluded.append({"path": str(p.relative_to(source)), "sha256": digest(p), "bytes": p.stat().st_size,
                         "reason": "superseded progress receipt" if p.name == "PROGRESS_SUMMARIES.json" else "private source, per-origin array, preprocessing cache or binary artifact"})
    return data, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    if root == REPOSITORY or REPOSITORY in root.parents:
        raise ValueError("Run evidence must remain outside the public repository")
    if digest(args.plan) != args.expected_plan_sha:
        raise AssertionError("Frozen plan digest differs")
    parse_json(args.plan.read_bytes(), args.plan)
    candidates = {run: stage_candidate(run, root, args.expected_plan_sha) for run in RUNS}
    summary = {run: {"public_json_files": len(data), "excluded_files": len(excluded)}
               for run, (data, excluded) in candidates.items()}
    if args.check_only:
        print(json.dumps({"status": "PASS_COMPLETED_EXPORT_CANDIDATE_NO_COPY", "runs": summary}))
        return

    target = STUDY / "results"
    receipt_path = STUDY / "provenance/RESULT_EXPORT_RECEIPT.json"
    if receipt_path.exists():
        old = parse_json(receipt_path.read_bytes(), receipt_path)
        if old["plan_sha256"] != args.expected_plan_sha or old["private_run_root"] != str(root):
            raise FileExistsError("Existing export receipt belongs to another run")
        for run, (data, _) in candidates.items():
            recorded = {row["path"]: row["sha256"] for row in old["runs"][run]["files"]}
            expected = {str(Path("results") / run / rel): hashlib.sha256(content).hexdigest()
                        for rel, content in data.items()}
            if recorded != expected:
                raise FileExistsError("Existing export inventory differs from the completed run")
            for rel, content in data.items():
                if (target / run / rel).read_bytes() != content:
                    raise FileExistsError("Existing public copy differs from its original receipt")
        print(json.dumps({"status": "ALREADY_EXPORTED_BYTE_IDENTICAL", "runs": summary}))
        return
    expected_target = {Path(run) / rel for run, (data, _) in candidates.items() for rel in data}
    if target.exists():
        unexpected_target = {p.relative_to(target) for p in target.rglob("*") if p.is_file()} - expected_target
        if unexpected_target:
            raise FileExistsError(f"Unexpected public result files: {sorted(map(str, unexpected_target))}")
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "EXPORTED_BYTE_IDENTICAL",
               "exporter_sha256": digest(__file__), "plan_sha256": args.expected_plan_sha,
               "private_run_root": str(root), "new_fit_or_forecast_score_calls": 0,
               "policy": "Exact complete JSON path allowlist; retain all arms, selection trials, training trajectories and authorized self-trained numeric states. Copy original bytes. Keep raw rows, per-origin arrays, online head paths, caches and binary states private.",
               "runs": {}}
    for run, (data, excluded) in candidates.items():
        files = []
        for rel, content in data.items():
            destination = target / run / rel
            if destination.exists():
                if destination.read_bytes() != content:
                    raise FileExistsError(f"Do not overwrite differing public evidence: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(content)
            if destination.read_bytes() != content:
                raise AssertionError("Copy is not byte identical")
            files.append({"path": str(destination.relative_to(STUDY)), "sha256": hashlib.sha256(content).hexdigest(),
                          "bytes": len(content)})
        receipt["runs"][run] = {"files": files, "excluded_private_files": excluded,
                                "public_json_files": len(files), "all_declared_arms_retained": True}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "EXPORTED_BYTE_IDENTICAL", "runs": summary,
                      "receipt": str(receipt_path), "receipt_sha256": digest(receipt_path)}))


if __name__ == "__main__":
    main()
