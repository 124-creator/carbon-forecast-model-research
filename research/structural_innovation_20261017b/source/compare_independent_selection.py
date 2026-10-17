"""Compare already produced D1 selections without importing either selector.

This is postprocessing, written after independent D1 selection. It does not
select a model, read forecasts or original prices, or change a study gate.
Only Python's standard library is required. Output must not already exist.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(independent: dict, primary: dict, metrics: list[dict]) -> dict:
    counts = {"discrete": 0, "numeric": 0}
    maximum = 0.0

    def exact(left, right, label):
        counts["discrete"] += 1
        if left != right:
            raise AssertionError(f"Discrete mismatch: {label}")

    def numeric(left, right, label):
        nonlocal maximum
        left, right = float(left), float(right)
        counts["numeric"] += 1
        if not (math.isfinite(left) and math.isfinite(right)):
            raise AssertionError(f"Nonfinite comparison: {label}")
        maximum = max(maximum, abs(left - right))
        if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"Numeric mismatch: {label}")

    exact(independent["phase"], "D1", "independent phase")
    exact(primary["stage"], "D1", "primary phase")
    exact(independent["selection_year"], primary["year"], "selection year")
    exact(independent["protocol_sha256"], primary["protocol_sha256"], "protocol")
    by_id = {row["id"]: row for row in metrics}
    exact(len(by_id), len(metrics), "unique independently recomputed jobs")

    def summary(left, right, label):
        exact(left["family"], right["family"], label + ": family")
        exact(left["value"], right["value"], label + ": value")
        numeric(left["score"], right["score"], label + ": score")
        exact(set(left["by_market"]), set(right["by_market"]), label + ": markets")
        primary_ids = []
        for market, lmarket in left["by_market"].items():
            rmarket = right["by_market"][market]
            numeric(lmarket["score"], rmarket["score"], label + ": " + market + " score")
            exact(len(lmarket["relative_mse"]), 20, label + ": independent all20")
            exact(len(rmarket["relative_mse"]), 20, label + ": primary all20")
            for h, (x, y) in enumerate(zip(lmarket["relative_mse"], rmarket["relative_mse"]), 1):
                numeric(x, y, f"{label}: {market} h{h}")
            expected_ids = sorted(job for job in left["job_ids"] if by_id[job]["market"] == market)
            exact(expected_ids, sorted(rmarket["job_ids"]), label + ": market job IDs")
            primary_ids.extend(rmarket["job_ids"])
            exact(sorted({by_id[job]["seed"] for job in expected_ids}),
                  sorted(rmarket["seeds"]), label + ": seeds")
            ns = {by_id[job]["n"] for job in expected_ids}
            exact(len(ns), 1, label + ": common independent n")
            exact(next(iter(ns)), rmarket["n"], label + ": origin count")
        exact(sorted(left["job_ids"]), sorted(primary_ids), label + ": full job IDs")
        if left["family"] != "ridge_val":
            ids = {by_id[job]["config_id"] for job in left["job_ids"]}
            exact(ids, {right["config_id"]}, label + ": config ID")

    exact(set(independent["selected"]), set(primary["selected"]), "selected families")
    for family in independent["selected"]:
        summary(independent["selected"][family], primary["selected"][family], "selected/" + family)

    exact(set(independent["matched_smp_alpha"]), set(primary["matched_to_smp"]), "matched families")
    for family in independent["matched_smp_alpha"]:
        summary(independent["matched_smp_alpha"][family], primary["matched_to_smp"][family], "matched/" + family)
    summary(independent["best_baseline"], primary["fixed_best_non_smp_baseline"], "D1 fixed best baseline")

    choices = independent["selected"]["ridge_val"]["pre_year_validation_choices"]
    for market, choice in choices.items():
        other = primary["selected"]["ridge_val"]["by_market"][market]
        exact(choice["value"], other["selected_value"], "ridge_val/" + market + ": alpha")
        exact([choice["job_id"]], other["job_ids"], "ridge_val/" + market + ": selected job")
        numeric(choice["validation_score"], other["selected_validation_score"], "ridge_val/" + market + ": validation")

    gate_mapping = {
        "full_below_rw": "better_than_rw",
        "full_below_same_alpha_iid": "better_than_matched_iid",
        "full_below_same_alpha_one_step": "better_than_matched_one_step",
        "full_within_2pct_of_best_baseline": "within_practical_margin_of_best_baseline",
    }
    exact(set(independent["gates"]), set(gate_mapping), "independent gate keys")
    exact(set(primary["gate_checks"]), set(gate_mapping.values()), "primary gate keys")
    for left, right in gate_mapping.items():
        exact(independent["gates"][left], primary["gate_checks"][right], "gate/" + left)
    exact(independent["advance_to_D2"], primary["advance_to_D2"], "advance to D2")
    exact(primary["confirmation_access"], False, "D1 does not authorize confirmation")

    return {
        "status": "PASS",
        "phase": "D1",
        "protocol_sha256": independent["protocol_sha256"],
        "selected_families": len(independent["selected"]),
        "matched_families": len(independent["matched_smp_alpha"]),
        "comparison_counts": counts,
        "maximum_numeric_absolute_difference": maximum,
        "numeric_tolerance": {"absolute": 1e-12, "relative": 1e-12},
        "exact_discrete_agreement": True,
        "gates": independent["gates"],
        "advance_to_D2": independent["advance_to_D2"],
        "scope": "All selected and matched family values/configuration job IDs; all selected market scores, all20 relative MSEs, n and seeds; local ridge validation choices; fixed strongest baseline; all four gates and advancement.",
        "limitations": "Postprocessing comparison, not a new preregistration or a second fitting run. Does not read price CSV, forecast NPZ or confirmation files; does not import either selector.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-dir", type=Path, required=True)
    parser.add_argument("--root-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    files = {
        "independent_selection": args.independent_dir / "SELECTION.json",
        "independent_recomputed_metrics": args.independent_dir / "RECOMPUTED_METRICS.json",
        "root_selection": args.root_selection,
    }
    objects = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in files.items()}
    result = compare(objects["independent_selection"], objects["root_selection"], objects["independent_recomputed_metrics"])
    result["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    result["comparator_sha256"] = sha256(Path(__file__))
    result["input_sha256"] = {key: sha256(path) for key, path in files.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "comparison_counts", "maximum_numeric_absolute_difference", "advance_to_D2")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
