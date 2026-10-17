"""Copy an isolated study with portable paths; never execute a market model.

Only source files named in each historical plan are copied. Historical results,
states, predictions, attempt markers and completion markers are excluded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from install_main_data import (DATA_FILES, json_bytes, require_new_output,
                               packaged_file, safe_local, safe_relative, verify_bytes, write_new)

HERE = Path(__file__).resolve().parent
STUDY_RELATIVE = "research/cross_market_innovation_20261017"
STAGES = {
    "D1": ("frozen_d1_source", "source", "D1_PRE_RUN.json"),
    "D2": ("d2_source", "d2_source", "D2_PRE_RUN.json"),
    "C": ("confirmation_source", "confirmation_source", "CONFIRMATION_PRE_RUN.json"),
}
PLAN_SHA256 = {
    "D1_PRE_RUN.json": "4c1a31662959c951548b19c068cf6e8f33d7ab9b86fbe519203575f4a3e593ba",
    "D2_PRE_RUN.json": "62e08b11471f065fa248deffd1d3b095794aebbf631e6bc304ed302e79eac3ff",
    "CONFIRMATION_PRE_RUN.json": "7baa32f9b9e6727b4bf030d02868d03e3313aca1b8048952d32e400139a45122",
    "LEARNING_DIAGNOSTIC_PRE_RUN.json": "fc6386e47aee83a25e35931a6b9400e59fd59579f39f37ced4a7fe9e9fef0251",
}
TOP_LEVEL = ["TIMING_PLAN.json", "CONFIRMATION_DESIGN_RULES.json", "aggregate_development.py",
             "make_figures.py"]
OPTIONAL_POSTPROCESSORS = ["aggregate_confirmation.py"]


def study_root(path: Path) -> Path:
    base = path.expanduser().resolve()
    if (base / "D1_PRE_RUN.json").is_file():
        return base
    candidate = base / STUDY_RELATIVE
    if (candidate / "D1_PRE_RUN.json").is_file():
        return candidate
    raise FileNotFoundError("Expected the repository root or the study directory")


def checked_blob(root: Path, relative: str, expected_sha: str | None = None) -> bytes:
    blob = safe_local(root, relative).read_bytes()
    if expected_sha is not None and hashlib.sha256(blob).hexdigest() != expected_sha:
        raise ValueError("Frozen source mismatch: " + relative)
    return blob


def collect_source_plan(study: Path) -> tuple[dict[str, bytes], dict]:
    files = {}
    summary = {}
    for stage, (old_directory, new_directory, name) in STAGES.items():
        plan_blob = checked_blob(study, name, PLAN_SHA256[name])
        plan = json.loads(plan_blob)
        files["historical_protocols/" + name] = plan_blob
        for relative, expected_sha in plan["source_sha256"].items():
            safe_relative(relative)
            destination = new_directory + "/" + relative
            if destination in files:
                raise ValueError("Colliding source destination")
            files[destination] = checked_blob(study, old_directory + "/" + relative, expected_sha)
        summary[stage] = dict(tasks=len(plan["tasks"]), source_files=len(plan["source_sha256"]),
                              historical_plan_sha256=hashlib.sha256(plan_blob).hexdigest())
    # The extra development diagnostic has a scalar source hash and reuses the
    # already copied D2 implementation. It never enters confirmation selection.
    learning_name = "LEARNING_DIAGNOSTIC_PRE_RUN.json"
    learning_blob = checked_blob(study, learning_name, PLAN_SHA256[learning_name])
    learning = json.loads(learning_blob)
    if learning["d2_plan_sha256"] != PLAN_SHA256["D2_PRE_RUN.json"]:
        raise ValueError("Historical learning diagnostic depends on a different D2 plan")
    if learning["changes_confirmation_selection"] or learning["confirmation_data_allowed"]:
        raise ValueError("Learning diagnostic must remain development-only")
    files["historical_protocols/" + learning_name] = learning_blob
    files["learning_diagnostic.py"] = checked_blob(study, "learning_diagnostic.py", learning["source_sha256"])
    summary["LEARNING"] = dict(tasks=len(learning["tasks"]), source_files=1,
                               historical_plan_sha256=hashlib.sha256(learning_blob).hexdigest(),
                               changes_confirmation_selection=False)
    for name in TOP_LEVEL:
        files[name] = checked_blob(study, name)
    for name in OPTIONAL_POSTPROCESSORS:
        if (study / name).is_file():
            files[name] = checked_blob(study, name)
    original_registry_blob = checked_blob(study, "provenance/DEVELOPMENT_REGISTRY.json")
    original_registry = json.loads(original_registry_blob)
    for stage, (_, _, name) in STAGES.items():
        plan = json.loads(files["historical_protocols/" + name])
        key = "data_registry_sha256" if stage == "C" else "registry_sha256"
        if hashlib.sha256(original_registry_blob).hexdigest() != plan[key]:
            raise ValueError("Original registry differs from historical plan: " + stage)
    specifications = [s for s in DATA_FILES if s["role"] == "development"]
    if [r["name"] for r in original_registry] != [s["name"] for s in specifications]:
        raise ValueError("Unexpected registry order or market set")
    runtime_registry = []
    for record, spec in zip(original_registry, specifications):
        if record["sha256"] != spec["sha256"] or record["price_column"] != "price":
            raise ValueError("Unexpected registry data identity")
        replacement = dict(record)
        replacement["path"] = "local_only/data/" + spec["path"]
        runtime_registry.append(replacement)
    files["provenance/DEVELOPMENT_REGISTRY.json"] = json_bytes(runtime_registry)
    # Sources and settings are tied to the same original protocol, independent of
    # an absolute path in the old runtime registry.
    d1 = json.loads(files["historical_protocols/D1_PRE_RUN.json"])
    c = json.loads(files["historical_protocols/CONFIRMATION_PRE_RUN.json"])
    if hashlib.sha256(files["TIMING_PLAN.json"]).hexdigest() != d1["timing_plan_sha256"]:
        raise ValueError("Historical timing plan changed")
    if hashlib.sha256(files["CONFIRMATION_DESIGN_RULES.json"]).hexdigest() != c["rules_sha256"]:
        raise ValueError("Historical confirmation rules changed")
    return files, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root of the exact five-file installed/package dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = require_new_output(args.output)
    study = study_root(args.study_dir)
    if output.is_relative_to(study):
        raise ValueError("Output must be outside the historical study directory")
    files, stages = collect_source_plan(study)
    if args.dry_run:
        print(json.dumps(dict(output=str(output), stage_plans=stages, source_files=len(files),
                              data_files_expected=DATA_FILES, data_bytes_opened=False,
                              model_executed=False, network_calls=0,
                              source_sha256_checked=True), ensure_ascii=False, indent=2))
        return
    data = args.data_dir.expanduser().resolve()
    if output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError("Study output and source data directories must be separate")
    # Binary identity checks/copy only. This preparer never parses price values.
    for spec in DATA_FILES:
        blob = packaged_file(data, spec).read_bytes()
        verify_bytes(blob, spec, spec["path"])
        files["local_only/data/" + spec["path"]] = blob
    for name in ("reproduce_study.py", "requirements-portable.txt", "requirements-figures.txt",
                 "DATA_METHODS_ZH.md", "REPRODUCE_PORTABLE_ZH.md", "MAIN_DATA_SOURCES.json"):
        files[name] = checked_blob(HERE, name)
    files[".gitignore"] = b"local_only/\nresults/\n__pycache__/\n*.pyc\n"
    context = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   kind="isolated_reproduction_of_an_existing_study", stages=stages,
                   runtime_registry_paths_relative_to_study=True,
                   run_via="reproduce_study.py (sets working directory before importing frozen sources)",
                   historical_result_files_copied=False, optimizer_executed_by_preparer=False,
                   new_independent_confirmation_claim=False,
                   data_preparation_may_parse_confirmation_prices=True,
                   interpretation="Reproducing the original benchmark does not create another unseen confirmation set.")
    files["provenance/REPRODUCTION_CONTEXT.json"] = json_bytes(context)
    file_manifest = [dict(path=name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
                     for name, blob in sorted(files.items())]
    output.mkdir(parents=True, exist_ok=False)
    for name, blob in files.items():
        write_new(output / name, blob)
    write_new(output / "provenance/WORKDIR_MANIFEST.json", json_bytes(dict(
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), files=file_manifest,
        copied_market_values_printed=False, model_fit=False, network_calls=0)))
    print(json.dumps(dict(status="PASS", output=str(output), files=len(files),
                          next_command="python -B reproduce_study.py d1 check",
                          historical_sources_unchanged=True, model_fit=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
