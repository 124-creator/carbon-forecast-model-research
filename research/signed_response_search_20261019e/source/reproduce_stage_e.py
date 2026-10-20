"""Relocate and reproduce the frozen stage-e experiment without changing it.

Standard-library launcher. --preflight-only checks identities, relocation and
fresh destinations; it never imports a model or starts a fit. All actual reruns
are explicitly labeled replication of already evaluated historical data.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
STUDY = SOURCE.parent
PLAN_FILE = STUDY/"audits/STATISTICAL_PLAN_BEFORE_FIT.json"
PLAN_SHA = "b6f33c9c328f37f9a283275d6cc36e4e48153bc3322d45e2b8deb74e3d1d987c"
ADDENDUM_FILE = STUDY/"audits/BUDGET_SENSITIVITY_BEFORE_RUN.json"
ADDENDUM_SHA = "ac1117d24f395cb7b3331ffd2e4de3d6e936f9b9891748204c4aa8df394de72c"
DATA_MANIFEST_FILE = STUDY/"provenance/RUN_DATASETS.json"
DATA_MANIFEST_SHA = "f28fae19e3bcbabb090d3d1e9ebe97aa8c22558114a5c346143ec03a70ab2b22"
DATA_SHA = {
    "TIANJIN_OFFICIAL_MEAN": "85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a",
    "CEA_OFFICIAL_MEAN": "f9f0e09b7500b2bdb4ec8b16c61d5d5fa2b64de63a24c0cf7bd1d269aa2be75d",
}
TRAINING_SOURCES = {"signed_response_models.py", "run_signed_response_search.py",
                    "run_neural_references.py", "run_budget_sensitivity.py"}
REPLICATION_SCOPE = (
    "Replication of the frozen, already evaluated stage-e historical experiment; "
    "not a newly unseen confirmation, external preregistration, or additional "
    "independent market sample. Historical role fields inside unmodified runner "
    "artifacts describe the original study only."
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Identity mismatch: {path}; expected {expected}, observed {actual}")
    return actual


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def verify_and_relocate(data_root):
    checked_hash(PLAN_FILE, PLAN_SHA)
    checked_hash(ADDENDUM_FILE, ADDENDUM_SHA)
    checked_hash(DATA_MANIFEST_FILE, DATA_MANIFEST_SHA)
    addendum = read(ADDENDUM_FILE)
    expected_sources = addendum["source_sha256"]
    if set(expected_sources) != TRAINING_SOURCES:
        raise RuntimeError("Unexpected training-source set in the frozen addendum")
    sources = {name: checked_hash(SOURCE/name, expected) for name,expected in expected_sources.items()}
    # The original runners also require all eight prior-stage source identities.
    # Read the literal mapping from the already SHA-verified core, without
    # importing torch/sklearn/ewtpy or executing any training module.
    tree = ast.parse((SOURCE/"run_signed_response_search.py").read_text())
    mappings = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "OLD_SHA" for target in node.targets)]
    if len(mappings) != 1 or len(mappings[0]) != 8:
        raise RuntimeError("Cannot verify original legacy-source dependencies")
    old_source = STUDY.parent/"literature_aligned_model_search_20261019d/source"
    legacy = {name: checked_hash(old_source/name, expected) for name,expected in mappings[0].items()}
    original = read(DATA_MANIFEST_FILE)
    relocated = copy.deepcopy(original)
    if {spec["market"] for spec in original["datasets"]} != set(DATA_SHA):
        raise RuntimeError("Unexpected canonical market list")
    for spec in relocated["datasets"]:
        market = spec["market"]
        if spec["sha256"] != DATA_SHA[market]:
            raise RuntimeError("Canonical manifest data identity differs")
        path = data_root/(market+".csv")
        checked_hash(path, DATA_SHA[market])
        spec["file"] = str(path)
    # The scientific/source metadata in the running manifest remain verbatim.
    restored = copy.deepcopy(relocated)
    for old,new in zip(original["datasets"], restored["datasets"]):
        new["file"] = old["file"]
    if restored != original:
        raise RuntimeError("Relocation changed non-path dataset metadata")
    return relocated, addendum, {"training_source_sha256": sources,
        "legacy_source_sha256": legacy, "statistical_plan_sha256": PLAN_SHA,
        "original_addendum_sha256": ADDENDUM_SHA,
        "original_dataset_manifest_sha256": DATA_MANIFEST_SHA,
        "canonical_data_sha256": DATA_SHA}


def relink_addendum(original, datasets, new_core_protocol, new_core_sha):
    """Only relink runtime provenance; all original scientific fields survive."""
    linked = copy.deepcopy(original)
    linked["original_core_protocol_sha256"] = new_core_sha
    new_paths = {spec["market"]:spec["file"] for spec in datasets["datasets"]}
    for item in linked["data_identity"]:
        item["file"] = new_paths[item["market"]]
    linked["original_addendum_sha256"] = ADDENDUM_SHA
    linked["original_frozen_core_protocol_sha256"] = original["original_core_protocol_sha256"]
    linked["replication_core_protocol_path"] = str(new_core_protocol)
    linked["replication_context"] = REPLICATION_SCOPE
    added = {"original_addendum_sha256", "original_frozen_core_protocol_sha256",
             "replication_core_protocol_path", "replication_context"}
    restored = copy.deepcopy(linked)
    for key in added:
        del restored[key]
    restored["original_core_protocol_sha256"] = original["original_core_protocol_sha256"]
    for old,new in zip(original["data_identity"], restored["data_identity"]):
        new["file"] = old["file"]
    if restored != original:
        raise RuntimeError("Budget relinking altered a scientific field")
    return linked


def context(identity, root, datasets, preflight, budget):
    return {"execution_role": "replication", "scope": REPLICATION_SCOPE,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "preflight_only": preflight, "with_budget320": budget,
            "run_root": str(root), "relocated_manifest": str(root/"RUN_DATASETS_RELOCATED.json"),
            "datasets": [{"market":s["market"],"file":s["file"],"sha256":s["sha256"],
                          "execution_role":"replication","historical_original_role":s["role"]}
                         for s in datasets["datasets"]],
            "identity": identity, "python_executable": sys.executable,
            "launcher_sha256": sha256(__file__),
            "historical_role_field_interpretation": "Retained verbatim in frozen runner outputs, never a claim of fresh confirmation in this replication.",
            "original_evidence_audit": "audit_budget_sensitivity.py and original diagnostic artifacts are pinned to the original run's protocol/addendum hashes. They are not automatically run against newly timestamped/relocated protocols.",
            "fits_if_executed": {"core_selection_and_final":224,"neural_selection_and_final":20,
                                 "budget_selection_and_final":170 if budget else 0,
                                 "total":414 if budget else 244},
            "preflight_model_fits":0 if preflight else None}


def execute(name, command, root, replication_context):
    print(f"REPLICATION START {name}: {json.dumps(command)}", flush=True)
    subprocess.run(command, check=True, cwd=STUDY)
    marker = read(root/name/"EXECUTION_COMPLETE.json")
    if (marker.get("status") != "completed" or not marker.get("source_hashes_unchanged")
            or not marker.get("plan_hash_unchanged")):
        raise RuntimeError(f"Incomplete or changed replication stage: {name}")
    # Write this sidecar after completion so the frozen runner still starts in
    # its required empty child directory. Root context labels the entire tree
    # from before the first model is run, including an interrupted child run.
    write(root/name/"REPLICATION_CONTEXT.json", replication_context)
    print(f"REPLICATION COMPLETE {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing the two exact canonical MARKET.csv files")
    parser.add_argument("--run-root", type=Path, required=True,
                        help="New or empty destination; use different roots for preflight and actual runs")
    parser.add_argument("--with-budget320", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    data_root, root = args.data_root.resolve(), args.run_root.resolve()
    if not data_root.is_dir():
        parser.error("--data-root must be an existing directory")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("Fresh run root required; existing artifacts are never overwritten")
    datasets, original_addendum, identity = verify_and_relocate(data_root)
    request = context(identity, root, datasets, args.preflight_only, args.with_budget320)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/"RUN_DATASETS_RELOCATED.json"
    write(manifest_path, datasets)
    write(root/"REPLICATION_CONTEXT.json", request)
    core_dir = root/"core"
    if args.preflight_only:
        # A placeholder hash exercises metadata-only roundtrip validation but
        # is never written as an executable addendum or passed to a runner.
        relink_addendum(original_addendum, datasets, core_dir/"PROTOCOL_BEFORE_RUN.json", "0"*64)
        receipt = {"status":"PASS_IDENTITIES_AND_RELOCATION_NO_FIT_NO_SCORE",
                   "execution_role":"replication_preflight", "model_imports":0,
                   "model_fits":0, "validation_or_test_forecast_scores":0,
                   "dataset_copy_changes":"file paths only", "budget_scientific_fields_roundtrip_check":True,
                   "budget_link_creation":"Only after a newly completed core protocol exists in an actual run",
                   "identity":identity,"run_root":str(root),"launcher_sha256":sha256(__file__)}
        write(root/"PREFLIGHT_ONLY.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return
    py = sys.executable
    execute("core", [py,str(SOURCE/"run_signed_response_search.py"),"--datasets",str(manifest_path),
            "--out",str(core_dir),"--freeze-plan",str(PLAN_FILE),"--freeze-plan-sha",PLAN_SHA,
            "--workers",str(args.workers)],root,request)
    execute("neural_references",[py,str(SOURCE/"run_neural_references.py"),"--datasets",str(manifest_path),
            "--out",str(root/"neural_references"),"--plan",str(PLAN_FILE),"--expected-plan-sha256",PLAN_SHA,
            "--workers",str(args.workers)],root,request)
    if args.with_budget320:
        fresh_protocol = core_dir/"PROTOCOL_BEFORE_RUN.json"
        linked = relink_addendum(original_addendum,datasets,fresh_protocol,sha256(fresh_protocol))
        linked_path = root/"BUDGET_ADDENDUM_RELINKED.json"
        write(linked_path, linked)
        execute("budget320",[py,str(SOURCE/"run_budget_sensitivity.py"),"--original-core-out",str(core_dir),
                "--addendum",str(linked_path),"--expected-addendum-sha",sha256(linked_path),
                "--out",str(root/"budget320"),"--workers",str(args.workers)],root,request)
    rows=[]
    stages=[("core","ALL_SUMMARIES.json"),("neural_references","SUMMARY.json")]
    if args.with_budget320:
        stages.append(("budget320","ALL_SUMMARIES.json"))
    for stage,filename in stages:
        for row in read(root/stage/filename):
            replica=copy.deepcopy(row)
            replica["historical_runner_role"]=replica.get("role")
            replica["role"]="replication"
            replica["execution_role"]="replication"
            replica["replicated_stage"]=stage
            rows.append(replica)
    write(root/"REPLICATION_SUMMARIES.json",rows)
    # Recheck all source and canonical identities at the end, too.
    verify_and_relocate(data_root)
    write(root/"REPLICATION_COMPLETE.json",{"status":"completed","execution_role":"replication",
            "scope":REPLICATION_SCOPE,"with_budget320":args.with_budget320,"summary_rows":len(rows),
            "identity":identity,"launcher_sha256":sha256(__file__),
            "original_bound_audits_automatically_run":False})
    print("REPLICATION COMPLETE: existing evidence reproduced; no fresh confirmation claim",flush=True)


if __name__ == "__main__":
    main()
