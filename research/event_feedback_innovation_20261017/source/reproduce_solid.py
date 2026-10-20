"""Prepare or explicitly execute a NEW replication of the bounded SOLID B1.

The five frozen runtime files are copied byte for byte. Preparation uses the
selected OFFLINE checkpoints already produced by a D1 run, never teacher_final.
No D1 fitting is initiated here. There is no confirmation-data argument.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ORIGINAL_B1_SHA256 = "b44357f30ce5c7f50f2023ebb0fd3fcc17e1598038724738b0b64b56c7ca76fd"
ORIGINAL_D1_SHA256 = "404c14867dcc5b43f456075d6b9edd4a5cd75cd99ad943a034e8da7172ef8a70"
OLD_FILES = {
    "GUANGZHOU": "GUANGZHOU_development.csv",
    "HUBEI": "HUBEI_development.csv",
    "SHANGHAI": "SHANGHAI_D4_development.csv",
    "EU_REPORTED": "EU_REPORTED_development.csv",
}
SELECTION_RULE = "Minimum pre-online validation score; exact ties prefer larger shrink"
TEACHER_KEYS = {
    "encoder.weight_ih_l0", "encoder.weight_hh_l0",
    "encoder.bias_ih_l0", "encoder.bias_hh_l0", "head.weight", "head.bias",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def canonical_state_hash(state):
    """Exactly the frozen rrc_core.digest_arrays convention, without importing a trainer."""
    import numpy as np
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = np.asarray(value)
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def checkpoint_identity(path, *, channels, expected_joint_state_sha256):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        joint = {key: archive[key].copy() for key in archive.files}
    if not joint or any(not key.startswith(("T.", "S.")) for key in joint):
        raise ValueError("Not an offline joint teacher/student pretrain archive")
    if canonical_state_hash(joint) != expected_joint_state_sha256:
        raise ValueError("Offline checkpoint differs from its pretrain receipt")
    teacher = {key[2:]: value for key, value in joint.items() if key.startswith("T.")}
    shapes = {
        "encoder.weight_ih_l0": (96, channels), "encoder.weight_hh_l0": (96, 32),
        "encoder.bias_ih_l0": (96,), "encoder.bias_hh_l0": (96,),
        "head.weight": (20, 32), "head.bias": (20,),
    }
    if set(teacher) != TEACHER_KEYS:
        raise ValueError("Expected exactly the fixed GRU32 teacher state")
    if any(a.shape != shapes[k] or a.dtype != np.dtype("float64") or
           not np.isfinite(a).all() for k, a in teacher.items()):
        raise ValueError("Unexpected teacher shape, precision or nonfinite state")
    return {"checkpoint_sha256": sha(path),
            "teacher_initial_state_sha256": canonical_state_hash(teacher),
            "joint_pretrain_state_sha256": canonical_state_hash(joint),
            "teacher_parameters": sum(a.size for a in teacher.values())}


def historical(study):
    b1_path = study / "provenance/SOLID_B1_PROTOCOL.json"
    d1_path = study / "provenance/D1_PROTOCOL.json"
    if sha(b1_path) != ORIGINAL_B1_SHA256 or sha(d1_path) != ORIGINAL_D1_SHA256:
        raise ValueError("The original historical protocol identity has changed")
    b1, d1 = read_json(b1_path), read_json(d1_path)
    for rel, expected in {**b1["source_sha256"], **d1["source_sha256"]}.items():
        if sha(study / rel) != expected:
            raise ValueError(f"Frozen historical source differs: {rel}")
    return b1_path, d1_path, b1, d1


def d1_identity(rrc_root, original):
    """Accept an unchanged original D1 or the existing reproduce_rrc replay format."""
    candidates = [rrc_root / "provenance/REPLAY_PROTOCOL.json",
                  rrc_root / "provenance/D1_PROTOCOL.json"]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise ValueError("D1 input needs exactly one original or new replay protocol")
    path = found[0]
    cfg = read_json(path)
    is_original = sha(path) == ORIGINAL_D1_SHA256
    if not is_original and cfg.get("replication", {}).get("original_protocol_sha256") != ORIGINAL_D1_SHA256:
        raise ValueError("The D1 input is not a declared replication of the fixed D1")
    actual, wanted = copy.deepcopy(cfg), copy.deepcopy(original)
    for item in (actual, wanted):
        for key in ("study_root", "created_utc", "deadline_utc", "phase", "replication"):
            item.pop(key, None)
        for market in item["markets"]:
            market.pop("path", None)
    if actual != wanted:
        raise ValueError("D1 algorithm, data identity or selection protocol changed")
    for rel, expected in original["source_sha256"].items():
        if sha(rrc_root / rel) != expected:
            raise ValueError(f"D1 source bytes differ: {rel}")
    return path, cfg, is_original


def new_destination(path):
    out = path.expanduser().resolve()
    if out.exists() or path.is_symlink():
        raise FileExistsError("Use a fresh output directory; no overwrite or restart")
    return out


def source_data_paths(b1, previous_study, eex_development):
    result = {}
    for entry in b1["jobs"]:
        name = entry["market"]
        path = (eex_development if name == "EEX_EUA_T3PA"
                else previous_study / "data" / OLD_FILES[name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing approved development input for {name}: {path}")
        result[name] = path
    return result


def required_paths(b1, d1, rrc_root):
    result = []
    for job in b1["jobs"]:
        folder = rrc_root / "local_only/D1" / f"{job['market']}_s{job['seed']}"
        result.extend([folder / "STARTED.json", folder / "PRETRAIN_SELECTION.json"])
        for shrink in d1["pretrain_shrink"]:
            result.extend([folder / f"pretrain_shrink{shrink}.json",
                           folder / f"pretrain_shrink{shrink}.npz"])
    for path in result:
        if not path.is_file():
            raise FileNotFoundError(f"Missing offline D1 input: {path}")
    return result


def collect_inputs(b1, d1, rrc_root, d1_protocol_path, data_paths):
    """Hash development bytes, but never parse prices or construct training data."""
    jobs, comparison = [], []
    for name, path in data_paths.items():
        expected = next(e["data_sha256"] for e in b1["jobs"] if e["market"] == name)
        if sha(path) != expected:
            raise ValueError(f"Development file identity differs: {name}")
    for old in b1["jobs"]:
        folder = rrc_root / "local_only/D1" / f"{old['market']}_s{old['seed']}"
        started = read_json(folder / "STARTED.json")
        if (started["seed"] != old["seed"] or started["market"] != old["market"] or
                started["protocol_sha256"] != sha(d1_protocol_path)):
            raise ValueError("D1 job's start receipt is not tied to the input protocol")
        choices = []
        for shrink in d1["pretrain_shrink"]:
            receipt = read_json(folder / f"pretrain_shrink{shrink}.json")
            score = float(receipt["validation_score"])
            if (not math.isfinite(score) or receipt["seed"] != old["seed"] or
                    receipt["shrink"] != shrink or receipt["cell"] != "gru"):
                raise ValueError("Unexpected offline pretraining receipt")
            choices.append((score, -shrink, shrink, receipt))
        score, _, shrink, receipt = min(choices, key=lambda item: item[:2])
        selection = read_json(folder / "PRETRAIN_SELECTION.json")
        if (selection["shrink"] != shrink or selection["score"] != score or
                selection["rule"] != SELECTION_RULE):
            raise ValueError("D1 selected state differs from its fixed validation selector")
        checkpoint = folder / f"pretrain_shrink{shrink}.npz"
        identity = checkpoint_identity(
            checkpoint, channels=5 if old["with_auction"] else 2,
            expected_joint_state_sha256=receipt["final_state_sha256"])
        entry = {**old, "data_path": str(data_paths[old["market"]]),
                 "checkpoint_path": str(checkpoint.resolve()), "selected_shrink": shrink,
                 "checkpoint_sha256": identity["checkpoint_sha256"],
                 "teacher_initial_state_sha256": identity["teacher_initial_state_sha256"]}
        jobs.append(entry)
        comparison.append({
            "market": old["market"], "seed": old["seed"],
            "checkpoint_relative_to_rrc": str(checkpoint.relative_to(rrc_root)),
            "selected_shrink": shrink, "original_selected_shrink": old["selected_shrink"],
            "same_selected_shrink_as_original_B1": shrink == old["selected_shrink"],
            **identity, "original_checkpoint_sha256": old["checkpoint_sha256"],
            "original_teacher_initial_state_sha256": old["teacher_initial_state_sha256"],
            "same_checkpoint_file_as_original_B1": identity["checkpoint_sha256"] == old["checkpoint_sha256"],
            "same_teacher_state_as_original_B1": identity["teacher_initial_state_sha256"] == old["teacher_initial_state_sha256"],
            "selection_receipt_sha256": sha(folder / "PRETRAIN_SELECTION.json"),
            "pretrain_receipt_sha256": sha(folder / f"pretrain_shrink{shrink}.json"),
        })
    return jobs, comparison


def prepare(study, args):
    b1_path, d1_path, b1, d1 = historical(study)
    rrc_root, previous = args.rrc_replay.resolve(), args.previous_study.resolve()
    input_protocol, _, input_is_original = d1_identity(rrc_root, d1)
    out = new_destination(args.output)
    if not 1 <= args.budget_minutes <= 240:
        raise ValueError("Use a finite replication deadline of 1 to 240 minutes")
    data_paths = source_data_paths(b1, previous, args.eex_development.resolve())
    paths = required_paths(b1, d1, rrc_root)
    scope = {
        "mode": "new_replication_of_published_SOLID_application_reference",
        "is_original_B1_execution": False, "is_external_preregistration": False,
        "is_confirmation": False, "confirmation_access": False,
        "does_not_reverse_RRC_stop": True, "original_B1_protocol_sha256": ORIGINAL_B1_SHA256,
        "D1_input_protocol_sha256": sha(input_protocol), "D1_input_is_original_run": input_is_original,
        "new_backbone_fits_in_this_entry": 0, "training_performed_by_prepare": False,
        "original_B1_counts": b1["counts"], "budget_minutes": args.budget_minutes,
        "output": str(out), "historical_results_copied": False,
        "checkpoint_policy": "Use the input D1's validated offline selection, never its online final teacher; disclose any difference from original B1.",
    }
    if args.paths_only:
        print(json.dumps({"paths_only": True, "no_output_created": True,
                          "development_file_contents_read": False,
                          "checkpoint_contents_read": False,
                          "data_hashes_verified": False, "state_hashes_verified": False,
                          "existing_offline_input_paths": len(paths), **scope}, indent=2))
        return
    jobs, comparison = collect_inputs(b1, d1, rrc_root, input_protocol, data_paths)
    scope["same_teacher_states_as_original_B1"] = all(e["same_teacher_state_as_original_B1"] for e in comparison)
    scope["same_checkpoint_files_as_original_B1"] = all(e["same_checkpoint_file_as_original_B1"] for e in comparison)
    if args.require_original_state and not scope["same_teacher_states_as_original_B1"]:
        raise ValueError("At least one new D1 teacher state differs from original B1")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "no_output_created": True, "market_fits": 0,
                          "development_byte_hashes_verified": True,
                          "prices_parsed": False, "state_identity_comparison": comparison,
                          **scope}, ensure_ascii=False, indent=2))
        return
    out.mkdir(parents=True, exist_ok=False)
    for rel, expected in b1["source_sha256"].items():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(study / rel, dest)
        if sha(dest) != expected:
            raise ValueError(f"Copy changed a frozen runtime source: {rel}")
    # Only source/protocol metadata are copied. Existing states remain explicit
    # read-only inputs from the user's D1 directory, never copied old forecasts.
    shutil.copyfile(Path(__file__).resolve(), out / "source/reproduce_solid.py")
    provenance = out / "provenance"
    provenance.mkdir(exist_ok=True)
    shutil.copyfile(b1_path, provenance / "ORIGINAL_SOLID_B1_PROTOCOL.json")
    shutil.copyfile(d1_path, provenance / "ORIGINAL_D1_PROTOCOL.json")
    now = datetime.now(timezone.utc)
    cfg = copy.deepcopy(b1)
    cfg.update(study_root=str(out), phase="SOLID_REPLAY", jobs=jobs,
               created_utc=now.isoformat(),
               deadline_utc=(now + timedelta(minutes=args.budget_minutes)).isoformat(),
               replication=scope)
    protocol = provenance / "SOLID_REPLAY_PROTOCOL.json"
    write_json(protocol, cfg)
    write_json(provenance / "INITIAL_STATE_COMPARISON.json", {
        **scope, "jobs": comparison,
        "meaning": "File identity and canonical teacher state identity are separate; differences are disclosed, not repaired with historical weights."})
    write_json(provenance / "PORTABLE_INPUT_REGISTRY.json", {
        "rebuild_paths_with": "prepare --rrc-replay ... --previous-study ... --eex-development ... --output NEW",
        "D1_relative_protocol": str(input_protocol.relative_to(rrc_root)),
        "old_development_relative_files": {k: "data/" + v for k, v in OLD_FILES.items()},
        "EEX_development_path_supplied_by_user": True,
        "checkpoints": [{"market": e["market"], "seed": e["seed"],
                         "relative_to_rrc": e["checkpoint_relative_to_rrc"],
                         "checkpoint_sha256": e["checkpoint_sha256"],
                         "teacher_initial_state_sha256": e["teacher_initial_state_sha256"]}
                        for e in comparison],
        "resolved_paths_are_in_new_protocol_only": True,
        "historical_protocol_paths_are_archival_not_opened": True,
    })
    (out / ".gitignore").write_text("local_only/\n__pycache__/\n*.pyc\n", encoding="utf-8")
    prepared = {**scope, "created_utc": now.isoformat(), "status": "PREPARED_NOT_EXECUTED",
                "protocol_sha256": sha(protocol),
                "entry_point_sha256": sha(out / "source/reproduce_solid.py"),
                "frozen_runtime_source_sha256": b1["source_sha256"],
                "state_comparison_sha256": sha(provenance / "INITIAL_STATE_COMPARISON.json"),
                "input_registry_sha256": sha(provenance / "PORTABLE_INPUT_REGISTRY.json"),
                "deadline_policy": "Deadline begins at preparation. If expired, prepare another fresh directory; never edit a prepared protocol.",
                "execution_command": [sys.executable, "-B", str(out / "source/reproduce_solid.py"),
                                      "run", "--protocol", str(protocol)]}
    write_json(out / "PREPARED.json", prepared)
    print(json.dumps(prepared, ensure_ascii=False, indent=2))


def run(protocol):
    protocol = protocol.resolve()
    out = protocol.parent.parent
    cfg, prepared = read_json(protocol), read_json(out / "PREPARED.json")
    if (sha(protocol) != prepared["protocol_sha256"] or
            sha(out / "source/reproduce_solid.py") != prepared["entry_point_sha256"] or
            prepared["status"] != "PREPARED_NOT_EXECUTED"):
        raise ValueError("Prepared replay identity has changed")
    original = out / "provenance/ORIGINAL_SOLID_B1_PROTOCOL.json"
    if sha(original) != ORIGINAL_B1_SHA256:
        raise ValueError("Historical B1 protocol copy differs")
    template = read_json(original)
    actual, wanted = copy.deepcopy(cfg), copy.deepcopy(template)
    for item in (actual, wanted):
        for key in ("study_root", "phase", "created_utc", "deadline_utc", "replication"):
            item.pop(key, None)
        for entry in item["jobs"]:
            for key in ("data_path", "checkpoint_path", "checkpoint_sha256",
                        "teacher_initial_state_sha256", "selected_shrink"):
                entry.pop(key)
    if actual != wanted or cfg["phase"] != "SOLID_REPLAY" or Path(cfg["study_root"]).resolve() != out:
        raise ValueError("Replay changed the fixed SOLID design or directory")
    for rel, expected in template["source_sha256"].items():
        if sha(out / rel) != expected:
            raise ValueError(f"Frozen runtime source changed: {rel}")
    for rel, key in (("provenance/INITIAL_STATE_COMPARISON.json", "state_comparison_sha256"),
                     ("provenance/PORTABLE_INPUT_REGISTRY.json", "input_registry_sha256")):
        if sha(out / rel) != prepared[key]:
            raise ValueError("Prepared input/state identity record changed")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(cfg["deadline_utc"]):
        raise TimeoutError("Prepared deadline expired; no execution or silent renewal")
    if (out / "results/SOLID_REPLAY").exists() or (out / "local_only/SOLID_REPLAY").exists():
        raise FileExistsError("Replay already started; no repeat execution")
    # This is the only path that executes real query-local adaptations.
    subprocess.run([sys.executable, "-B", str(out / "source/run_solid_reference.py"),
                    "--protocol", str(protocol)], check=True, cwd=out)
    complete = read_json(out / "results/SOLID_REPLAY/COMPLETE.json")
    required = {"jobs": 10, "streams": 60, "prediction_series": 80,
                "equivalent_derived_series": 20, "query_local_adaptations": 25332,
                "new_backbone_pretraining_fits": 0, "metrics": 480,
                "confirmation_data_read": False, "timing_only": False}
    if any(complete.get(key) != value for key, value in required.items()):
        raise ValueError("Replay is incomplete or differs from the declared finite run")
    write_json(out / "REPLAY_COMPLETE.json", {
        **prepared["original_B1_counts"], "is_original_B1_execution": False,
        "is_confirmation": False, "confirmation_access": False,
        "protocol_sha256": sha(protocol),
        "completion_sha256": sha(out / "results/SOLID_REPLAY/COMPLETE.json"),
        "same_teacher_states_as_original_B1": prepared["same_teacher_states_as_original_B1"],
        "same_checkpoint_files_as_original_B1": prepared["same_checkpoint_files_as_original_B1"],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "New SOLID reference replication; no D1 refit, no new innovation claim or reopening of the RRC stop decision. This entry does not label itself an independent statistical audit.",
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, default=Path(__file__).resolve().parent.parent)
    sub = ap.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="Verify inputs and prepare only; never fit")
    prep.add_argument("--rrc-replay", type=Path, required=True)
    prep.add_argument("--previous-study", type=Path, required=True)
    prep.add_argument("--eex-development", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--budget-minutes", type=float, default=40.)
    mode = prep.add_mutually_exclusive_group()
    mode.add_argument("--paths-only", action="store_true", help="No data/checkpoint contents read; no output")
    mode.add_argument("--dry-run", action="store_true", help="Verify bytes/states; no output or execution")
    prep.add_argument("--require-original-state", action="store_true",
                      help="Fail if any selected new teacher differs from the historical B1 teacher")
    execute = sub.add_parser("run", help="Explicitly execute a previously prepared NEW replication")
    execute.add_argument("--protocol", type=Path, required=True)
    args = ap.parse_args()
    if args.action == "prepare":
        prepare(args.study.resolve(), args)
    else:
        run(args.protocol)


if __name__ == "__main__":
    main()
