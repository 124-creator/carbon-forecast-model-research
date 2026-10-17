"""Pure synthetic preparation tests; no market file or model execution."""
from __future__ import annotations
import argparse
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import numpy as np

STUDY = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("reproduce_solid_under_test",
                                             STUDY / "source/reproduce_solid.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)


def main():
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def rejects(name, function, error=ValueError):
        try:
            function()
        except error:
            checks.append(name)
        else:
            raise AssertionError(name)
    rng = np.random.default_rng(2601017)
    with tempfile.TemporaryDirectory(prefix="solid_portability_synthetic_") as tmp:
        root = Path(tmp)
        historic, old_data = root / "synthetic_study", root / "synthetic_previous"
        historic.mkdir(); (old_data / "data").mkdir(parents=True)
        b1 = rep.read_json(STUDY / "provenance/SOLID_B1_PROTOCOL.json")
        d1 = rep.read_json(STUDY / "provenance/D1_PROTOCOL.json")
        for rel in {**b1["source_sha256"], **d1["source_sha256"]}:
            dest = historic / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(STUDY / rel, dest)
        data_paths = {}
        for market in d1["markets"]:
            name = market["name"]
            path = (root / "EEX_EUA_T3PA_daily_development.csv" if name == "EEX_EUA_T3PA"
                    else old_data / "data" / rep.OLD_FILES[name])
            path.write_text("date,price\n2000-01-01,1\n2000-01-02,2\n")
            market["path"], market["sha256"] = str(path), rep.sha(path)
            data_paths[name] = path
        d1["study_root"] = str(historic)
        d1_path = historic / "provenance/D1_PROTOCOL.json"
        rep.write_json(d1_path, d1)
        for job in b1["jobs"]:
            folder = historic / "local_only/D1" / f"{job['market']}_s{job['seed']}"
            folder.mkdir(parents=True)
            channels = 5 if job["with_auction"] else 2
            for shrink, score in ((0.1, .9), (1.0, .8)):
                shapes = {"encoder.weight_ih_l0": (96, channels), "encoder.weight_hh_l0": (96, 32),
                          "encoder.bias_ih_l0": (96,), "encoder.bias_hh_l0": (96,),
                          "head.weight": (20, 32), "head.bias": (20,)}
                joint = {"T." + key: rng.normal(size=shape).astype(np.float64)
                         for key, shape in shapes.items()}
                joint["S.synthetic_fixture"] = np.zeros((2,), dtype=np.float64)
                np.savez_compressed(folder / f"pretrain_shrink{shrink}.npz", **joint)
                rep.write_json(folder / f"pretrain_shrink{shrink}.json",
                               {"seed": job["seed"], "shrink": shrink, "cell": "gru",
                                "validation_score": score,
                                "final_state_sha256": rep.canonical_state_hash(joint)})
            rep.write_json(folder / "PRETRAIN_SELECTION.json",
                           {"shrink": 1.0, "score": .8, "rule": rep.SELECTION_RULE})
            rep.write_json(folder / "STARTED.json", {"market": job["market"], "seed": job["seed"],
                                                     "protocol_sha256": rep.sha(d1_path)})
            selected = folder / "pretrain_shrink1.0.npz"
            identity = rep.checkpoint_identity(
                selected, channels=channels,
                expected_joint_state_sha256=rep.read_json(folder / "pretrain_shrink1.0.json")["final_state_sha256"])
            job.update(data_path=str(data_paths[job["market"]]),
                       data_sha256=rep.sha(data_paths[job["market"]]), selected_shrink=1.0,
                       checkpoint_path=str(selected), checkpoint_sha256=identity["checkpoint_sha256"],
                       teacher_initial_state_sha256=identity["teacher_initial_state_sha256"])
        b1["study_root"] = str(historic)
        b1_path = historic / "provenance/SOLID_B1_PROTOCOL.json"
        rep.write_json(b1_path, b1)
        args = argparse.Namespace(rrc_replay=historic, previous_study=old_data,
            eex_development=data_paths["EEX_EUA_T3PA"], output=root / "paths_only",
            budget_minutes=40., paths_only=True, dry_run=False, require_original_state=False)
        # Only the fixture constants are replaced in memory. All real frozen
        # runtime source bytes are unchanged; no synthetic protocol is shipped.
        with patch.object(rep, "ORIGINAL_B1_SHA256", rep.sha(b1_path)), \
             patch.object(rep, "ORIGINAL_D1_SHA256", rep.sha(d1_path)):
            original_open = Path.open
            def forbid_data_open(path, *pargs, **kwargs):
                if Path(path).suffix in (".csv", ".npz"):
                    raise AssertionError("Paths-only tried to read a data/checkpoint file")
                return original_open(path, *pargs, **kwargs)
            captured = io.StringIO()
            with patch.object(Path, "open", forbid_data_open), contextlib.redirect_stdout(captured):
                rep.prepare(historic, args)
            check("paths_only_never_opens_csv_or_npz", json.loads(captured.getvalue())["state_hashes_verified"] is False)
            check("paths_only_creates_no_output", not args.output.exists())
            args.paths_only, args.dry_run = False, True
            with contextlib.redirect_stdout(io.StringIO()):
                rep.prepare(historic, args)
            check("full_dry_run_creates_no_output", not args.output.exists())
            jobs, identities = rep.collect_inputs(b1, d1, historic, d1_path, data_paths)
            check("ten_job_file_and_teacher_identities_match", len(jobs) == 10 and all(
                e["same_checkpoint_file_as_original_B1"] and e["same_teacher_state_as_original_B1"]
                for e in identities))
            selected = Path(jobs[0]["checkpoint_path"])
            with np.load(selected, allow_pickle=False) as archive:
                joint = {key: archive[key].copy() for key in archive.files}
            alternate = root / "same_arrays_uncompressed.npz"
            np.savez(alternate, **joint)
            alt = rep.checkpoint_identity(alternate, channels=5,
                        expected_joint_state_sha256=rep.canonical_state_hash(joint))
            check("npz_file_difference_does_not_imply_state_difference",
                  alt["checkpoint_sha256"] != jobs[0]["checkpoint_sha256"] and
                  alt["teacher_initial_state_sha256"] == jobs[0]["teacher_initial_state_sha256"])
            rejects("checkpoint_wrong_joint_digest_rejected",
                    lambda: rep.checkpoint_identity(selected, channels=5, expected_joint_state_sha256="0"*64))
            rejects("checkpoint_wrong_channels_rejected",
                    lambda: rep.checkpoint_identity(selected, channels=2,
                        expected_joint_state_sha256=rep.canonical_state_hash(joint)))
            online = root / "teacher_final.npz"
            np.savez(online, **{k[2:]: v for k, v in joint.items() if k.startswith("T.")})
            rejects("online_final_archive_not_accepted_as_offline",
                    lambda: rep.checkpoint_identity(online, channels=5,
                                                    expected_joint_state_sha256="0"*64))
            selection_path = selected.parent / "PRETRAIN_SELECTION.json"
            selection_original = selection_path.read_bytes()
            bad = rep.read_json(selection_path); bad["shrink"] = .1
            rep.write_json(selection_path, bad)
            rejects("selection_not_matching_validation_rule_rejected",
                    lambda: rep.collect_inputs(b1, d1, historic, d1_path, data_paths))
            selection_path.write_bytes(selection_original)
            corrupt_data = data_paths["GUANGZHOU"]
            clean = corrupt_data.read_bytes(); corrupt_data.write_bytes(clean+b"\n")
            rejects("development_identity_change_rejected",
                    lambda: rep.collect_inputs(b1, d1, historic, d1_path, data_paths))
            corrupt_data.write_bytes(clean)
            altered = copy.deepcopy(b1); altered["jobs"][0]["teacher_initial_state_sha256"] = "0"*64
            _, changed = rep.collect_inputs(altered, d1, historic, d1_path, data_paths)
            check("different_historical_state_disclosed_without_substitution",
                  changed[0]["same_teacher_state_as_original_B1"] is False and
                  changed[0]["teacher_initial_state_sha256"] == jobs[0]["teacher_initial_state_sha256"])
            args.dry_run, args.output = False, root / "prepared"
            with contextlib.redirect_stdout(io.StringIO()):
                rep.prepare(historic, args)
            check("five_frozen_runtime_sources_copied_exactly", all(
                rep.sha(args.output / rel) == expected for rel, expected in b1["source_sha256"].items()))
            check("no_historical_results_or_checkpoints_copied",
                  not (args.output / "results").exists() and not (args.output / "local_only").exists())
            check("new_protocol_does_not_claim_original_or_confirmation",
                  rep.read_json(args.output / "provenance/SOLID_REPLAY_PROTOCOL.json")["replication"]["is_confirmation"] is False)
            rejects("prepare_refuses_existing_output", lambda: rep.prepare(historic, args), FileExistsError)
            protocol = args.output / "provenance/SOLID_REPLAY_PROTOCOL.json"
            class ExplicitDispatchReached(RuntimeError):
                pass
            dispatch = []
            def no_execution(command, **kwargs):
                dispatch.append((command, kwargs))
                raise ExplicitDispatchReached()
            with patch.object(rep.subprocess, "run", no_execution):
                rejects("only_explicit_run_reaches_original_runner_dispatch",
                        lambda: rep.run(protocol), ExplicitDispatchReached)
            check("dispatch_uses_copied_original_runner_and_new_protocol",
                  len(dispatch) == 1 and dispatch[0][0][2:] == [
                      str(args.output / "source/run_solid_reference.py"), "--protocol", str(protocol)])
            blob = protocol.read_bytes()
            tampered = rep.read_json(protocol); tampered["neighbors"] = 11
            rep.write_json(protocol, tampered)
            rejects("prepared_protocol_tampering_rejected", lambda: rep.run(protocol))
            protocol.write_bytes(blob)
            copied_source = args.output / "source/rrc_core.py"
            source_bytes = copied_source.read_bytes()
            copied_source.write_bytes(source_bytes+b"\n")
            rejects("runtime_source_tampering_rejected", lambda: rep.run(protocol))
            copied_source.write_bytes(source_bytes)
            with patch.object(rep, "datetime") as clock:
                clock.now.return_value = datetime.now(timezone.utc) + timedelta(days=1)
                clock.fromisoformat.side_effect = datetime.fromisoformat
                rejects("expired_prepared_deadline_rejected", lambda: rep.run(protocol), TimeoutError)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
              "scope": "Synthetic fixtures only; subprocess dispatch intercepted before execution",
              "checks": checks, "check_count": len(checks),
              "entry_point_sha256": rep.sha(STUDY / "source/reproduce_solid.py"),
              "test_script_sha256": rep.sha(Path(__file__)),
              "real_market_file_reads": 0, "real_checkpoint_reads": 0,
              "confirmation_reads": 0, "model_fits": 0, "head_adaptations": 0,
              "full_market_replication_performed": False,
              "test_fixture_note": "Only synthetic protocol constants changed in the test module memory; actual frozen study sources/protocols were not edited."}
    rep.write_json(STUDY / "backup_review/SOLID_PORTABILITY_SYNTHETIC.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
