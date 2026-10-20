#!/usr/bin/env python3
"""Artificial-file tests of the reproduction preparer; no market files or fits.

Temporary synthetic protocol/data identities replace constants only in an
imported test module. The distributed helper and frozen runtime bytes are never
edited. Child process launch is intercepted rather than executed.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2)+"\n").encode()


def load(path):
    spec = importlib.util.spec_from_file_location("synthetic_reproduction_preparer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(helper_path):
    helper = load(helper_path)
    labels = []
    def require(value, label):
        if not value:
            raise AssertionError(label)
        labels.append(label)
    def rejects(func, expected, label):
        try:
            func()
        except expected:
            labels.append(label)
        else:
            raise AssertionError("Expected rejection: "+label)
    with tempfile.TemporaryDirectory(prefix="prior_repro_artificial_") as tmp:
        top = Path(tmp)
        repo = top/"source_repo"
        study = repo/helper.PREFIX
        study.mkdir(parents=True)
        source_bytes = {f"source/runtime_{i:02d}.py": f"# Artificial file {i}\n".encode() for i in range(11)}
        protocol = dict(source_sha256={name: sha(raw) for name, raw in source_bytes.items()},
                        year=2022, parent_commit="SYNTHETIC_NOT_A_REAL_COMMIT")
        source_bytes["provenance/PROTOCOL.json"] = encoded(protocol)
        source_bytes["methods/METHOD_ZH.md"] = b"Artificial method fixture; no market model.\n"
        source_bytes["source/prepare_reproduction.py"] = b"# Not executed: artificial copied preparer marker\n"
        source_bytes["audits/ORIGINAL_CHECK.json"] = encoded(dict(synthetic=True))
        parent_rows = []
        for i in range(32):
            name = f"research/parent_artificial/source_{i:02d}.txt"
            raw = f"synthetic parent bytes {i}\n".encode()
            helper.write_new(repo/name, raw)
            parent_rows.append(dict(path=name, **helper.identity(raw)))
        parent_blob = encoded(dict(count=32, files=parent_rows))
        source_bytes[helper.PARENT_INDEX] = parent_blob
        helper.PROTOCOL_SHA256 = sha(source_bytes["provenance/PROTOCOL.json"])
        helper.METHOD_SHA256 = sha(source_bytes["methods/METHOD_ZH.md"])
        helper.DEPENDENCIES_SHA256 = sha(parent_blob)
        source_rows = []
        for name, raw in source_bytes.items():
            helper.write_new(study/name, raw)
            destination = "historical/pre_run_evidence/"+name if name.startswith("audits/") else name
            source_rows.append(dict(source_path=name, destination_path=destination, **helper.identity(raw)))
        source_blob = encoded(dict(protocol_sha256=helper.PROTOCOL_SHA256, files=source_rows))
        helper.write_new(study/helper.SOURCES_INDEX, source_blob)
        # Intentionally excluded artifacts: preparation must never copy these.
        helper.write_new(study/"results/OLD_RESULT.json", b'{"synthetic_old_result":true}\n')
        helper.write_new(study/"local_only/old_prediction.npz", b"not a real npz\n")
        eex = top/"artificial_development.csv"
        eex_blob = b"date,price\n2000-01-03,123\n"
        eex.write_bytes(eex_blob)
        helper.EEX_SHA256 = sha(eex_blob)
        def args(name, dry=False):
            return SimpleNamespace(source_dir=repo, output=top/name, eex_dev=eex, dry_run=dry)
        quiet = contextlib.redirect_stdout(io.StringIO())
        with quiet:
            plan = helper.prepare(args("dry", True))
        require(not (top/"dry").exists(), "dry_run_creates_no_directory")
        require(plan["network_requests"] == plan["market_fits"] == plan["confirmation_inputs"] == 0,
                "prepare_declares_no_network_fit_or_confirmation")
        with contextlib.redirect_stdout(io.StringIO()):
            helper.prepare(args("replay"))
        output = top/"replay"
        copied = output/helper.PREFIX
        require(all((copied/name).read_bytes() == raw for name, raw in source_bytes.items() if not name.startswith("audits/")),
                "all_synthetic_runtime_and_protocol_bytes_preserved")
        require((copied/"historical/pre_run_evidence/audits/ORIGINAL_CHECK.json").read_bytes() == source_bytes["audits/ORIGINAL_CHECK.json"],
                "original_receipt_archived_without_claiming_new_pass")
        require(not (copied/"results").exists() and not (copied/"local_only").exists(), "old_results_and_predictions_not_copied")
        require((output/"local_only/inputs/EEX_EUA_T3PA_daily_development.csv").read_bytes() == eex_blob, "only_explicit_development_input_copied")
        require("local_only/" in (output/".gitignore").read_text(), "private_directory_ignored")
        require(all((output/row["path"]).read_bytes() == (repo/row["path"]).read_bytes() for row in parent_rows),
                "all_32_parent_fixture_bytes_and_relative_paths_preserved")
        commands = json.loads((output/"REPRODUCTION_COMMANDS.json").read_text())
        require("results/D1" in commands["independent_audit"] and "audits/D1_independent" in commands["independent_audit"], "audit_paths_match_original_layout")
        require("source/analyze_results.py" in commands["aggregate_tables"] and "--no-plots" in commands["aggregate_tables"], "aggregate_cli_present")
        rejects(lambda: helper.prepare(args("replay")), FileExistsError, "existing_output_rejected")
        for name in ("../bad", "/absolute", "x/../bad", "x\\bad", "", "x//bad"):
            rejects(lambda value=name: helper.relative(value), ValueError, "unsafe_manifest_path_"+repr(name))
        rejects(lambda: helper.prepare(SimpleNamespace(**{**vars(args("unused")), "output": repo/"new"})), ValueError, "output_inside_source_rejected")
        rejects(lambda: helper.new_output(top/"sibling"/".."/"source_repo"/"new"), ValueError, "parent_traversal_containment_bypass_rejected")
        eex.write_bytes(eex_blob+b"changed\n")
        rejects(lambda: helper.prepare(args("wrong_eex")), ValueError, "changed_eex_identity_rejected_before_creation")
        require(not (top/"wrong_eex").exists(), "wrong_eex_leaves_no_output")
        eex.write_bytes(eex_blob)
        victim = repo/parent_rows[0]["path"]
        old = victim.read_bytes(); victim.write_bytes(b"changed\n")
        rejects(lambda: helper.prepare(args("wrong_parent")), ValueError, "changed_parent_identity_rejected")
        victim.write_bytes(old)
        runtime = study/"source/runtime_00.py"
        old = runtime.read_bytes(); runtime.write_bytes(b"changed\n")
        rejects(lambda: helper.read_indices(repo), ValueError, "changed_frozen_runtime_rejected")
        runtime.write_bytes(old)
        duplicate = json.loads(source_blob); duplicate["files"].append(duplicate["files"][0])
        (study/helper.SOURCES_INDEX).write_bytes(encoded(duplicate))
        rejects(lambda: helper.read_indices(repo), ValueError, "duplicate_allowlist_rejected")
        (study/helper.SOURCES_INDEX).write_bytes(source_blob)
        link = top/"symlink_output_parent"
        link.symlink_to(top/"source_repo", target_is_directory=True)
        rejects(lambda: helper.new_output(link/"unused"), ValueError, "symlink_output_ancestor_rejected")
        alias = repo/"alias_source.txt"; alias.symlink_to(victim)
        rejects(lambda: helper.regular(repo, "alias_source.txt"), ValueError, "symlink_source_rejected")
        captured = []
        real_run = helper.subprocess.run
        helper.subprocess.run = lambda command, **kwargs: captured.append((command, kwargs))
        try:
            helper.run(SimpleNamespace(workspace=output))
            require(len(captured) == 1 and captured[0][0][3] == "run", "one_explicit_child_command_intercepted_not_executed")
            require(str(copied/"results/D1") in captured[0][0] and str(output/"local_only/D1") in captured[0][0], "run_child_paths_match_auditor_and_analysis")
            require((output/"MARKET_RUN_LAUNCHED.json").is_file(), "launch_marker_saved_before_child")
            rejects(lambda: helper.run(SimpleNamespace(workspace=output)), FileExistsError, "second_launch_rejected_without_retry")
            with contextlib.redirect_stdout(io.StringIO()):
                helper.prepare(args("private_exists"))
            guarded = top/"private_exists"
            (guarded/"local_only/D1").mkdir()
            rejects(lambda: helper.run(SimpleNamespace(workspace=guarded)), FileExistsError, "preexisting_private_run_directory_rejected")
            require(not (guarded/"MARKET_RUN_LAUNCHED.json").exists() and len(captured) == 1, "guard_failure_never_launches_child")
        finally:
            helper.subprocess.run = real_run
    return dict(status="PASS_SYNTHETIC_FILE_PREPARATION", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                checks=len(labels), labels=labels, helper_sha256=sha(helper_path.read_bytes()),
                checker_sha256=sha(Path(__file__).read_bytes()),
                scope=dict(real_market_csv_reads=0, real_result_reads=0, confirmation_reads=0,
                           network_requests=0, model_imports=0, model_fits=0, child_processes_executed=0),
                synthetic_constants_replaced_in_memory_only=True,
                limitation="This tests copying and path/identity guards with artificial bytes; it is not a replay or numerical forecast audit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, default=Path(__file__).with_name("prepare_reproduction.py"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"audits/REPRODUCTION_PREPARATION_CHECK.json")
    args = parser.parse_args()
    result = check(args.helper)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded(result))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
