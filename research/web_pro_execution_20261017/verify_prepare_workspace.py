#!/usr/bin/env python3
"""Exercise isolated recovery and synthetic/support CLIs; never run market fits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--approved-local-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Verification output already exists")
    source, approved_root = args.source_dir.resolve(), args.approved_local_dir.resolve()
    stage = source / "research/web_pro_execution_20261017"
    spec = importlib.util.spec_from_file_location("workspace_prepare_under_test", stage / "prepare_workspace.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    checks, commands = [], []

    def record(name, **details):
        checks.append({"check": name, "passed": True, **details})

    def rejected(name, action, expected_text=None):
        try:
            action()
        except (entry.PreparationError, FileExistsError) as error:
            if expected_text is not None:
                assert expected_text in str(error), (name, str(error), expected_text)
            record(name, error_type=type(error).__name__, error_matched=expected_text)
        else:
            raise AssertionError("An unsafe operation was accepted: " + name)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    frozen = {}
    for label in ("A", "B"):
        existing = stage / f"STAGE_{label}_PRE_RUN.json"
        if existing.exists():
            row = json.loads(existing.read_text(encoding="utf-8"))
            frozen.update(row["source_sha256"])
    before = {name: digest(source / name) for name in frozen}
    assert before == frozen, "Previously frozen source already differs"
    index_bytes = entry.regular_bytes(stage, entry.INDEX)
    index = entry.parse_index(index_bytes)
    record("pinned_index", files=21, sha256=entry.INDEX_SHA256)
    rejected("changed_index_rejected", lambda: entry.parse_index(index_bytes + b" "), "SHA256 mismatch")
    for value in ("../escape", "/absolute", "a/../../escape", "a\\escape", "a//b", "a/./b", "C:/escape", "a/\x00b"):
        rejected("unsafe_relative_path", lambda value=value: entry.relative_path(value))
    with tempfile.TemporaryDirectory(prefix="carbon-reproduce-check-") as temporary:
        base = Path(temporary)
        raw = base / "size_fixture.csv"
        raw.write_bytes(b"x" * 29205)  # Synthetic bytes, not a later market observation.
        with patch.object(entry.os, "open", side_effect=AssertionError("Unexpected content read")):
            rejected("oversized_csv_refused_before_open", lambda: entry.regular_bytes(
                base, raw.name, 29204, entry.APPROVED_SHA256), "before read")
        tiny = base / "hash_fixture.txt"
        tiny.write_bytes(b"abc")
        rejected("same_size_wrong_hash", lambda: entry.regular_bytes(base, tiny.name, 3, "0" * 64), "SHA256 mismatch")
        link = base / "symbolic.txt"
        link.symlink_to(tiny)
        rejected("symlink_source_refused", lambda: entry.regular_bytes(base, link.name), "Symbolic link")
        occupied = base / "occupied"
        occupied.mkdir()
        witness = occupied / "witness.txt"
        witness.write_bytes(b"must remain identical\n")
        witness_hash = digest(witness)
        rejected("existing_directory_refused", lambda: entry.prepare(source, occupied, approved_root), "already exists")
        assert digest(witness) == witness_hash and len(list(occupied.iterdir())) == 1
        rejected("existing_file_refused", lambda: entry.prepare(source, witness, approved_root), "already exists")
        rejected("output_inside_source_refused", lambda: entry.destination(
            source / "reproduction-must-not-be-created", source, approved_root), "isolated outside")
        dangling = base / "dangling-output"
        dangling.symlink_to(base / "absent-target", target_is_directory=True)
        rejected("dangling_output_symlink_refused", lambda: entry.prepare(source, dangling, approved_root), "Symbolic link")
        output = base / "restored"
        # A network call here is a test failure, even if it would target a pinned URL.
        with patch.object(entry.urllib.request, "build_opener", side_effect=AssertionError("Unexpected network")):
            dry_network = entry.prepare(source, output, dry_run=True)
            assert dry_network["network_requests_performed"] == 0 and not output.exists()
            record("network_dry_run_has_no_network_or_writes")
            dry_local = entry.prepare(source, output, approved_root, dry_run=True)
            assert dry_local["approved_files_verified"] == 21 and not output.exists()
            record("local_dry_run_verifies_21_without_writes")
            restored = entry.prepare(source, output, approved_root)
            assert restored["status"] == "COMPLETE" and restored["network_requests_performed"] == 0
            for name, expected in restored["output_file_sha256"].items():
                assert digest(output / name) == expected
            restored_stage = output / entry.STAGE
            assert not (restored_stage / "local_only").exists()
            assert not (restored_stage / "STAGE_A_PRE_RUN.json").exists()
            assert not (restored_stage / "STAGE_B_PRE_RUN.json").exists()
            assert not (restored_stage / "STAGE_A_RESULTS.json").exists()
            assert not (restored_stage / "STAGE_B_RESULTS.json").exists()
            assert not list(output.rglob("*.npz"))
            assert (output / entry.APPROVED_CSV).stat().st_size == 29204
            record("complete_local_recovery", approved_files=21, source_files=restored["source_file_count"],
                   output_files=restored["files_written"], all_written_hashes_rechecked=True,
                   old_stage_results_copied=False, point_arrays_copied=False)
            rejected("second_prepare_refused", lambda: entry.prepare(source, output, approved_root), "already exists")
            damaged = base / "damaged-approved-cache"
            damaged.mkdir()
            for row in index["files"]:
                target = damaged / row["runtime_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                content = entry.regular_bytes(approved_root, row["runtime_path"], row["bytes"], row["sha256"])
                target.write_bytes(content)
            victim = damaged / index["files"][0]["runtime_path"]
            content = victim.read_bytes()
            victim.write_bytes(bytes([content[0] ^ 1]) + content[1:])
            bad_output = base / "damaged-output"
            rejected("corrupt_approved_source_stops_before_output", lambda: entry.prepare(
                source, bad_output, damaged), "SHA256 mismatch")
            assert not bad_output.exists()

        # Controlled responses only: no real HTTP request occurs in these checks.
        row = index["files"][0]
        payload = entry.regular_bytes(approved_root, row["runtime_path"], row["bytes"], row["sha256"])

        class Response:
            status = 200
            def __init__(self, length):
                self.headers = {} if length is None else {"Content-Length": str(length)}
                self.read_sizes = []
            def geturl(self):
                return row["raw_url"]
            def read(self, count):
                self.read_sizes.append(count)
                assert count == row["bytes"], "Read beyond fixed length"
                return payload
            def __enter__(self):
                return self
            def __exit__(self, *unused):
                return False

        class Opener:
            def __init__(self, response):
                self.response, self.urls = response, []
            def open(self, request, timeout):
                self.urls.append(request.full_url)
                assert request.full_url == row["raw_url"] and timeout == 30
                return self.response

        response = Response(row["bytes"])
        opener = Opener(response)
        assert entry.download_approved(row, opener) == payload
        assert response.read_sizes == [row["bytes"]] and len(opener.urls) == 1
        record("mock_http_exact_url_length_and_hash", real_network_requests=0, read_limit=row["bytes"])
        for length in (None, row["bytes"] + 1):
            response = Response(length)
            rejected("mock_http_length_refused_without_body_read", lambda response=response:
                     entry.download_approved(row, Opener(response)), "Content-Length")
            assert not response.read_sizes
        request = entry.urllib.request.Request(row["raw_url"])
        rejected("redirect_refused_without_followup_request", lambda: entry.NoRedirect().redirect_request(
            request, None, 302, "Moved", {}, "https://example.invalid/unapproved.csv"), "redirect refused")

        cli_output = base / "cli-restored"
        cli = subprocess.run([
            sys.executable, "-B", str(stage / "prepare_workspace.py"),
            "--source-dir", str(source), "--approved-local-dir", str(approved_root),
            "--output", str(cli_output),
        ], text=True, capture_output=True, timeout=30, check=False)
        assert cli.returncode == 0, cli.stderr
        cli_result = json.loads(cli.stdout)
        assert cli_result["status"] == "COMPLETE" and cli_result["approved_files_verified"] == 21
        assert cli_result["network_requests_performed"] == 0
        record("actual_prepare_cli_local_recovery_passes", approved_files=21, network_requests=0,
               source_files=cli_result["source_file_count"])
        restored_stage = cli_output / entry.STAGE

        def command(arguments):
            result = subprocess.run([sys.executable, "-B", *arguments], cwd=restored_stage,
                                    text=True, capture_output=True, timeout=60, check=False)
            if result.returncode:
                raise AssertionError("Command failed: " + " ".join(arguments) + "\n" + result.stderr[-4000:])
            serialized = result.stdout
            if arguments == ["stage_b.py", "check"]:
                assert serialized.startswith("CHECK "), "Unexpected Stage B check output prefix"
                serialized = serialized[len("CHECK "):]
            value = json.loads(serialized)
            commands.append({"argv": ["python", "-B", *arguments], "exit_code": result.returncode,
                             "market_fits_performed": 0})
            return value

        environment = command(["check_environment.py"])
        assert environment["passed"]
        record("copied_environment_checker_passes", core=environment["observed"]["imported_versions"])
        a = command(["stage_a.py", "check"])
        assert a["passed"]
        record("copied_stage_a_synthetic_check_passes")
        support_a = command(["audit/verify_stage_a.py", "--support-only"])
        assert support_a["rw"]["main_origins"] == 451 and support_a["source"]["records"] == 950
        record("copied_stage_a_support_audit_passes", market_records=950, score_origins=451)
        b = command(["stage_b.py", "check"])
        assert b["passed"]
        record("copied_stage_b_synthetic_check_passes")
        support_b = command(["audit/verify_stage_b.py", "--support-only"])
        assert support_b["source"]["records"] == 950
        record("copied_stage_b_support_audit_passes", market_records=950)
        # Synthetic checks may create synthetic receipts/cache, never market plans.
        assert not list(restored_stage.glob("STAGE_*_PRE_RUN.json"))
        assert not list(restored_stage.glob("STAGE_*_TIMING_PLAN.json"))
        assert not list(restored_stage.glob("stage_*_receipts"))
        record("no_market_timing_or_formal_run_artifacts_created")
    after = {name: digest(source / name) for name in frozen}
    assert after == before
    record("original_formal_source_hashes_unchanged", frozen_paths_checked=len(before))
    result = {
        "passed": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Actual isolated local restoration, controlled HTTP response checks, environment and synthetic/support CLIs",
        "real_network_requests": 0, "network_download_route_retested_against_github": False,
        "market_fits_performed": 0, "full_training_reproduction_claimed": False,
        "point_predictions_or_market_weights_saved_in_public_validation": False,
        "temporary_restoration_removed": True, "checks": checks, "commands": commands,
        "prepare_workspace_sha256": digest(stage / "prepare_workspace.py"),
        "source_manifest_sha256": digest(stage / entry.BUNDLE_MANIFEST),
        "verifier_sha256": digest(Path(__file__)),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "checks": len(checks), "commands": len(commands),
                      "market_fits_performed": 0, "real_network_requests": 0}, indent=2))


if __name__ == "__main__":
    main()
