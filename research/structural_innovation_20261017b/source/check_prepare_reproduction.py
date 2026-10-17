"""Source-only and artificial-file checks for the SMP replay preparation helper.

No real CSV, price archive, model state, result, or confirmation file is read.
The real protocol/source bundle is hashed, but no market replay is prepared.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def load_helper(path):
    spec = importlib.util.spec_from_file_location("smp_prepare_reproduction_check_target", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_checks(source_root):
    source_root = Path(source_root).resolve()
    helper_path = source_root / "source/prepare_reproduction.py"
    helper = load_helper(helper_path)
    checks = []

    def require(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(dict(name=name, passed=True))

    def rejects(name, exception, action):
        try:
            action()
        except exception:
            checks.append(dict(name=name, passed=True))
            return
        raise AssertionError(name + ": expected rejection")

    root, protocol, protocol_body, frozen, auxiliary = helper.inspect_source_bundle(source_root)
    require("actual_source_count_12", len(frozen) == 12)
    require("actual_auxiliary_inventory", set(auxiliary) == {"source/vendor/LICENSE", helper.AUDIT_RELATIVE})
    require("actual_protocol_identity", helper.sha256_bytes(protocol_body) == helper.EXPECTED_PROTOCOL_SHA256)
    require("actual_scope_before_2024", protocol["data_end_exclusive"] == "2024-01-01")
    require("actual_exact_five_names", {m["name"] for m in protocol["markets"]} == helper.SERIES)
    require("auxiliary_not_in_training_manifest", not (set(auxiliary) & set(frozen)))
    require("copy_plan_contains_no_results_or_data",
            all(Path(name).parts[0] not in {"local_only", "results", "data"} for name in frozen | auxiliary))
    require("no_original_model_weights_in_plan",
            all(Path(name).suffix not in {".npz", ".npy", ".pt", ".pth"} for name in frozen | auxiliary))

    with tempfile.TemporaryDirectory(prefix="smp_reproduction_artificial_") as tmp_string:
        tmp = Path(tmp_string)
        # Copy only source bytes and original protocol to a temporary source
        # bundle. There is no data map, STARTED record, fit, or market replay.
        clone = tmp / "source_only_clone"
        clone.mkdir()
        helper.write_exclusive_bytes(clone / helper.PROTOCOL_RELATIVE, protocol_body)
        for name, body in {**frozen, **auxiliary}.items():
            helper.write_exclusive_bytes(clone / name, body)
        require("source_only_clone_protocol_bytes_equal",
                (clone / helper.PROTOCOL_RELATIVE).read_bytes() == protocol_body)
        clone_checked = helper.inspect_source_bundle(clone)
        require("source_only_clone_12_training_and_two_auxiliaries",
                len(clone_checked[3]) == 12 and len(clone_checked[4]) == 2)
        require("source_only_clone_no_result_directory", not (clone / "results").exists())
        require("source_only_clone_no_private_directory", not (clone / "local_only").exists())
        require("source_only_clone_no_data_map", not (clone / "data_map.json").exists())
        rejects("exclusive_source_write_refuses_overwrite", FileExistsError,
                lambda: helper.write_exclusive_bytes(clone / helper.PROTOCOL_RELATIVE, b"modified"))
        require("refused_write_keeps_protocol", (clone / helper.PROTOCOL_RELATIVE).read_bytes() == protocol_body)
        # Negative source checks modify only the temporary code copy.
        chosen = next(iter(frozen))
        (clone / chosen).write_bytes(frozen[chosen] + b" ")
        rejects("modified_temporary_source_rejected", ValueError,
                lambda: helper.inspect_source_bundle(clone))
        (clone / chosen).write_bytes(frozen[chosen])
        (clone / "source/vendor/LICENSE").unlink()
        rejects("missing_temporary_vendor_license_rejected", ValueError,
                lambda: helper.inspect_source_bundle(clone))
        helper.write_exclusive_bytes(clone / "source/vendor/LICENSE", auxiliary["source/vendor/LICENSE"])

        fixture = tmp / "fixture"
        fixture.mkdir()
        mapping_file = fixture / "map.json"
        fake_markets, fake_mapping = [], {}
        for index, name in enumerate(sorted(helper.SERIES)):
            filename = "artificial_" + str(index) + ".csv"
            body = ("ARTIFICIAL FILE " + str(index) + "; NO MARKET DATA").encode()
            (fixture / filename).write_bytes(body)
            fake_markets.append(dict(name=name, sha256=helper.sha256_bytes(body)))
            fake_mapping[name] = filename
        mapping_file.write_text(json.dumps(fake_mapping), encoding="utf-8")
        opened = []
        original_hash = helper.hash_local_file

        def tracked_hash(path):
            require("hashed_file_stays_inside_artificial_fixture_" + str(len(opened)),
                    path.is_relative_to(fixture))
            opened.append(path)
            return original_hash(path)

        with patch.object(helper, "hash_local_file", tracked_hash):
            mapping, data_receipt, map_sha = helper.verify_data_mapping(mapping_file, fake_markets)
        require("exactly_five_artificial_files_hashed", len(opened) == 5)
        require("relative_map_paths_resolve_at_json_parent",
                mapping == {name: str((fixture / filename).resolve()) for name, filename in fake_mapping.items()})
        require("output_map_paths_are_absolute", all(Path(value).is_absolute() for value in mapping.values()))
        require("input_map_identity_uses_exact_bytes", map_sha == helper.sha256_bytes(mapping_file.read_bytes()))
        require("five_artificial_receipts", len(data_receipt) == 5)
        mapping_file.write_text(json.dumps(mapping), encoding="utf-8")
        absolute_mapping, _, _ = helper.verify_data_mapping(mapping_file, fake_markets)
        require("absolute_and_relative_map_semantics_match", absolute_mapping == mapping)

        first = sorted(fake_mapping)[0]
        wrong_markets = [dict(m) for m in fake_markets]
        wrong_markets[0]["sha256"] = "0" * 64
        rejects("wrong_artificial_data_hash_rejected", ValueError,
                lambda: helper.verify_data_mapping(mapping_file, wrong_markets))

        def write_map_then_verify(obj):
            mapping_file.write_text(json.dumps(obj), encoding="utf-8")
            return helper.verify_data_mapping(mapping_file, fake_markets)

        rejects("missing_name_rejected", ValueError,
                lambda: write_map_then_verify({key: value for key, value in mapping.items() if key != first}))
        rejects("extra_name_rejected", ValueError,
                lambda: write_map_then_verify({**mapping, "EXTRA": "not_opened.csv"}))
        rejects("non_object_map_rejected", ValueError,
                lambda: write_map_then_verify(list(mapping)))
        rejects("empty_path_rejected", ValueError,
                lambda: write_map_then_verify({**mapping, first: ""}))
        rejects("non_string_path_rejected", ValueError,
                lambda: write_map_then_verify({**mapping, first: 123}))
        rejects("missing_mapped_file_rejected", ValueError,
                lambda: write_map_then_verify({**mapping, first: "absent.csv"}))
        mapping_file.write_text('{"duplicate":1,"duplicate":2}', encoding="utf-8")
        rejects("duplicate_json_keys_rejected", ValueError,
                lambda: helper.verify_data_mapping(mapping_file, fake_markets))

        sealed_fixture = fixture / "sealed_confirmation"
        sealed_fixture.mkdir()
        artificial_forbidden = sealed_fixture / "artificial.csv"
        artificial_forbidden.write_bytes(b"ARTIFICIAL FORBIDDEN DIRECTORY FIXTURE")
        mapping_file.write_text(json.dumps({**mapping, first: str(artificial_forbidden)}), encoding="utf-8")
        with patch.object(helper, "hash_local_file", side_effect=AssertionError("hash must not be called")):
            rejects("confirmation_directory_refused_before_hash", ValueError,
                    lambda: helper.verify_data_mapping(mapping_file, fake_markets))

        existing = tmp / "existing"
        existing.mkdir()
        args = SimpleNamespace(work_root=existing, source_root=source_root,
                               data_map=tmp / "nonexistent_map.json", dry_run=True)
        with patch.object(helper, "inspect_source_bundle", side_effect=AssertionError("source not reached")), \
                patch.object(helper, "verify_data_mapping", side_effect=AssertionError("data not reached")):
            rejects("existing_workroot_refused_before_source_or_data_read", FileExistsError,
                    lambda: helper.prepare(args))
        rejects("existing_file_destination_refused", FileExistsError,
                lambda: helper.new_destination(mapping_file))
        dangling = tmp / "dangling"
        dangling.symlink_to(tmp / "absent_target")
        rejects("dangling_destination_symlink_refused", FileExistsError,
                lambda: helper.new_destination(dangling))
        new_root = helper.new_destination(tmp / "new_work")
        require("new_destination_check_does_not_create_output", not new_root.exists())

        bad_paths = ["../escape.py", "/absolute.py", "source/../escape.py",
                     "source//same.py", "results/x.json", "local_only/x.txt",
                     "data/x.csv", "source/weights.npz", "source/weights.pt",
                     "source" + chr(92) + "escape.py"]
        for index, bad in enumerate(bad_paths):
            rejects("unsafe_copy_path_rejected_" + str(index), ValueError,
                    lambda bad=bad: helper.safe_source_relative(bad))
        require("regular_source_relative_accepted",
                helper.safe_source_relative("source/models.py").as_posix() == "source/models.py")

        commands = helper.replay_commands(tmp / "work with spaces")
        by_name = {entry["name"]: entry for entry in commands}
        require("six_manual_commands_only", len(commands) == 6)
        require("D1_runner_receives_map", "--data-map" in by_name["D1_run"]["argv"])
        require("D2_runner_receives_map", "--data-map" in by_name["D2_run"]["argv"])
        require("D1_independent_output_explicit", "--output" in by_name["D1_independent_results"]["argv"])
        require("D2_independent_uses_independent_D1_selection",
                str(tmp / "work with spaces/audits/D1_independent/SELECTION.json")
                in by_name["D2_independent_results"]["argv"])
        require("D2_requires_audit_and_gate",
                "audit PASS" in by_name["D2_run"]["condition"]
                and "advance_to_D2=true" in by_name["D2_run"]["condition"])
        require("no_confirmation_command",
                all("C" not in entry["argv"] for entry in commands))
        require("shell_paths_with_spaces_are_quoted",
                all("'" in entry["shell"] for entry in commands))

    # Verify the real frozen bundle once more after all temporary negative cases.
    final_bundle = helper.inspect_source_bundle(source_root)
    require("original_protocol_unchanged_after_tests", final_bundle[2] == protocol_body)
    require("original_frozen_sources_unchanged_after_tests", final_bundle[3] == frozen)
    require("original_auxiliaries_unchanged_after_tests", final_bundle[4] == auxiliary)
    return dict(status="PASS", created_utc=datetime.now(timezone.utc).isoformat(),
                checks=len(checks), check_details=checks,
                protocol_sha256=helper.EXPECTED_PROTOCOL_SHA256,
                helper_sha256=helper.sha256_bytes(helper_path.read_bytes()),
                checker_sha256=helper.sha256_bytes(Path(__file__).read_bytes()),
                frozen_source_sha256={name: helper.sha256_bytes(body) for name, body in frozen.items()},
                auxiliary_sha256={name: helper.sha256_bytes(body) for name, body in auxiliary.items()},
                real_market_files_opened=0, real_confirmation_files_opened=0,
                market_fits=0, optimizers_steps=0, network_requests=0,
                real_market_replay_prepared=False, production_fit_or_metric_functions_imported=False,
                historical_results_copied=False,
                scope="Real frozen source identity and source-only temporary copy; artificial five-file map and negative guards. No full real-data prepare or numerical replay was executed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Audit output must not exist.")
    result = run_checks(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ("status", "checks", "helper_sha256",
                      "checker_sha256", "real_market_files_opened",
                      "real_confirmation_files_opened", "market_fits")}, indent=2))


if __name__ == "__main__":
    main()
