"""Synthetic checks only. No market CSV/workbook, network or model execution."""
from __future__ import annotations

import contextlib
import copy
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import install_main_data as data
import prepare_reproduction_workdir as workdir
import reproduce_study as launch


def identity(blob):
    return dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())


class PortabilityChecks(unittest.TestCase):
    def test_reported_string_preservation_and_cutoff(self):
        blob = ("date,author_reported_price,source_excel_row,source_date_kind\r\n"
                "2023-12-28,14,2,excel_serial\r\n"
                "2023-12-29,14.50,3,excel_serial\r\n"
                "2024-01-02,99,4,excel_serial\r\n").encode()
        result, metadata = data.reported_development(blob)
        expected = ("date,price,source_excel_row,source_date_kind\n"
                    "2023-12-28,14,2,excel_serial\n"
                    "2023-12-29,14.50,3,excel_serial\n").encode()
        self.assertEqual(result, expected)
        self.assertEqual(metadata["development_records"], 2)
        self.assertEqual(metadata["excluded_previously_available_suffix_records"], 1)
        with self.assertRaises(ValueError):
            data.reported_development(blob.replace(b"2023-12-29", b"2023-12-28"))

    def test_shanghai_format_sort_partition_and_no_imputation(self):
        import openpyxl
        book = openpyxl.Workbook(); sheet = book.active; sheet.title = "Sheet1"
        sheet.append(["TradingDate", "CityName", "收盘价", "碳配额"])
        sheet.append([dt.datetime(2024, 1, 15), "上海", 100, 1])
        sheet.append([dt.datetime(2023, 12, 29), "上海", 12, 1])
        sheet.append([dt.datetime(2026, 1, 5), "上海", 9, 1])
        sheet.append([dt.datetime(2025, 9, 25), "上海", 20, 1])
        buffer = io.BytesIO(); book.save(buffer); book.close()
        result = data.shanghai_partitions(buffer.getvalue())
        self.assertEqual(result["development/SHANGHAI_D4.csv"],
                         b"date,price,source_row\n2023-12-29,12,3\n")
        self.assertEqual(result["confirmation_sealed/SHANGHAI_D4.csv"],
                         b"date,price,source_row\n2024-01-15,100,2\n2025-09-25,20,5\n")

    def test_hash_failure_and_local_path_guards(self):
        blob = b"synthetic bytes only\n"
        data.verify_bytes(blob, identity(blob), "synthetic")
        with self.assertRaises(ValueError):
            data.verify_bytes(blob + b"x", identity(blob), "synthetic")
        for value in ("../escape", "/absolute", "a/../b", "a\\b", "C:drive", "a//b"):
            with self.subTest(path=value), self.assertRaises(ValueError):
                data.safe_relative(value)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "existing").mkdir()
            (base / "existing/sentinel").write_text("preserve")
            with self.assertRaises(FileExistsError): data.require_new_output(base / "existing")
            self.assertEqual((base / "existing/sentinel").read_text(), "preserve")
            (base / "dangling").symlink_to(base / "absent")
            with self.assertRaises(FileExistsError): data.require_new_output(base / "dangling")
            (base / "link").symlink_to(base / "existing/sentinel")
            with self.assertRaises(ValueError): data.safe_local(base, "link")
            spec = dict(name="HUBEI", role="development", path="development/HUBEI.csv")
            (base / "HUBEI_development.csv").write_bytes(blob)
            self.assertEqual(data.packaged_file(base, spec).name, "HUBEI_development.csv")
            data.write_new(base / "development/HUBEI.csv", blob)
            with self.assertRaises(ValueError): data.packaged_file(base, spec)

    def test_zip_traversal_and_symlink_rejected(self):
        for bad_member in ("../escape.xlsx", "/escape.xlsx"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive: archive.writestr(bad_member, b"x")
            with self.assertRaises(ValueError): data.extract_shanghai(buffer.getvalue())
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            item = zipfile.ZipInfo("data/link")
            item.create_system = 3; item.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(item, "target")
        with self.assertRaises(ValueError): data.extract_shanghai(buffer.getvalue())

    def test_dry_run_does_not_read_or_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "new-data"
            stdout = io.StringIO()
            with mock.patch.object(sys, "argv", ["install_main_data.py", "--download", "--output", str(target), "--dry-run"]), \
                    mock.patch.object(data, "fetch_source", side_effect=AssertionError("No network permitted")), \
                    mock.patch.object(Path, "read_bytes", side_effect=AssertionError("No data reads permitted")), \
                    contextlib.redirect_stdout(stdout):
                data.main()
            result = json.loads(stdout.getvalue())
            self.assertFalse(target.exists()); self.assertEqual(result["network_calls"], 0)
            self.assertEqual(len(result["expected_files"]), 5)

    def test_semantic_plan_check_preserves_parameters_and_sources(self):
        before = dict(created_utc="old", source_sha256={"model.py": "a"}, registry_sha256="old-registry",
                      tasks=[dict(seed=17, config=dict(hidden=32), id="one")],
                      selected_baseline=dict(config=dict(hidden=16), scope="pooled", score=1.0))
        moved = copy.deepcopy(before)
        moved.update(created_utc="new", registry_sha256="new-registry", sealed_path="relative/path.csv")
        moved["selected_baseline"]["score"] = 1.000000000001
        self.assertEqual(launch.canonical_semantics(before), launch.canonical_semantics(moved))
        changed = copy.deepcopy(moved); changed["tasks"][0]["seed"] = 29
        self.assertNotEqual(launch.canonical_semantics(before), launch.canonical_semantics(changed))
        changed = copy.deepcopy(moved); changed["source_sha256"]["model.py"] = "b"
        self.assertNotEqual(launch.canonical_semantics(before), launch.canonical_semantics(changed))

    def test_runtime_source_check_skips_sealed_bytes_and_detects_edit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            body = b"# Synthetic immutable source\n"
            data.write_new(base / "source/model.py", body)
            manifest = dict(files=[dict(path="source/model.py", **identity(body)),
                                   dict(path="local_only/data/confirmation_sealed/SHANGHAI_D4.csv",
                                        bytes=123, sha256="not-opened")])
            data.write_new(base / "provenance/WORKDIR_MANIFEST.json", data.json_bytes(manifest))
            with mock.patch.object(launch, "ROOT", base):
                # There is no sealed file in this fixture. Access would fail.
                launch.verify_copied_runtime_files()
                (base / "source/model.py").write_bytes(body + b"# Changed\n")
                with self.assertRaises(ValueError): launch.verify_copied_runtime_files()

    def test_learning_dependency_is_rebound_and_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            old_d2 = dict(created_utc="old", tasks=[dict(seed=17)], source_sha256={"model.py": "same"})
            new_d2 = dict(old_d2, created_utc="new")
            old_d2_blob, new_d2_blob = data.json_bytes(old_d2), data.json_bytes(new_d2)
            old_learning = dict(created_utc="old", tasks=[dict(seed=17, cap=100)], source_sha256="same-root-source",
                                d2_plan_sha256=identity(old_d2_blob)["sha256"],
                                changes_confirmation_selection=False, confirmation_data_allowed=False)
            new_learning = dict(old_learning, created_utc="new", d2_plan_sha256=identity(new_d2_blob)["sha256"])
            old_learning_blob = data.json_bytes(old_learning)
            data.write_new(base / "historical_protocols/D2_PRE_RUN.json", old_d2_blob)
            data.write_new(base / "historical_protocols/LEARNING_DIAGNOSTIC_PRE_RUN.json", old_learning_blob)
            data.write_new(base / "D2_PRE_RUN.json", new_d2_blob)
            data.write_new(base / "LEARNING_DIAGNOSTIC_PRE_RUN.json", data.json_bytes(new_learning))
            data.write_new(base / "provenance/D2_REPLAY_PLAN_CHECK.json", data.json_bytes(dict(plan_sha256=identity(new_d2_blob)["sha256"])))
            pins = {"D2_PRE_RUN.json": identity(old_d2_blob)["sha256"],
                    "LEARNING_DIAGNOSTIC_PRE_RUN.json": identity(old_learning_blob)["sha256"]}
            with mock.patch.object(launch, "ROOT", base), mock.patch.object(launch, "HISTORICAL_PLAN_SHA256", pins):
                self.assertEqual(launch.verify_plan("learning"), new_learning)
                changed = dict(new_learning, d2_plan_sha256="wrong-dependency")
                (base / "LEARNING_DIAGNOSTIC_PRE_RUN.json").write_bytes(data.json_bytes(changed))
                with self.assertRaises(ValueError): launch.verify_plan("learning")
                changed = dict(new_learning, source_sha256="modified-source")
                (base / "LEARNING_DIAGNOSTIC_PRE_RUN.json").write_bytes(data.json_bytes(changed))
                with self.assertRaises(ValueError): launch.verify_plan("learning")

    def invoke_launcher_fixture(self, base, argv, module, plan):
        with mock.patch.object(launch, "ROOT", base), mock.patch.object(launch.os, "chdir"), \
                mock.patch.object(launch, "verify_copied_runtime_files"), \
                mock.patch.object(launch, "check_environment", return_value={"synthetic": True}), \
                mock.patch.object(launch, "verify_plan", return_value=plan), \
                mock.patch.object(launch, "module_from", return_value=module) as loader, \
                mock.patch.object(sys, "argv", ["reproduce_study.py"] + argv):
            launch.main()
            return loader.call_args

    def test_learning_cli_init_run_and_no_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan = dict(tasks=[dict(seed=17)], confirmation_data_allowed=False)
            data.write_new(base / "provenance/REPRODUCTION_CONTEXT.json", b"{}\n")
            for stage in ("D1", "D2"):
                data.write_new(base / (stage + "_RUN_COMPLETE.json"), data.json_bytes(dict(all_completed=True)))
            data.write_new(base / "historical_protocols/LEARNING_DIAGNOSTIC_PRE_RUN.json", data.json_bytes(plan))
            module = mock.Mock()
            module.init.side_effect = lambda: data.write_new(base / "LEARNING_DIAGNOSTIC_PRE_RUN.json", data.json_bytes(plan))
            module.run.side_effect = lambda: data.write_new(base / "LEARNING_DIAGNOSTIC_STARTED.json", b"{}\n")
            called = self.invoke_launcher_fixture(base, ["learning", "init"], module, plan)
            self.assertEqual(called, mock.call(".", "learning_diagnostic"))
            self.assertEqual(module.init.call_count, 1)
            self.assertTrue((base / "provenance/LEARNING_REPLAY_PLAN_CHECK.json").is_file())
            self.invoke_launcher_fixture(base, ["learning", "run"], module, plan)
            self.assertEqual(module.run.call_count, 1)
            with self.assertRaises(FileExistsError):
                self.invoke_launcher_fixture(base, ["learning", "run"], module, plan)
            self.assertEqual(module.run.call_count, 1)

    def test_confirmation_selection_does_not_require_learning(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); plan = dict(tasks=[dict(seed=17)])
            data.write_new(base / "provenance/REPRODUCTION_CONTEXT.json", b"{}\n")
            for stage in ("D1", "D2"):
                data.write_new(base / (stage + "_RUN_COMPLETE.json"), data.json_bytes(dict(all_completed=True)))
            data.write_new(base / "DEVELOPMENT_AGGREGATE.json", b"{}\n")
            data.write_new(base / "historical_protocols/CONFIRMATION_PRE_RUN.json", data.json_bytes(plan))
            module = mock.Mock()
            module.init.side_effect = lambda: data.write_new(base / "CONFIRMATION_PRE_RUN.json", data.json_bytes(plan))
            called = self.invoke_launcher_fixture(base, ["c", "init"], module, plan)
            self.assertEqual(called, mock.call("confirmation_source", "run_confirmation"))
            self.assertEqual(module.DEFAULT_SEALED, Path("local_only/data/confirmation_sealed/SHANGHAI_D4.csv"))
            self.assertFalse((base / "LEARNING_DIAGNOSTIC_RESULTS.json").exists())
            self.assertFalse((base / "LEARNING_DIAGNOSTIC_COMPLETE.json").exists())

    def test_figures_dispatch_and_output_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); module = mock.Mock()
            data.write_new(base / "provenance/REPRODUCTION_CONTEXT.json", b"{}\n")
            for stage in ("D1", "D2", "CONFIRMATION"):
                data.write_new(base / (stage + "_RUN_COMPLETE.json"), data.json_bytes(dict(all_completed=True)))
            for name in ("CONFIRMATION_SUMMARY.json", "CONFIRMATION_INFERENCE.json"):
                data.write_new(base / name, b"{}\n")
            # The special completion filename is required before plotting.
            with self.assertRaises(FileNotFoundError):
                self.invoke_launcher_fixture(base, ["figures"], module, {})
            data.write_new(base / "LEARNING_DIAGNOSTIC_COMPLETE.json", data.json_bytes(dict(all_completed=True)))
            called = self.invoke_launcher_fixture(base, ["figures"], module, {})
            self.assertEqual(called, mock.call(".", "make_figures"))
            module.main.assert_called_once_with()
            for target in launch.FIGURE_OUTPUTS:
                path = base / target; data.write_new(path, b"synthetic output sentinel\n")
                with self.assertRaises(FileExistsError):
                    self.invoke_launcher_fixture(base, ["figures"], module, {})
                self.assertEqual(path.read_bytes(), b"synthetic output sentinel\n")
                path.unlink()
            module.main.assert_called_once_with()

    def test_materialize_synthetic_study_then_move(self):
        # The production pin dictionaries are patched only inside this synthetic
        # unit-test process. No CLI accepts alternate identities or unverified data.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); study = base / "history"; cache = base / "package"
            study.mkdir(); cache.mkdir(); (study / "provenance").mkdir()
            specs = []
            for index, template in enumerate(data.DATA_FILES):
                blob = ("date,price\n2020-01-01," + str(index + 1) + "\n").encode()
                spec = dict(template, **identity(blob), records=1)
                specs.append(spec); data.write_new(cache / spec["path"], blob)
            registry = [dict(name=s["name"], path="/old/unused/" + s["name"], price_column="price", sha256=s["sha256"])
                        for s in specs if s["role"] == "development"]
            registry_blob = data.json_bytes(registry)
            data.write_new(study / "provenance/DEVELOPMENT_REGISTRY.json", registry_blob)
            timing = data.json_bytes(dict(synthetic=True))
            rules = data.json_bytes(dict(synthetic_rules=True))
            data.write_new(study / "TIMING_PLAN.json", timing)
            data.write_new(study / "CONFIRMATION_DESIGN_RULES.json", rules)
            data.write_new(study / "aggregate_development.py", b"# Synthetic fixture; never executed.\n")
            data.write_new(study / "make_figures.py", b"# Synthetic figure fixture; never executed.\n")
            plan_pins = {}
            for stage, (source_dir, _, name) in workdir.STAGES.items():
                source = ("# Synthetic source for " + stage + "; never executed.\n").encode()
                data.write_new(study / source_dir / "model.py", source)
                plan = dict(stage=stage, tasks=[dict(seed=17)], source_sha256={"model.py": hashlib.sha256(source).hexdigest()},
                            registry_sha256=hashlib.sha256(registry_blob).hexdigest(),
                            data_registry_sha256=hashlib.sha256(registry_blob).hexdigest(),
                            timing_plan_sha256=hashlib.sha256(timing).hexdigest(), rules_sha256=hashlib.sha256(rules).hexdigest())
                blob = data.json_bytes(plan); data.write_new(study / name, blob)
                plan_pins[name] = hashlib.sha256(blob).hexdigest()
            learning_source = b"# Synthetic learning source; never executed.\n"
            data.write_new(study / "learning_diagnostic.py", learning_source)
            learning = dict(tasks=[dict(seed=17)], source_sha256=identity(learning_source)["sha256"],
                            d2_plan_sha256=plan_pins["D2_PRE_RUN.json"],
                            changes_confirmation_selection=False, confirmation_data_allowed=False)
            learning_blob = data.json_bytes(learning)
            data.write_new(study / "LEARNING_DIAGNOSTIC_PRE_RUN.json", learning_blob)
            plan_pins["LEARNING_DIAGNOSTIC_PRE_RUN.json"] = identity(learning_blob)["sha256"]
            # Deliberately excluded historical artifacts.
            data.write_new(study / "D1_STARTED.json", b"{}\n")
            data.write_new(study / "local_only/old_prediction.npz", b"must not copy\n")
            target = base / "new-study"
            argv = ["prepare_reproduction_workdir.py", "--study-dir", str(study), "--data-dir", str(cache), "--output", str(target)]
            with mock.patch.object(workdir, "DATA_FILES", specs), mock.patch.object(workdir, "PLAN_SHA256", plan_pins), \
                    mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                workdir.main()
            runtime = json.loads((target / "provenance/DEVELOPMENT_REGISTRY.json").read_text())
            self.assertTrue(all(not Path(r["path"]).is_absolute() for r in runtime))
            self.assertFalse((target / "D1_STARTED.json").exists())
            self.assertFalse((target / "local_only/old_prediction.npz").exists())
            self.assertEqual((target / "learning_diagnostic.py").read_bytes(), learning_source)
            self.assertEqual((target / "historical_protocols/LEARNING_DIAGNOSTIC_PRE_RUN.json").read_bytes(), learning_blob)
            self.assertTrue((target / "make_figures.py").is_file())
            self.assertTrue((target / "requirements-figures.txt").is_file())
            for r in runtime:
                self.assertEqual(hashlib.sha256((target / r["path"]).read_bytes()).hexdigest(), r["sha256"])
            moved = base / "moved-study"; target.rename(moved)
            for r in runtime: self.assertTrue((moved / r["path"]).is_file())
            with mock.patch.object(sys, "argv", argv[:-1] + [str(moved)]), self.assertRaises(FileExistsError):
                workdir.main()


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PortabilityChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert "torch" not in sys.modules, "Synthetic portability checks must not import a model runtime"
    report = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  passed=result.wasSuccessful(), tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  market_csv_or_workbook_read=False, market_confirmation_read=False,
                  network_called=False, torch_imported=False, market_fit=False,
                  synthetic_fixtures_only=True,
                  actual_raw_source_download_via_new_helper_tested=False,
                  exact_market_derived_bytes_regenerated_via_new_helper=False,
                  full_market_study_replay_executed=False)
    (Path(__file__).resolve().parent / "PORTABILITY_VALIDATION.json").write_bytes(data.json_bytes(report))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == "__main__":
    main()
