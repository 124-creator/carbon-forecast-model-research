#!/usr/bin/env python3
"""Regression tests for the public, raw-data-free metadata audit."""
import copy
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from verify_public_release import DATA_FILES, verify  # noqa: E402


class PublicReleaseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = {
            name: json.loads((HERE / file).read_text(encoding="utf-8"))
            for name, file in DATA_FILES.items()
        }

    def test_original_snapshot_passes(self):
        result = verify(copy.deepcopy(self.baseline), license_present=False)
        self.assertEqual(result["metadata_integrity"], "PASS")
        self.assertEqual(result["checks"], 42)
        self.assertEqual(result["problems"], [])
        self.assertFalse(result["full_public_raw_dataset_release_ready"])
        self.assertEqual(result["model_fits"], 0)
        self.assertEqual(result["comparisons"], 0)

    def test_policy_record_count_tampering_is_detected(self):
        sample = copy.deepcopy(self.baseline)
        sample["gate"]["policy_document_gate"]["total_http200_across_both_catalogs"] = 15
        result = verify(sample)
        self.assertIn("policy_combined_http200", result["problems"])

    def test_missing_ccer_original_cannot_be_relabelled_complete(self):
        sample = copy.deepcopy(self.baseline)
        sample["ccer_2025"]["acquisition"]["all_expected_notices_originally_downloaded"] = True
        result = verify(sample)
        self.assertIn("ccer25_incomplete_flag", result["problems"])

    def test_old_relay_prefix_error_cannot_be_deleted(self):
        sample = copy.deepcopy(self.baseline)
        sample["relay_rebuild"]["prior_canonical_missing"] = 0
        result = verify(sample)
        self.assertIn("historical_omission_retained", result["problems"])

    def test_policy_not_backdated_into_pit(self):
        sample = copy.deepcopy(self.baseline)
        sample["policy"]["documents"][0]["point_in_time_model_eligible"] = True
        result = verify(sample)
        self.assertIn("policy_base_pit_false", result["problems"])

    def test_no_trade_not_mislabelled_numeric(self):
        sample = copy.deepcopy(self.baseline)
        sample["ccer_2026"]["strict_daily_numeric_amount_count"] = 192
        result = verify(sample)
        self.assertIn("ccer26_numeric_count", result["problems"])
        self.assertIn("ccer26_numeric_plus_no_trade", result["problems"])


if __name__ == "__main__":
    unittest.main()
