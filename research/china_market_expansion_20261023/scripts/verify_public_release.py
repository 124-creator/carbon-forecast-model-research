#!/usr/bin/env python3
"""Read-only consistency check for publicly releasable audit metadata.

This is NOT a data downloader, model fit, forecast, or upstream rights grant.
Run from any cwd: python research/china_market_expansion_20261023/scripts/verify_public_release.py
"""
import json
from pathlib import Path

DATA_FILES = {
    "gate": "audit/PRIMARY_PILOTS_POLICY_GATE_20261023.json",
    "policy": "policies/policy_event_ledger.json",
    "policy_extra": "policies/new_verified_gov_notice_20261023.json",
    "relay_2026": "audit/sceex_2026_download_audit.json",
    "relay_numeric": "audit/sceex_numeric_coverage_counts.json",
    "relay_rebuild": "audit/PRIVATE_REBUILD_2018_2025_RECEIPT_20261023.json",
    "ccer_2025": "audit/CCER_2025_OFFICIAL_RECONCILIATION_20261023.json",
    "ccer_2026": "audit/ccer_2026_official_daily_notices.json",
}


def verify(data, license_present=False):
    """Verify fixed October 23 audit snapshot. Never drop exceptions or rows."""
    problems = []
    checks = 0

    def require(ok, label):
        nonlocal checks
        checks += 1
        if not ok:
            problems.append(label)

    gate = data["gate"]
    policy = data["policy"]
    extra = data["policy_extra"]
    relay = data["relay_2026"]
    numeric = data["relay_numeric"]
    rebuild = data["relay_rebuild"]
    c25 = data["ccer_2025"]
    c26 = data["ccer_2026"]
    pd = gate["policy_document_gate"]
    documents = policy["documents"]
    extras = extra["official_documents"]

    require(len(documents) == 15 == policy["candidate_document_count"], "policy_base_count")
    require(sum(d["download_status"] == "http_200" for d in documents) == 13, "policy_base_http200")
    require(sum(d["download_status"] == "http_412_blocked" for d in documents) == 2, "policy_base_http412")
    require(all(d["point_in_time_model_eligible"] is False for d in documents), "policy_base_pit_false")
    require(len(extras) == 1 and extras[0]["http_status"] == 200, "policy_extra_one_http200")
    require(all(d["point_in_time_model_eligible"] is False for d in extras), "policy_extra_pit_false")
    require(pd["total_candidates_across_both_catalogs"] == 16, "policy_combined_total")
    require(pd["total_http200_across_both_catalogs"] == 14, "policy_combined_http200")
    require(pd["total_blocked_http412_across_both_catalogs"] == 2, "policy_combined_http412")
    require(pd["total_point_in_time_model_eligible"] == 0, "policy_eligible_zero")
    require(pd["total_immutable_first_public_version_certified"] == 0, "policy_first_public_uncertified")

    require(gate["HBEA"]["data_rows"] == 9, "hubei_preserved_nine_rows")
    require(gate["HBEA"]["reported_trade_price_600000_count"] == 9, "hubei_preserved_ambiguous_price")
    require(gate["HBEA"]["amount_divided_by_volume_vs_reported_daily_yield_price_flags"] == 0, "hubei_ratio_flags")
    require(gate["HBEA"]["new_Oct14_Oct23_usable_daily_records"] == 0, "hubei_latest_gap")
    require(gate["BEA"]["current_2026_Oct14_Oct23_daily_records_added"] == 0, "beijing_latest_gap")
    require(gate["GDEA"]["current_2026_Oct14_Oct23_daily_records_added"] == 0, "guangdong_latest_gap")

    require(rebuild["raw_original_pages"] == 22, "historical_source_page_count")
    require(rebuild["raw_page_sha256_matches_metadata"] == 22, "historical_source_hashes")
    require(rebuild["raw_rows_reparsed"] == 21430, "historical_full_rows")
    require(rebuild["prior_canonical_csv_rows"] + rebuild["prior_canonical_missing"] == 21430,
            "historical_omission_retained")
    require(rebuild["prior_csv_is_exact_prefix_of_reparsed_rows"] is True, "historical_prefix")
    require(rebuild["recovered_private_csv"]["record_count"] == 21430, "historical_private_receipt")
    require(relay["raw_table_rows"] == 1459, "relay_2026_raw_rows")
    require(relay["last_date"] == "2026-10-13", "relay_2026_latest_date")
    require(numeric["data"]["2013_2017"]["rows"] == 2823, "relay_2016_17_raw_rows")
    require(numeric["data"]["2026"]["rows"] == 1459, "relay_2026_numeric_tally")
    require(2823 + 21430 + 1459 == 25712, "relay_all_raw_rows_not_clean_daily_samples")

    require(c25["index"]["index_2025_entries"] == 203, "ccer25_expected")
    require(c25["acquisition"]["http_200"] == 202, "ccer25_originals")
    require(c25["acquisition"]["http_404"] == 1, "ccer25_failure_retained")
    require(c25["acquisition"]["all_expected_notices_originally_downloaded"] is False,
            "ccer25_incomplete_flag")
    require(c25["acquisition"]["adjacent_cumulative_mismatch"] == 0, "ccer25_cumulative_checks")
    require(c25["explicit_missing_day_reconstruction_not_source_data"]["point_in_time_model_eligible"] is False,
            "ccer25_derived_day_ineligible")
    require(c26["eligible_document_count"] == 192 == c26["download_success_count"],
            "ccer26_original_count")
    require(c26["strict_daily_numeric_amount_count"] == 186, "ccer26_numeric_count")
    require(c26["explicit_no_trade_notice_count"] == 6, "ccer26_no_trade_preserved")
    require(c26["strict_daily_numeric_amount_count"] + c26["explicit_no_trade_notice_count"] == 192,
            "ccer26_numeric_plus_no_trade")

    require(gate["scope"]["model_fits"] == 0 and gate["scope"]["scored_origins"] == 0,
            "no_new_model_runs")
    require(gate["scope"]["confirmation_2026"] is False, "no_2026_confirmation")
    require(gate["scope"]["raw_rows_deleted"] == 0 and gate["scope"]["missing_values_imputed"] == 0,
            "no_deleted_or_imputed_rows")
    require(rebuild["publisher_redistribution_permission_verified"] is False,
            "raw_redistribution_not_authorized_in_audit")

    return {
        "snapshot": "CHINA-DATA-20261023",
        "checks": checks,
        "metadata_integrity": "PASS" if not problems else "FAIL",
        "problems": problems,
        "code_license_present": bool(license_present),
        "upstream_raw_redistribution_rights_verified": False,
        "full_public_raw_dataset_release_ready": False,
        "model_fits": 0,
        "comparisons": 0,
        "scope": "metadata and code only; historical point-in-time model certification NOT granted",
    }


def main():
    folder = Path(__file__).resolve().parents[1]
    repository = folder.parents[1]
    docs = {key: json.loads((folder / rel).read_text(encoding="utf-8"))
            for key, rel in DATA_FILES.items()}
    licensed = any((repository / name).is_file()
                   for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"))
    report = verify(docs, license_present=licensed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["metadata_integrity"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
