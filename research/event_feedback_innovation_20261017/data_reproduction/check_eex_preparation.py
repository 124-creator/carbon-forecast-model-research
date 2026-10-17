"""Synthetic-only checks; no source archive or observed market values read."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import prepare_eex_data as p


def fake_row(**changes):
    row = dict(date="2013-01-07", time=dt.time(9, 0), auction_name="TEST_A", contract="T3PA",
               price=2, volume=10, revenue=20, status=None, certificate=None,
               total_bids=30, cover_ratio=3, total_bidders=4, successful_bidders=2,
               source_year=2013, source_member="synthetic.xls", excel_row=4,
               source_cells=["T3PA", 2, 10, 20], _has_status=False)
    row.update(changes)
    return row


def run():
    completed = []
    row = fake_row()
    assert p.validate_completed(row, False) == ("inferred_completed", None)
    modern = fake_row(date="2024-01-15", source_year=2024, status="successful", _has_status=True)
    assert p.validate_completed(modern, True) == ("explicit_successful", None)
    completed.append("legacy inferred and modern explicit completion labels stay distinct")
    assert p.validate_completed(fake_row(status="cancelled", source_cells=["cancelled"]), True) == (None, "explicit_cancelled")
    assert p.validate_completed(fake_row(source_cells=["Auction cancelled"]), False) == (None, "failure_marker")
    assert p.validate_completed(fake_row(price=None), False) == (None, "missing_or_nonpositive_required_value")
    assert p.validate_completed(fake_row(revenue=20.02), False) == (None, "revenue_identity_failure")
    assert p.validate_completed(fake_row(revenue=20.01), False) == ("inferred_completed", None)
    completed.append("cancellation, required values and fixed one-cent reconciliation")
    for contract in ("T2PA", "EAA2", "EAA3"):
        assert p.validate_completed(fake_row(contract=contract), False) == (None, "excluded_contract")
    try:
        p.validate_completed(fake_row(contract="UNKNOWN"), False)
        raise AssertionError("unknown contract accepted")
    except ValueError:
        pass
    completed.append("strict approved contract scope")
    a = p.make_component(row, "inferred_completed")
    b = p.make_component(fake_row(auction_name="TEST_B", price=6, volume=30, revenue=180,
                                  total_bids=90, total_bidders=8, successful_bidders=6), "inferred_completed")
    daily = p.daily_aggregate([a, b])[0]
    assert daily["price"] == Decimal(5)
    assert daily["auction_volume"] == Decimal(40)
    assert daily["total_bid_volume"] == Decimal(120)
    assert daily["cover_ratio"] == Decimal(3)
    assert daily["bidder_count"] == Decimal(7)
    assert daily["successful_bidder_count"] == Decimal(5)
    assert daily["auction_count"] == 2
    completed.append("known two-auction volume weighting and bid-volume ratio")
    missing = copy.deepcopy(b)
    missing["total_bid_volume"] = None
    missing["bidder_count"] = None
    result = p.daily_aggregate([a, missing])[0]
    assert result["total_bid_volume"] is None and result["cover_ratio"] is None
    assert result["bidder_count"] is None and result["price"] == Decimal(5)
    assert set(result["missing_fields"].split("|")) == {"total_bid_volume", "cover_ratio", "bidder_count"}
    completed.append("missing components propagate missing covariates without zero-fill")
    unique, duplicate = p.deduplicate([row, {**row, "excel_row": 5}])
    assert len(unique) == 1 and len(duplicate) == 1
    try:
        p.deduplicate([row, {**row, "price": 3}])
        raise AssertionError("conflicting duplicate accepted")
    except ValueError:
        pass
    completed.append("official-identity exact duplicate handling and conflict refusal")
    assert p.partition("2023-12-31") == "development"
    assert p.partition("2024-01-01") == "confirmation_2024"
    assert p.partition("2025-12-31") == "confirmation_2025"
    for date in ("2011-12-31", "2026-01-01"):
        try:
            p.partition(date)
            raise AssertionError("out-of-scope date accepted")
        except ValueError:
            pass
    completed.append("pre-frozen development and confirmation boundaries; 2026 excluded")
    wrong = {**a, "contract": "T2PA"}
    try:
        p.daily_aggregate([a, wrong])
        raise AssertionError("mixed contracts accepted")
    except ValueError:
        pass
    completed.append("no contract mixing inside a daily aggregate")
    assert p.decimal_text(Decimal("0.00")) == "0"
    assert p.decimal_text(Decimal("10")) == "10"
    assert p.decimal_text(Decimal("5.2500")) == "5.25"
    assert p.decimal_text(None) == ""
    completed.append("deterministic decimal formatting and explicit empty missing cells")
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Synthetic receipt exists")
    passed = run()
    here = Path(__file__).resolve().parent
    receipt = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "passed": True,
               "synthetic_checks": passed, "check_count": len(passed),
               "market_source_archive_read": False, "market_fit": False,
               "source_code_sha256": {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                                      for name in ("check_eex_preparation.py", "prepare_eex_data.py", "inspect_eex_source.py")}}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
