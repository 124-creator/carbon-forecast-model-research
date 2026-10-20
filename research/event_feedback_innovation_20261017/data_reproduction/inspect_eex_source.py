"""Source-only EEX structural audit; no forecasting or price statistics.

All price cells are inspected only for type/sign, cancellation markers and
price-times-volume reconciliation. No numeric price/volume/bid summaries are
emitted. This does not decide inclusion of pre-2020 rows without Status.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import re
import warnings
import zipfile

ARCHIVE_SHA256 = "c3ff6ffaaf348cbd17f61d4726fdbd45a7ead089a0006568a490dd63bb080231"
SCOPE_SHA256 = "05fef41b2a591f0570cef37f1aa34b24e48c7d9a105eb819a1603275a88b650f"
SHEET = "Primary Market Auction"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def serializable(value):
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return {"excel_temporal_value": value.isoformat(), "type": type(value).__name__}
    return value


def read_workbook(blob, suffix):
    """Return literal source cells, with Excel date types faithfully decoded."""
    if suffix == ".xls":
        import xlrd
        book = xlrd.open_workbook(file_contents=blob)
        result = {}
        for sheet in book.sheets():
            rows = []
            for r in range(sheet.nrows):
                values = []
                for c in range(sheet.ncols):
                    cell = sheet.cell(r, c)
                    if cell.ctype == xlrd.XL_CELL_DATE:
                        value = xlrd.xldate.xldate_as_datetime(cell.value, book.datemode)
                    elif cell.ctype == xlrd.XL_CELL_ERROR:
                        value = {"excel_error_code": int(cell.value)}
                    else:
                        value = cell.value
                    values.append(value)
                rows.append(values)
            result[sheet.name] = rows
        return result
    import openpyxl
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style", category=UserWarning)
        book = openpyxl.load_workbook(io.BytesIO(blob), data_only=True, read_only=True)
        result = {s.title: [list(r) for r in s.iter_rows(values_only=True)] for s in book.worksheets}
        book.close()
    return result


def source_books(path):
    blob = path.read_bytes()
    if digest(blob) != ARCHIVE_SHA256:
        raise ValueError("Archive SHA256 does not match acquired source")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        records = []
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            p = Path(entry.filename)
            if p.is_absolute() or ".." in p.parts or p.suffix not in {".xls", ".xlsx"}:
                raise ValueError("Unexpected archive member")
            match = re.fullmatch(r"emission-spot-primary-market-auction-report-(20\d\d)-data\.(xls|xlsx)", p.name)
            if not match:
                raise ValueError("Unexpected annual source name")
            year = int(match.group(1))
            raw = archive.read(entry)
            records.append((year, entry.filename, raw, read_workbook(raw, p.suffix)))
        if sorted(r[0] for r in records) != list(range(2012, 2026)):
            raise ValueError("Annual source inventory differs from 2012-2025")
        yield from sorted(records)


def header_and_indices(rows):
    candidates = []
    for i, row in enumerate(rows[:12]):
        names = [norm(c) for c in row]
        if "date" in names and "contract" in names and any(s.startswith("auction price ") for s in names):
            candidates.append((i, row, names))
    if len(candidates) != 1:
        raise ValueError("Expected one authenticated auction header")
    index, header, names = candidates[0]
    aliases = {
        "date": {"date"}, "time": {"time", "auction time"},
        "auction_name": {"auction name"}, "contract": {"contract"},
        "status": {"status"}, "certificate": {"certificate"},
        "price": {"auction price €/tco2", "auction price eur/tco2"},
        "volume": {"auction volume tco2", "auction volume"},
        "total_bids": {"total amount of bids"},
        "cover_ratio": {"cover ratio"},
        "total_bidders": {"total number of bidders"},
        "successful_bidders": {"number of successful bidders", "number of succesful bidders"},
        "revenue": {"total revenue €", "total revenue eur"},
    }
    indices = {}
    for field, accepted in aliases.items():
        found = [j for j, name in enumerate(names) if name in accepted]
        if len(found) > 1 or (len(found) != 1 and field not in {"status", "certificate"}):
            raise ValueError(f"Ambiguous or absent source field {field}")
        indices[field] = found[0] if found else None
    return index, header, indices


def as_date(value):
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, str):
        for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
            try:
                return dt.datetime.strptime(value.strip(), fmt).date().isoformat()
            except ValueError:
                pass
    raise ValueError("Unrecognized auction date cell; manual schema review required")


def as_decimal(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        result = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def source_rows(year, member, sheets):
    rows = sheets[SHEET]
    hi, header, indices = header_and_indices(rows)
    parsed = []
    without_date = []
    for i, row in enumerate(rows[hi + 1:], start=hi + 2):
        values = {k: row[j] if j is not None else None for k, j in indices.items()}
        try:
            date = as_date(values["date"])
        except ValueError:
            raise ValueError(f"Unrecognized date at year {year}, Excel row {i}") from None
        if date is None:
            if any(v is not None and v != "" for v in row):
                without_date.append(i)
            continue
        if not date.startswith(str(year)):
            raise ValueError(f"Year/date mismatch at year {year}, Excel row {i}")
        values.update(date=date, source_year=year, source_member=member, excel_row=i,
                      source_cells=row)
        parsed.append(values)
    return hi, header, indices, parsed, without_date


def audit(path):
    reports, date_contracts = [], defaultdict(set)
    identity_locations = defaultdict(list)
    cancellation_text = Counter()
    source_dates = []
    for year, member, raw, sheets in source_books(path):
        hi, header, indices, parsed, no_date = source_rows(year, member, sheets)
        counts = Counter()
        contracts = Counter()
        statuses = Counter()
        certificate_values = Counter()
        per_contract_dates = defaultdict(list)
        markers = defaultdict(Counter)
        flagged = []
        for row in parsed:
            contract = str(row["contract"]).strip()
            status = norm(row["status"]) if indices["status"] is not None else "<column absent>"
            contracts[contract] += 1
            statuses[status] += 1
            if indices["certificate"] is not None:
                certificate_values[str(row["certificate"]).strip()] += 1
            date_contracts[row["date"]].add(contract)
            per_contract_dates[contract].append(row["date"])
            source_dates.append(row["date"])
            identity = (row["date"], str(serializable(row["time"])), str(row["auction_name"]), contract)
            identity_locations[identity].append([year, row["excel_row"]])
            numbers = {k: as_decimal(row[k]) for k in ("price", "volume", "revenue")}
            positive = all(v is not None and v > 0 for v in numbers.values())
            counts["all_three_price_volume_revenue_positive" if positive else "has_missing_nonnumeric_or_nonpositive_required_field"] += 1
            valid = all(v is not None for v in numbers.values())
            matched = bool(valid and abs(numbers["price"] * numbers["volume"] - numbers["revenue"]) <= Decimal("0.01"))
            if positive:
                counts["positive_required_fields_revenue_matches_within_one_cent" if matched else "positive_required_fields_revenue_mismatch"] += 1
            for field, value in numbers.items():
                if value is None:
                    raw_value = row[field]
                    # Only marker strings, never source numeric strings, may be displayed.
                    marker = str(raw_value).strip() if raw_value is not None else "<empty>"
                    if len(marker) > 180 or re.search(r"\d", marker):
                        marker = "<nonnumeric text redacted; see private source>"
                    markers[field][marker or "<empty>"] += 1
            if not positive or not matched:
                flagged.append({"excel_row": row["excel_row"], "date": row["date"],
                                "contract": contract, "status": status,
                                "positive_price_volume_revenue": positive,
                                "revenue_reconciles_within_one_cent": matched})
            for cell in row["source_cells"]:
                if isinstance(cell, str) and re.search(r"cancel|unsuccess|not successful|not held", cell, re.I):
                    text = cell.strip()
                    # Cancellation notices can embed dates/numbers: retain full text privately,
                    # output only a standardized categorical flag in public-safe audit.
                    cancellation_text[f"{year}: cancellation/unsuccessful textual notice"] += 1
        caption_rows = []
        for sheet_name, values in sheets.items():
            if sheet_name == "Caption":
                for row in values:
                    text = [str(v).strip() for v in row if isinstance(v, str) and v.strip()]
                    if any(re.search(r"T[23]PA|EAA[23]|European (?:Emission|Aviation) Allowances|EUA[23]\.Phase|EUAA[23]\.Phase", v, re.I) for v in text):
                        caption_rows.append(text)
        reports.append({
            "year": year, "source_member": member, "source_bytes": len(raw), "source_sha256": digest(raw),
            "sheet": SHEET, "worksheet_rows": len(sheets[SHEET]), "header_excel_row": hi + 1,
            "header": [str(v).strip() if v is not None else "" for v in header],
            "field_column_indices_zero_based": indices, "dated_auction_rows": len(parsed),
            "min_date": min(r["date"] for r in parsed), "max_date": max(r["date"] for r in parsed),
            "contract_row_counts": dict(contracts), "status_row_counts": dict(statuses),
            "certificate_row_counts": dict(certificate_values),
            "contract_date_ranges": {k: [min(v), max(v)] for k, v in sorted(per_contract_dates.items())},
            "required_field_audit_counts": dict(counts),
            "nonnumeric_required_field_markers": {k: dict(v) for k, v in markers.items()},
            "flagged_identity_and_boolean_checks_only": flagged,
            "nonempty_rows_without_auction_date": no_date,
            "official_caption_product_mapping_text": caption_rows,
        })
    return {
        "created_utc": utc(), "archive_sha256": ARCHIVE_SHA256,
        "scope_sha256": SCOPE_SHA256, "inspection_script_sha256": digest(Path(__file__).read_bytes()),
        "purpose": "Mechanical source audit before inclusion addendum. No final inclusion or price aggregate is produced.",
        "confirmation_numeric_price_volume_bid_data_displayed": False,
        "market_fit": False, "annual_reports": reports,
        "mixed_contract_dates": [{"date": d, "contracts": sorted(v)} for d, v in sorted(date_contracts.items()) if len(v) > 1],
        "duplicate_official_identity_locations": [{"date": k[0], "contract": k[3], "locations": v} for k, v in identity_locations.items() if len(v) > 1],
        "cancellation_or_unsuccessful_text_notice_counts": dict(cancellation_text),
        "all_dated_rows": len(source_dates), "min_date": min(source_dates), "max_date": max(source_dates),
        "legacy_status_inclusion_not_decided": True,
        "source_warning": "openpyxl's missing default style warning is formatting only; no data interpolation or correction performed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output exists; refusing overwrite")
    result = audit(args.archive)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("created_utc", "all_dated_rows", "min_date", "max_date", "mixed_contract_dates", "duplicate_official_identity_locations", "cancellation_or_unsuccessful_text_notice_counts")}, ensure_ascii=False, indent=2))
    for row in result["annual_reports"]:
        print(json.dumps({k: row[k] for k in ("year", "dated_auction_rows", "min_date", "max_date", "contract_row_counts", "status_row_counts", "contract_date_ranges", "required_field_audit_counts", "nonnumeric_required_field_markers", "flagged_identity_and_boolean_checks_only", "official_caption_product_mapping_text")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
