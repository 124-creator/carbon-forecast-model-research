"""Prepare the four already-exposed development panels, without fitting a model.

The default inputs are the four byte-identical CSVs listed in SOURCES. Dates are
parsed and checked before any CSV price is converted to a number. An explicitly
provided, byte-pinned PLOS workbook can add Shenzhen 2020/2021 only after a frozen
market protocol's SHA-256 is supplied. The code never downloads data. Do not run
this module until the owning experiment has frozen its protocol.

The mature xlrd library decodes the Shenzhen worksheet, potentially including its
2022 cells. Only rows dated through 2021 may enter experiment arrays. This is not
a claim that the workbook or 2022 cells have never been decoded. No 2022 price is
exported or used for fitting/scoring; 2024+ dates always cause a rejection.

One private NPZ is written per market-year, with common origins for h=1,5,20.
For role in train/validation/test:
  role_X                  [n,20,3], standardized historical features;
  role_daily_scale        [n], origin causal daily EWMA scale;
  role_future_returns     [n,20], raw future log-return increments;
  role_future_normalized  [n,20], increments / origin daily scale (NO sqrt(h));
  role_y                  [n,3], future h-step sums / (origin scale * sqrt(h));
  role_unit               [n,3], origin scale * sqrt(h);
  role_indices/dates      [n], source quote index / ISO date;
  role_origin_price       [n], unmodified source price at the origin;
  role_target_price       [n,3], unmodified source prices at horizons 1,5,20;
  role_target_indices/dates [n,3], explicit maturity coordinates.

all_returns[0] is a ZERO SENTINEL, not a measured return. all_returns[1:] are
successive log-price differences. For the separate GAS benchmark, reconstruct
x_t^(h) = sum(all_returns[t-h+1:t+1]), t>=h. Fit that mature h-step target process
only through train_label_end; this preparation does NOT fit GAS to daily returns.

Every output containing prices, features, returns or targets remains outside the
Git repository. JSON receipts contain only source hashes, counts, dates, index
boundaries and procedural metadata. No fitted moments or price summaries are
written to those receipts. Existing output files are never overwritten.

Example (after authorization):
  python prepare_data.py --data-dir /tmp/carbon-distribution-update-data/development \
      --local-dir /tmp/carbon-conditional-persistence-runs
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import datetime as dt
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MARKETS = ("EUA_LONG", "GUANGZHOU", "HUBEI", "SHANGHAI")
YEARS = (2021, 2022, 2023)
SHENZHEN_MARKET = "SHENZHEN_PLOS"
SHENZHEN_YEARS = (2020, 2021)
HORIZONS = (1, 5, 20)
MAX_HORIZON = 20
FIRST_ORIGIN = 39
HISTORY = 20
VALIDATION_ORIGINS = 126
MIN_TRAIN_ORIGINS = 128
EWMA_DECAY = 0.94
DEVELOPMENT_END_EXCLUSIVE = "2024-01-01"
DEFAULT_DATA_DIR = Path("/tmp/carbon-distribution-update-data/development")
DEFAULT_LOCAL_DIR = Path("/tmp/carbon-conditional-persistence-runs")
REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCES = {
    "EUA_LONG": {
        "filename": "EUA_LONG_development.csv", "rows": 2839,
        "sha256": "142f391305c482eb3bd58dcf8df477db90cf54402495527749a53e126cdf0e7f",
        "first_date": "2013-01-02", "last_date": "2023-12-29",
    },
    "GUANGZHOU": {
        "filename": "GUANGZHOU_development.csv", "rows": 1434,
        "sha256": "8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051",
        "first_date": "2018-01-03", "last_date": "2023-12-29",
    },
    "HUBEI": {
        "filename": "HUBEI_development.csv", "rows": 1408,
        "sha256": "04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797",
        "first_date": "2018-01-02", "last_date": "2023-12-29",
    },
    "SHANGHAI": {
        "filename": "SHANGHAI_development.csv", "rows": 1507,
        "sha256": "425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1",
        "first_date": "2013-12-19", "last_date": "2023-12-29",
    },
}
EXCLUSIONS = {
    ("SHANGHAI", 2021): "Pre-existing whole-year exclusion: 47 reported quotes, sparse/gapped year."
}
SHENZHEN_METADATA_ONLY = {
    "doi": "10.1371/journal.pone.0296105",
    "supplement_url": "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0296105.s001",
    "sha256": "90c49501afa9beb7370c4409084a28729daa4904a7e7e5de310d0f6bd7a3ba10",
    "finite_numeric_quotes_in_old_mechanical_receipt": 1893,
    "first_date": "2013-08-05", "last_date": "2022-03-11",
    "candidate_year_counts": {"2020": 201, "2021": 206},
    "positive_price_domain_verified": False,
    "price_access_status": "CONDITIONAL: explicit workbook plus frozen market-protocol identity required after synthetic gate.",
    "excluded_year": "2022: incomplete calendar year; no subperiod selection.",
    "exposure": "Old mechanical finite-value checks exist; 2020/2021 have no documented prior model scoring in this project.",
}


class CellExcluded(ValueError):
    """A predeclared date/count rule excludes an evaluation cell."""


class SourceExcluded(ValueError):
    """A frozen price-domain rule excludes the complete proposed new source."""

    def __init__(self, message: str, receipt: dict):
        super().__init__(message)
        self.receipt = receipt


def verify_frozen_protocol(path: Path, expected_sha256: str):
    """Bind an explicit new-source read to the caller's already-frozen protocol."""
    if path is None or not expected_sha256 or len(expected_sha256) != 64:
        raise ValueError("New-source access requires a frozen market protocol and its expected SHA-256")
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError("Frozen market-protocol SHA-256 mismatch; Shenzhen remains unopened")
    protocol = json.loads(raw)
    data_policy = protocol.get("data", {})
    if (protocol.get("synthetic_decision") != "PASS_FINITE_SYNTHETIC_RESOURCE_GATE"
            or data_policy.get("new_source_sha256") != SHENZHEN_METADATA_ONLY["sha256"]
            or data_policy.get("new_source_bytes") != 13625856
            or data_policy.get("new_source_expected_finite_rows_through_2021") != 1857
            or data_policy.get("additional_panels") != ["SHENZHEN_PLOS_2020", "SHENZHEN_PLOS_2021"]):
        raise ValueError("Frozen protocol does not authorize the exact post-gate Shenzhen source and panels")
    return {"filename": Path(path).name, "sha256": actual,
            "expected_frozen_identity_verified_before_workbook_open": True}


def _plos_date(cell, datemode, xlrd):
    """The exact date rules of the prior mechanical author-supplement audit."""
    if cell.ctype == xlrd.XL_CELL_DATE:
        return xlrd.xldate_as_datetime(cell.value, datemode).date()
    if isinstance(cell.value, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y.%m.%d"):
            try:
                return dt.datetime.strptime(cell.value.strip(), fmt).date()
            except ValueError:
                pass
    return None


def read_shenzhen_plos(workbook_path: Path, protocol_path: Path, protocol_sha256: str):
    """Read only the predeclared <=2021 experiment subset of the fixed workbook.

    xlrd's sheet decoding may include 2022 cells. The explicit date-first row
    loop never accesses their price field, and none enters arrays or exports.
    Missing/nonnumeric values use the old source's mechanical exclusion rule.
    Any finite nonpositive included value rejects the entire Shenzhen source.
    """
    authorization = verify_frozen_protocol(protocol_path, protocol_sha256)
    import xlrd

    path = Path(workbook_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular, nonsymlink PLOS source workbook")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != 13625856 or digest != SHENZHEN_METADATA_ONLY["sha256"]:
        raise ValueError("PLOS source byte identity changed; do not substitute or re-clean it")
    book = xlrd.open_workbook(file_contents=raw, on_demand=True)
    try:
        if tuple(book.sheet_names()) != ("shenzhen", "guangdong", "hubei", "shanghai"):
            raise ValueError("Unexpected author-workbook sheet identities")
        sheet = book.sheet_by_name("shenzhen")
        if (sheet.nrows, sheet.ncols) != (2295, 73):
            raise ValueError("Unexpected Shenzhen worksheet dimensions")
        if sheet.cell_value(0, 0) != "indicator" or sheet.cell_value(0, 1) != "close price":
            raise ValueError("Unexpected author date/price header identity")
        # First pass: date fields only. The library-level decoding caveat above
        # is recorded, not concealed by a claim of cell-selective BIFF parsing.
        dated_rows = [(row, _plos_date(sheet.cell(row, 0), book.datemode, xlrd))
                      for row in range(2, sheet.nrows)]
        valid_dates = [date for _, date in dated_rows if date is not None]
        if not valid_dates or max(valid_dates) >= dt.date(2024, 1, 1):
            raise ValueError("PLOS date guard failed; no price-row access is allowed")
        records = []
        missing_or_nonnumeric = 0
        excluded_later_dates = 0
        for row, date in dated_rows:
            if date is None:
                continue
            if date >= dt.date(2022, 1, 1):
                excluded_later_dates += 1
                continue
            cell = sheet.cell(row, 1)
            if cell.ctype != xlrd.XL_CELL_NUMBER or not math.isfinite(cell.value):
                missing_or_nonnumeric += 1
                continue
            records.append((date, float(cell.value)))
        if len(records) != 1857:
            raise ValueError("The frozen old mechanical rules should yield exactly 1857 <=2021 quotes")
        dates = pd.DatetimeIndex([row[0] for row in records])
        if not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError("Included Shenzhen dates must already be unique and increasing")
        year_counts = {str(y): int(np.sum(dates.year == y)) for y in sorted(set(dates.year))}
        expected_years = {"2013": 89, "2014": 233, "2015": 236, "2016": 223,
                          "2017": 219, "2018": 225, "2019": 225, "2020": 201, "2021": 206}
        if year_counts != expected_years:
            raise ValueError("Included Shenzhen date counts differ from the prior mechanical receipt")
        price = np.asarray([row[1] for row in records], dtype=np.float64)
        source = {
            "market": SHENZHEN_MARKET, "filename": path.name, "sha256": digest,
            "bytes": len(raw), "rows": len(price), "first_date": str(dates[0].date()),
            "last_date": str(dates[-1].date()), "year_counts": year_counts,
            "article_doi": SHENZHEN_METADATA_ONLY["doi"],
            "supplement_url": SHENZHEN_METADATA_ONLY["supplement_url"],
            "raw_workbook_hash_verified_before_decoding": True,
            "date_guard_before_explicit_price_row_access": True,
            "library_decoding_caveat": "xlrd worksheet decoding may include 2022 cells; this is not a never-decoded claim.",
            "allowed_array_period": "through 2021-12-31 inclusive",
            "later_dated_rows_excluded_before_price_row_access": excluded_later_dates,
            "included_period_dated_missing_or_nonnumeric_rows_excluded": missing_or_nonnumeric,
            "invalid_date_rows_excluded": int(sum(date is None for _, date in dated_rows)),
            "nonpositive_finite_included_quotes": int(np.sum(price <= 0)),
            "positive_price_domain_verified": bool(np.all(price > 0)),
            "missing_rule": "Original numeric cell type and finite value required; no imputation or string-number conversion.",
            "source_prices_rounded_shifted_or_filled": False,
            "technical_indicators_used": False,
            "exposure_group": "First model scoring of this author source in 2020/2021; older mechanical price-cell checks already exist.",
            "frozen_market_protocol": authorization,
        }
        if source["nonpositive_finite_included_quotes"]:
            raise SourceExcluded("Complete Shenzhen source excluded: at least one finite <=2021 quote is nonpositive", source)
        return dates, price, source
    finally:
        book.release_resources()


def read_development(data_dir: Path, market: str):
    """Read a whitelisted source snapshot, refusing reserved dates before prices."""
    if market not in SOURCES:
        raise ValueError("Only the four fixed development sources are authorized; Shenzhen is blocked")
    expected = SOURCES[market]
    path = Path(data_dir) / expected["filename"]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, nonsymlink source file: {path}")
    # One immutable in-memory byte snapshot prevents a date/price read race.
    # Taking a hash or parsing usecols=['date'] does not convert price fields.
    raw = path.read_bytes()
    date_frame = pd.read_csv(BytesIO(raw), usecols=["date"], dtype={"date": "string"})
    dates = pd.DatetimeIndex(pd.to_datetime(date_frame["date"], format="%Y-%m-%d", errors="raise"))
    if not len(dates) or dates.isna().any():
        raise ValueError("Empty or missing date column")
    if dates.max() >= pd.Timestamp(DEVELOPMENT_END_EXCLUSIVE):
        raise ValueError("Reserved 2024+ dates detected; no price conversion was performed")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Dates must already be unique and increasing; no reordering or deletion is allowed")
    if (len(dates) != expected["rows"] or str(dates[0].date()) != expected["first_date"]
            or str(dates[-1].date()) != expected["last_date"]):
        raise ValueError("Source date/row identity differs from the frozen development panel")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected["sha256"]:
        raise ValueError(f"Source SHA-256 mismatch for {market}; refusing price parsing")
    prices_as_text = pd.read_csv(BytesIO(raw), usecols=["price"], dtype={"price": "string"})
    price = pd.to_numeric(prices_as_text["price"], errors="raise").to_numpy(dtype=np.float64)
    if len(price) != len(dates) or not np.all(np.isfinite(price)) or np.any(price <= 0):
        raise ValueError("Invalid price domain; stop without dropping rows, filling, rounding, or shifting prices")
    receipt = {"market": market, **expected, "bytes": len(raw),
               "date_guard_before_price_conversion": True,
               "source_hash_verified_before_price_conversion": True,
               "rows_deleted_or_imputed": 0,
               "exposure_group": "Previously trained/validated/scored development data; not independent confirmation."}
    return dates, price, receipt


def create_split(dates: pd.DatetimeIndex, market: str, year: int):
    """Use max-h=20 for all roles and horizons, with strict maturity."""
    allowed_years = SHENZHEN_YEARS if market == SHENZHEN_MARKET else YEARS if market in MARKETS else ()
    if year not in allowed_years:
        raise ValueError("Unknown market/year; this protocol is closed to extra cells")
    if (market, year) in EXCLUSIONS:
        raise CellExcluded(EXCLUSIONS[(market, year)])
    years = dates.year.to_numpy()
    year_ids = np.flatnonzero(years == year)
    if not len(year_ids):
        raise CellExcluded("No quoted dates in the prespecified evaluation year")
    first_test = int(year_ids[0])
    val_end = first_test - MAX_HORIZON - 1
    val_start = val_end - VALIDATION_ORIGINS + 1
    train_end = val_start - MAX_HORIZON - 1
    if train_end - FIRST_ORIGIN + 1 < MIN_TRAIN_ORIGINS:
        raise CellExcluded("Fewer than 128 mature training origins under the fixed split")
    train = np.arange(FIRST_ORIGIN, train_end + 1, dtype=np.int64)
    validation = np.arange(val_start, val_end + 1, dtype=np.int64)
    test = np.flatnonzero((years[:-MAX_HORIZON] == year) & (years[MAX_HORIZON:] == year))
    if not len(test):
        raise CellExcluded("No origin and max-h endpoint both in the prespecified year")
    if len(validation) != VALIDATION_ORIGINS or test[0] != first_test:
        raise AssertionError("Validation count or first annual origin is inconsistent")
    if not (train[-1] + MAX_HORIZON < validation[0]
            and validation[-1] + MAX_HORIZON < test[0]):
        raise AssertionError("A maximum-horizon label crosses a role boundary")
    if not np.all((years[test] == year) & (years[test + MAX_HORIZON] == year)):
        raise AssertionError("Test origin/endpoint crosses the annual boundary")
    return {"train": train, "validation": validation, "test": test}


def causal_daily_scale(returns: np.ndarray, floor: float):
    """Causal EWMA with a 20-return prefix initializer and no index-0 units bug.

    At indices 1..20 the variance is the mean squared return of the available
    prefix. At 21+ it is .94*v[t-1]+.06*r[t]^2. Used feature rows start at 20;
    their values match the prior experiment's 20-return initializer. Index 0 is
    a zero-observation sentinel, assigned floor only, never a model feature.
    """
    variance = np.zeros(len(returns), dtype=np.float64)
    prefix_end = min(HISTORY, len(returns) - 1)
    if prefix_end:
        variance[1:prefix_end + 1] = np.cumsum(returns[1:prefix_end + 1] ** 2) / np.arange(1, prefix_end + 1)
    for t in range(HISTORY + 1, len(returns)):
        variance[t] = EWMA_DECAY * variance[t - 1] + 0.06 * returns[t] ** 2
    return np.maximum(np.sqrt(variance), floor)


def role_metadata(ids: np.ndarray, dates: pd.DatetimeIndex):
    first, last = int(ids[0]), int(ids[-1])
    return {
        "count": int(len(ids)), "first_origin": first, "last_origin": last,
        "first_date": str(dates[first].date()), "last_date": str(dates[last].date()),
        "origin_calendar_span_days": int((dates[last] - dates[first]).days),
        "first_feature_index": first - HISTORY + 1,
        "first_feature_date": str(dates[first - HISTORY + 1].date()),
        "last_feature_index": last, "last_feature_date": str(dates[last].date()),
        "last_label_index": last + MAX_HORIZON,
        "last_label_date": str(dates[last + MAX_HORIZON].date()),
        "by_horizon": {str(h): {"count": int(len(ids)), "first_target_index": first + h,
                                  "last_target_index": last + h,
                                  "first_target_date": str(dates[first + h].date()),
                                  "last_target_date": str(dates[last + h].date())}
                       for h in HORIZONS},
    }


def prepare_arrays(dates: pd.DatetimeIndex, price: np.ndarray, source: dict, market: str, year: int):
    """Prepare arrays using training-only moments; no estimator/optimizer calls."""
    expected_hash = SHENZHEN_METADATA_ONLY["sha256"] if market == SHENZHEN_MARKET else SOURCES.get(market, {}).get("sha256")
    if not expected_hash or source.get("sha256") != expected_hash:
        raise ValueError("Prepared arrays require a fixed-source read_development receipt")
    if dates.max() >= pd.Timestamp(DEVELOPMENT_END_EXCLUSIVE):
        raise ValueError("2024+ data are refused before calculating price-derived arrays")
    if market == SHENZHEN_MARKET and dates.max() >= pd.Timestamp("2022-01-01"):
        raise ValueError("Only Shenzhen dates through 2021 may enter experimental arrays")
    roles = create_split(dates, market, year)
    train = roles["train"]
    train_label_end = int(train[-1] + MAX_HORIZON)
    logprice = np.log(price)
    returns = np.r_[0.0, np.diff(logprice)]
    floor = max(1e-8, 0.1 * float(np.median(np.abs(returns[1:train_label_end + 1]))))
    daily_scale = causal_daily_scale(returns, floor)
    calendar_gap = np.r_[1.0, np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(np.float64)]
    if np.any(calendar_gap <= 0):
        raise AssertionError("Nonpositive date interval")
    features = np.stack((returns / daily_scale, np.log(daily_scale), np.log1p(calendar_gap)), axis=1)
    offsets = np.arange(-HISTORY + 1, 1)
    train_windows = features[train[:, None] + offsets]
    # Overlapping training windows receive exactly their specified multiplicity.
    feature_mean = train_windows.mean(axis=(0, 1))
    feature_sd = np.maximum(train_windows.std(axis=(0, 1)), 1e-5)
    horizons = np.asarray(HORIZONS, dtype=np.int64)
    iso_dates = dates.strftime("%Y-%m-%d").to_numpy(dtype="U10")
    arrays = {
        "horizons": horizons, "max_horizon": np.array(MAX_HORIZON, dtype=np.int64),
        "all_returns": returns, "all_dates": iso_dates,
        "train_label_end": np.array(train_label_end, dtype=np.int64),
        "feature_mean": feature_mean, "feature_sd": feature_sd, "floor": np.array(floor),
        "source_sha256": np.array(source["sha256"], dtype="U64"),
    }
    for role, ids in roles.items():
        future = returns[ids[:, None] + np.arange(1, MAX_HORIZON + 1)]
        future_normalized = future / daily_scale[ids, None]
        unit = daily_scale[ids, None] * np.sqrt(horizons)[None, :]
        cumulative = np.cumsum(future, axis=1)[:, horizons - 1]
        y = cumulative / unit
        target_indices = ids[:, None] + horizons
        # Internal consistency only: no data-dependent row selection or fitting.
        np.testing.assert_allclose(cumulative, logprice[target_indices] - logprice[ids, None], rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(y, np.cumsum(future_normalized, axis=1)[:, horizons - 1] / np.sqrt(horizons), rtol=1e-11, atol=1e-11)
        arrays.update({
            f"{role}_X": (features[ids[:, None] + offsets] - feature_mean) / feature_sd,
            f"{role}_daily_scale": daily_scale[ids],
            f"{role}_future_returns": future, f"{role}_future_normalized": future_normalized,
            f"{role}_y": y, f"{role}_unit": unit,
            f"{role}_indices": ids, f"{role}_dates": iso_dates[ids],
            f"{role}_origin_price": price[ids], f"{role}_target_price": price[target_indices],
            f"{role}_target_indices": target_indices, f"{role}_target_dates": iso_dates[target_indices],
        })
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise AssertionError(f"Nonfinite prepared array: {key}")
    receipt = {
        "schema_version": 1, "cell": f"{market}_{year}", "market": market, "year": year,
        "source": source, "horizons": list(HORIZONS), "split_max_horizon": MAX_HORIZON,
        "train_label_end": train_label_end, "train_label_end_date": str(dates[train_label_end].date()),
        "roles": {role: role_metadata(ids, dates) for role, ids in roles.items()},
        "maturity": {
            "train_target_before_first_validation_origin": bool(train_label_end < roles["validation"][0]),
            "validation_target_before_first_test_origin": bool(roles["validation"][-1] + MAX_HORIZON < roles["test"][0]),
            "common_test_origins_and_h20_endpoints_in_year": True,
        },
        "preprocessing": {
            "feature_rows": HISTORY, "feature_names": ["log_return / current_causal_daily_scale", "log_current_causal_daily_scale", "log1p_calendar_gap_days"],
            "first_origin": FIRST_ORIGIN, "earliest_used_feature_index": FIRST_ORIGIN - HISTORY + 1,
            "history_note": "40 quotes through first origin 39; actual network feature tensor has 20 rows, indices 20..39. EWMA uses the causal prefix.",
            "ewma_decay": EWMA_DECAY, "ewma_prefix": "At t=1..20 mean(r[1:t+1]^2); t>=21 uses recursive EWMA; r[0]=0 sentinel excluded.",
            "floor_fit_first_return_index": 1, "floor_fit_last_return_index": train_label_end,
            "floor_rule": "max(1e-8, .1*median(abs(training-prefix daily returns)))",
            "feature_moments_fit_origins": [int(train[0]), int(train[-1])],
            "feature_sd_floor": 1e-5, "fitted_moments_only_in_private_npz": True,
            "future_normalization": "Each increment / origin daily scale; horizon endpoint / (origin daily scale * sqrt(h))",
        },
        "gas_interface": {
            "all_returns_index0": "zero sentinel, not a measured observation",
            "target_process": "x_t^h = sum(all_returns[t-h+1:t+1]), t>=h",
            "fit_target_end_inclusive": train_label_end,
            "fit_target_start_inclusive_by_horizon": {str(h): h for h in HORIZONS},
            "filter_maturity": "At forecast origin o, only target-process observations with maturity t<=o may update filter state.",
        },
        "model_fits": 0, "price_based_row_changes": 0, "reserved_price_access": False,
        "release_policy": "NPZ private/local only; receipt has no price, return, feature, factor, or point-prediction values.",
    }
    return arrays, receipt


def ensure_local_output(local_dir: Path):
    resolved = Path(local_dir).resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError("Private arrays must not be written anywhere inside the Git repository")
    return resolved


def write_prepared(local_dir: Path, arrays: dict, receipt: dict):
    """Write a private NPZ plus value-free receipt; never overwrite an artifact."""
    prepared = ensure_local_output(local_dir) / "prepared"
    if prepared.is_symlink():
        raise ValueError("The private prepared directory must not be a symlink")
    prepared = ensure_local_output(prepared)
    prepared.mkdir(parents=True, exist_ok=True)
    path_npz = prepared / f"{receipt['cell']}.npz"
    path_json = prepared / f"{receipt['cell']}.json"
    if path_npz.exists() or path_json.exists():
        raise FileExistsError(f"Prepared cell already exists: {receipt['cell']}")
    # O_EXCL prevents accidental replacement. A failed write removes only the
    # brand-new file opened by this call, never a pre-existing file.
    with path_npz.open("xb") as handle:
        try:
            np.savez_compressed(handle, **arrays)
        except BaseException:
            handle.close()
            path_npz.unlink()
            raise
    final_receipt = {**receipt, "prepared_sha256": hashlib.sha256(path_npz.read_bytes()).hexdigest(),
                     "prepared_bytes": path_npz.stat().st_size}
    with path_json.open("x", encoding="utf-8") as handle:
        json.dump(final_receipt, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return final_receipt


def prepare_cell(data_dir: Path, local_dir: Path, market: str, year: int, *,
                 shenzhen_workbook: Path | None = None, market_protocol: Path | None = None,
                 market_protocol_sha256: str | None = None):
    """Convenience entry point for one fixed market-year; no model fitting."""
    ensure_local_output(local_dir)
    if (market, year) in EXCLUSIONS:
        raise CellExcluded(EXCLUSIONS[(market, year)])
    if market == SHENZHEN_MARKET:
        if shenzhen_workbook is None:
            raise ValueError("An explicit private Shenzhen workbook is required")
        dates, price, source = read_shenzhen_plos(shenzhen_workbook, market_protocol, market_protocol_sha256)
    else:
        dates, price, source = read_development(data_dir, market)
    arrays, receipt = prepare_arrays(dates, price, source, market, year)
    return write_prepared(local_dir, arrays, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing exact copies of the four frozen development CSVs")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR,
                        help="Private output directory outside the Git repository")
    parser.add_argument("--shenzhen-workbook", type=Path,
                        help="Optional exact PLOS S1 XLS; only <=2021 rows enter arrays")
    parser.add_argument("--market-protocol", type=Path,
                        help="Already-frozen market protocol, required with a Shenzhen workbook")
    parser.add_argument("--market-protocol-sha256",
                        help="Expected SHA-256 of that protocol, fixed before opening Shenzhen")
    args = parser.parse_args()
    local_dir = ensure_local_output(args.local_dir)
    use_shenzhen = args.shenzhen_workbook is not None
    if use_shenzhen:
        verify_frozen_protocol(args.market_protocol, args.market_protocol_sha256)
    elif args.market_protocol is not None or args.market_protocol_sha256 is not None:
        raise ValueError("Protocol identity was provided without a Shenzhen workbook")
    aggregate = local_dir / "DATA_PREPARATION_RECEIPTS.json"
    if aggregate.exists():
        raise FileExistsError("Aggregate preparation receipt already exists; choose a fresh local directory")
    cells = [(market, year) for market in MARKETS for year in YEARS if (market, year) not in EXCLUSIONS]
    if use_shenzhen:
        cells += [(SHENZHEN_MARKET, year) for year in SHENZHEN_YEARS]
    for market, year in cells:
        for suffix in (".npz", ".json"):
            if (local_dir / "prepared" / f"{market}_{year}{suffix}").exists():
                raise FileExistsError("An output cell exists; choose a fresh local directory")
    receipts, exclusions = [], []
    for market in MARKETS:
        dates, price, source = read_development(args.data_dir, market)
        for year in YEARS:
            try:
                arrays, receipt = prepare_arrays(dates, price, source, market, year)
            except CellExcluded as exc:
                exclusions.append({"cell": f"{market}_{year}", "reason": str(exc)})
                continue
            receipts.append(write_prepared(local_dir, arrays, receipt))
    shenzhen_source_exclusion = None
    if use_shenzhen:
        try:
            dates, price, source = read_shenzhen_plos(args.shenzhen_workbook, args.market_protocol, args.market_protocol_sha256)
        except SourceExcluded as exc:
            shenzhen_source_exclusion = {"reason": str(exc), "source": exc.receipt}
            exclusions.extend({"cell": f"{SHENZHEN_MARKET}_{year}", "reason": str(exc)} for year in SHENZHEN_YEARS)
        else:
            for year in SHENZHEN_YEARS:
                arrays, receipt = prepare_arrays(dates, price, source, SHENZHEN_MARKET, year)
                receipts.append(write_prepared(local_dir, arrays, receipt))
    expected_cells = 13 if use_shenzhen and shenzhen_source_exclusion is None else 11
    summary = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "prepared_cells": len(receipts), "expected_cells": expected_cells,
               "receipts": receipts, "exclusions": exclusions,
               "new_shenzhen": {**SHENZHEN_METADATA_ONLY,
                                "status_this_run": "PREPARED_THROUGH_2021" if use_shenzhen and shenzhen_source_exclusion is None else "EXCLUDED_BY_SOURCE_DOMAIN" if shenzhen_source_exclusion else "NOT_REQUESTED",
                                "positive_price_domain_verified": bool(use_shenzhen and shenzhen_source_exclusion is None)},
               "shenzhen_workbook_requested": use_shenzhen,
               "shenzhen_source_exclusion": shenzhen_source_exclusion,
               "model_fits": 0, "reserved_price_access": False}
    if len(receipts) != expected_cells:
        raise AssertionError("The frozen identities and eligibility rules produced an unexpected cell count; stop for audit")
    local_dir.mkdir(parents=True, exist_ok=True)
    with aggregate.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"prepared_cells": len(receipts), "excluded_cells": len(exclusions),
                      "model_fits": 0, "private_receipt": str(aggregate)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
