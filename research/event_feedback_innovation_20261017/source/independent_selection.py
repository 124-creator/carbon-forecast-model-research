"""Independent D1 selection/advancement; standard library, no model imports.

Freeze this implementation before reading D1 outcomes. Select every family on
2022 only; evaluate those fixed identities on 2023. No new significance test,
grid, training, confirmation-data read or independent-market assumption.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

PROTOCOL_SHA256 = "404c14867dcc5b43f456075d6b9edd4a5cd75cd99ad943a034e8da7172ef8a70"
BASELINE_FAMILIES = ("er", "masked", "dsof", "self_td", "static_gru",
                     "static_teacher_student", "online_gru", "rw")
HORIZONS = (1, 5, 20)
SELECTION_YEAR = 2022
ADVANCEMENT_YEAR = 2023
EEX = "EEX_EUA_T3PA"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def protocol(path):
    blob = Path(path).read_bytes()
    if sha(blob) != PROTOCOL_SHA256:
        raise ValueError("D1 protocol differs from frozen identity")
    cfg = json.loads(blob)
    if cfg["selection_year"] != SELECTION_YEAR or cfg["advancement_year"] != ADVANCEMENT_YEAR:
        raise ValueError("Unexpected selection/advancement years")
    if cfg["data_end_exclusive"] != "2024-01-01" or cfg["confirmation_access"]:
        raise ValueError("D1-only information boundary violated")
    return cfg


def config_registry(cfg):
    registry = {k: {"family": k} for k in ("rw", "static_gru", "static_teacher_student", "online_gru")}
    calibrators = {c["id"]: c for c in cfg["calibrators"]}
    for cal in cfg["calibrators"]:
        if not cal["old_target"]:
            registry["output_" + cal["id"]] = {"family": "output_only", "calibrator": cal["id"],
                                                   "mode": cal["mode"], "alpha": cal["alpha"]}
    for item in cfg["student_configs"]:
        if item["id"] in registry:
            raise ValueError("Duplicate configuration identity")
        metadata = dict(item)
        if item.get("calibrator"):
            cal = calibrators[item["calibrator"]]
            metadata.update(mode=cal["mode"], alpha=cal["alpha"], old_target=cal["old_target"])
        registry[item["id"]] = metadata
    rrc = [v for v in registry.values() if v["family"] == "rrc"]
    if len(rrc) != 8 or any(v["mode"] not in ("full", "linear") for v in rrc):
        raise ValueError("The primary family must be the eight frozen full/linear candidates")
    return registry


def rational(value):
    value = str(value)
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite candidate outcome; advancement cannot pass")
    return Fraction(value)


def mean(values):
    return sum(values, Fraction(0)) / len(values)


def validate_cells(rows, cfg, registry):
    names = [m["name"] for m in cfg["markets"]]
    keys, counts, scores = set(), {}, {}
    for row in rows:
        key = (row["config_id"], int(row["year"]), row["market"], int(row["seed"]), int(row["horizon"]))
        if key in keys:
            raise ValueError("Duplicate metric cell")
        keys.add(key)
        config, year, market, seed, horizon = key
        if config not in registry or row["family"] != registry[config]["family"]:
            raise ValueError("Unknown/mislabelled configuration")
        origin_count = int(row["origins"])
        if origin_count <= 0:
            raise ValueError("Empty evaluated cell")
        support_key = (year, market, seed)
        if support_key in counts and counts[support_key] != origin_count:
            raise ValueError("Different origin supports across configurations/horizons")
        counts[support_key] = origin_count
        numeric = {k: rational(row[k]) for k in ("mse", "rw_mse", "relative_mse", "mae", "directional_accuracy")}
        if any(numeric[k] < 0 for k in ("mse", "relative_mse", "mae")) or numeric["rw_mse"] <= 0:
            raise ValueError("Invalid loss or RW denominator")
        if not 0 <= numeric["directional_accuracy"] <= 1:
            raise ValueError("Invalid directional accuracy")
        if not math.isclose(float(numeric["mse"] / numeric["rw_mse"]), float(numeric["relative_mse"]), rel_tol=5e-12, abs_tol=1e-12):
            raise ValueError("Metric ratio inconsistency")
        scores[key] = numeric["relative_mse"]
    expected = {(c, y, m, s, h) for c in registry for y in (SELECTION_YEAR, ADVANCEMENT_YEAR)
                for m in names for s in cfg["seeds"] for h in HORIZONS}
    if keys != expected:
        raise ValueError("Metric cube has missing/extra planned series, seed, horizon or year cells")
    return scores, counts


def summarize(rows, cfg):
    registry = config_registry(cfg)
    cells, supports = validate_cells(rows, cfg, registry)
    names = [m["name"] for m in cfg["markets"]]
    score, series = {}, {}
    for config in registry:
        for year in (SELECTION_YEAR, ADVANCEMENT_YEAR):
            for name in names:
                series[config, year, name] = mean([cells[config, year, name, s, h]
                                                 for s in cfg["seeds"] for h in HORIZONS])
            score[config, year] = mean([series[config, year, name] for name in names])
    by_family = defaultdict(list)
    for config, meta in registry.items():
        by_family[meta["family"]].append(config)
    selected = {family: min(configs, key=lambda c: (score[c, SELECTION_YEAR], c))
                for family, configs in by_family.items()}
    baseline = min((selected[f] for f in BASELINE_FAMILIES), key=lambda c: (score[c, SELECTION_YEAR], c))
    chosen = selected["rrc"]
    current, reference = score[chosen, ADVANCEMENT_YEAR], score[baseline, ADVANCEMENT_YEAR]
    wins = [name for name in names if series[chosen, ADVANCEMENT_YEAR, name] < series[baseline, ADVANCEMENT_YEAR, name]]
    no_prefix, output = selected["no_prefix"], selected["output_only"]
    no_prefix_margin = score[no_prefix, ADVANCEMENT_YEAR] - current
    output_margin = score[output, ADVANCEMENT_YEAR] - current
    dsof = selected["dsof"]
    eex_rrc, eex_dsof = series[chosen, ADVANCEMENT_YEAR, EEX], series[dsof, ADVANCEMENT_YEAR, EEX]
    parts = {
        "primary": bool(reference > 0 and current <= Fraction(99, 100) * reference and current < 1),
        "breadth": len(wins) >= 3,
        "mechanism": bool(no_prefix_margin >= Fraction(1, 10000) and output_margin >= Fraction(1, 10000)),
        "EEX_relevance": bool(eex_rrc < 1 and eex_rrc < eex_dsof),
        "finite": True,
    }
    family_rows = [{"family": family, "selected_config_2022": config,
                    "selection_score_2022": float(score[config, SELECTION_YEAR]),
                    "unchanged_config_score_2023": float(score[config, ADVANCEMENT_YEAR])}
                   for family, config in sorted(selected.items())]
    overall_rows = [{"config_id": config, "family": meta["family"], "year": year,
                     "equal_cell_score": float(score[config, year]),
                     "source_cell_count": len(names) * len(cfg["seeds"]) * len(HORIZONS)}
                    for config, meta in registry.items() for year in (SELECTION_YEAR, ADVANCEMENT_YEAR)]
    series_rows = [{"config_id": config, "family": meta["family"], "year": year, "series": name,
                    "equal_seed_horizon_score": float(series[config, year, name]),
                    "source_cell_count": len(cfg["seeds"]) * len(HORIZONS)}
                   for config, meta in registry.items() for year in (SELECTION_YEAR, ADVANCEMENT_YEAR) for name in names]
    matched = []
    for config, meta in registry.items():
        if meta["family"] != "rrc":
            continue
        counterparts = [("output_same_calibrator", "output_" + meta["calibrator"], True, False)]
        for other, reference_meta in registry.items():
            same_lr = reference_meta.get("lr") == meta["lr"]
            same_alpha = reference_meta.get("alpha") == meta["alpha"]
            if reference_meta["family"] in ("no_prefix", "intercept") and same_lr and same_alpha:
                counterparts.append((reference_meta["family"] + "_alpha_lr_matched", other, True, True))
            if reference_meta["family"] == "dsof" and same_lr:
                counterparts.append(("dsof_lr_matched", other, False, True))
            if reference_meta["family"] == "rrc" and same_lr and same_alpha and reference_meta["mode"] != meta["mode"]:
                counterparts.append(("full_vs_linear_alpha_lr_matched", other, True, True))
            if reference_meta["family"] == "old_target" and same_lr:
                counterparts.append(("old_target_sanity_only", other, same_alpha, True))
        for label, other, alpha_matched, lr_matched in counterparts:
            for year in (SELECTION_YEAR, ADVANCEMENT_YEAR):
                matched.append({"rrc_config": config, "comparison": label, "reference_config": other,
                                "year": year, "rrc_score": float(score[config, year]),
                                "reference_score": float(score[other, year]),
                                "reference_minus_rrc": float(score[other, year] - score[config, year]),
                                "alpha_matched": alpha_matched, "student_lr_matched": lr_matched})
    summary = {
        "selection_year": SELECTION_YEAR, "advancement_year": ADVANCEMENT_YEAR,
        "selection_source": "2022 only; no 2023 reselection, including the baseline family choice",
        "selected_rrc": chosen, "selected_baseline": baseline,
        "selected_baseline_family": registry[baseline]["family"],
        "selected_no_prefix": no_prefix, "selected_output_only": output, "selected_dsof": dsof,
        "rrc_score_2023": float(current), "baseline_score_2023": float(reference),
        "primary_relative_improvement": float(1 - current / reference) if reference > 0 else None,
        "breadth_series_won": wins, "breadth_count": len(wins),
        "no_prefix_minus_rrc_2023": float(no_prefix_margin), "output_only_minus_rrc_2023": float(output_margin),
        "EEX_rrc_2023": float(eex_rrc), "EEX_dsof_2023": float(eex_dsof),
        "gate_parts": parts, "advance": all(parts.values()),
        "all_38_configs_finite": len(registry) == 38,
        "metric_cells": len(cells), "configurations": len(registry),
        "exact_ties": "Ascending config id after exact rational mean of finite decimal CSV scores",
        "threshold_arithmetic": "Exact rational comparisons of emitted CSV decimal scores; no added tolerance at advancement thresholds",
        "replication_limit": "Five price series represent four connected market systems. EEX and EU_REPORTED are not independent markets; seeds/horizons are not independent market observations.",
        "inference_limit": "Development advancement heuristic, not statistical significance or proof of publication readiness.",
    }
    return summary, family_rows, overall_rows, series_rows, matched


def synthetic(cfg):
    registry = config_registry(cfg)
    families = defaultdict(list)
    for config, meta in registry.items():
        families[meta["family"]].append(config)
    chosen = sorted(families["rrc"])[0]
    tempting = sorted(families["rrc"])[-1]
    dsof = sorted(families["dsof"])[0]
    dsof_other = sorted(families["dsof"])[-1]
    rows = []
    for config, meta in registry.items():
        for year in (2022, 2023):
            value = "1.2" if year == 2022 else "1.1"
            if config == "rw": value = "1"
            if config == chosen: value = "0.6" if year == 2022 else "0.8"
            if config == tempting: value = "0.7" if year == 2022 else "0.1"
            if config == dsof: value = "0.7" if year == 2022 else "0.9"
            if config == dsof_other: value = "0.8" if year == 2022 else "0.2"
            for market in cfg["markets"]:
                for seed in cfg["seeds"]:
                    for horizon in HORIZONS:
                        rows.append(dict(config_id=config, family=meta["family"], year=year,
                                         market=market["name"], seed=seed, horizon=horizon, origins=100,
                                         mse=value, rw_mse="1", relative_mse=value, mae="0.1", directional_accuracy="0.5"))
    first = summarize(rows, cfg)[0]
    assert first["selected_rrc"] == chosen and first["selected_baseline"] == dsof
    assert first["advance"] and first["breadth_count"] == 5
    assert summarize(list(reversed(rows)), cfg)[0] == first
    checks = ["2022 RRC and baseline identities retained despite better 2023 alternatives",
              "all four advancement components pass known synthetic case", "row-order invariant exact averaging"]
    tied = [dict(r) for r in rows]
    for row in tied:
        if row["family"] == "rrc" and row["year"] == 2022:
            row["mse"] = row["relative_mse"] = "0.6"
    assert summarize(tied, cfg)[0]["selected_rrc"] == sorted(families["rrc"])[0]
    checks.append("exact score ties use config-id order")
    boundary = [dict(r) for r in rows]
    no_prefix = first["selected_no_prefix"]
    output = first["selected_output_only"]
    for row in boundary:
        if row["year"] != 2023: continue
        if row["config_id"] == chosen: value = "0.99"
        elif row["config_id"] == dsof: value = "1"
        elif row["config_id"] in (no_prefix, output): value = "0.9901"
        else: continue
        row["mse"] = row["relative_mse"] = value
    exact = summarize(boundary, cfg)[0]
    assert exact["advance"]
    checks.append("exact 1 percent and 0.0001 thresholds admitted without floating-point drift")
    below_margin = [dict(r) for r in boundary]
    for row in below_margin:
        if row["config_id"] == no_prefix and row["year"] == 2023:
            row["mse"] = row["relative_mse"] = "0.990099999999"
    assert not summarize(below_margin, cfg)[0]["gate_parts"]["mechanism"]
    checks.append("below-threshold mechanism margin fails")
    for invalid in (rows[:-1], rows + [rows[0]]):
        try:
            summarize(invalid, cfg)
            raise AssertionError("unbalanced cells accepted")
        except ValueError:
            pass
    corrupt = [dict(r) for r in rows]
    corrupt[0]["relative_mse"] = "nan"
    try:
        summarize(corrupt, cfg)
        raise AssertionError("nonfinite outcome accepted")
    except ValueError:
        pass
    checks.append("missing, duplicate and nonfinite cells cannot silently advance")
    return {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "passed": True,
            "checks": checks, "synthetic_cells": len(rows), "market_results_read": False,
            "market_fit": False, "confirmation_read": False,
            "source_sha256": sha(Path(__file__).read_bytes()), "protocol_sha256": PROTOCOL_SHA256}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(cfg, protocol_path, output, override_root=None):
    root = Path(override_root or cfg["study_root"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("Selection destination exists; no overwrite or hidden reselection")
    completed_path = root / "results/D1/COMPLETE.json"
    completed = json.loads(completed_path.read_bytes())
    if completed["protocol_sha256"] != PROTOCOL_SHA256 or completed["jobs"] != cfg["counts"]["jobs"] or completed["failures"]:
        raise ValueError("D1 is incomplete or not the frozen protocol")
    for rel, expected in cfg["source_sha256"].items():
        if sha((root / rel).read_bytes()) != expected:
            raise ValueError("Frozen experiment source changed")
    table = root / "results/D1/ALL_METRICS.csv"
    with table.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if completed["metric_rows"] != len(rows):
        raise ValueError("Metric table incomplete")
    summary, family, overall, series, matched = summarize(rows, cfg)
    summary.update(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), source_sha256=sha(Path(__file__).read_bytes()),
                   protocol_sha256=PROTOCOL_SHA256, metric_table_sha256=sha(table.read_bytes()),
                   completion_receipt_sha256=sha(completed_path.read_bytes()),
                   market_fit=False, confirmation_read=False, raw_forecasts_read=False)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "INDEPENDENT_SELECTION.json", summary)
    write_csv(output / "FAMILY_SELECTION_2022.csv", family)
    write_csv(output / "CONFIG_YEAR_SCORES.csv", overall)
    write_csv(output / "CONFIG_YEAR_SERIES_SCORES.csv", series)
    write_csv(output / "MATCHED_ABLATIONS.csv", matched)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("check", "run"):
        sub = commands.add_parser(action)
        sub.add_argument("--protocol", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        if action == "run":
            sub.add_argument("--study-root", type=Path)
    args = parser.parse_args()
    cfg = protocol(args.protocol)
    if args.action == "check":
        if args.output.exists(): raise FileExistsError("Synthetic receipt exists")
        receipt = synthetic(cfg)
        write_json(args.output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    else:
        run(cfg, args.protocol, args.output, args.study_root)


if __name__ == "__main__":
    main()
