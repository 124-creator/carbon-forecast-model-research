"""Independent synthetic audit of prior-transport data construction.

This program never calls load_series, reads a market CSV/NPZ/checkpoint, or
performs an optimizer step. Its reference calculations use generated prices,
calendar dates, explicit loops and scalar formulas. A PASS covers the tested
data API and static input-scope declarations, not an unseen runner.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np


class Audit:
    def __init__(self):
        self.checks = []
        self.comparisons = 0
        self.max_abs_error = 0.0
        self.boundary_exception_types = {}

    def truth(self, label, value):
        if not bool(value):
            raise AssertionError(label)
        self.checks.append(label)
        self.comparisons += 1

    def equal(self, label, actual, expected, exact=False):
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape:
            raise AssertionError(f"{label}: shape {a.shape} != {b.shape}")
        if a.dtype.kind in "iufb" and b.dtype.kind in "iufb":
            error = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.0
            self.max_abs_error = max(self.max_abs_error, error)
            passed = np.array_equal(a, b) if exact else np.allclose(a, b, rtol=3e-13, atol=3e-13)
        else:
            passed = np.array_equal(a, b)
        if not passed:
            raise AssertionError(f"{label}: value mismatch")
        self.checks.append(label)
        self.comparisons += a.size

    def rejected(self, label, call, permitted=(ValueError,)):
        try:
            call()
        except permitted as error:
            self.boundary_exception_types[label] = type(error).__name__
            self.checks.append(label)
            self.comparisons += 1
        else:
            raise AssertionError(f"{label}: invalid request was accepted")


def generate_fixtures(names):
    start, end = dt.date(2016, 1, 3), dt.date(2023, 12, 31)
    calendar = [start + dt.timedelta(days=i) for i in range((end-start).days+1)]
    fixtures = {}
    for k, name in enumerate(names):
        kept = []
        for i, day in enumerate(calendar):
            eligible = (
                (day.weekday() < 5 and i % 11 != 0) if k == 0 else
                (i % 3 != 0) if k == 1 else
                (day.weekday() < 5 and i % 9 not in (1, 2)) if k == 2 else
                (i % 2 == 0 and i % 13 != 0) if k == 3 else
                (day.weekday() < 5 and i % 17 != 2)
            )
            if eligible:
                kept.append(day.isoformat())
        increments = np.array([.065*math.sin(.17*i+k)+.035*math.cos(.031*i)+.007*(k+1)
                               for i in range(len(kept))])
        increments[150:159] = 0.0
        price = 110.0 + 8*k + np.cumsum(increments)
        fixtures[name] = (np.array(kept, dtype="U10"), price)
    return fixtures


def expected_split(dates, budget, year):
    cut, next_cut = f"{year}-01-01", f"{year+1}-01-01"
    available = [i for i in range(60, len(dates)-20) if dates[i+20] < cut]
    val = available[-80:]
    meta = [i for i in available if dates[i+20] < dates[val[0]]][-80:]
    support = [i for i in available if dates[i+20] < dates[meta[0]]][-240:]
    test = [i for i in range(60, len(dates)-20)
            if cut <= dates[i] < next_cut and dates[i+20] < next_cut]
    if budget == 100:
        support = [support[round(j*239/59)] for j in range(60)]
        meta = [meta[round(j*79/19)] for j in range(20)]
        val = [val[round(j*79/19)] for j in range(20)]
    return {k: np.array(v, dtype=np.int64) for k, v in
            dict(support=support, meta=meta, val=val, test=test).items()}


def reference_at(dates, price, origin):
    increments = [0.0] + [float(price[j]-price[j-1]) for j in range(1, len(price))]
    initial = math.fsum(v*v for v in increments[1:61]) / 60
    floor = max(initial*1e-8, 1e-16)
    variance = max(initial, floor)
    for j in range(61, origin+1):
        variance = max(.94*variance + .06*increments[j]**2, floor)
    scale = math.sqrt(variance)
    rows = []
    for j in range(origin-19, origin+1):
        gap = (dt.date.fromisoformat(str(dates[j]))-dt.date.fromisoformat(str(dates[j-1]))).days
        row = [increments[j]/scale, math.log1p(gap)/math.log(8.),
               math.log(scale/math.sqrt(initial))]
        rows.append([max(-10., min(10., z)) for z in row])
    sigma = np.array([scale*math.sqrt(h) for h in range(1, 21)])
    delta = np.array([price[origin+h]-price[origin] for h in range(1, 21)])
    return np.array(rows), sigma, delta


def same_batch(audit, label, a, b, keys=None):
    keys = tuple(a) if keys is None else keys
    for key in keys:
        audit.equal(label+"/"+key, a[key], b[key], exact=True)


def perform(module, scope, source_sha):
    audit = Audit()
    names = ("EEX_EUA_T3PA", "GUANGZHOU", "HUBEI", "SHANGHAI", "EU_REPORTED")
    audit.equal("exact_five_names", [x[0] for x in module.SPECS], names, exact=True)
    audit.truth("all_five_sha256_declared", all(len(x[2]) == 64 for x in module.SPECS))
    audit.truth("H20_L20", module.H == module.L == 20)
    audit.equal("score_horizon_indices", module.HORIZONS, [0, 4, 19], exact=True)
    audit.truth("scope_year_2022", scope["year"] == 2022)
    audit.equal("scope_budgets", scope["target_supervision_budgets"], [100, 400], exact=True)
    fixtures = generate_fixtures(names)

    def new_series(overrides=None):
        overrides = overrides or {}
        return {name: module.Series(name, dates.copy(), overrides.get(name, price).copy())
                for name, (dates, price) in fixtures.items()}

    # An accidental call cannot silently turn the synthetic audit into a data run.
    def forbidden_loader(*args, **kwargs):
        raise AssertionError("Market loader is forbidden in this synthetic audit")
    module.load_series = forbidden_loader
    series = new_series()
    cases = {}
    calendar_summary = {}
    for target in names:
        dates, price = fixtures[target]
        big = series[target].split(400, 2022)
        small = series[target].split(100, 2022)
        for budget, got in ((100, small), (400, big)):
            ref = expected_split(dates, budget, 2022)
            for role in ("support", "meta", "val", "test"):
                audit.equal(f"{target}/{budget}/{role}/oracle", got[role], ref[role], exact=True)
            audit.truth(f"{target}/{budget}/support_meta_purge",
                        dates[got["support"][-1]+20] < dates[got["meta"][0]])
            audit.truth(f"{target}/{budget}/meta_val_purge",
                        dates[got["meta"][-1]+20] < dates[got["val"][0]])
            audit.truth(f"{target}/{budget}/annual_maturity", dates[got["val"][-1]+20] < "2022-01-01")
            audit.truth(f"{target}/{budget}/test_same_year",
                        all(str(dates[t]).startswith("2022") and str(dates[t+20]).startswith("2022")
                            for t in got["test"]))
            case = module.make_case(series, target, budget, 2022)
            cases[(target, budget)] = case
            audit.truth(f"{target}/{budget}/target_not_source", target not in case["sources"])
            cutoff = str(dates[big["meta"][0]])
            audit.truth(f"{target}/{budget}/fixed_source_cut", case["source_cutoff"] == cutoff)
            expected_names = [name for name in names if name != target and
                              {name, target} != {"EEX_EUA_T3PA", "EU_REPORTED"}]
            audit.equal(f"{target}/{budget}/source_names", list(case["sources"]), expected_names, exact=True)
            for source_name, batch in case["sources"].items():
                src_dates, _ = fixtures[source_name]
                wanted = [j for j in range(60, len(src_dates)-20) if src_dates[j+20] < cutoff]
                audit.equal(f"{target}/{budget}/{source_name}/source_oracle",
                            batch["origin"], wanted, exact=True)
                audit.truth(f"{target}/{budget}/{source_name}/source_calendar_maturity",
                            all(d < cutoff for d in batch["last_label_date"]))
                rng = np.random.default_rng(713 + budget)
                for episode in range(3):
                    support, query = module.source_episode(batch, rng)
                    audit.truth(f"{target}/{budget}/{source_name}/{episode}/episode_counts",
                                len(support["origin"]) == 64 and len(query["origin"]) == 32)
                    audit.truth(f"{target}/{budget}/{source_name}/{episode}/episode_calendar_purge",
                                max(support["last_label_date"]) < min(query["date"]))
                    audit.truth(f"{target}/{budget}/{source_name}/{episode}/episode_complete",
                                max(query["last_label_date"]) < cutoff)
            for role in ("support", "meta", "val", "test"):
                role_data = case["target_"+role]
                audit.equal(f"{target}/{budget}/{role}/receipt_n",
                            case["receipt"]["roles"][role]["n"], len(role_data["origin"]), exact=True)
                audit.equal(f"{target}/{budget}/{role}/receipt_last_target",
                            case["receipt"]["roles"][role]["last_label_date"],
                            role_data["last_label_date"][-1], exact=True)

        audit.equal(f"{target}/same_test", small["test"], big["test"], exact=True)
        for role in ("support", "meta", "val"):
            audit.truth(f"{target}/{role}/nested", set(small[role]) <= set(big[role]))
            audit.equal(f"{target}/{role}/same_span", small[role][[0,-1]], big[role][[0,-1]], exact=True)
        for source_name in cases[(target, 100)]["sources"]:
            same_batch(audit, f"{target}/{source_name}/fixed_rich_source",
                       cases[(target, 100)]["sources"][source_name],
                       cases[(target, 400)]["sources"][source_name])

        points = [60, 61, int(big["support"][0]), int(big["support"][-1]),
                  int(big["meta"][0]), int(big["val"][-1]), int(big["test"][0])]
        for origin in points:
            out = series[target].batch([origin], origin+20, labels=True)
            rx, rs, rd = reference_at(dates, price, origin)
            audit.equal(f"{target}/{origin}/feature_oracle", out["x"][0], rx)
            audit.equal(f"{target}/{origin}/sigma_oracle", out["sigma"][0], rs)
            audit.equal(f"{target}/{origin}/delta_oracle", out["delta"][0], rd)
            audit.equal(f"{target}/{origin}/target_oracle", out["y"][0], rd/rs)
        q = int(big["test"][0])
        audit.rejected(f"{target}/immature_H20", lambda q=q: series[target].batch([q], q+19))
        audit.rejected(f"{target}/future_origin", lambda q=q: series[target].batch([q+1], q, labels=False))
        audit.rejected(f"{target}/origin_before60", lambda: series[target].batch([59], 100, labels=False))
        visible = series[target].batch([q], q, labels=False)
        audit.truth(f"{target}/query_has_no_labels",
                    not {"y", "delta", "last_label_date"} & set(visible))
        changed = price.copy()
        changed[q+1:] *= 1.3 + .002*np.arange(len(changed)-q-1)
        poison = module.Series(target, dates, changed)
        same_batch(audit, f"{target}/query_future_poison", visible, poison.batch([q], q, labels=False))
        audit.truth(f"{target}/future_poison_is_effective",
                    not np.array_equal(series[target].batch([q], q+20)["delta"],
                                       poison.batch([q], q+20)["delta"]))
        calendar_summary[target] = {
            "synthetic_rows": len(dates),
            "support400_counts": {r: len(big[r]) for r in ("support", "meta", "val", "test")},
            "source_count": len(cases[(target,400)]["sources"]),
            "source_cutoff": cases[(target,400)]["source_cutoff"],
        }

    # Poison all values at/after the external yearly boundary. Every training,
    # meta/validation feature and source batch must remain exactly unchanged.
    future = {}
    for name, (dates, price) in fixtures.items():
        copy = price.copy()
        copy[dates >= "2022-01-01"] *= 1.8
        future[name] = copy
    poisoned = new_series(future)
    for target in names:
        for budget in (100, 400):
            other = module.make_case(poisoned, target, budget, 2022)
            base = cases[(target,budget)]
            for role in ("support", "meta", "val"):
                same_batch(audit, f"{target}/{budget}/annual_poison/{role}",
                           base["target_"+role], other["target_"+role])
            for src in base["sources"]:
                same_batch(audit, f"{target}/{budget}/annual_poison/{src}",
                           base["sources"][src], other["sources"][src])

    # A source after its earlier target-meta cutoff must also be irrelevant.
    target = "GUANGZHOU"
    cutoff = cases[(target,400)]["source_cutoff"]
    overrides = {}
    for name, (dates, price) in fixtures.items():
        changed = price.copy()
        if name != target:
            changed[dates >= cutoff] *= 1.6
        overrides[name] = changed
    altered = module.make_case(new_series(overrides), target, 400, 2022)
    for source_name in altered["sources"]:
        same_batch(audit, "source_meta_cutoff_poison/"+source_name,
                   cases[(target,400)]["sources"][source_name], altered["sources"][source_name])

    # Constant starts and too-short/empty evaluation supports are rejected,
    # never filled by future volatility or silently reduced training budgets.
    dates, price = fixtures["GUANGZHOU"]
    audit.rejected("constant_first60_explicit_rejection",
                   lambda: module.Series("constant", dates, np.ones(len(dates))*100.0))
    audit.rejected("misordered_dates", lambda: module.Series("unordered", dates[::-1], price))
    audit.rejected("nonfinite_price", lambda: module.Series("nan", dates, np.full(len(dates), np.nan)))
    audit.rejected("short_series", lambda: module.Series("short", dates[:100], price[:100]))
    audit.rejected("unfrozen_budget", lambda: series["GUANGZHOU"].split(101, 2022))
    audit.rejected("no_prehistory_support", lambda: series["GUANGZHOU"].split(400, 2015),
                   permitted=(ValueError, IndexError))
    audit.rejected("missing_evaluation_year", lambda: series["GUANGZHOU"].split(400, 2025))
    short = module.Series("short_history", dates[:400], price[:400])
    audit.rejected("insufficient_purged_supervision", lambda: short.split(400, int(str(dates[380])[:4])),
                   permitted=(ValueError, IndexError))

    return {
        "status":"PASS_SYNTHETIC_DATA_AND_STATIC_SCOPE",
        "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha256":source_sha,
        "checks":len(audit.checks),
        "comparisons":int(audit.comparisons),
        "maximum_absolute_numeric_error":audit.max_abs_error,
        "check_names":audit.checks,
        "boundary_exception_types":audit.boundary_exception_types,
        "synthetic_calendar_summary":calendar_summary,
        "runtime":{"python":sys.version.split()[0],"numpy":np.__version__},
        "scope":{"generated_series":5,"market_file_reads":0,"npz_reads":0,"checkpoint_reads":0,
                 "old_confirmation_reads":0,"new_confirmation_reads":0,"optimizer_steps":0,
                 "production_loader_calls":0},
        "static_only":["load_series exact SHA and pre2024 date guard inspected; loader never executed"],
        "limits":[
            "No market values or real-market supports were read or checked.",
            "make_case includes evaluation labels for later scoring; an independent runner review must verify these never enter training/meta/earlystop.",
            "Future-poison checks cover visible features/scales/supports and generated labels, not fitted-model predictions.",
            "A constant first60 volatility is intentionally rejected, not repaired with future data.",
            "Exception type for empty unsupported splits is recorded separately; rejection itself is the tested invariant.",
            "No scoring, model mathematics, novelty or publication claim is certified here."
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", type=Path, default=Path(__file__).with_name("data.py"))
    parser.add_argument("--scope", type=Path, default=Path(__file__).resolve().parents[1]/"provenance/PRE_FIT_SCOPE.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"audits/INDEPENDENT_DATA_AUDIT.json")
    args = parser.parse_args()
    source = args.data_source.resolve()
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    scope = json.loads(args.scope.read_text())
    module_spec = importlib.util.spec_from_file_location("prior_transport_synthetic_data_only", source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    result = perform(module, scope, source_sha)
    result["audit_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["scope_sha256"] = hashlib.sha256(args.scope.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("status","checks","comparisons","maximum_absolute_numeric_error",
                                           "source_sha256","audit_source_sha256","runtime")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
