"""Independent finite-measure audit; NumPy/stdlib only, no market fitting.

The auditor consumes this study's saved forecast NPZ files and public metadata.
It never imports the production runner, scores, data, statistics or models;
never reads source price CSVs/checkpoints; and never produces per-origin output.
The source must be frozen after synthetic verification before an audit is run.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys

import numpy as np


STATISTICAL = ("fhs_fixed", "fhs_rolling", "ewma_normal", "ewma_t5",
               "garch_normal", "garch_t5", "garch_fhs")
NEURAL = ("global_shape", "conditional_shape", "scale_only", "shape_scale",
          "deepnpts_rps", "deepnpts_crps")
FAMILIES = STATISTICAL + NEURAL
LEVELS = (50, 80, 90)
HORIZONS = (1, 5, 20)
ARRAY_KEYS = ("atoms", "probabilities", "truth", "sigma", "origin", "date")
EXPECTED_KEYS = set(ARRAY_KEYS) | {"validation_" + k for k in ARRAY_KEYS}
UNIT_SCALE = ("fhs_fixed", "fhs_rolling", "ewma_normal", "ewma_t5",
              "global_shape", "conditional_shape", "deepnpts_rps", "deepnpts_crps")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid market/family path component")
    return value


class Checks:
    def __init__(self):
        self.checks = 0
        self.numeric_comparisons = 0
        self.exact_comparisons = 0
        self.maximum_absolute_difference = 0.0

    def require(self, condition, message):
        self.checks += 1
        if not bool(condition):
            raise AssertionError(message)

    def equal(self, actual, expected, message):
        left, right = np.asarray(actual), np.asarray(expected)
        self.exact_comparisons += left.size
        self.require(left.shape == right.shape and np.array_equal(left, right), message)

    def near(self, actual, expected, message, atol=2e-10, rtol=2e-11):
        left, right = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        self.require(left.shape == right.shape, message + "/shape")
        self.require(np.isfinite(left).all() and np.isfinite(right).all(), message + "/finite")
        delta = np.abs(left-right)
        self.numeric_comparisons += left.size
        self.maximum_absolute_difference = max(self.maximum_absolute_difference,
                                              float(delta.max(initial=0.0)))
        self.require(np.all(delta <= atol + rtol*np.abs(right)), message)

    def summary(self):
        return dict(checks=self.checks, numeric_comparisons=self.numeric_comparisons,
                    exact_comparisons=self.exact_comparisons,
                    maximum_absolute_difference=self.maximum_absolute_difference)


def finite_crps(atoms, probabilities, truth):
    """Direct double sum, with explicit atom loops, not production scoring code."""
    a, p, y = np.asarray(atoms), np.asarray(probabilities), np.asarray(truth)
    k = a.shape[-1]
    absolute = np.zeros_like(y, dtype=np.float64)
    pair = np.zeros_like(y, dtype=np.float64)
    for i in range(k):
        absolute += p[..., i] * np.abs(a[..., i]-y)
        for j in range(k):
            pair += p[..., i] * p[..., j] * np.abs(a[..., i]-a[..., j])
    return absolute - pair/2


def inverse_cdf(atoms, probabilities, tau):
    """Smallest sorted atom whose cumulative probability reaches tau."""
    if not 0 < tau < 1:
        raise ValueError("An interior quantile level is required")
    order = np.argsort(atoms, axis=-1, kind="stable")
    a = np.take_along_axis(atoms, order, axis=-1)
    p = np.take_along_axis(probabilities, order, axis=-1)
    cumulative = np.cumsum(p, axis=-1)
    index = np.sum(cumulative < tau, axis=-1)
    if np.any(index >= a.shape[-1]):
        raise ValueError("Probability mass does not reach requested quantile")
    return np.take_along_axis(a, index[..., None], axis=-1)[..., 0]


def summarize_measure(atoms, probabilities, truth):
    loss = finite_crps(atoms, probabilities, truth)
    means = np.zeros_like(truth, dtype=np.float64)
    for i in range(atoms.shape[-1]):
        means += probabilities[..., i] * atoms[..., i]
    rw = np.mean(truth**2, axis=0)
    if np.any(rw <= 0):
        raise ValueError("RW MSE must be positive to audit ratio metrics")
    mse = np.mean((means-truth)**2, axis=0)
    metrics = dict(n=len(truth), crps=np.mean(loss, axis=0).tolist(),
                   mean_mse=mse.tolist(), rw_mse=rw.tolist(),
                   mean_mse_ratio=(mse/rw).tolist(),
                   median_mae=np.mean(np.abs(inverse_cdf(atoms, probabilities, .5)-truth), axis=0).tolist(),
                   max_abs_predictive_change_mean=float(np.max(np.abs(means))),
                   minimum_crps=float(np.min(loss)),
                   probability_sum_error=float(np.max(np.abs(np.sum(probabilities, axis=-1)-1))))
    for level in LEVELS:
        tail = (1-level/100)/2
        lower = inverse_cdf(atoms, probabilities, tail)
        upper = inverse_cdf(atoms, probabilities, 1-tail)
        width = np.mean(upper-lower, axis=0)
        metrics["coverage_"+str(level)] = np.mean((truth >= lower) & (truth <= upper), axis=0).tolist()
        metrics["width_"+str(level)] = width.tolist()
        metrics["width_over_rw_rms_"+str(level)] = (width/np.sqrt(rw)).tolist()
    return metrics, means


def aggregate_metrics(records, market_names):
    """Protocol-specified equal market/horizon ratios; no model selection."""
    market_scores = {family: {} for family in FAMILIES}
    scores = {}
    for family in FAMILIES:
        for market in market_names:
            value = records[market, family]["metrics"]["crps"]
            reference = records[market, "fhs_fixed"]["metrics"]["crps"]
            ratios = []
            for horizon in HORIZONS:
                if reference[horizon-1] <= 0:
                    raise ValueError("Fixed-FHS reference CRPS is not positive")
                ratios.append(value[horizon-1]/reference[horizon-1])
            market_scores[family][market] = math.fsum(ratios)/len(HORIZONS)
        scores[family] = math.fsum(market_scores[family].values())/len(market_names)
    global_competitor = min(scores[f] for f in ("fhs_fixed", "fhs_rolling", "global_shape"))
    best_statistical = min(scores[f] for f in STATISTICAL)
    shape_breadth = sum(market_scores["conditional_shape"][m] < market_scores["global_shape"][m]
                       for m in market_names)
    scale_breadth = sum(market_scores["shape_scale"][m] < market_scores["scale_only"][m]
                       for m in market_names)
    gate = dict(
        conditional_shape_1pct_better_than_static_and_rolling=scores["conditional_shape"] <= .99*global_competitor,
        shape_scale_1pct_better_than_scale_only=scores["shape_scale"] <= .99*scores["scale_only"],
        shape_scale_better_than_best_statistical=scores["shape_scale"] < best_statistical,
        conditional_shape_beats_global_in_3_series=shape_breadth >= 3,
        shape_scale_beats_scale_only_in_3_series=scale_breadth >= 3,
    )
    return dict(scores=scores, market_scores=market_scores, gate=gate,
                advance=all(gate.values()),
                gate_evidence=dict(shape_breadth=shape_breadth, scale_breadth=scale_breadth,
                    shape_reference_score=global_competitor, best_statistical_score=best_statistical))


def compare_mapping(checks, actual, expected, label):
    checks.equal(sorted(actual), sorted(expected), label+"/keys")
    for key in actual:
        a, b = actual[key], expected[key]
        if isinstance(a, dict):
            compare_mapping(checks, a, b, label+"/"+key)
        elif isinstance(a, (bool, str, int)):
            checks.equal(a, b, label+"/"+key)
        else:
            checks.near(a, b, label+"/"+key)


def inspect_npz(path, family, year, checks):
    with np.load(path, allow_pickle=False) as archive:
        checks.equal(sorted(archive.files), sorted(EXPECTED_KEYS), str(path.name)+"/NPZ keys")
        arrays = {key: archive[key].copy() for key in EXPECTED_KEYS}
    computed = {}
    for prefix, output_key in (("", "metrics"), ("validation_", "validation_metrics")):
        a, p, y, sigma, origins, dates = [arrays[prefix+k] for k in ARRAY_KEYS]
        tag = path.name+"/"+(prefix or "evaluation")
        checks.require(a.ndim == 3 and a.shape[1:] == (20,31), tag+"/atoms shape")
        checks.equal(p.shape, a.shape, tag+"/probabilities shape")
        checks.equal(y.shape, a.shape[:2], tag+"/truth shape")
        checks.equal(sigma.shape, y.shape, tag+"/sigma shape")
        checks.equal(origins.shape, (len(y),), tag+"/origins shape")
        checks.equal(dates.shape, (len(y),), tag+"/dates shape")
        checks.require(len(y) > 0, tag+"/nonempty")
        checks.require(all(np.isfinite(v).all() for v in (a,p,y,sigma)), tag+"/finite")
        checks.require(all(v.dtype == np.float64 for v in (a,p,y,sigma)), tag+"/float64")
        checks.require(np.issubdtype(origins.dtype, np.integer), tag+"/integer origins")
        checks.require(np.all(np.diff(origins) == 1), tag+"/consecutive observed-index support")
        checks.require(np.all(sigma > 0) and np.all(p > 0), tag+"/positive sigma and mass")
        checks.require(np.all(np.diff(a, axis=-1) > 0), tag+"/ordered support")
        checks.near(p.sum(-1), np.ones_like(y), tag+"/mass", atol=3e-13, rtol=0)
        parsed = [date.fromisoformat(str(day)) for day in dates]
        checks.require(all(parsed[i] < parsed[i+1] for i in range(len(parsed)-1)), tag+"/ordered dates")
        if prefix:
            checks.require(all(day < date(year,1,1) for day in parsed), tag+"/pre-year validation")
        else:
            checks.require(all(day.year == year for day in parsed), tag+"/same-year evaluation")
        # Verify overlapping cumulative labels using NPZ labels only. This is
        # an internal identity check, not a substitute for source-price audit.
        if len(y) > 1:
            checks.near(y[:-1,1:]-y[:-1,:1], y[1:,:-1], tag+"/overlapping labels")
        metrics, means = summarize_measure(a,p,y)
        checks.equal(metrics["n"], len(origins), tag+"/metric n")
        checks.near(means/sigma, np.zeros_like(y), tag+"/zero standardized mean", atol=2e-11, rtol=0)
        checks.require(metrics["minimum_crps"] >= -2e-10, tag+"/nonnegative CRPS")
        checks.near(metrics["mean_mse"], metrics["rw_mse"], tag+"/RW mean risk")
        variance = np.zeros_like(y)
        for i in range(a.shape[-1]):
            variance += p[...,i]*(a[...,i]-means)**2
        ratio = np.sqrt(variance)/sigma
        if family in UNIT_SCALE:
            checks.near(ratio, np.ones_like(y), tag+"/common scale", atol=3e-11, rtol=0)
        if family in ("scale_only", "shape_scale"):
            checks.require(np.all(ratio >= math.exp(-1)-1e-11) and
                           np.all(ratio <= math.exp(1)+1e-11), tag+"/bounded scale multiplier")
        if family.startswith("deepnpts_"):
            checks.require(float(p.min()) >= .03/31-1e-14, tag+"/author floor")
        if not prefix:
            computed["distribution"] = dict(max_abs_mean=float(np.max(np.abs(means))),
                scale_ratio_min=float(ratio.min()), scale_ratio_max=float(ratio.max()),
                minimum_probability=float(p.min()))
        computed[output_key] = metrics
    checks.require(int(arrays["validation_origin"][-1])+20 < int(arrays["origin"][0]),
                   path.name+"/validation labels before evaluation")
    return computed, arrays


def audit_published_rows(computed, published, checks):
    by = {(r["market"], r["family"]): r for r in published}
    checks.require(len(by) == len(published), "unique published rows")
    checks.equal(sorted(by), sorted(computed), "published row grid")
    for key, own in computed.items():
        row = by[key]
        checks.equal(row["status"], "complete", str(key)+"/complete")
        for field in ("metrics", "validation_metrics", "distribution"):
            compare_mapping(checks, own[field], row[field], str(key)+"/"+field)
    return by


def compare_aggregate(own, published, checks):
    for field in ("scores", "market_scores", "gate"):
        compare_mapping(checks, own[field], published[field], "aggregate/"+field)
    checks.equal(own["advance"], published["advance"], "aggregate/advance")


def audit(args):
    started = utc()
    checks = Checks()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    freeze = read_json(args.freeze)
    checks.equal(freeze["source_sha256"], digest(__file__), "independent source freeze")
    checks.equal(freeze["status"], "FROZEN_AFTER_SYNTHETIC_PASS", "freeze status")
    protocol_path = Path(args.protocol).resolve()
    protocol = read_json(protocol_path)
    checks.require("DRAFT" not in protocol.get("status", "").upper(), "formal protocol required")
    checks.equal(protocol["H"], 20, "H20")
    checks.equal(protocol["K"], 31, "K31")
    checks.equal(protocol["year"], 2022, "only authorized D1 year")
    checks.equal(protocol["score_horizons"], list(HORIZONS), "score horizons")
    checks.equal(protocol["statistical_families"], list(STATISTICAL), "statistical family grid")
    checks.equal(protocol["neural_families"], list(NEURAL), "neural family grid")
    markets = [component(s["name"]) for s in protocol["markets"]]
    checks.require(len(markets) == 5 and len(set(markets)) == 5, "five unique series")
    public = Path(args.results).resolve()
    private = Path(args.private_output).resolve()
    pre_run = read_json(public/"PRE_RUN.json")
    checks.equal(pre_run["protocol_sha256"], digest(protocol_path), "run protocol identity")
    checks.require(datetime.fromisoformat(freeze["created_utc"]) < datetime.now(timezone.utc),
                   "freeze precedes first audit read")
    effective = {s["name"]: s["sha256"] for s in pre_run["effective_sources"]}
    checks.equal(effective, {s["name"]:s["sha256"] for s in protocol["markets"]}, "source declarations")
    computed, file_records, support_by_market = {}, [], {}
    test_rows = validation_rows = distinct_rows = 0
    for spec in protocol["markets"]:
        market = spec["name"]
        common = None
        for family in FAMILIES:
            path = private/component(market)/(component(family)+".npz")
            own, arrays = inspect_npz(path, family, protocol["year"], checks)
            computed[market,family] = own
            if common is None:
                common = {k:arrays[k] for k in EXPECTED_KEYS if not ("atoms" in k or "probabilities" in k)}
                distinct_rows += len(arrays["origin"])
            else:
                for key, expected in common.items():
                    checks.equal(arrays[key], expected, market+"/"+family+"/common "+key)
            test_rows += len(arrays["origin"])
            validation_rows += len(arrays["validation_origin"])
            file_records.append(dict(market=market,family=family,sha256=digest(path),
                                     evaluation_n=len(arrays["origin"]),validation_n=len(arrays["validation_origin"])))
        # Only public identities and support are read here, never the source CSV.
        market_public = read_json(public/(market+".json"))
        support = market_public["support"]
        support_by_market[market] = support
        checks.equal(support["name"], market, market+"/support name")
        checks.equal(support["sha256"], spec["sha256"], market+"/declared source identity")
        checks.equal(support["evaluation_n"], len(common["origin"]), market+"/support evaluation n")
        checks.equal(support["validation_n"], len(common["validation_origin"]), market+"/support validation n")
        checks.equal(support["year"], 2022, market+"/support year")
        checks.equal(support["first_evaluation_origin"], int(common["origin"][0]), market+"/first origin")
        checks.equal(support["first_validation_origin"], int(common["validation_origin"][0]), market+"/first validation")
        checks.equal(support["last_validation_label"], int(common["validation_origin"][-1])+20, market+"/last validation label")
        checks.equal(support["last_evaluation_label"], int(common["origin"][-1])+20, market+"/last evaluation label")
        checks.require(support["last_train_label"] < support["first_validation_origin"], market+"/training purge")
        checks.require(support["last_evaluation_label"] < support["rows"], market+"/label inside declared data")
        checks.equal(support["evaluation_first_date"], str(common["date"][0]), market+"/first date")
        checks.equal(support["evaluation_last_date"], str(common["date"][-1]), market+"/last date")
        for field in ("quote_imputation", "future_calendar_used", "covariates_used"):
            checks.equal(support[field], False, market+"/declared "+field)
        audit_published_rows({k:v for k,v in computed.items() if k[0] == market}, market_public["rows"], checks)
    # Freeze independent aggregation before opening the runner's aggregate file.
    own_aggregate = aggregate_metrics(computed, markets)
    published = read_json(public/"RESULTS.json")
    rows = audit_published_rows(computed, published["rows"], checks)
    checks.equal(published["status"], "complete", "complete aggregate")
    checks.equal(published["expected_jobs"], 65, "65 forecast series")
    checks.equal(published["failed"], 0, "zero reported failures")
    compare_aggregate(own_aggregate, published, checks)
    neural_fits = statistical_fits = steps = epochs = epoch_zero = 0
    for (market, family), row in rows.items():
        expected_fitted = family in NEURAL or family in ("garch_normal", "garch_t5")
        checks.equal(row["fitted"], expected_fitted, market+"/"+family+"/fit identity")
        if family in NEURAL:
            neural_fits += int(row["fitted"])
            checks.equal(row["seed"], protocol["neural"]["seed"], family+"/seed")
            trace = row["trace"]
            checks.equal([v["epoch"] for v in trace], list(range(len(trace))), family+"/trace epochs")
            best_epoch, best_score = 0, trace[0]["validation_score"]
            for entry in trace[1:]:
                if entry["validation_score"] < best_score-protocol["neural"]["min_delta"]:
                    best_epoch, best_score = entry["epoch"], entry["validation_score"]
            checks.equal(row["selected_epoch"], best_epoch, family+"/validation-only checkpoint")
            checks.near(row["validation_score"], best_score, family+"/selected trace score")
            value = computed[market,family]["validation_metrics"]["crps"]
            reference = computed[market,"fhs_fixed"]["validation_metrics"]["crps"]
            selected_score = math.fsum(value[h-1]/reference[h-1] for h in HORIZONS)/3
            checks.near(selected_score, best_score, family+"/selected NPZ validation score")
            checks.equal(row["completed_epochs"], len(trace)-1, family+"/completed epochs")
            expected_steps = math.ceil(support_by_market[market]["train_n"]/protocol["neural"]["batch_size"])*(len(trace)-1)
            checks.equal(row["optimizer_steps"], expected_steps, family+"/optimizer steps")
            epoch_zero += int(best_epoch == 0)
        else:
            statistical_fits += int(row["fitted"])
            checks.equal(row["optimizer_steps"], 0, family+"/no neural steps")
        steps += row.get("optimizer_steps",0)
        epochs += row.get("completed_epochs",0)
    counts = dict(actual_neural_fits=neural_fits,actual_statistical_fits=statistical_fits,
                  optimizer_steps=steps,completed_neural_epochs=epochs,selected_epoch_zero=epoch_zero)
    for key,value in counts.items():
        checks.equal(value,published[key],"aggregate counts/"+key)
    checks.equal(neural_fits, 30, "30 neural fits")
    checks.equal(statistical_fits, 10, "10 statistical fits")
    report = dict(created_utc=utc(), started_utc=started, status="PASS", scope="D1_2022_saved_forecasts_only",
        independent_source_sha256=digest(__file__), freeze_sha256=digest(args.freeze),
        protocol_sha256=digest(protocol_path), runner_results_sha256=digest(public/"RESULTS.json"),
        comparisons=checks.summary(), files=file_records, forecast_files_read=len(file_records),
        evaluation_prediction_rows=test_rows, validation_prediction_rows=validation_rows,
        distinct_series_evaluation_origins=distinct_rows,
        evaluation_marginal_distributions=test_rows*20, validation_marginal_distributions=validation_rows*20,
        audited_family_market_cells=len(computed), **counts, independent_aggregate=own_aggregate,
        aggregate_metrics=[dict(market=m,family=f,**computed[m,f]) for m in markets for f in FAMILIES],
        market_source_csv_reads=0, checkpoint_reads=0, confirmation_reads=0, new_fits=0,
        bootstrap_replicates=0, D2_runs=0, C_runs=0,
        interpretation="Numerical and scope audit of saved issued distributions, not independent author-benchmark retraining. Related EEX/EU and overlapping origins are not IID replicates.",
        decision="Further fixed-need research may be considered under a new protocol; no novelty or publication certification." if own_aggregate["advance"] else "STOP fixed conditional-shape necessity experiment; do not alter gates or append tuning.")
    write_new(output,report)
    return {key:report[key] for key in ("status","comparisons","forecast_files_read","evaluation_prediction_rows",
                                      "validation_prediction_rows","audited_family_market_cells","decision")}


def cdf_integral_single(atoms, probabilities, target):
    """Independent piecewise-CDF integral used only in synthetic tests."""
    knots = sorted(set([float(target)] + [float(v) for v in atoms]))
    total = 0.0
    for lo,hi in zip(knots[:-1],knots[1:]):
        middle = (lo+hi)/2
        cdf = math.fsum(float(p) for a,p in zip(atoms,probabilities) if a <= middle)
        observation = float(target <= middle)
        total += (hi-lo)*(cdf-observation)**2
    return total


def synthetic(args):
    checks = Checks()
    rng = np.random.default_rng(88421)
    a = rng.normal(size=(7,4,6))
    p = rng.uniform(.1,1,size=a.shape)
    p /= p.sum(-1,keepdims=True)
    y = rng.normal(size=(7,4))
    direct = finite_crps(a,p,y)
    integral = np.array([[cdf_integral_single(a[i,h],p[i,h],y[i,h]) for h in range(4)] for i in range(7)])
    checks.near(direct,integral,"CRPS double sum versus CDF integration",atol=2e-13,rtol=0)
    permutation = rng.permutation(6)
    checks.near(finite_crps(a[...,permutation],p[...,permutation],y),direct,"atom permutation",atol=2e-13,rtol=0)
    checks.near(finite_crps(4*a+7,p,4*y+7),4*direct,"CRPS affine unit behavior",atol=2e-12,rtol=0)
    two = np.array([[[-1.,1.]]])
    for weight, expected in ((.1,.82),(.5,.5),(.9,.82)):
        prob = np.array([[[weight,1-weight]]])
        risk = (float(finite_crps(two,prob,np.array([[-1.]]))[0,0])+
                float(finite_crps(two,prob,np.array([[1.]]))[0,0]))/2
        checks.near(risk,expected,"two-point proper-score risk",atol=2e-14,rtol=0)
    qa=np.array([[[2.,-2.,0.]]]); qp=np.array([[[.25,.5,.25]]])
    checks.equal(inverse_cdf(qa,qp,.5),np.array([[-2.]]),"lower inverse CDF tie")
    checks.equal(inverse_cdf(qa,qp,.75),np.array([[0.]]),"second inverse CDF tie")
    markets=["A","B","C","D","E"]
    fake={(m,f):dict(metrics=dict(crps=[1.]*20)) for m in markets for f in FAMILIES}
    for m in markets:
        fake[m,"conditional_shape"]["metrics"]["crps"]=[.98]*20
        fake[m,"shape_scale"]["metrics"]["crps"]=[.97]*20
    checks.equal(aggregate_metrics(fake,markets)["advance"],True,"five fixed gates pass fixture")
    for m in markets:
        fake[m,"shape_scale"]["metrics"]["crps"]=[1.]*20
    checks.equal(aggregate_metrics(fake,markets)["advance"],False,"no post hoc gate substitution")
    fixture_files=[]
    if bool(args.synthetic_private) != bool(args.synthetic_results):
        raise ValueError("Both explicitly synthetic artifact paths are required together")
    if args.synthetic_private:
        published=read_json(args.synthetic_results)
        checks.equal(sorted({r["market"] for r in published["rows"]}),["SYNTHETIC"],"synthetic-only published identity")
        computed={}
        for family in FAMILIES:
            path=Path(args.synthetic_private)/"SYNTHETIC"/(family+".npz")
            own,_=inspect_npz(path,family,2022,checks)
            computed["SYNTHETIC",family]=own
            fixture_files.append(dict(family=family,sha256=digest(path)))
        audit_published_rows(computed,published["rows"],checks)
        compare_aggregate(aggregate_metrics(computed,["SYNTHETIC"]),published,checks)
    result=dict(created_utc=utc(),status="PASS",source_sha256=digest(__file__),
                environment=dict(python=platform.python_version(),numpy=np.__version__),
                comparisons=checks.summary(),synthetic_saved_forecast_files=len(fixture_files),
                synthetic_files=fixture_files,market_source_csv_reads=0,formal_prediction_reads=0,
                checkpoint_reads=0,confirmation_reads=0,new_fits=0,
                scope="Analytic synthetic checks and optional explicitly labelled SYNTHETIC saved forecasts only")
    write_new(args.output,result)
    return {k:v for k,v in result.items() if k != "synthetic_files"}


def freeze_source(args):
    receipt=read_json(args.synthetic_receipt)
    if receipt.get("status") != "PASS" or receipt.get("source_sha256") != digest(__file__):
        raise ValueError("A passing synthetic receipt for these exact source bytes is required")
    result=dict(created_utc=utc(),status="FROZEN_AFTER_SYNTHETIC_PASS",source_sha256=digest(__file__),
                synthetic_receipt_sha256=digest(args.synthetic_receipt),
                purpose="Freeze independent implementation before first formal D1 forecast/results read",
                command_reads="This script and the supplied synthetic receipt only; no formal results, CSVs, checkpoints or confirmation files")
    write_new(args.output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    check=sub.add_parser("check")
    check.add_argument("--synthetic-private")
    check.add_argument("--synthetic-results")
    check.add_argument("--output",required=True)
    freeze=sub.add_parser("freeze")
    freeze.add_argument("--synthetic-receipt",required=True)
    freeze.add_argument("--output",required=True)
    run=sub.add_parser("audit")
    run.add_argument("--protocol",required=True)
    run.add_argument("--results",required=True,help="Directory containing PRE_RUN, market JSONs and RESULTS")
    run.add_argument("--private-output",required=True,help="Directory containing MARKET/FAMILY.npz")
    run.add_argument("--freeze",required=True)
    run.add_argument("--output",required=True)
    args=parser.parse_args()
    result={"check":synthetic,"freeze":freeze_source,"audit":audit}[args.command](args)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__ == "__main__":
    main()
