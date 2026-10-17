"""Narrow independent mathematical review of the development bootstrap.

Uses only invented arrays and the public independent_results.py source.
Never calls the audit CLI, loads forecasts, reads quotes, or fits a model.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np


def manual_counts(days, block, rng):
    # Explicit loops deliberately differ from the reviewed vectorized path.
    starts = rng.integers(0, days, size=(days + block - 1) // block)
    result = np.zeros(days)
    filled = 0
    for start in starts:
        for offset in range(block):
            if filled == days:
                return result
            result[(int(start) + offset) % days] += 1
            filled += 1
    return result


def manual_score(series, weights, year=2023, fixed_denominator=False):
    totals_a, totals_b = [], []
    start = np.datetime64(str(year) + "-01-01")
    for item in series.values():
        indices = (np.asarray(item["date"], dtype="datetime64[D]") - start).astype(int)
        local = weights[indices]
        if local.sum() == 0:
            return None
        for h in (0, 4, 19):
            denominator = (item["rw_loss"][:, h].mean() if fixed_denominator else
                           np.dot(local, item["rw_loss"][:, h]) / local.sum())
            if denominator <= 0:
                return None
            totals_a.append(np.dot(local, item["loss_a"][:, h]) / local.sum() / denominator)
            totals_b.append(np.dot(local, item["loss_b"][:, h]) / local.sum() / denominator)
    return np.array([sum(totals_a) / 15, sum(totals_b) / 15])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "source/independent_results.py"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    ns = runpy.run_path(str(source))
    checks, max_error, compared_values = [], 0., 0

    def check(name, condition):
        if not bool(condition): raise AssertionError(name)
        checks.append(name)

    def close(name, actual, expected, atol=3e-14):
        nonlocal max_error, compared_values
        a, b = np.asarray(actual), np.asarray(expected)
        check(name, a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=atol))
        max_error = max(max_error, float(np.max(np.abs(a-b))))
        compared_values += a.size

    target = np.ones((3, 20))
    seed_pred = np.stack([np.zeros_like(target), 2 * np.ones_like(target)])
    risk = ns["mean_seed_squared_loss"](seed_pred, target)
    close("seed losses averaged after squaring", risk, np.ones_like(target))
    check("seed risk differs from ensemble risk", np.all(risk > np.square(seed_pred.mean(0)-target)))

    class FixedStart:
        def integers(self, low, high, size): return np.full(size, high-1)
    for year, days in [(2023, 365), (2024, 366)]:
        for block in (14, 28, 56):
            got = ns["calendar_day_counts"](days, block, FixedStart())
            oracle = manual_counts(days, block, FixedStart())
            close(f"circular wrapping and final truncation {year}/{block}", got, oracle, atol=0.)
            check(f"exact natural-calendar weight total {year}/{block}", got.sum() == days)

    series = {}
    for market in range(5):
        day = np.arange(2 + market, 363, 17 + market)
        h = np.arange(1, 21)[None, :]
        rw = (1 + market + day[:, None] / 90) * h + .2
        a = rw * (.75 + .2 * np.sin(day[:, None] / 35 + h / 10))
        b = rw * (.95 + .15 * np.cos(day[:, None] / 45 - h / 20))
        series[f"S{market}"] = dict(date=np.datetime64("2023-01-01") + day.astype("timedelta64[D]"),
                                    loss_a=a, loss_b=b, rw_loss=rw)
    wrong_fixed_denominator_difference = 0.
    for block in (14, 28, 56):
        got = ns["paired_calendar_bootstrap"](series, year=2023, block_days=block,
                    repetitions=40, seed=314159, max_attempts=100, return_samples=True)
        check(f"synthetic bootstrap completed {block}", got["status"] == "completed")
        rng, oracle, attempts = np.random.default_rng(314159), [], 0
        while len(oracle) < 40 and attempts < 100:
            attempts += 1
            weights = manual_counts(365, block, rng)
            score = manual_score(series, weights)
            if score is not None:
                oracle.append(score)
                wrong = manual_score(series, weights, fixed_denominator=True)
                wrong_fixed_denominator_difference = max(wrong_fixed_denominator_difference,
                                                         float(np.max(abs(score-wrong))))
        oracle = np.array(oracle)
        point = manual_score(series, np.ones(365))
        close(f"equal 15-cell point score {block}", [got["point_score_a"], got["point_score_b"]], point)
        close(f"shared calendar weights and recalculated risk {block}",
              np.array([got["score_a_samples"], got["score_b_samples"]]).T, oracle)
        close(f"paired differences {block}", got["differences"], oracle[:, 0] - oracle[:, 1])
        close(f"frozen percentile interval {block}", got["percentile_95_interval"],
              np.quantile(oracle[:, 0]-oracle[:, 1], [.025, .975], method="linear"))
        check(f"invalid draws counted {block}", got["attempts"] == attempts and got["invalid_repetitions"] == attempts - 40)
    check("test separates recalculated from fixed RW denominator", wrong_fixed_denominator_difference > 1e-4)

    sparse_rw = np.zeros((365, 20)); sparse_rw[0] = 1.
    invalid_series = {f"S{k}": dict(date=np.datetime64("2023-01-01") + np.arange(365).astype("timedelta64[D]"),
                        loss_a=.8*sparse_rw, loss_b=sparse_rw.copy(), rw_loss=sparse_rw.copy()) for k in range(5)}
    failed = ns["paired_calendar_bootstrap"](invalid_series, year=2023, block_days=1,
                   repetitions=10, max_attempts=10, seed=42, return_samples=True)
    check("invalid denominator draws stop at attempt budget", failed["status"] == "insufficient_valid_replicates" and failed["attempts"] == 10)
    check("insufficient draws cannot open resource gate", failed["upper_below_zero"] is False and failed["percentile_95_interval"] is None)
    rng = np.random.default_rng(42)
    valid = sum(manual_score(invalid_series, manual_counts(365, 1, rng)) is not None for _ in range(10))
    check("independent invalid-draw count", failed["valid_repetitions"] == valid and failed["invalid_repetitions"] == 10-valid)

    class Poison(dict):
        def __getitem__(self, key): raise AssertionError("A failed point gate accessed inputs")
        def __iter__(self): raise AssertionError("A failed point gate iterated inputs")
        def items(self): raise AssertionError("A failed point gate inspected inputs")
    stopped = ns["d2_uncertainty"]({"point_gate_pass": False}, Poison(), Poison())
    check("failed point gate accesses no bootstrap inputs", stopped["bootstrap_repetitions_executed"] == 0)
    check("failed point gate leaves confirmation closed", stopped["confirmation_opened"] is False and stopped["eligible_for_separate_confirmation_freeze"] is False)
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    check("reviewed source did not change during checks", before == after)
    receipt = dict(status="PASS", created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=before, audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks_passed=len(checks), compared_values=compared_values, max_absolute_difference=max_error,
        checks=checks, oracle="Explicit nested circular-block loops and per-series weighted scalar ratios",
        fixed_denominator_counterexample_max_difference=wrong_fixed_denominator_difference,
        raw_market_reads=0, development_result_reads=0, saved_prediction_or_weight_reads=0,
        confirmation_reads=0, model_fits=0, optimizer_steps=0,
        static_gate_review="D1/D2 fixed thresholds, same-alpha matching, pre-year ridge validation ties and fixed D1 baseline agree with PROTOCOL; full inventory/source truth are outside this narrow bootstrap review.",
        environment=dict(python=sys.version.split()[0], numpy=np.__version__),
        limits=["The bootstrap presumes an appropriate stable/locally stationary dependent sampling distribution within the year; arbitrary drift is not covered.",
                "Common day weights retain observed cross-series dependence in sampled blocks; five series are not independent market replicates.",
                "Rejecting draws with undefined RW denominators conditions on admissible draws; invalid rates and insufficient-valid status must remain visible.",
                "Percentile intervals are development resource checks, not independent confirmation, universal publication criteria, or simultaneous 95% coverage claims.",
                "This reviewer did not independently reload source prices or inspect market predictions; the separate results auditor owns those declared scopes."])
    output = args.output or root / "audits/SMP_STATISTICS_REVIEW.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status":"PASS", "checks":len(checks), "compared_values":compared_values,
                      "max_absolute_difference":max_error, "output":str(output)}))


if __name__ == "__main__":
    main()
