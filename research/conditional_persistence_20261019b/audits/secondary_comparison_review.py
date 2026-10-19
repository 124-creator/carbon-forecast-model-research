"""Independent synthetic mapping check for the frozen secondary comparison.

No market observations, forecasts, checkpoints, or model fits are accessed.
The reference calculation expands sampled union dates into observed row indices
instead of using the production multiplicity/einsum implementation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import compare_benchmark_families as production


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expanded_date_reference(losses, dates):
    rng = np.random.default_rng(20261019)
    result = np.zeros((1999, len(production.PAIRS)))
    pairs = [(production.FAMILIES.index(a), production.FAMILIES.index(b))
             for a, b in production.PAIRS]
    min_observed = 10**9
    for year in (2021, 2022, 2023):
        panels = [p for p in production.PANELS if p.endswith(str(year))]
        calendar = sorted(set(day for p in panels for day in dates[p].tolist()))
        n = len(calendar)
        starts = rng.integers(0, n, size=(1999, (n + 39) // 40))
        draws = (starts[:, :, None] + np.arange(40)).reshape(1999, -1)[:, :n] % n
        for panel in panels:
            available = {day: i for i, day in enumerate(dates[panel].tolist())}
            mapping = np.array([available.get(day, -1) for day in calendar])
            for b, draw in enumerate(draws):
                rows = mapping[draw]
                rows = rows[rows >= 0]  # absent quotes contribute no row
                min_observed = min(min_observed, len(rows))
                if not len(rows):
                    raise ValueError("Synthetic reference encountered an empty panel")
                means = losses[panel][rows].mean(axis=0)
                for j, (candidate, baseline) in enumerate(pairs):
                    for horizon in range(2):
                        result[b, j] += (
                            means[horizon, candidate] / means[horizon, baseline] - 1
                        ) / 22
    return result, min_observed


def main():
    source = ROOT / "source" / "compare_benchmark_families.py"
    spec = ROOT / "provenance" / "SECONDARY_COMPARISON_SPEC.json"
    identities = {"source_sha256": digest(source), "spec_sha256": digest(spec)}
    rng = np.random.default_rng(1951)
    losses, dates = {}, {}
    missing_counts = {}
    for i, panel in enumerate(production.PANELS):
        year = int(panel.rsplit("_", 1)[1])
        calendar = pd.bdate_range(f"{year}-01-04", periods=127).strftime("%Y-%m-%d").to_numpy()
        keep = (np.arange(127) + i) % (5 + i % 3) != 0
        dates[panel] = calendar[keep]
        missing_counts[panel] = int((~keep).sum())
        values = .05 + rng.lognormal(mean=-1, sigma=.6, size=(keep.sum(), 2, len(production.FAMILIES)))
        values *= 1 + np.arange(keep.sum())[:, None, None] / 100
        losses[panel] = values
    actual = production.synchronized_resamples(losses, dates)
    reference, min_observed = expanded_date_reference(losses, dates)
    err = float(np.max(np.abs(actual - reference)))
    np.testing.assert_allclose(actual, reference, atol=3e-14, rtol=3e-13)
    repeated = np.broadcast_to(np.array([1.0, .3, 1.0 / 7]), (3, 3)).copy()
    np.testing.assert_array_equal(production.seed_mean(repeated), repeated[0])
    assert len(production.PANELS) * len(production.HORIZONS) == 22
    assert identities == {"source_sha256": digest(source), "spec_sha256": digest(spec)}
    receipt = {
        "status": "PASS_READ_ONLY_STATISTICAL_REVIEW",
        **identities,
        "market_values_or_forecasts_opened": False,
        "new_model_fits": 0,
        "synthetic_panels": 11,
        "horizons": [5, 20],
        "paired_draws_checked": 1999,
        "synthetic_missing_dates_per_panel": missing_counts,
        "smallest_retained_panel_draw": min_observed,
        "max_absolute_synchronized_bootstrap_difference": err,
        "seed_mean_preserves_exact_ties": True,
        "manual_review": [
            "Per-seed scores are averaged before cell means; forecasts are not pooled into an ensemble.",
            "Each panel-horizon relative CRPS contributes exactly one twenty-second.",
            "The same date multiplicities bind all markets within a year and all horizons and families.",
            "Missing panel dates contribute no value and are not forward-filled, back-filled, or assigned another market's observation.",
            "Empty panel draws raise instead of hidden redraw.",
            "Secondary synchronized intervals do not replace the frozen primary gate.",
        ],
        "scope_limit": "Synthetic computation and source/spec review; no claim of independent re-scoring of the market density outputs.",
    }
    path = ROOT / "audits" / "SECONDARY_COMPARISON_REVIEW.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
