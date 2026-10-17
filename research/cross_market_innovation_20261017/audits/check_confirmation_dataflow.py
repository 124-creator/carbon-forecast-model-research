"""Only synthetic arrays: audit C data_for IDs, mature prefixes and boundaries."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "carbon_workspace/research/cross_market_innovation_20261017"
SOURCE = ROOT / "confirmation_source"
sys.path.insert(0, str(SOURCE))
c = importlib.import_module("run_confirmation")


def forbidden(*args, **kwargs):
    raise AssertionError("Market loader forbidden in this synthetic-only review")


def main():
    # Imports perform no model fit or price access; disable the loader entrypoints
    # before calling any audited function so inputs must come from this script.
    c.get_markets = forbidden
    c.open_markets = forbidden
    names = ["GUANGZHOU", "HUBEI", "EU_REPORTED", "SHANGHAI_D4"]
    days = dt.date(2020, 1, 1).toordinal() + np.arange(731) * 3
    dates = np.array([dt.date.fromordinal(int(d)).isoformat() for d in days])
    index = np.arange(len(days))
    markets = []
    for mid, name in enumerate(names):
        price = 100 + 30 * mid + 0.02 * index + (mid + 1) * np.sin(index / (7 + mid))
        use = np.ones(len(days), bool) if mid == 3 else dates < "2024-01-01"
        markets.append(c.Market(name, dates[use], days[use], price[use], "synthetic", "synthetic-no-file"))
    count = 0
    cases = []
    for cutoff in ("2024-01-01", "2025-01-01"):
        local_task = dict(market_names=[names[3]], cutoff=cutoff)
        pooled_task = dict(market_names=names, cutoff=cutoff)
        lt, lv, le = c.data_for(markets, local_task)
        pt, pv, pe = c.data_for(markets, pooled_task)
        assert le["market_names"] == [names[3]] and np.all(le["market"] == 0)
        assert pe["market_names"] == names and np.all(pe["market"] == 3)
        for key in ("origin", "dates", "levels", "scale", "raw_y", "y", "stat_raw", "stat_normalized", "full", "short", "sparse", "full_mask", "short_mask", "sparse_mask"):
            assert np.array_equal(le[key], pe[key]), key
            count += np.asarray(le[key]).size
        for pair in ((lt, pt), (lv, pv)):
            local, pooled = pair
            use = pooled["market"] == 3
            for key in ("origin", "dates", "levels", "scale", "raw_y", "y", "stat_raw", "stat_normalized", "full", "short", "sparse"):
                assert np.array_equal(local[key], pooled[key][use]), key
                count += np.asarray(local[key]).size
        sh = markets[3]
        end = int(np.searchsorted(sh.dates, cutoff))
        perturbed = []
        for m in markets:
            p = m.price.copy()
            p[m.dates >= cutoff] += 10000 + 5 * np.arange(np.sum(m.dates >= cutoff))
            perturbed.append(c.Market(m.name, m.dates, m.day, p, "synthetic", "synthetic-no-file"))
        ft, fv, _ = c.data_for(perturbed, pooled_task)
        for before, after in ((pt, ft), (pv, fv)):
            assert before["supports"] == after["supports"]
            for key in before:
                if isinstance(before[key], np.ndarray):
                    assert np.array_equal(before[key], after[key]), (cutoff, key)
                    count += before[key].size
        for name, sup in pt["supports"].items():
            assert max(sup["train"]) + 20 < min(sup["validation"])
            assert max(sup["validation"]) + 20 < sup["end"]
            m = markets[names.index(name)]
            assert m.dates[max(sup["validation"]) + 20] < cutoff
        # All targets finish within the forecast year. Next-origin input changes
        # under future perturbation are allowed; the first origin's own future
        # must never alter its inputs or anchor.
        origins = pe["origin"]
        assert np.all(np.array([d[:4] for d in sh.dates[origins]]) == cutoff[:4])
        assert np.all(np.array([d[:4] for d in sh.dates[origins + 20]]) == cutoff[:4])
        first = int(origins[0])
        p = sh.price.copy(); p[first + 1:] += 2222
        altered = list(markets)
        altered[3] = c.Market(sh.name, sh.dates, sh.day, p, "synthetic", "synthetic-no-file")
        _, _, fe = c.data_for(altered, pooled_task)
        for key in ("levels", "scale", "full", "short", "sparse", "stat_raw", "stat_normalized"):
            assert np.array_equal(pe[key][0], fe[key][0]), key
            count += np.asarray(pe[key][0]).size
        assert not np.array_equal(pe["raw_y"][0], fe["raw_y"][0])
        cases.append(dict(cutoff=cutoff, sh_training_end=end, eval_origins=len(origins),
                          local_market_id=0, pooled_market_id=3,
                          future_training_and_validation_difference=0.0,
                          first_forecast_future_input_difference=0.0))
    receipt = dict(
        status="PASS", scope="synthetic C dataflow only; init/run and market loaders not called",
        audit_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256={name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                       for name in ("run_confirmation.py", "data_core.py", "run_d2.py", "statistical_reference.py")},
        synthetic_cases=cases, elementwise_comparisons=count,
        market_file_reads=0, confirmation_file_reads=0,
        model_fits=0, optimizer_steps=0,
        observed_confirmation_started=(ROOT / "CONFIRMATION_STARTED.json").exists(),
        static_findings=[
            "Shared Shanghai evaluation restores four-market vocabulary and ID=3 only after local levels/statistical anchors are attached.",
            "Both annual cutoffs use only fully mature training and validation labels; 2025 uses 2024 labels only through the frozen refresh rule.",
            "The complete task list and selected configurations must still be frozen and hashed before run opens sealed prices.",
        ],
    )
    (HERE / "CONFIRMATION_DATAFLOW_PREFLIGHT.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
