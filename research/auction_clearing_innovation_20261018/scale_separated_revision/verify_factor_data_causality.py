"""Independent input identities, source ratios and counterfactual time audit.

Clones only corrected development inputs into temporary directories. Perturbed
event fields are never written to the study. No fitting or scoring occurs.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)


def main():
    root = Path(__file__).resolve().parent
    parent = root.parent / "calendar_correction_20261018/rerun"
    builder = root / "prepare_factorized_examples.py"
    freeze_path = root / "PROTOCOL_FREEZE.json"
    freeze_sha = "327c96fb050e60b86cb58dfcf23c47ca971ddfe910201687cf1c54f7f8917a62"
    assert sha(freeze_path) == freeze_sha
    freeze = json.loads(freeze_path.read_text())
    for rel, expected in freeze["frozen_sha256"].items():
        assert sha(root / rel) == expected, rel
    old = arrays(parent / "local_only/examples_development.npz")
    new = arrays(root / "local_only/examples_development.npz")
    meta = pd.read_csv(parent / "local_only/example_metadata.csv")
    new_meta = pd.read_csv(root / "local_only/example_metadata.csv")
    pd.testing.assert_frame_equal(meta, new_meta, check_exact=True)
    ev = pd.read_csv(parent / "local_only/auction_events_development.csv")
    ev = ev.sort_values(["date", "platform"], kind="stable").reset_index(drop=True)
    assert ev.date.max() < "2024-01-01" and meta.target_date.max() < "2024-01-01"
    assert (meta.history_last_date < meta.issue_date).all()
    assert (meta.origin_date < meta.issue_date).all()
    assert (meta.calendar_available_date <= meta.issue_date).all()
    assert len(new_meta) == 4141
    immutable = [k for k in old if k not in ("base", "context")]
    assert all(equal(old[k], new[k]) for k in immutable)
    assert equal(old["context"], new["context"][:, :old["context"].shape[1]])

    # Independent source calculation uses sorted array positions/searchsorted,
    # not the builder's per-origin dataframe selection/cache implementation.
    p = ev.clearing_price_eur_per_tco2.to_numpy(float)
    m = ev.bid_median_eur_per_tco2.to_numpy(float)
    good = np.isfinite(p) & (p > 0) & np.isfinite(m) & (m > 0)
    eligible = np.flatnonzero(good)
    dates = ev.date.to_numpy()[eligible]
    id_to_index = dict(zip(ev.event_id, range(len(ev))))
    ref = np.array([id_to_index[x] for x in meta.reference_event_id], dtype=int)
    assert np.all(ev.date.to_numpy()[ref] <= meta.origin_date.to_numpy())
    assert np.all(ev.date.to_numpy()[ref] < meta.issue_date.to_numpy())
    assert np.array_equal(m[ref], new["last_bid_median"])
    independent_micro, last_observed = [], []
    for i, row in enumerate(meta.itertuples(index=False)):
        end = int(np.searchsorted(dates, row.origin_date, side="right"))
        selected = eligible[end-60:end]
        assert len(selected) == 60
        assert np.all(ev.date.to_numpy()[selected] < row.issue_date)
        independent_micro.append(max(float(np.median(np.abs(p[selected]-m[selected]))), .01, .0001*m[ref[i]]))
        last_observed.append(str(ev.date.to_numpy()[selected].max()))
    independent_micro = np.asarray(independent_micro, dtype=np.float64)
    assert equal(independent_micro, new["micro_scale"])
    b0 = ev.total_bid_volume_tco2.to_numpy(float)[ref]
    s0 = ev.supply_tco2.to_numpy(float)[ref]
    cal = pd.read_csv(parent / "local_only/calendar_vintages_development.csv").fillna("")
    cmap = {(r.vintage_id, r.scheduleddate, r.platform): r for r in cal.itertuples(index=False)}
    plans = [cmap[(r.calendar_vintage_id, r.target_date, r.platform)] for r in meta.itertuples(index=False)]
    assert all(p.revision_available_date <= r.issue_date for p, r in zip(plans, meta.itertuples(index=False)))
    assert all(not p.valid_until_exclusive or r.issue_date < p.valid_until_exclusive for p, r in zip(plans, meta.itertuples(index=False)))
    planned = np.array([float(p.plannedvolume) for p in plans])
    physical = np.column_stack([s0/b0, planned/b0, new["scale"]/new["anchor"],
                                independent_micro/new["last_bid_median"], new["last_bid_median"]/new["anchor"]])
    assert new["base"].dtype == np.float64 and equal(physical, new["base"])
    extra_context = np.column_stack([np.log(independent_micro/new["anchor"]),
                                     np.log(new["last_bid_median"]/new["anchor"])]).astype(np.float32)
    assert equal(extra_context, new["context"][:, -2:])

    protected = ["x", "context", "base", "anchor", "scale", "last_bid_median", "micro_scale"]
    source_files = ["local_only/examples_development.npz", "local_only/example_metadata.csv",
                    "local_only/auction_events_development.csv", "local_only/calendar_vintages_development.csv", "EXAMPLE_AUDIT.json"]
    source_hashes = {rel: sha(parent / rel) for rel in source_files}
    tests = []
    with tempfile.TemporaryDirectory(prefix="factor_causality_") as scratch:
        scratch = Path(scratch)
        clone = scratch / "source"
        (clone / "local_only").mkdir(parents=True)
        for rel in source_files:
            shutil.copyfile(parent / rel, clone / rel)

        def run(tag):
            output = scratch / tag
            subprocess.run([sys.executable, str(builder), "--source-study", str(clone), "--study-dir", str(output)],
                           text=True, capture_output=True, check=True)
            return arrays(output / "local_only/examples_development.npz")

        replay = run("unperturbed")
        assert set(replay) == set(new) and all(equal(replay[k], new[k]) for k in new)
        with (parent / "local_only/auction_events_development.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, source_rows = list(reader.fieldnames), list(reader)

        def mutate(event_id, factors):
            # Preserve every unmodified lexical cell to avoid whole-file
            # dataframe float round-trip changes in a counterfactual test.
            changed = [dict(row) for row in source_rows]
            hits = 0
            for row in changed:
                if row["event_id"] == event_id:
                    hits += 1
                    for key, factor in factors.items():
                        row[key] = repr(float(row[key]) * factor)
            assert hits == 1
            with (clone / "local_only/auction_events_development.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(changed)

        choices = []
        for year in (2021, 2022, 2023):
            eligible_rows = ev[(ev.date >= f"{year}-06-01") & (ev.date <= f"{year}-06-10")
                               & ev.event_id.isin(meta.result_event_id) & np.isfinite(ev.bid_median_eur_per_tco2)
                               & np.isfinite(ev.clearing_price_eur_per_tco2)]
            chosen = eligible_rows.iloc[0]
            choices.append(chosen)
            mutate(chosen.event_id, {"bid_median_eur_per_tco2": 3.7, "clearing_price_eur_per_tco2": 1.9,
                                      "total_bid_volume_tco2": 2.3, "supply_tco2": 1.6})
            altered = run(f"future_event_{year}")
            before = (meta.origin_date < chosen.date).to_numpy()
            unchanged = {k: equal(new[k][before], altered[k][before]) for k in protected}
            assert all(unchanged.values())
            labels_changed = ~(np.equal(new["future_median_label"], altered["future_median_label"])
                               | (np.isnan(new["future_median_label"]) & np.isnan(altered["future_median_label"])))
            assert (labels_changed & before).any(), "Mutation must actually reach a future label"
            tests.append({"case": "future_raw_event_fields", "event_date": chosen.date,
                          "event_platform": chosen.platform, "origins_strictly_before_count": int(before.sum()),
                          "protected_inputs_unchanged": unchanged,
                          "future_labels_changed_before_event": int((labels_changed & before).sum()),
                          "pass": True})
        chosen = choices[1]
        mutate(chosen.event_id, {"supply_tco2": 1.73})
        altered = run("actual_supply_only")
        assert equal(new["base"][:, 1], altered["base"][:, 1])
        historical_supply_effect = int(np.count_nonzero(new["base"][:, 0] != altered["base"][:, 0]))
        assert historical_supply_effect > 0, "Positive control: source supply must change a later historical reference"
        tests.append({"case": "actual_supply_only", "event_date": chosen.date, "event_platform": chosen.platform,
                      "planned_supply_base_column_1_unchanged_all_4141_examples": True,
                      "historical_reference_base_column_0_changed_later_examples": historical_supply_effect,
                      "pass": True})
    assert all(sha(parent / rel) == expected for rel, expected in source_hashes.items())
    for rel, expected in freeze["frozen_sha256"].items():
        assert sha(root / rel) == expected, rel
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "scope": "Independent source identities, past-only 60-auction micro scale and raw physical ratios; exact clone replay; 4 counterfactual development-data cases; no fitting or scoring",
        "freeze_id": freeze["freeze_id"], "freeze_sha256": freeze_sha,
        "auditor_sha256": sha(Path(__file__).resolve()), "builder_sha256": sha(builder),
        "n_examples": len(meta), "n_origins": int(meta.origin_date.nunique()),
        "immutable_parent_arrays_identical": immutable,
        "metadata_all_fields_identical_to_corrected": True,
        "old_context_columns_identical": True, "source_history_strictly_before_issue": True,
        "reference_event_strictly_before_issue": True, "all_micro_histories_have_60_eligible_past_events": True,
        "independent_micro_scale_exact_match": True, "base_float64_exact_source_ratios": True,
        "base_column_1_source": "corrected as-of calendar plannedvolume divided by known reference-event total bid volume",
        "future_actual_supply_not_used_in_base_column_1": True,
        "new_context_exact_past_only_values": True, "temporary_builder_replay_all_arrays_identical": True,
        "counterfactuals": tests, "source_input_hashes_unchanged": source_hashes,
        "all_frozen_hashes_unchanged": True, "frozen_file_count": len(freeze["frozen_sha256"]),
        "confirmation_year_contents_read": False,
        "limits": "This audit verifies the data builder and inference input arrays. Later model training must still mask auxiliary labels by maturity and keep them out of inference; training implementation is a separate review. Annual historical result fields retain the project's next-day availability assumption.",
    }
    destination = root / "results/FACTOR_DATA_CAUSALITY_CHECK.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "n_examples": len(meta), "counterfactual_cases": len(tests),
                      "all_frozen_hashes_unchanged": True, "output": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
