"""Fixed later-period official Tianjin evaluation; no edits to prior sources.

Primary identity was selected after earlier v1-v3/SWEWT results, before this
official tail's model evaluation. The published rounded daily mean is retained.
--preflight/--self-test performs identity, boundary and forward checks only.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_new_market_replication as replication
import run_swewt_gru_comparison as swewt

common = replication.common
SOURCE = Path(__file__).resolve().parent
MARKET, FOLD = "TIANJIN_OFFICIAL_MEAN", "fixed602020"
DATA_FILE = Path("/tmp/carbon-next-market-data-20261019d/TIANJIN_OFFICIAL_MEAN.csv")
DATA_SHA = "85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a"
PLAN_FILE = common.STUDY / "audits/V4_STATISTICAL_AUDIT_PLAN.json"
PLAN_SHA = "8645aa650d5202852d00700f5bb7a47f907c5c7476db7dd84a923cecc7eae913"
N, TRAIN_END, VALIDATION_END = 1077, 646, 861
COUNTS = {"train": 562, "val": 195, "final": 777, "test": 196}
FAMILIES = [*replication.FAMILIES, *swewt.FAMILIES]
CLASSICAL = ["rw", "drift", "ridge_price", "ridge_full", "hgb_full"]
OLD_SOURCE_SHA = {
    "run_model_search.py": "26347433fe6416298e4969da618cbba291999f3a66131172bc8f2e510f3633bf",
    "innovation_models.py": "cca83f45474c5110f180ae8b7763131f33596a664939c4ebd34c32db24b4dc1e",
    "run_liquidity_extension.py": "95665858a1945b00bba6d6233022c34c0a58993899ccadda8c895f64ffe69807",
    "liquidity_models.py": "91dd66c0b33b8d75100e37cc382f01be44021ae538f6cbbedddda7f9b9386e0b",
    "run_new_market_replication.py": "8c617a714bb0b47d117b213de9d299c3bd6a534c8211c28c8d039d454df12e56",
    "literature_diagnostic.py": "e9053bfd34bce9e60c67c0590245271ccecaaf0bcf009a1e715be9abe7ada2e5",
    "run_swewt_gru_comparison.py": "0b3395f547fad792c6880a9d25819c6e326ab48ef733240ad690e4833b63218d",
}


def checked_plan():
    assert common.digest(PLAN_FILE) == PLAN_SHA, "Frozen v4 analysis plan changed"
    plan = json.loads(PLAN_FILE.read_text())
    assert plan["data_identity"]["prepared_sha256"] == DATA_SHA
    assert plan["neural_families"] == FAMILIES
    assert plan["split"]["row_boundaries"] == [0, TRAIN_END, VALIDATION_END, N]
    assert plan["split"]["counts"] == COUNTS
    assert common.LRS == [.003, .01] and common.SEEDS == [17, 29, 43]
    assert (common.MAX_EPOCHS, common.MIN_EPOCHS, common.PATIENCE, common.BATCH_SIZE) == (80, 15, 12, 128)
    assert common.LOOKBACK == 64 and common.HORIZONS.tolist() == [1, 5, 20]
    assert common.ENERGY_LAG_DAYS == 14
    for name, sha in OLD_SOURCE_SHA.items():
        assert common.digest(SOURCE / name) == sha, f"Frozen source changed: {name}"
    return plan


def load_official(market=MARKET):
    assert market == MARKET
    assert common.digest(DATA_FILE) == DATA_SHA, "Official prepared data identity changed"
    df = pd.read_csv(DATA_FILE)
    assert list(df) == ["date", "price", "volume_tons", "turnover_yuan"]
    assert len(df) == N and df.date.is_unique and df.date.is_monotonic_increasing
    assert [df.date.iloc[0], df.date.iloc[-1]] == ["2013-12-26", "2026-09-30"]
    price, volume, turnover = [df[c].to_numpy(float) for c in ["price", "volume_tons", "turnover_yuan"]]
    assert np.isfinite(np.column_stack([price, volume, turnover])).all()
    assert (price > 0).all() and (volume > 0).all() and (turnover >= 0).all()
    assert np.max(np.abs(price * 100 - np.round(price * 100))) < 1e-9
    # Published two-decimal prices are the target; turnover/volume is never
    # substituted. The half-cent discrepancy is only an identity/QC check.
    assert np.max(np.abs(price - turnover / volume)) <= .005000001
    dates = pd.to_datetime(df.date).to_numpy(dtype="datetime64[D]")
    tokens = [np.r_[0., np.diff(price)]]
    sources = [{"path": str(DATA_FILE), "sha256": DATA_SHA,
                "official_url": "https://www.chinatcx.com.cn/tshq",
                "source": "first TJEA table, daily-total published rounded mean and total tonnes",
                "license": "no redistribution license established; raw and prepared source remain private"}]
    alignments = []
    for name in ["brent", "wti", "henryhub"]:
        path = common.REPO / "data/parsed" / f"{name}_eia_daily.csv"
        ef = pd.read_csv(path).sort_values("date")
        missing = int(ef.spot_price.isna().sum())
        ef = ef.dropna(subset=["spot_price"])
        ed = pd.to_datetime(ef.date).to_numpy(dtype="datetime64[D]")
        ep = ef.spot_price.to_numpy(float)
        j = np.searchsorted(ed, dates - np.timedelta64(14, "D"), side="right") - 1
        assert (j >= 0).all() and np.isfinite(ep[j]).all()
        assert (ed[j] <= dates - np.timedelta64(14, "D")).all()
        tokens.append(np.r_[0., np.diff(ep[j])])
        sources.append({"path": str(path.relative_to(common.REPO)), "sha256": common.digest(path)})
        alignments.append({"series": name, "minimum_age_days": int(np.min((dates-ed[j]).astype(int))),
                           "maximum_age_days": int(np.max((dates-ed[j]).astype(int))),
                           "source_missing_price_rows_excluded_before_backward_asof": missing})
    logs = np.log1p(volume)
    context = np.zeros((N, 2), dtype="float32")
    for t in range(10, N):
        past = logs[max(0, t-60):t]
        context[t] = [np.clip((logs[t]-np.median(past))/max(float(np.std(past)), .25), -5, 5), 1.]
    return {"dates": dates, "price": price, "tokens": np.column_stack(tokens),
            "volume_context": context, "sources": sources, "alignment": alignments}


def prepare_official(market=MARKET, fold=FOLD):
    assert market == MARKET and fold == FOLD
    d = load_official(market)
    dates, p, tokens = d["dates"], d["price"], d["tokens"]
    origins = np.arange(64, N-20)
    masks = {"train": origins+20 < TRAIN_END,
             "val": (origins >= TRAIN_END) & (origins+20 < VALIDATION_END),
             "final": origins+20 < VALIDATION_END,
             "test": origins >= VALIDATION_END}
    assert {k: int(v.sum()) for k, v in masks.items()} == COUNTS
    assert origins[masks["train"]][-1]+20 < origins[masks["val"]][0]
    assert origins[masks["val"]][-1]+20 < origins[masks["test"]][0]
    assert str(dates[VALIDATION_END]) == "2023-04-19"
    assert str(dates[origins[masks["test"]][-1]]) == "2026-07-02"
    scale = np.sqrt(np.mean(tokens[1:TRAIN_END]**2, axis=0)).clip(1e-7)
    x = np.stack([tokens[t-63:t+1]/scale for t in origins]).astype("float32")
    context = np.stack([d["volume_context"][t-63:t+1] for t in origins])
    x = np.concatenate([x, context], axis=-1)
    y = np.stack([p[origins+h]-p[origins] for h in common.HORIZONS], axis=1)
    variance = np.mean(y[masks["train"]]**2, axis=0).clip(scale[0]**2*1e-6)
    manifest = {
        "market": market, "fold": fold, "legacy_year_field_is_fold_id": True,
        "target": "official TJEA daily total transaction published two-decimal mean, CNY/tonne; not close",
        "source_processing": "all1077 official daily-total records retained and sorted; no interpolation, deletion, monthly outlier replacement or recomputation from turnover/volume",
        "source_vintage": "current public historical snapshot; per-day publication vintages unverified",
        "row_boundaries": [0, TRAIN_END, VALIDATION_END, N], "counts": COUNTS,
        "origin_ranges": {k: [int(origins[v][0]), int(origins[v][-1])] for k, v in masks.items()},
        "first_train_origin": str(dates[origins[masks["train"]][0]]),
        "last_train_label": str(dates[origins[masks["train"]][-1]+20]),
        "first_validation_origin": str(dates[origins[masks["val"]][0]]),
        "last_validation_label": str(dates[origins[masks["val"]][-1]+20]),
        "first_test_origin": str(dates[origins[masks["test"]][0]]),
        "last_test_origin": str(dates[origins[masks["test"]][-1]]),
        "last_test_label": str(dates[origins[masks["test"]][-1]+20]),
        "training_rms": scale.tolist(), "train_target_second_moment": variance.tolist(),
        "sources": d["sources"], "energy_alignment": d["alignment"],
        "volume": {"unit": "tonnes", "valid_rows": N, "usable_context_rows": N-10,
                   "normalization": "current log1p total tonnes minus prior60record median, divided by prior std floored.25, clipped[-5,5]; >=10 past observations",
                   "information": "current official price and total volume known after origin report; turnover not an input",
                   "raw_information_span": "64 feature tokens plus up to60 earlier records for volume references"},
        "role": "fixed later-period replication on one official market; not a cross-market macro",
    }
    return {**d, "origins": origins, "x": x, "y": y, "masks": masks,
            "scale": scale, "variance": variance, "manifest": manifest}


def prepare_official_swewt(market=MARKET, fold=FOLD):
    d = prepare_official(market, fold)
    base = copy.deepcopy(d["manifest"])
    d["x"] = swewt.build_sequences(d["price"], d["origins"], d["scale"][0], d["x"])
    d["manifest"] = {**base, "swewt_adaptation": {
        "classification": "controlled multistep adaptation of Energies2024 DOI10.3390/en17174358, not numerical author reproduction",
        "ewt_window_price_levels": 64, "retained_sequence_length": 17, "modes": 3,
        "decomposition_information": "separate EWT only of price[t-63:t+1] for each origin; never a full-series decomposition",
        "columns": ["raw_minus_origin", "low_band_minus_origin", "band2", "band3", "brent", "wti", "henryhub", "volume_surprise", "volume_valid"],
        "normalization": "four price channels divide original training target RMS; final17 energy/volume tokens retain common transforms",
        "arms": {"swewt_gru_price": 4, "swewt_gru_full": 9},
        "original_official_manifest": base}}
    return d


def make_official_model(family, seed):
    if family in swewt.FAMILIES:
        return swewt.make_swewt_model(family, seed)
    assert family in replication.FAMILIES
    return replication.make_replication_model(family, seed)


def install_worker_hooks(family, data_file):
    global DATA_FILE
    DATA_FILE = Path(data_file)
    common.prepare = prepare_official_swewt if family in swewt.FAMILIES else prepare_official
    common.make_model = make_official_model


def run_job(family, out, data_file):
    # Reset hooks for every task: a process may run ordinary and EWT arms.
    install_worker_hooks(family, data_file)
    if family == "classical":
        return common.classical_job(MARKET, FOLD, out)
    return common.neural_job(MARKET, FOLD, family, out)


def data_gate(d, plan):
    tail = d["dates"][VALIDATION_END:]
    assert tail[0] > np.datetime64("2017-12-31")
    checks = []
    for old in plan["launch_date_gate"]["independent_old_source_date_checks"]:
        path = common.STUDY / "data/sciencedb_819_v1/prepared" / (old["market"]+".csv")
        assert common.digest(path) == old["source_sha256"]
        dates = pd.to_datetime(pd.read_csv(path).date).to_numpy(dtype="datetime64[D]")
        overlap = len(np.intersect1d(tail, dates))
        assert overlap == 0
        checks.append({"old_market": old["market"], "old_source_sha256": old["source_sha256"], "tail_date_overlap": overlap})
    return checks


def preflight():
    torch.set_num_threads(1)
    plan = checked_plan()
    d, e = prepare_official(), prepare_official_swewt()
    gate = data_gate(d, plan)
    assert d["x"].shape == (993, 64, 6) and e["x"].shape == (993, 17, 9)
    for key in ["origins", "y", "scale", "variance"]:
        assert np.array_equal(d[key], e[key])
    assert all(np.array_equal(d["masks"][k], e["masks"][k]) for k in d["masks"])
    assert np.array_equal(e["x"][..., 4:], d["x"][:, -17:, 1:])
    check_origin = TRAIN_END
    perturbed = d["price"].copy()
    perturbed[check_origin+1:] += 12345.
    a = swewt.price_sequence_at_origin(d["price"], check_origin, d["scale"][0])
    b = swewt.price_sequence_at_origin(perturbed, check_origin, d["scale"][0])
    assert np.array_equal(a, b)
    models = {}
    for family in FAMILIES:
        install_worker_hooks(family, str(DATA_FILE))
        expected = prepare_official_swewt if family in swewt.FAMILIES else prepare_official
        assert common.prepare is expected and common.make_model is make_official_model
        model = common.make_model(family, 17).eval()
        x = (e if family in swewt.FAMILIES else d)["x"][:3]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            prediction = model(torch.from_numpy(x))
        assert prediction.shape == (3, 20) and bool(torch.isfinite(prediction).all())
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        models[family] = {"input_shape": list(x.shape), "output_shape": list(prediction.shape),
                          "parameters": sum(p.numel() for p in model.parameters()), "finite": True}
    install_worker_hooks("classical", str(DATA_FILE))
    assert common.prepare is prepare_official
    checked_plan()
    return {"status": "PASS_NO_FIT_NO_SCORE", "new_fit_count": 0, "optimizer_steps": 0,
            "new_tail_model_performance_scores_computed": 0, "source_sha256": source_hashes(),
            "statistics_plan_sha256": PLAN_SHA, "input_manifest": d["manifest"],
            "source_date_gate": gate, "swewt_future_perturbation_max_gap": 0.,
            "labels_masks_scales_unchanged_between_arms": True, "worker_routing_checked": True,
            "forward_checks": models, "runtime": swewt.dependency_receipt(),
            "planned_neural_tasks": 50, "planned_selection_tasks": 20, "planned_refit_tasks": 30,
            "planned_classical_estimator_fits": 33, "expected_summary_rows": 15}


def source_hashes():
    return {name: common.digest(SOURCE/name) for name in [*OLD_SOURCE_SHA, Path(__file__).name]}


def freeze_run(out):
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Preserve earlier results; use a new empty output directory")
    plan = checked_plan()
    d = prepare_official()
    gate = data_gate(d, plan)
    e = prepare_official_swewt()
    protocol = {
        "stage": plan["stage"], "created_unix": time.time(), "output_directory": str(out.resolve()),
        "design_timing": plan["design_timing"], "market": MARKET, "markets": [MARKET], "fold": FOLD,
        "fixed_primary_candidate": "allocation_readout", "prespecified_secondary_candidate": "liquidity_full",
        "target": plan["target_definition"], "data_identity": plan["data_identity"],
        "source_limitation": "official current public snapshot; daily vintage unverified; preserve published rounded price, no monthly anomaly replacement",
        "source_date_gate": gate, "split": plan["split"], "neural_families": FAMILIES, "classical_families": CLASSICAL,
        "statistics_plan_path": str(PLAN_FILE), "statistics_plan_sha256": PLAN_SHA,
        "source_sha256": source_hashes(), "runtime": swewt.dependency_receipt(),
        "learning_rates": common.LRS, "selection_seed": 17, "refit_seeds": common.SEEDS,
        "maximum_epochs": 80, "minimum_epochs_before_stopping": 15, "patience": 12, "batch_size": 128,
        "horizons_observation_steps": [1, 5, 20], "lookback": 64,
        "training_and_information": plan["training_and_information"], "estimand": plan["estimand"],
        "planned_neural_tasks": 50, "planned_selection_tasks": 20, "planned_refit_constructions": 30,
        "planned_classical_estimator_fits": 33,
        "classical_count_definition": "two Ridge families*(8selection+1refit)=18, HGB(4configs+1refit)*3horizons=15; total33 regression fits, excluding2 scaler fits and formula-only RW/drift",
        "expected_summary_rows": 15, "neural_family_jobs": 10, "classical_worker_jobs": 1,
        "swewt_worker_routing": "each EWT worker uses prepare_official_swewt with independent past64 windows; eight other neural arms and all classical families use ordinary64x6 prepare_official",
        "swewt_adaptation": e["manifest"]["swewt_adaptation"],
        "claims": plan["conclusions"], "raw_data_distribution": "raw and canonical source files remain private; code, summaries and authorized numeric learned parameters may be public",
    }
    common.write_json(out/"PROTOCOL_BEFORE_RUN.json", protocol)
    common.write_json(out/f"{MARKET}_{FOLD}"/"INPUT_MANIFEST.json", d["manifest"])
    common.write_json(out/f"{MARKET}_{FOLD}"/"SWEWT_INPUT_MANIFEST.json", e["manifest"])
    return protocol


def main():
    global DATA_FILE
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight", "--self-test", dest="preflight", action="store_true")
    parser.add_argument("--preflight-report", type=Path, default=common.STUDY/"audits/OFFICIAL_TIANJIN_RUNNER_PREFLIGHT.json")
    args = parser.parse_args()
    DATA_FILE = args.data_file.resolve()
    if args.preflight:
        receipt = preflight()
        common.write_json(args.preflight_report, receipt)
        print(json.dumps({"status": receipt["status"], "report": str(args.preflight_report),
                          "runner_sha256": receipt["source_sha256"][Path(__file__).name], "counts": COUNTS}), flush=True)
        return
    if args.out is None:
        parser.error("--out is required unless --preflight/--self-test is supplied")
    protocol = freeze_run(args.out)
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_job, family, str(args.out), str(DATA_FILE)): family for family in ["classical", *FAMILIES]}
        for future in as_completed(jobs):
            result = future.result()
            rows = result if isinstance(result, list) else [result]
            summaries.extend(rows)
            print(json.dumps({"completed": jobs[future], "scores": [{k: r[k] for k in ["family", "mean_seed_mse_over_rw", "ensemble_mse_over_rw"]} for r in rows]}), flush=True)
            common.write_json(args.out/"PROGRESS_SUMMARIES.json", summaries)
    checked_plan()
    assert source_hashes() == protocol["source_sha256"]
    assert common.digest(DATA_FILE) == DATA_SHA
    assert swewt.dependency_receipt()["ewtpy_module_sha256"] == protocol["runtime"]["ewtpy_module_sha256"]
    assert len(summaries) == 15 and {r["family"] for r in summaries} == set(FAMILIES+CLASSICAL)
    receipts = [json.loads(p.read_text()) for p in (args.out/f"{MARKET}_{FOLD}").glob("*/selection_lr*.json")]
    refits = [json.loads(p.read_text()) for p in (args.out/f"{MARKET}_{FOLD}").glob("*/refit_seed*.json")]
    assert len(receipts) == 20 and len(refits) == 30
    common.write_json(args.out/"ALL_SUMMARIES.json", summaries)
    common.write_json(args.out/"EXECUTION_COMPLETE.json", {
        "status": "completed", "worker_jobs": 11, "summary_rows": 15,
        "selection_tasks": len(receipts), "refit_constructions": len(refits), "neural_tasks": 50,
        "selection_tasks_with_positive_steps": sum(r["gradient_positive_steps"] > 0 for r in receipts),
        "refit_tasks_with_positive_steps": sum(r["gradient_positive_steps"] > 0 for r in refits),
        "classical_estimator_fits": 33, "source_hashes_unchanged": True,
        "statistics_plan_hash_unchanged": True, "data_hash_unchanged": True, "ewtpy_module_hash_unchanged": True})


if __name__ == "__main__":
    main()
