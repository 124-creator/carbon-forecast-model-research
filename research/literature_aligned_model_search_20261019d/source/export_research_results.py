"""Export aggregate results and complete fit logs without copying source price paths.

This is a read-only postprocessing operation on completed experiments.  It never
fits a model, changes a selected checkpoint, or selects which cells to report.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
DEFAULT_RUN_ROOT = Path("/tmp/carbon-literature-aligned-runs-20261019d")
RUN_NAMES = ["main_v1", "liquidity_v2", "new_market_v3", "swewt_gru_supplement", "official_tianjin_v4"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_price(market, official_data_root):
    path = (official_data_root/f"{market}.csv" if market == "TIANJIN_OFFICIAL_MEAN" else
            STUDY/"data/sciencedb_819_v1/prepared"/f"{market}.csv"
            if market.endswith("_MEAN") else
            REPO/"research/cross_market_innovation_20261017/data"/f"{market}_development.csv")
    df = pd.read_csv(path).sort_values("date").reset_index(drop=True)
    return df.price.to_numpy(float)


def export_one(root, name, official_data_root):
    src, dest = root/name, STUDY/"results"/name
    complete = json.loads((src/"EXECUTION_COMPLETE.json").read_text())
    assert complete["status"] == "completed" and complete["source_hashes_unchanged"]
    dest.mkdir(parents=True, exist_ok=True)
    for filename in ["PROTOCOL_BEFORE_RUN.json", "EXECUTION_COMPLETE.json", "ALL_SUMMARIES.json"]:
        shutil.copyfile(src/filename, dest/filename)
    manifests, choices, traces, checkpoints, prediction_files, native = [], [], [], [], [], []
    for path in sorted(src.glob("*/INPUT_MANIFEST.json")):
        manifests.append(json.loads(path.read_text()))
    additional_manifests = [{"relative_path": str(p.relative_to(src)), **json.loads(p.read_text())}
                            for p in sorted(src.glob("*/SWEWT_INPUT_MANIFEST.json"))]
    for pattern in ["*/*/CHOICE_BEFORE_TEST.json", "*/*/SELECTION.json"]:
        for path in sorted(src.glob(pattern)):
            choices.append({"relative_path": str(path.relative_to(src)), **json.loads(path.read_text())})
    for pattern in ["*/*/selection_lr*.json", "*/*/refit_seed*.json"]:
        for path in sorted(src.glob(pattern)):
            trace = json.loads(path.read_text())
            checkpoint = path.with_suffix(".pt")
            assert trace["checkpoint_sha256"] == sha(checkpoint)
            traces.append({"relative_path": str(path.relative_to(src)), **trace})
            checkpoints.append({"relative_path": str(checkpoint.relative_to(src)),
                                "sha256": sha(checkpoint), "bytes": checkpoint.stat().st_size})
    for path in sorted(src.glob("*/*/predictions.npz")):
        summary = json.loads(path.with_name("SUMMARY.json").read_text())
        market, family, fold = summary["market"], summary["family"], summary["year"]
        price = load_price(market, official_data_root)
        with np.load(path) as z:
            predictions, observed, origins, horizons = [z[k].copy() for k in ["predictions", "observed", "origins", "horizons"]]
        level = np.stack([price[origins+h] for h in horizons], axis=1)
        expected = level-price[origins,None]
        assert np.allclose(observed, expected, atol=1e-12, rtol=1e-12)
        errors = predictions-observed[None]
        for j,h in enumerate(horizons):
            per_seed_mse = np.mean(errors[:,:,j]**2, axis=1)
            rw_mse = float(np.mean(observed[:,j]**2))
            actual_variance = float(np.mean((level[:,j]-level[:,j].mean())**2))
            native.append({
                "market": market, "fold": fold, "family": family, "horizon": int(h), "n_origins": len(origins),
                "mean_seed_mse_over_rw": float(per_seed_mse.mean()/rw_mse),
                "mean_seed_mse": float(per_seed_mse.mean()),
                "mean_seed_rmse": float(np.sqrt(per_seed_mse).mean()),
                "rmse_of_mean_seed_mse": float(np.sqrt(per_seed_mse.mean())),
                "mean_seed_mae": float(np.mean(np.abs(errors[:,:,j]))),
                "mean_seed_mape_percent": float(100*np.mean(np.abs(errors[:,:,j])/np.abs(level[None,:,j]))),
                "mean_seed_r2_price_levels": float(1-per_seed_mse.mean()/actual_variance) if actual_variance > 0 else None,
                "rw_mse": rw_mse,
            })
        prediction_files.append({"relative_path": str(path.relative_to(src)), "sha256": sha(path),
                                 "bytes": path.stat().st_size, "seeds": len(predictions), "origins": len(origins)})
    write(dest/"INPUT_MANIFESTS.json", manifests)
    if additional_manifests:
        write(dest/"ADDITIONAL_INPUT_MANIFESTS.json", additional_manifests)
    write(dest/"ALL_SELECTIONS.json", choices)
    write(dest/"CHECKPOINT_MANIFEST.json", checkpoints)
    write(dest/"PREDICTION_FILE_MANIFEST.json", prediction_files)
    with (dest/"FIT_TRACES.json.gz").open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as f:
            f.write(json.dumps(traces, ensure_ascii=False, separators=(",", ":")).encode())
    ledger = {
        "neural_tasks": len(traces),
        "selection_tasks": sum(t["mode"] == "train" for t in traces),
        "refit_tasks": sum(t["mode"] == "final" for t in traces),
        "tasks_with_positive_gradient_steps": sum(t["gradient_positive_steps"] > 0 for t in traces),
        "refits_with_positive_gradient_steps": sum(t["mode"] == "final" and t["gradient_positive_steps"] > 0 for t in traces),
        "epoch_zero_refits": sum(t["mode"] == "final" and t["epochs_executed"] == 0 for t in traces),
        "selection_epochs_executed": sum(t["epochs_executed"] for t in traces if t["mode"] == "train"),
        "refit_epochs_executed": sum(t["epochs_executed"] for t in traces if t["mode"] == "final"),
        "classical_estimator_fits": 0 if name == "swewt_gru_supplement" else 33*len(manifests),
        "classical_count_definition": "each cell: two Ridge families x (8 selection + 1 final) multi-output fits; HGB four configurations x three horizons + three final scalar-output fits = 33. RW/drift have no estimator.fit.",
        "unique_origins": sum(m["counts"]["test"] for m in manifests),
        "origin_horizon_targets": 3*sum(m["counts"]["test"] for m in manifests),
        "scope": "formal market pipeline only; synthetic implementation checks and independent read-only numerical replays counted separately",
    }
    write(dest/"EXECUTION_LEDGER.json", ledger)
    pd.DataFrame(native).to_csv(dest/"PAPER_STYLE_NATIVE_METRICS.csv", index=False)
    table = pd.DataFrame(native)
    table.groupby(["market","fold","family"], as_index=False)["mean_seed_mse_over_rw"].mean().to_csv(dest/"CELL_METRICS.csv", index=False)
    write(dest/"EXPORT_RECEIPT.json", {
        "source_run_directory": str(src), "formal_fit_count_added_by_export": 0,
        "full_trace_sha256": sha(dest/"FIT_TRACES.json.gz"), "ledger": ledger,
        "public_output_scope": "aggregates, selections, complete fit traces, source and checkpoint/prediction hashes; no newly acquired unlicensed author's raw data or per-origin target/prediction paths copied by this exporter",
        "metrics": "MSE/RMSE/MAE in original source price units; MAPE uses actual future price level; R2 uses future price-level variance within the same market/fold/horizon; average individual seed metrics as named. No new selection or hypothesis tests.",
    })
    return {"run": name, **ledger}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--official-data-root", type=Path, default=Path("/tmp/carbon-next-market-data-20261019d"))
    parser.add_argument("--runs", nargs="+", choices=RUN_NAMES, default=RUN_NAMES)
    args = parser.parse_args()
    print(json.dumps([export_one(args.run_root, name, args.official_data_root) for name in args.runs], indent=2))


if __name__ == "__main__":
    main()
