"""Export complete readable metrics and deterministic, byte-preserving archives.

No model training, data selection or inferential changes occur in this script.
Run against the completed original study directory.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import tarfile

ROOT = Path(__file__).resolve().parent


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def archive(path, files):
    members = []
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
                for p in sorted(files):
                    b = p.read_bytes()
                    rel = p.relative_to(ROOT).as_posix()
                    info = tarfile.TarInfo(rel)
                    info.size = len(b)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    tf.addfile(info, io.BytesIO(b))
                    members.append({"path": rel, "bytes": len(b),
                                    "sha256": hashlib.sha256(b).hexdigest()})
    manifest = {"archive": path.name, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "member_count": len(members),
                "uncompressed_member_bytes": sum(x["bytes"] for x in members),
                "members": members,
                "note": "Original member bytes preserved; deterministic archive container metadata."}
    path.with_name(path.name.replace(".tar.gz", "_MANIFEST.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "records").mkdir(exist_ok=True)
    fit_rows, metric_rows = [], []
    stage_files = {
        "D1": ("D1_RESULTS.json", "d1_receipts"),
        "D2": ("D2_RESULTS.json", "d2_receipts"),
        "LEARNING": ("LEARNING_DIAGNOSTIC_RESULTS.json", "learning_receipts"),
        "CONFIRMATION": ("CONFIRMATION_RESULTS.json", "confirmation_receipts"),
    }
    for stage, (filename, receipt_dir) in stage_files.items():
        doc = json.loads((ROOT / filename).read_text())
        for result in doc["results"]:
            task = result["task"]
            cfg = task["config"]
            receipt_rel = f"{receipt_dir}/{task['id']}.json"
            receipt = json.loads((ROOT / receipt_rel).read_text())
            row = {
                "stage": stage, "task_id": task["id"],
                "config_id": cfg["id"], "family": cfg["family"],
                "scope": task.get("scope", "pooled"),
                "markets": "|".join(task["market_names"]),
                "cutoff": task["cutoff"], "seed": task["seed"],
                "train_cap": task.get("train_cap", task.get("cap")),
                "selection": task.get("selection", "recent"),
                "purpose": task.get("purpose", ""),
                "epochs": result.get("epochs", receipt.get("epochs")),
                "selected_epoch": result.get("selected_epoch", receipt.get("selected_epoch")),
                "seconds": result.get("seconds", receipt.get("seconds")),
                "parameter_count": receipt.get("parameter_count"),
                "validation_score": receipt.get("validation_score"),
                "full_score": result["metrics"]["full"]["score"],
                "state_sha256": receipt.get("state_sha256"),
                "receipt_path": receipt_rel,
                "receipt_sha256": hashlib.sha256((ROOT / receipt_rel).read_bytes()).hexdigest(),
                "config_json": encode(cfg), "task_json": encode(task),
            }
            fit_rows.append(row)
            for view, view_result in result["metrics"].items():
                for cell in view_result["cells"]:
                    metric_rows.append({"stage": stage, "task_id": task["id"],
                                        "config_id": cfg["id"], "scope": row["scope"],
                                        "seed": task["seed"], "view": view,
                                        **cell})
        for cutoff, controls in doc.get("controls", {}).items():
            for name, data in controls.items():
                for cell in data["cells"]:
                    metric_rows.append({"stage": stage, "task_id": f"control_{cutoff}_{name}",
                                        "config_id": name, "scope": "deterministic_control",
                                        "seed": "", "view": "full", **cell})
    assert len(fit_rows) == 385
    assert len({(x["stage"], x["task_id"]) for x in fit_rows}) == 385
    csv_write(ROOT / "results/ALL_FITS.csv", fit_rows)
    csv_write(ROOT / "results/ALL_METRICS.csv", metric_rows)
    cfgs = json.loads((ROOT / "DEVELOPMENT_AGGREGATE.json").read_text())["development_summary"]
    config_rows = []
    for c in cfgs:
        config_rows.append({
            "scope": c["scope"], "config_id": c["config_id"], "train_cap": c["train_cap"],
            "fits": c["fits"], "seeds": "|".join(map(str, c["seeds"])),
            "score": c["score"], "selected_epoch_zero": c["selected_epoch_zero"],
            "config_json": encode(c["config"]), "per_seed_json": encode(c["per_seed"]),
            "per_market_json": encode(c["per_market"]), "per_year_json": encode(c["per_year"]),
            "per_horizon_json": encode(c["per_horizon"]), "views_json": encode(c["views"]),
            "task_ids_json": encode(c["task_ids"]),
        })
    assert len(config_rows) == 49
    csv_write(ROOT / "results/ALL_CONFIGS.csv", config_rows)
    summary = json.loads((ROOT / "CONFIRMATION_SUMMARY.json").read_text())
    confirmation_rows = []
    for c in summary["models"]:
        confirmation_rows.append({
            "name": c["name"], "score": c["score"], "year_2024": c["per_year"]["2024"],
            "year_2025": c["per_year"]["2025"], "seeds": "|".join(map(str, c["seeds"])),
            "h1": c["per_horizon"]["1"], "h5": c["per_horizon"]["5"],
            "h20": c["per_horizon"]["20"], "per_seed_json": encode(c.get("per_seed", {})),
        })
    csv_write(ROOT / "results/CONFIRMATION_MODEL_SUMMARY.csv", confirmation_rows)
    # Freeze records and per-fit training traces; do not rewrite their JSON.
    record_files = []
    prefixes = ("D1_", "D2_", "LEARNING_DIAGNOSTIC_", "CONFIRMATION_", "TIMING_")
    for p in ROOT.iterdir():
        if p.is_file() and p.name.startswith(prefixes) and p.suffix in {".json", ".jsonl", ".log"}:
            record_files.append(p)
    for directory in ("d1_receipts", "d2_receipts", "learning_receipts", "confirmation_receipts"):
        record_files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    record_files.extend(ROOT / name for name in
                        ("POSTPROCESSING_CORRECTION.json", "IMPLEMENTATION_CHECK.json"))
    record_manifest = archive(ROOT / "records/RUN_RECORDS.tar.gz", record_files)
    c_files = sorted((ROOT / "local_only/confirmation").glob("*.npz"))
    assert len(c_files) == 130  # 62 predictor states + 62 predictions + 6 imputer states
    array_manifest = archive(ROOT / "records/CONFIRMATION_ARRAYS.tar.gz", c_files)
    receipt = {
        "formal_predictor_rows": len(fit_rows), "metric_rows": len(metric_rows),
        "configuration_groups": len(config_rows), "confirmation_model_rows": len(confirmation_rows),
        "record_archive": {k: v for k, v in record_manifest.items() if k != "members"},
        "confirmation_archive": {k: v for k, v in array_manifest.items() if k != "members"},
        "new_market_fits": 0, "new_statistical_tests": 0,
        "development_state_arrays_in_public_archive": False,
        "confirmation_state_arrays_in_public_archive": "all 62 predictors and 6 imputers",
    }
    (ROOT / "results/DELIVERY_EXPORT_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
