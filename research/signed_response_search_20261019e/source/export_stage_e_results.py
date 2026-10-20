"""Export authorized numerical research summaries, never raw row-level files.

Self-trained numeric state_dict values are intentionally retained under the
existing authorization. Source raw tables, NPZ arrays and binary checkpoints
are excluded by an exact JSON-only output allowlist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def allowed(p):
    return p.suffix == ".json" and (
        p.name in {"PROTOCOL_BEFORE_RUN.json", "INPUT_MANIFEST.json", "EXECUTION_COMPLETE.json",
                   "ALL_SUMMARIES.json", "SUMMARY.json", "REFIT.json", "SELECTION.json"}
        or p.name.startswith("selection_") or p.name.startswith("refit_seed"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--include-budget320", action="store_true",
                        help="Export completed post-exposure sensitivity with a separate receipt")
    args = parser.parse_args()
    target = STUDY/"results"
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "runs": {},
               "policy": "JSON allowlist; self-trained numeric parameter states retained; third-party raw, per-origin predictions and binary weights are not copied."}
    run_names = ["core", "neural_references"]
    if args.include_budget320:
        run_names.append("budget320")
    for name in run_names:
        src = args.run_root/name
        complete = json.loads((src/"EXECUTION_COMPLETE.json").read_text())
        assert complete["status"] == "completed" and complete["source_hashes_unchanged"]
        selected = sorted(p for p in src.rglob("*.json") if allowed(p))
        records = []
        for p in selected:
            json.loads(p.read_text())
            rel = p.relative_to(src)
            out = target/name/rel
            if out.exists() and digest(out) != digest(p):
                raise FileExistsError(f"Refuse to overwrite differing exported evidence: {out}")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, out)
            records.append({"path": str(out.relative_to(STUDY)), "sha256": digest(out), "bytes": out.stat().st_size})
        receipt["runs"][name] = {"files": records,
            "private_prediction_files": [{"filename": str(p.relative_to(src)), "sha256": digest(p), "bytes": p.stat().st_size}
                                         for p in sorted(src.rglob("predictions.npz"))],
            "excluded_binary_checkpoints": sum(1 for _ in src.rglob("*.pt"))}
    receipt_name = "RESULT_EXPORT_RECEIPT_WITH_BUDGET.json" if args.include_budget320 else "RESULT_EXPORT_RECEIPT.json"
    path = STUDY/"provenance"/receipt_name
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: {"json_files": len(v["files"]), "private_prediction_count": len(v["private_prediction_files"]),
                         "excluded_pt": v["excluded_binary_checkpoints"]} for k, v in receipt["runs"].items()}))


if __name__ == "__main__":
    main()
