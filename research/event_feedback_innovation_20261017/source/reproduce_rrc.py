"""Reproduce the frozen D1 design in a new directory, without editing history.

`public-audit` needs only the published aggregate files. `replay` requires the
four old development CSVs and the separately prepared official EEX development
CSV. A replay is a NEW replication, never the original pre-registered run.
No confirmation file is requested, read or scored by this entry point.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ORIGINAL_PROTOCOL_SHA256 = "404c14867dcc5b43f456075d6b9edd4a5cd75cd99ad943a034e8da7172ef8a70"
OLD_FILES = {
    "GUANGZHOU": "GUANGZHOU_development.csv",
    "HUBEI": "HUBEI_development.csv",
    "SHANGHAI": "SHANGHAI_D4_development.csv",
    "EU_REPORTED": "EU_REPORTED_development.csv",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original(study):
    pp = study / "provenance/D1_PROTOCOL.json"
    if digest(pp) != ORIGINAL_PROTOCOL_SHA256:
        raise ValueError("Historical D1 protocol has changed")
    cfg = json.loads(pp.read_bytes())
    for rel, expected in cfg["source_sha256"].items():
        if digest(study / rel) != expected:
            raise ValueError(f"Frozen source has changed: {rel}")
    return pp, cfg


def destination(path):
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError("Use a new output directory; no overwrite or automatic restart")
    return path


def command(*args):
    subprocess.run([sys.executable, "-B", *map(str, args)], check=True)


def public_audit(study, args):
    pp, cfg = original(study)
    out = destination(args.output)
    # The original independent selector verifies the frozen protocol identity
    # and every aggregate cell. It does not open the paths inside the protocol.
    command(study / "source/independent_selection.py", "run", "--protocol", pp,
            "--study-root", study, "--output", out)


def replay(study, args):
    pp, cfg = original(study)
    out = destination(args.output)
    old_data = args.previous_study.resolve() / "data"
    for entry in cfg["markets"]:
        path = (args.eex_development.resolve() if entry["name"] == "EEX_EUA_T3PA"
                else old_data / OLD_FILES[entry["name"]])
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError(f"Development source mismatch: {entry['name']}")
        entry["path"] = str(path)
    if not 1 <= args.budget_minutes <= 240:
        raise ValueError("Set a finite replication budget of 1–240 minutes")
    plan = {"mode": "new_replication_of_failed_fixed_RRC_candidate",
            "original_protocol_sha256": ORIGINAL_PROTOCOL_SHA256,
            "market_names": [m["name"] for m in cfg["markets"]],
            "counts": cfg["counts"], "confirmation_access": False,
            "budget_minutes": args.budget_minutes, "output": str(out),
            "historical_results_overwritten": False,
            "performance_directed_grid_changes": False}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "market_fits": 0,
                          "no_output_created": True, **plan}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=False)
    (out / "source").mkdir()
    (out / "provenance").mkdir()
    for rel in cfg["source_sha256"]:
        shutil.copyfile(study / rel, out / rel)
    for name in ("select_rrc.py", "independent_selection.py"):
        shutil.copyfile(study / "source" / name, out / "source" / name)
    cfg["study_root"] = str(out)
    cfg["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    cfg["deadline_utc"] = (dt.datetime.now(dt.timezone.utc) +
                           dt.timedelta(minutes=args.budget_minutes)).isoformat()
    cfg["replication"] = plan
    cfg["phase"] = "New replication of original D1; not a new advancement attempt"
    rp = out / "provenance/REPLAY_PROTOCOL.json"
    rp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    (out / "provenance/ORIGINAL_D1_PROTOCOL.json").write_bytes(pp.read_bytes())
    (out / ".gitignore").write_text("local_only/\n__pycache__/\n*.pyc\n")
    command(out / "source/run_rrc_development.py", "--protocol", rp)
    command(out / "source/select_rrc.py", "--protocol", pp, "--metrics",
            out / "results/D1/ALL_METRICS.csv", "--out", out / "results/D1")
    # The independent CLI deliberately accepts only the ORIGINAL completion
    # receipt. For this new replication, reuse its unchanged pure summarizer;
    # do not forge an original protocol identity on the replay completion.
    spec = importlib.util.spec_from_file_location("replay_independent_selection",
                                                 out / "source/independent_selection.py")
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    with (out / "results/D1/ALL_METRICS.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary, families, overall, series, matched = independent.summarize(rows, cfg)
    summary["replication_of_protocol_sha256"] = ORIGINAL_PROTOCOL_SHA256
    summary["is_original_execution"] = False
    summary["confirmation_read"] = False
    independent_out = out / "results/D1/independent_selection"
    independent_out.mkdir()
    independent.write_json(independent_out / "INDEPENDENT_SELECTION.json", summary)
    for filename, table in (("FAMILY_SELECTION_2022.csv", families),
                            ("CONFIG_YEAR_SCORES.csv", overall),
                            ("CONFIG_YEAR_SERIES_SCORES.csv", series),
                            ("MATCHED_ABLATIONS.csv", matched)):
        independent.write_csv(independent_out / filename, table)
    (out / "REPLICATION_COMPLETE.json").write_text(json.dumps({
        **plan, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "replay_protocol_sha256": digest(rp),
        "replication_metrics_sha256": digest(out / "results/D1/ALL_METRICS.csv"),
        "original_metrics_sha256": digest(study / "results/D1/ALL_METRICS.csv"),
        "interpretation": "Numerical replication cannot reverse the original fixed-candidate stop decision."
    }, ensure_ascii=False, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, default=Path(__file__).resolve().parent.parent)
    sub = ap.add_subparsers(dest="action", required=True)
    audit = sub.add_parser("public-audit")
    audit.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("replay")
    run.add_argument("--previous-study", type=Path, required=True)
    run.add_argument("--eex-development", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--budget-minutes", type=float, default=40.)
    run.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.action == "public-audit":
        public_audit(args.study.resolve(), args)
    else:
        replay(args.study.resolve(), args)


if __name__ == "__main__":
    main()
