#!/usr/bin/env python3
"""Create a new local replay workspace; never train during preparation.

The distributed source directory must be a pristine checkout or extracted
delivery. Network is not used. Only the approved four parent CSVs and an
explicitly supplied, hash-verified EEX development CSV are materialized.
The run subcommand is the separate, explicit market-training action.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

PREFIX = "research/prior_transport_learning_20261018"
PROTOCOL_SHA256 = "204534271be271dd010e2978980d155c71a0ce28699b03bfbaa6bae76148b657"
METHOD_SHA256 = "df16d308f0f13831a7f2c66077f6a35a00b40b081bf256def45f0827975d1ea9"
DEPENDENCIES_SHA256 = "14c11cbc8108f2be832ad6f99a136287b7157f05b927cf733782a214edb167a4"
EEX_SHA256 = "5a0c7580edebc2a7a5e66d7487568f6d99a8e08cca0785676c2aaeccad7db6bb"
SOURCES_INDEX = "provenance/REPRODUCTION_SOURCES.json"
PARENT_INDEX = "provenance/REPRODUCTION_PARENT_DEPENDENCIES.json"


def identity(blob):
    return dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
                git_blob_sha1=hashlib.sha1(b"blob "+str(len(blob)).encode()+b"\0"+blob).hexdigest())


def relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
        raise ValueError("Unsafe relative path: "+str(name))
    return path


def regular(base, name):
    rel = relative(name)
    base = Path(base).resolve(strict=True)
    path = base.joinpath(*rel.parts)
    cursor = base
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Symlink is not an approved source: "+str(path))
    if not path.is_file() or not path.resolve().is_relative_to(base):
        raise ValueError("Missing or unsafe source file: "+str(path))
    return path


def checked(base, row, key="path"):
    path = regular(base, row[key])
    blob = path.read_bytes()
    got = identity(blob)
    for field in ("bytes", "sha256", "git_blob_sha1"):
        if field in row and got[field] != row[field]:
            raise ValueError("Source integrity failure: "+row[key]+"/"+field)
    return blob


def read_indices(repo):
    study = Path(repo).resolve() / PREFIX
    protocol_blob = regular(study, "provenance/PROTOCOL.json").read_bytes()
    if identity(protocol_blob)["sha256"] != PROTOCOL_SHA256:
        raise ValueError("Historical frozen protocol identity mismatch")
    if identity(regular(study, "methods/METHOD_ZH.md").read_bytes())["sha256"] != METHOD_SHA256:
        raise ValueError("Historical method identity mismatch")
    parent_blob = regular(study, PARENT_INDEX).read_bytes()
    if identity(parent_blob)["sha256"] != DEPENDENCIES_SHA256:
        raise ValueError("Fixed parent dependency index identity mismatch")
    parents = json.loads(parent_blob)
    sources = json.loads(regular(study, SOURCES_INDEX).read_bytes())
    protocol = json.loads(protocol_blob)
    if sources["protocol_sha256"] != PROTOCOL_SHA256 or len(protocol["source_sha256"]) != 11:
        raise ValueError("Unexpected frozen runtime declaration")
    if parents["count"] != 32 or len(parents["files"]) != 32:
        raise ValueError("Unexpected parent dependency count")
    for rows, src_key, dst_key in ((parents["files"], "path", "path"),
                                   (sources["files"], "source_path", "destination_path")):
        for row in rows:
            relative(row[src_key]); relative(row[dst_key])
        if len({r[src_key] for r in rows}) != len(rows) or len({r[dst_key] for r in rows}) != len(rows):
            raise ValueError("Duplicate allowlist source or destination")
    actual_sources = {r["source_path"]: r for r in sources["files"]}
    for name, sha in protocol["source_sha256"].items():
        row = actual_sources.get(name)
        if row is None or row["sha256"] != sha or row["destination_path"] != name:
            raise ValueError("Frozen runtime source must retain its original path: "+name)
    for row in sources["files"]:
        name = row["source_path"]
        if any(part in {"local_only", "reference", "results"} for part in PurePosixPath(name).parts):
            raise ValueError("Private or historical result path is not a reproduction source")
        if PurePosixPath(name).suffix.lower() in {".npz", ".npy", ".pt", ".pth", ".csv", ".xls", ".xlsx", ".zip", ".pdf"}:
            raise ValueError("Non-source container in reproduction allowlist")
        checked(study, row, "source_path")
    return study, sources, parents, protocol


def new_output(path):
    path = Path(path).expanduser()
    if ".." in path.parts:
        raise ValueError("Output may not contain parent-directory traversal")
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError("Output must be new; no overwrite, resume or restart")
    for ancestor in path.parents:
        if ancestor.is_symlink():
            raise ValueError("Output ancestors must not be symlinks")
    return path.resolve(strict=False)


def write_new(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(blob)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()


def commands(workspace):
    workspace = Path(workspace).absolute()
    study = workspace / PREFIX
    eex = workspace / "local_only/inputs/EEX_EUA_T3PA_daily_development.csv"
    public = study / "results/D1"
    private = workspace / "local_only/D1"
    audit = study / "audits/D1_independent"
    python = [sys.executable, "-B"]
    rows = {
        "synthetic_smoke": python+[str(study/"source/synthetic_run.py"), "smoke"],
        "synthetic_timing": python+[str(study/"source/synthetic_run.py"), "timing"],
        "synthetic_data_audit": python+[str(study/"source/independent_data.py"), "--output", str(study/"audits/INDEPENDENT_DATA_AUDIT.json")],
        "prepare_causal_data": python+[str(study/"source/run.py"), "prepare", "--eex-dev", str(eex), "--private-dir", str(private)],
        "run_350_prediction_flows": python+[str(study/"source/prepare_reproduction.py"), "run", "--workspace", str(workspace)],
        "summarize": python+[str(study/"source/run.py"), "summarize", "--public-dir", str(public), "--private-dir", str(private)],
        "independent_audit": python+[str(study/"source/independent_results.py"), "--protocol", str(study/"provenance/PROTOCOL.json"), "--public-dir", str(public), "--private-dir", str(private), "--output-dir", str(audit)],
        "aggregate_tables": python+[str(study/"source/analyze_results.py"), "--study", str(study), "--no-plots"],
        "scientific_figures_optional": python+[str(study/"source/analyze_results.py"), "--study", str(study)],
        "post_run_mechanism_optional": python+[str(study/"source/analyze_mechanism.py"), "--study", str(study),
                                                "--trace-dir", str(private), "--output-json", str(public/"MECHANISM_ANALYSIS.json"),
                                                "--output-md", str(study/"methods/POST_RUN_MECHANISM_ZH.md")],
    }
    return {key: shlex.join(value) for key, value in rows.items()}


def prepare(args):
    repo = args.source_dir.expanduser().resolve(strict=True)
    study, sources, parents, protocol = read_indices(repo)
    destination = new_output(args.output)
    if destination.is_relative_to(repo):
        raise ValueError("Use a new output outside the source checkout/delivery")
    # Read bytes only to establish the fixed identity; no CSV parsing, prices,
    # distributions, predictions or scores are displayed or computed here.
    dependency_bytes = [(row, checked(repo, row)) for row in parents["files"]]
    if args.eex_dev.is_symlink() or not args.eex_dev.is_file():
        raise ValueError("EEX development input must be a regular local CSV")
    eex_blob = args.eex_dev.read_bytes()
    if identity(eex_blob)["sha256"] != EEX_SHA256:
        raise ValueError("Only the exact approved EEX development CSV is accepted")
    plan = dict(schema_version=1, source_dir=str(repo), workspace=str(destination),
                study_relative_path=PREFIX, protocol_sha256=PROTOCOL_SHA256,
                reproduction_sources_sha256=identity((study/SOURCES_INDEX).read_bytes())["sha256"],
                parent_dependency_files=len(parents["files"]), source_files=len(sources["files"]),
                eex_development_sha256=EEX_SHA256, network_requests=0, market_fits=0,
                confirmation_inputs=0, copied_old_results=0, copied_checkpoints=0,
                commands=commands(destination))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, output_created=False, **plan), ensure_ascii=False, indent=2))
        return plan
    destination.mkdir(parents=True, exist_ok=False)
    new_study = destination / PREFIX
    for row in sources["files"]:
        write_new(new_study/row["destination_path"], checked(study, row, "source_path"))
    write_new(new_study/SOURCES_INDEX, (study/SOURCES_INDEX).read_bytes())
    for row, blob in dependency_bytes:
        write_new(destination/row["path"], blob)
    write_new(destination/"local_only/inputs/EEX_EUA_T3PA_daily_development.csv", eex_blob)
    write_new(destination/".gitignore", b"local_only/\n**/results/\n**/fresh_run/\n**/__pycache__/\n*.pyc\n*.pt\n*.npz\n")
    write_new(destination/"local_only/.gitignore", b"*\n!.gitignore\n")
    plan["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_new(destination/"REPRODUCTION_WORKSPACE.json", json_bytes(plan))
    write_new(destination/"REPRODUCTION_COMMANDS.json", json_bytes(plan["commands"]))
    print(json.dumps(dict(prepared=True, **plan), ensure_ascii=False, indent=2))
    return plan


def run(args):
    workspace = args.workspace.expanduser().resolve(strict=True)
    receipt = json.loads(regular(workspace, "REPRODUCTION_WORKSPACE.json").read_bytes())
    if receipt["protocol_sha256"] != PROTOCOL_SHA256 or receipt["study_relative_path"] != PREFIX:
        raise ValueError("This is not a prepared replay of the frozen study")
    study = workspace/PREFIX
    protocol = json.loads(regular(study, "provenance/PROTOCOL.json").read_bytes())
    if identity((study/"provenance/PROTOCOL.json").read_bytes())["sha256"] != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol changed")
    for name, sha in protocol["source_sha256"].items():
        checked(study, {"path": name, "sha256": sha})
    eex = regular(workspace, "local_only/inputs/EEX_EUA_T3PA_daily_development.csv")
    if identity(eex.read_bytes())["sha256"] != EEX_SHA256:
        raise ValueError("Prepared EEX development byte identity changed")
    public = new_output(study/"results/D1")
    private = new_output(workspace/"local_only/D1")
    marker = workspace/"MARKET_RUN_LAUNCHED.json"
    if marker.exists() or marker.is_symlink():
        raise FileExistsError("A market run has already been launched; no automatic retry")
    command = [sys.executable, "-B", str(study/"source/run.py"), "run", "--eex-dev", str(eex),
               "--protocol", str(study/"provenance/PROTOCOL.json"), "--public-dir", str(public),
               "--private-dir", str(private), "--year", "2022"]
    write_new(marker, json_bytes(dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                                    protocol_sha256=PROTOCOL_SHA256, command=command,
                                    warning="Explicit launch; interrupted or failed runs are retained without automatic retry")))
    subprocess.run(command, cwd=study, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    default_repo = Path(__file__).resolve().parents[3]
    check = sub.add_parser("check", help="Check code/protocol and dependency paths only; no CSV bytes or network")
    check.add_argument("--source-dir", type=Path, default=default_repo)
    prep = sub.add_parser("prepare", help="Hash-check approved inputs and create a new workspace; no fitting")
    prep.add_argument("--source-dir", type=Path, default=default_repo)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--eex-dev", type=Path, required=True)
    prep.add_argument("--dry-run", action="store_true")
    launch = sub.add_parser("run", help="Explicitly execute the frozen 350-flow experiment in a prepared workspace")
    launch.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run":
        run(args)
    else:
        _, sources, parents, _ = read_indices(args.source_dir)
        for row in parents["files"]:
            regular(args.source_dir, row["path"])
        print(json.dumps(dict(passed=True, source_files=len(sources["files"]),
                              parent_paths_checked=len(parents["files"]), csv_bytes_read=0,
                              old_results_read=0, network_requests=0, market_fits=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
