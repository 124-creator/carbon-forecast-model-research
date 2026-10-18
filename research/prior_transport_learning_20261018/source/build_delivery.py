#!/usr/bin/env python3
"""Prepare/freeze an additive prior-transport delivery, without Git writes.

freeze-reproduction reads approved source and pre-run evidence only.
build is explicit, requires a final independent PASS, and reads public results.
No branch, commit, push, pull request, network request or market fit is made.
"""
from __future__ import annotations
import argparse
import base64
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

PREFIX = "research/prior_transport_learning_20261018"
PARENT = "1cba2ebaedd0fecd1891c90e322f0ccbcc5e2a99"
PARENT_TREE = "3f0cec9fd972092719b8ee99d284f5649f2cb2d8"
PARENT_BLOBS = 2138
PROTOCOL_SHA256 = "204534271be271dd010e2978980d155c71a0ce28699b03bfbaa6bae76148b657"
METHOD_SHA256 = "df16d308f0f13831a7f2c66077f6a35a00b40b081bf256def45f0827975d1ea9"
DEPENDENCIES_SHA256 = "14c11cbc8108f2be832ad6f99a136287b7157f05b927cf733782a214edb167a4"
SOURCE_INDEX = "provenance/REPRODUCTION_SOURCES.json"
DEPENDENCY_INDEX = "provenance/REPRODUCTION_PARENT_DEPENDENCIES.json"
PRUNE = {"local_only", "__pycache__", "fresh_run", ".git"}
CONTAINERS = {".npz", ".npy", ".xlsx", ".xls", ".zip", ".pt", ".pth", ".pkl", ".pickle", ".parquet", ".gz"}
R2D2_PUBLIC = {"LICENSE", "README.md", "environment.yml", "SOURCE_MANIFEST.json",
               "fewshots/models/r2d2.py", "fewshots/models/adjust.py"}


def identity(blob):
    return dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
                git_blob_sha1=hashlib.sha1(b"blob "+str(len(blob)).encode()+b"\0"+blob).hexdigest())


def relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
        raise ValueError("Unsafe relative path: "+name)
    return path


def dump(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_regular(base, name):
    rel = relative(name)
    base = Path(base).resolve(strict=True)
    cursor = base
    for part in rel.parts:
        cursor = cursor/part
        if cursor.is_symlink():
            raise ValueError("No source symlinks: "+name)
    if not cursor.is_file() or not cursor.resolve().is_relative_to(base):
        raise ValueError("Missing or unsafe file: "+name)
    return cursor.read_bytes()


def verify_frozen(study):
    blob = read_regular(study, "provenance/PROTOCOL.json")
    if identity(blob)["sha256"] != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol changed")
    protocol = json.loads(blob)
    if protocol["parent_commit"] != PARENT or len(protocol["source_sha256"]) != 11:
        raise ValueError("Wrong protocol parent or runtime source count")
    for name, expected in protocol["source_sha256"].items():
        if identity(read_regular(study, name))["sha256"] != expected:
            raise ValueError("Frozen runtime changed: "+name)
    if identity(read_regular(study, "methods/METHOD_ZH.md"))["sha256"] != METHOD_SHA256:
        raise ValueError("Frozen method changed")
    return protocol


def parent_metadata(path):
    tree = json.loads(path.read_bytes())
    if tree["sha"] != PARENT_TREE or tree["truncated"] is not False:
        raise ValueError("Wrong or incomplete parent tree")
    rows = {r["path"]: r for r in tree["tree"] if r["type"] == "blob"}
    if len(rows) != PARENT_BLOBS:
        raise ValueError("Parent blob count changed")
    return rows


def parent_dependencies(study, inherited):
    raw = read_regular(study, DEPENDENCY_INDEX)
    if identity(raw)["sha256"] != DEPENDENCIES_SHA256:
        raise ValueError("Parent dependency index changed")
    index = json.loads(raw)
    if index["count"] != 32 or len(index["files"]) != 32:
        raise ValueError("Expected exactly 32 inherited dependencies")
    for row in index["files"]:
        relative(row["path"])
        previous = inherited[row["path"]]
        if row["git_blob_sha1"] != previous["sha"] or row["bytes"] != previous["size"]:
            raise ValueError("Dependency disagrees with fixed parent tree: "+row["path"])
    return index["files"]


def freeze_reproduction(study):
    protocol = verify_frozen(study)
    source_names = list(protocol["source_sha256"])
    source_names += ["source/synthetic_run.py", "source/independent_data.py",
                     "source/independent_results.py", "source/check_independent_results.py",
                     "source/prepare_reproduction.py", "source/check_reproduction.py", "source/analyze_results.py", "source/analyze_mechanism.py",
                     "requirements.txt", "requirements-figures.txt", "REPRODUCE_ZH.md",
                     "provenance/PROTOCOL.json", "provenance/PRE_FIT_SCOPE.json",
                     "methods/METHOD_ZH.md", DEPENDENCY_INDEX]
    rows = []
    for name in dict.fromkeys(source_names):
        blob = read_regular(study, name)
        rows.append(dict(source_path=name, destination_path=name, **identity(blob),
                         role="frozen_runtime" if name in protocol["source_sha256"] else "reproduction_source_or_protocol"))
    existing = {row["source_path"] for row in rows}
    for name, expected in protocol["pre_run_evidence_sha256"].items():
        if name in existing:
            continue
        blob = read_regular(study, name)
        if identity(blob)["sha256"] != expected:
            raise ValueError("Historical pre-run evidence changed: "+name)
        rows.append(dict(source_path=name,
                         destination_path="historical/pre_run_evidence/"+name,
                         **identity(blob), role="original_pre_run_evidence_not_a_new_replay_pass"))
    value = dict(schema_version=1, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 status="DELIVERY_AUXILIARY_IDENTITY_FREEZE; not a new pre-training protocol",
                 protocol_sha256=PROTOCOL_SHA256, frozen_runtime_files=11,
                 repository_prefix=PREFIX, source_file_count=len(rows), files=rows,
                 excluded=["Original formal results", "per-origin data and forecasts", "model checkpoints",
                           "EEX quotation files", "Auto-Lambda original source and paper", "Deep Factors reading archive"],
                 manifest_self_excluded=True, market_results_read_by_this_operation=0)
    dump(study/SOURCE_INDEX, value)
    print(json.dumps(dict(frozen=True, path=str(study/SOURCE_INDEX), source_files=len(rows),
                          sha256=identity((study/SOURCE_INDEX).read_bytes())["sha256"],
                          market_results_read=0, market_fits=0), ensure_ascii=False))
    return value


def public_files(study):
    stack, files = [study], []
    while stack:
        for path in stack.pop().iterdir():
            if path.name in PRUNE:
                continue  # prune before traversal, including unread private data
            if path.is_symlink():
                raise ValueError("No public symlinks: "+str(path))
            rel = path.relative_to(study).as_posix()
            if path.name == "reference":
                raise ValueError("Reading archives and restricted author source must not be published: "+rel)
            if path.is_dir():
                stack.append(path)
                continue
            if not path.is_file() or path.name == "PUBLIC_DELIVERY_MANIFEST.json" or path.suffix == ".pyc":
                continue
            suffix = path.suffix.lower()
            if suffix in CONTAINERS:
                raise ValueError("Private data/model container: "+rel)
            if "SOURCE_PAYLOAD" in path.name or "GITHUB_SOURCE_PAYLOAD" in path.name:
                raise ValueError("Raw acquisition payload is not public source: "+rel)
            if rel.startswith("vendor/") and (not rel.startswith("vendor/r2d2/") or rel.removeprefix("vendor/r2d2/") not in R2D2_PUBLIC):
                raise ValueError("Only the fixed MIT R2D2 subset is permitted in vendor: "+rel)
            if suffix == ".csv" and not (rel.startswith("results/") or rel.startswith("audits/")):
                raise ValueError("Only aggregate result/audit CSVs may be public: "+rel)
            if suffix in {".pdf", ".png"} and not (rel.startswith("figures/") or rel.startswith("results/figures/")):
                raise ValueError("Only study-generated scientific figures may be binary documents: "+rel)
            files.append(path)
    return sorted(files)


def check_public_bytes(path, blob):
    if path.suffix.lower() == ".csv":
        rows = csv.DictReader(io.StringIO(blob.decode("utf-8-sig")))
        fields = set(rows.fieldnames or [])
        forbidden = {"date", "origin", "price", "prediction", "predictions", "truth", "delta", "sigma", "target_change"}
        if fields & forbidden:
            raise ValueError("Potential per-origin quotation/prediction table: "+str(path))
    elif path.suffix.lower() not in {".png", ".pdf"}:
        blob.decode("utf-8")


def no_existing(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError("Do not overwrite delivery artifact: "+str(path))


def build(args):
    study = args.repo_root/PREFIX
    pub, snap, zip_path = args.publication_dir, args.snapshot, args.zip
    for path in (snap, zip_path, pub/"TRANSPORT_INDEX.json", study/"PUBLIC_DELIVERY_MANIFEST.json"):
        no_existing(path)
    qa_path = study/"audits/FINAL_REPORT_REVIEW.json"
    qa = json.loads(qa_path.read_bytes())
    if qa.get("status") != "PASS":
        raise ValueError("Final independent delivery PASS is required before build")
    verify_frozen(study)
    inherited = parent_metadata(args.parent_tree)
    dependency_rows = parent_dependencies(study, inherited)
    reproduction = json.loads(read_regular(study, SOURCE_INDEX))
    for row in reproduction["files"]:
        got = identity(read_regular(study, row["source_path"]))
        if any(got[k] != row[k] for k in ("bytes", "sha256", "git_blob_sha1")):
            raise ValueError("Reproduction source changed since delivery auxiliary freeze: "+row["source_path"])
    files = public_files(study)
    if not (study/"REPORT_ZH.md") in files or not (study/"REPRODUCE_ZH.md") in files:
        raise ValueError("Final report and reproduction instructions are required")
    blobs_by_rel = {}
    for path in files:
        name = path.relative_to(study).as_posix()
        if PREFIX+"/"+name in inherited:
            raise ValueError("New study must not replace a parent file: "+name)
        raw = path.read_bytes()
        check_public_bytes(path, raw)
        blobs_by_rel[name] = raw
    # Verify every inherited dependency before writing any snapshot artifact.
    dep_blobs = {}
    bases = [args.repo_root]+list(args.fallback)
    for row in dependency_rows:
        selected = next((base for base in bases if (base/row["path"]).is_file()), None)
        if selected is None:
            raise ValueError("Missing inherited dependency: "+row["path"])
        raw = read_regular(selected, row["path"])
        got = identity(raw)
        if any(got[k] != row[k] for k in ("bytes", "git_blob_sha1")) or ("sha256" in row and got["sha256"] != row["sha256"]):
            raise ValueError("Inherited dependency byte identity changed: "+row["path"])
        dep_blobs[row["path"]] = raw
    records = [dict(path=name, **identity(blob)) for name, blob in sorted(blobs_by_rel.items())]
    manifest = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    repository="124-creator/carbon-forecast-model-research", repository_prefix=PREFIX,
                    additive_parent_commit=PARENT, base_tree=PARENT_TREE, parent_blobs_to_preserve=PARENT_BLOBS,
                    decision_location="REPORT_ZH.md and results: fixed development evidence, not a new confirmation certificate",
                    frozen_training_source_files=11, final_review_sha256=identity(qa_path.read_bytes())["sha256"],
                    builder_sha256=identity(Path(__file__).read_bytes())["sha256"],
                    public_file_count_excluding_this_manifest=len(records),
                    public_bytes_excluding_this_manifest=sum(row["bytes"] for row in records),
                    excluded=["private paths pruned before traversal", "per-origin quotes, labels, forecasts, scales, weights",
                              "Auto-Lambda original source/LICENSE/paper", "Deep Factors reading archive", "paper PDFs"],
                    files=records, self_hash="Self excluded; ZIP and Git blob identities cover this manifest")
    dump(study/"PUBLIC_DELIVERY_MANIFEST.json", manifest, exclusive=True)
    blobs_by_rel["PUBLIC_DELIVERY_MANIFEST.json"] = (study/"PUBLIC_DELIVERY_MANIFEST.json").read_bytes()
    snap.mkdir(parents=True, exist_ok=False)
    pub.mkdir(parents=True, exist_ok=True)
    entries, inline, large = [], [], []
    for name, raw in sorted(blobs_by_rel.items()):
        path = PREFIX+"/"+name
        dest = snap/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        entry = dict(path=path, mode="100644", type="blob", **identity(raw))
        if len(raw) <= 131072 and PurePosixPath(name).suffix.lower() not in {".png", ".pdf"}:
            entry["transport"] = "inline"
            inline.append(dict(path=path, mode="100644", type="blob", content=raw.decode("utf-8")))
        else:
            entry["transport"] = "blob"
            encoded = base64.b64encode(raw).decode("ascii")
            parts = []
            for j, start in enumerate(range(0, len(encoded), 36000)):
                part = pub/f"blob_{len(large):03d}_part_{j:03d}.txt"
                with part.open("x", encoding="ascii") as stream:
                    stream.write(encoded[start:start+36000])
                parts.append(dict(path=str(part), characters=part.stat().st_size))
            large.append(dict(**entry, encoding="base64", encoded_characters=len(encoded), parts=parts))
        entries.append(entry)
    batches, current, size = [], [], 0
    for row in inline:
        n = len(json.dumps(row, ensure_ascii=False).encode())
        if current and size+n > 65000:
            batches.append(current); current, size = [], 0
        current.append(row); size += n
    if current:
        batches.append(current)
    chunks = []
    for i, rows in enumerate(batches):
        path = pub/f"inline_{i:03d}.json"
        dump(path, rows, exclusive=True)
        chunks.append(dict(index=i, path=str(path), bytes=path.stat().st_size, entries=len(rows)))
    dump(pub/"TRANSPORT_INDEX.json", dict(parent_commit=PARENT, base_tree=PARENT_TREE,
         parent_blobs_to_preserve=PARENT_BLOBS, additive_prefix=PREFIX,
         github_mutations_performed=0, entry_count=len(entries), inline_count=len(inline),
         blob_count=len(large), inline_chunks=chunks, blobs=large, files=entries,
         public_file_bytes=sum(row["bytes"] for row in entries)), exclusive=True)
    for name, raw in sorted(dep_blobs.items()):
        dest = snap/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    guide = f"""# 碳价先验运输学习交付包

本轮入口：[完整报告]({PREFIX}/REPORT_ZH.md)、[方法]({PREFIX}/methods/METHOD_ZH.md)、[复现说明]({PREFIX}/REPRODUCE_ZH.md)。结论、固定门槛和失败项以该报告与独立审核为准。本包不作新的原创性或发表资格认证。

GitHub 增量仅新增 `{PREFIX}/`，基于父提交 `{PARENT}`、完整父树 `{PARENT_TREE}`。全部 {PARENT_BLOBS} 个父文件 blob 保留，未在此打包过程中写入 GitHub。

ZIP 提供新研究全部公开交付，以及父提交已公开的 4 份开发 CSV、26 项 EEX 复现/来源文件和 2 份数据归属说明，共 32 个必要父依赖。它们保持原相对路径并逐字节匹配父 Git blob。本 ZIP 不是整个历史仓库镜像。

EEX 报价、任何确认报价、逐起点预测/目标/尺度、检查点和模型权重均不在包内。重新拟合需要本地固定 SHA 的 EEX 开发 CSV。Auto-Lambda 受限原源码/论文和 Deep Factors 阅读归档不提供；R2D2 的 MIT 原始子集及外部兼容层单独标注。

`BUNDLE_MANIFEST.json` 覆盖所有内容文件的 SHA-256 和 Git blob 身份（自身除外）。新工作区的执行必须留下新收据，不能继承原运行 PASS。
"""
    (snap/"BUNDLE_README_ZH.md").write_text(guide, encoding="utf-8")
    members = sorted(p for p in snap.rglob("*") if p.is_file())
    bundle = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), parent_commit=PARENT,
                  base_tree=PARENT_TREE, new_study_files=len(entries), prior_public_dependency_files=32,
                  prior_dependency_blob_identity_verified=True,
                  files=[dict(path=p.relative_to(snap).as_posix(), **identity(p.read_bytes())) for p in members],
                  self_excluded="BUNDLE_MANIFEST.json")
    dump(snap/"BUNDLE_MANIFEST.json", bundle, exclusive=True)
    members.append(snap/"BUNDLE_MANIFEST.json")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(members):
            archive.write(path, path.relative_to(snap).as_posix())
    expected = {p.relative_to(snap).as_posix(): identity(p.read_bytes())["sha256"] for p in members}
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(expected):
            raise ValueError("ZIP integrity or member inventory failure")
        for name, sha in expected.items():
            if identity(archive.read(name))["sha256"] != sha:
                raise ValueError("ZIP content mismatch: "+name)
    receipt = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), zip=str(zip_path),
                   **identity(zip_path.read_bytes()), members=len(members), new_study_files=len(entries),
                   prior_public_dependency_files=32, all_prior_dependency_blobs_match_parent=True,
                   parent_commit=PARENT, base_tree=PARENT_TREE, parent_blobs_to_preserve=PARENT_BLOBS,
                   public_manifest_sha256=identity((study/"PUBLIC_DELIVERY_MANIFEST.json").read_bytes())["sha256"],
                   all_zip_member_bytes_verified=True, all_member_paths_unique=True,
                   no_private_directory_traversal=True, frozen_training_sources_verified=11,
                   github_mutations_performed=0, market_fits=0)
    dump(pub/"BUNDLE_RECEIPT.json", receipt, exclusive=True)
    print(json.dumps(dict(receipt=receipt, inline_chunks=len(chunks), blob_count=len(large)), ensure_ascii=False))


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("freeze-reproduction", "check", "build"))
    p.add_argument("--repo-root", type=Path, default=root/"carbon_workspace")
    p.add_argument("--publication-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--parent-tree", type=Path, default=Path(__file__).resolve().parent/"PARENT_TREE.json")
    p.add_argument("--snapshot", type=Path, default=root/"scratch/prior_delivery_20261018")
    p.add_argument("--zip", type=Path, default=root/"carbon_research_prior_transport_20261018.zip")
    p.add_argument("--fallback", type=Path, action="append", default=[])
    args = p.parse_args()
    study = args.repo_root/PREFIX
    if args.action == "freeze-reproduction":
        freeze_reproduction(study)
    elif args.action == "check":
        verify_frozen(study)
        inherited = parent_metadata(args.parent_tree)
        rows = parent_dependencies(study, inherited)
        print(json.dumps(dict(passed=True, runtime_sources=11, parent_blobs=len(inherited),
                              parent_dependency_paths=len(rows), result_files_read=0,
                              csv_bytes_read=0, market_fits=0, delivery_built=False)))
    else:
        build(args)


if __name__ == "__main__":
    main()
