"""Independent source/job/gate audit using synthetic aggregates only.

No trainer is imported. No market file, weights, forecasts or result values are
read. Pure function ASTs are executed with in-memory rows and intercepted IO.
Run from any directory with Python and NumPy; --root defaults to this study.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = ["provenance/DESIGN_BEFORE_RUN.json", "source/run_development.py",
             "source/selection.py", "source/models.py", "source/data.py", "source/fit.py"]
    contents = {p: (root / p).read_text() for p in paths}
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
    protocol = json.loads(contents[paths[0]])
    trees = {p: ast.parse(s) for p, s in contents.items() if p.endswith(".py")}
    families = next(ast.literal_eval(n.value) for n in trees["source/models.py"].body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FAMILIES" for t in n.targets))
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def extract(path, names, namespace):
        body = [n for n in trees[path].body if isinstance(n, ast.FunctionDef) and n.name in names]
        check("AST function inventory " + ",".join(names), {n.name for n in body} == set(names))
        exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), namespace)

    ns = {"FAMILIES": families}
    extract("source/run_development.py", ["config_name", "jobs_from_protocol"], ns)
    make_jobs = ns["jobs_from_protocol"]
    check("exact twelve model families", tuple(protocol["neural_families"]) == families and len(families) == 12)
    d1jobs = make_jobs(protocol, "D1")
    check("D1 exact counts", dict(Counter(j["kind"] for j in d1jobs)) == protocol["D1_exact_job_counts"])
    check("D1 160 unique jobs", len(d1jobs) == len({j["id"] for j in d1jobs}) == 160)
    check("D1 only 2022 and prescribed seeds", all(j["year"] == 2022 and j["seed"] in (protocol["D1_seeds"] if j["kind"] == "neural" else [0]) for j in d1jobs))
    d2counts = {}
    for case, other in [("minimum", .1), ("maximum", .0)]:
        selected = {f: {"value": .1 if f == "smp_full" else other} for f in families}
        jobs = make_jobs(protocol, "D2", {"advance_to_D2": True, "selected": selected})
        counts = dict(Counter(j["kind"] for j in jobs))
        d2counts[case] = counts
        check("D2 " + case + " unique jobs", len(jobs) == len({j["id"] for j in jobs}))
        check("D2 " + case + " exact count", len(jobs) == (220 if case == "minimum" else 385))
        check("D2 " + case + " only 2023 and prescribed seeds", all(j["year"] == 2023 and j["seed"] in (protocol["D2_seeds"] if j["kind"] == "neural" else [0]) for j in jobs))
        for f in families:
            check("D2 " + case + " union " + f, {j["value"] for j in jobs if j["family"] == f} == {selected[f]["value"], .1})
    check("D2 maximum counts equal design", d2counts["maximum"] == protocol["D2_max_job_counts"])
    try:
        make_jobs(protocol, "D2", {"advance_to_D2": False})
    except RuntimeError:
        checks.append("D2 rejects failed D1 gate")
    else:
        raise AssertionError("D2 allowed a failed D1 gate")

    class MemoryJSON:
        def __init__(self, value): self.value = value
        def read_text(self): return json.dumps(self.value)

    main_node = next(n for n in trees["source/run_development.py"].body if isinstance(n, ast.FunctionDef) and n.name == "main")
    map_node = next(n for n in main_node.body if isinstance(n, ast.If) and ast.unparse(n.test) == "args.data_map")
    mapping = {m["name"]: "/tmp/smp_synthetic_path/" + m["name"] + ".csv" for m in protocol["markets"]}
    map_code = compile(ast.Module(body=[map_node], type_ignores=[]), "isolated_data_map_AST", "exec")
    mapped_protocol = deepcopy(protocol)
    exec(map_code, {"args": SimpleNamespace(data_map=MemoryJSON(mapping)), "json": json, "protocol": mapped_protocol, "Path": Path})
    for old, new in zip(protocol["markets"], mapped_protocol["markets"]):
        check("mapping preserves source identity " + old["name"], {k:v for k,v in old.items() if k != "path"} == {k:v for k,v in new.items() if k != "path"})
        check("mapping changes path only " + old["name"], new["path"] == mapping[old["name"]])
    for label, bad in [("missing", dict(list(mapping.items())[1:])), ("extra", dict(mapping, EXTRA="/tmp/x")), ("non-string", dict(mapping, GUANGZHOU=3))]:
        try:
            exec(map_code, {"args": SimpleNamespace(data_map=MemoryJSON(bad)), "json": json, "protocol": deepcopy(protocol), "Path": Path})
        except ValueError:
            checks.append("mapping rejects " + label)
        else:
            raise AssertionError("mapping allowed " + label)
    run_source, data_source = contents["source/run_development.py"], contents["source/data.py"]
    check("mapped worker uses original source SHA", 'load_market(job["market"], protocol["data_end_exclusive"])' in run_source and 'actual != spec["sha256"]' in data_source)
    check("data-map hash recorded", 'data_map_sha256=sha(args.data_map)' in run_source)
    check("development suffix rejected before numerical price parsing", data_source.index("dates >= end_exclusive") < data_source.index('float(r["price"])'))
    check("feature-only query before issuance and scoring", run_source.index("labels=False") < run_source.index("normalized = predict(net, query)") < run_source.index("truth = data.batch") < run_source.index("result = metrics(raw_forecast, truth)"))
    check("training does not receive query", 'fit_neural(train, val,' in run_source)
    check("job deadline before market loading", run_source.index("time.monotonic() >= deadline") < run_source.index('market = load_market'))
    check("two single-thread workers declared", protocol["cpu_processes"] == 2 and protocol["threads_per_process"] == 1 and 'max_workers=protocol["cpu_processes"]' in run_source)
    check("confirmation remains prohibited", protocol["confirmation_access"] is False and protocol["data_end_exclusive"] == "2024-01-01")

    captured, current_rows = {}, {}
    env = {"FAMILIES": families, "np": np, "defaultdict": defaultdict, "json": json,
           "datetime": datetime, "timezone": timezone, "sha": lambda _: "synthetic-protocol-hash",
           "load_results": lambda _, phase: current_rows[phase], "print": lambda *a, **k: None,
           "write_new": lambda p, value: captured.__setitem__(p.name, deepcopy(value))}
    extract("source/selection.py", ["aggregate", "choose_d1", "assess_d2"], env)
    fake_protocol = MemoryJSON(protocol)

    def rows(jobs, full=.85, iid=.95, one=.95, baseline=.9):
        out = []
        for j in jobs:
            f = j["family"]
            score = full if f == "smp_full" else iid if f == "smp_iid_state" else one if f == "smp_one_step" else 1. if f == "rw" else baseline
            out.append(dict(j, market=j["market"]["name"], relative_mse=[score] * 20,
                            score=score, n=123, selected_validation_score=2., status="completed"))
        return out

    def select_d1(**kwargs):
        current_rows["D1"] = rows(d1jobs, **kwargs)
        env["choose_d1"](root, fake_protocol)
        return captured["D1_SELECTION.json"]

    d1 = select_d1()
    check("D1 all gates pass on valid synthetic case", d1["advance_to_D2"] and all(d1["gate_checks"].values()))
    check("exact ties choose larger neural alpha", all(d1["selected"][f]["value"] == .1 for f in families))
    check("exact ties choose larger ridge alpha", d1["selected"]["ridge"]["value"] == 100.)
    check("local ridge uses validation and larger-alpha tie", all(v["selected_value"] == 100. for v in d1["selected"]["ridge_val"]["by_market"].values()))
    for key, kw in [("better_than_rw", dict(full=1., iid=1.1, one=1.1, baseline=1.1)),
                    ("better_than_matched_iid", dict(iid=.85)),
                    ("better_than_matched_one_step", dict(one=.85)),
                    ("within_practical_margin_of_best_baseline", dict(baseline=.8))]:
        rejected = select_d1(**kw)
        check("D1 failure case " + key, rejected["gate_checks"][key] is False and not rejected["advance_to_D2"])
    d1 = select_d1()
    d2jobs = make_jobs(protocol, "D2", d1)
    current_rows["D2"] = rows(d2jobs, full=.85, iid=.95, one=.95, baseline=.9)
    env["assess_d2"](root, fake_protocol, MemoryJSON(d1))
    d2 = captured["D2_ASSESSMENT.json"]
    check("D2 all seven gates on valid synthetic case", len(d2["gate_checks"]) == 7 and d2["point_gate_pass"])
    check("D2 does not reselect configurations", all(d2["selected"][f]["config_id"] == row["config_id"] for f, row in d1["selected"].items()))
    check("D2 preserves fixed D1 baseline identity", d2["fixed_best_non_smp_baseline"]["config_id"] == d1["fixed_best_non_smp_baseline"]["config_id"])
    check("D2 point gate never opens confirmation", d2["confirmation_ready"] is False)
    failure_cases = [
        ("practical_rw_improvement", dict(full=1., iid=1.1, one=1.1, baseline=1.1)),
        ("better_than_all_selected_primary_baselines", dict(baseline=.8)),
        ("better_than_matched_iid", dict(iid=.85)),
        ("better_than_matched_one_step", dict(one=.85)),
        ("EEX_better_than_rw", dict(full=1., iid=1.1, one=1.1, baseline=1.1)),
        ("at_least_three_series_better_than_rw", dict(full=1., iid=1.1, one=1.1, baseline=1.1)),
        ("at_least_two_seeds_better_than_rw", dict(full=1., iid=1.1, one=1.1, baseline=1.1)),
    ]
    for key, kw in failure_cases:
        current_rows["D2"] = rows(d2jobs, **kw)
        env["assess_d2"](root, fake_protocol, MemoryJSON(d1))
        rejected = captured["D2_ASSESSMENT.json"]
        check("D2 failure case " + key, rejected["gate_checks"][key] is False and not rejected["point_gate_pass"])

    formal_path = root / "provenance/PROTOCOL.json"
    formal_review = None
    if formal_path.exists():
        formal = json.loads(formal_path.read_text())
        for key, value in protocol.items():
            if key not in ("created_utc", "status", "uncertainty"):
                check("formal protocol preserves " + key, formal.get(key) == value)
        for key, value in protocol["uncertainty"].items():
            check("formal uncertainty preserves " + key, formal["uncertainty"].get(key) == value)
        check("formal protocol binds reviewed design", formal["design_sha256"] == hashes["provenance/DESIGN_BEFORE_RUN.json"])
        for relative, expected in formal["source_sha256"].items():
            path = root / relative
            check("formal source path is public Python " + relative,
                  relative.startswith("source/") and path.suffix == ".py" and path.resolve().is_relative_to((root / "source").resolve()))
            check("formal source bytes " + relative, hashlib.sha256(path.read_bytes()).hexdigest() == expected)
        formal_review = dict(sha256=hashlib.sha256(formal_path.read_bytes()).hexdigest(),
            created_utc=formal["created_utc"], source_count=len(formal["source_sha256"]),
            added_uncertainty_fields=sorted(set(formal["uncertainty"]) - set(protocol["uncertainty"])),
            note="Calendar-grid/invalid-replicate clarifications were recorded in the formal freeze; bootstrap implementation is reviewed separately.")

    receipt = dict(status="PASS", created_utc=datetime.now(timezone.utc).isoformat(),
        checks_passed=len(checks), checks=checks, source_sha256=hashes,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        D1_counts=dict(Counter(j["kind"] for j in d1jobs)), D2_counts=d2counts,
        formal_started_exists_at_receipt=(root / "results/D1/STARTED.json").exists(),
        IO_scope="Only listed source/design bytes; all gate rows and paths are synthetic; source AST functions only, no trainer imports",
        raw_market_reads=0, saved_weight_or_prediction_reads=0, confirmation_reads=0,
        model_fits=0, optimizer_steps=0, market_result_reads=0,
        original_static_review_communicated_before_formal_start=True,
        review_timing_note="Original exact-source static PASS preceded the formal freeze; this synthetic receipt may be serialized after formal execution starts.",
        formal_protocol_followup=formal_review,
        limits=["PASS establishes consistency of the reviewed design and executable paths, not predictive signal or novelty.",
                "Reviewed training/job/point-gate fields must remain fixed; formal metadata and inference clarifications are disclosed separately; no result-dependent edit is authorized.",
                "Two one-thread CPU workers use a cooperative deadline: already-started final validation/restoration/prediction may finish later; not a hard wall kill.",
                "Post-run independent audit must verify complete unique inventory, support/denominator equality and saved predictions; aggregate loader alone is not that audit.",
                "D1/D2 are exposed development periods; resource gates are not publication criteria or independent confirmation.",
                "A point gate cannot establish the need for every component: no_clock/no_sign/pooled_mark/no_aux remain separate attribution evidence.",
                "Calendar bootstrap implementation is outside this preflight and will receive separate review."])
    output = args.output or root / "audits/SMP_RUNNER_PREFLIGHT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "checks": len(checks), "D1_jobs": len(d1jobs),
                      "D2_min_jobs": 220, "D2_max_jobs": 385, "output": str(output)}))


if __name__ == "__main__":
    main()
