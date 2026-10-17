"""External launcher for an isolated copy; never modify frozen model sources.

Each invocation executes just one explicit stage command. There is no automatic
training, overwrite, restart, deadline extension or candidate reselection bypass.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    "d1": ("source", "run_development", "D1_PRE_RUN.json", "D1_STARTED.json"),
    "d2": ("d2_source", "run_d2", "D2_PRE_RUN.json", "D2_STARTED.json"),
    "learning": (".", "learning_diagnostic", "LEARNING_DIAGNOSTIC_PRE_RUN.json", "LEARNING_DIAGNOSTIC_STARTED.json"),
    "c": ("confirmation_source", "run_confirmation", "CONFIRMATION_PRE_RUN.json", "CONFIRMATION_STARTED.json"),
}
HISTORICAL_PLAN_SHA256 = {
    "D1_PRE_RUN.json": "4c1a31662959c951548b19c068cf6e8f33d7ab9b86fbe519203575f4a3e593ba",
    "D2_PRE_RUN.json": "62e08b11471f065fa248deffd1d3b095794aebbf631e6bc304ed302e79eac3ff",
    "CONFIRMATION_PRE_RUN.json": "7baa32f9b9e6727b4bf030d02868d03e3313aca1b8048952d32e400139a45122",
    "LEARNING_DIAGNOSTIC_PRE_RUN.json": "fc6386e47aee83a25e35931a6b9400e59fd59579f39f37ced4a7fe9e9fef0251",
}
VOLATILE = {"created_utc", "registry_sha256",
            "d1_complete_sha256", "d2_plan_sha256", "data_registry_sha256", "selection_sha256", "sealed_path",
            "reproduction_context", "prices_opened_before_freeze"}
FIGURE_OUTPUTS = ("figures/confirmation_results.png", "figures/confirmation_results.svg",
                  "figures/learning_curves.png", "figures/learning_curves.svg",
                  "results/LEARNING_CURVES.csv")


def canonical_semantics(plan: dict) -> dict:
    out = {key: copy.deepcopy(value) for key, value in plan.items() if key not in VOLATILE}
    if isinstance(out.get("selected_baseline"), dict):
        # Development losses may differ across hardware; the chosen configuration
        # and scope still must match, as must the complete task order and seeds.
        out["selected_baseline"].pop("score", None)
    return out


def verify_plan(stage: str) -> dict:
    name = STAGES[stage][2]
    plan = json.loads((ROOT / name).read_text())
    historical_blob = (ROOT / "historical_protocols" / name).read_bytes()
    if hashlib.sha256(historical_blob).hexdigest() != HISTORICAL_PLAN_SHA256[name]:
        raise ValueError("Historical protocol identity changed")
    historical = json.loads(historical_blob)
    if canonical_semantics(plan) != canonical_semantics(historical):
        raise ValueError("Reproduced task/config/seed protocol differs; stopped without repair: " + stage)
    if stage == "learning":
        # The D2 plan hash changes with runtime paths/time but must refer to the
        # exact verified D2 plan in this workdir, never an arbitrary dependency.
        verify_plan("d2")
        d2_blob = (ROOT / "D2_PRE_RUN.json").read_bytes()
        d2_receipt = json.loads((ROOT / "provenance/D2_REPLAY_PLAN_CHECK.json").read_text())
        d2_sha = hashlib.sha256(d2_blob).hexdigest()
        if plan["d2_plan_sha256"] != d2_sha or d2_receipt["plan_sha256"] != d2_sha:
            raise ValueError("Learning diagnostic D2 dependency changed")
    return plan


def dump_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify_copied_runtime_files() -> None:
    manifest = json.loads((ROOT / "provenance/WORKDIR_MANIFEST.json").read_text())
    required_top = {"reproduce_study.py", "aggregate_development.py", "aggregate_confirmation.py",
                    "learning_diagnostic.py", "make_figures.py",
                    "TIMING_PLAN.json", "CONFIRMATION_DESIGN_RULES.json",
                    "provenance/DEVELOPMENT_REGISTRY.json", "provenance/REPRODUCTION_CONTEXT.json"}
    prefixes = ("source/", "d2_source/", "confirmation_source/", "historical_protocols/")
    for item in manifest["files"]:
        name = item["path"]
        if name not in required_top and not name.startswith(prefixes):
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Unsafe runtime manifest path")
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Runtime source leaves the reproduction workdir")
        blob = path.read_bytes()
        if len(blob) != item["bytes"] or hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise ValueError("Copied runtime file changed: " + name)
    # Data-file entries are deliberately skipped here: no confirmation price
    # bytes are needed to validate source identities or to initialize D1/D2.


def module_from(directory: str, name: str):
    sys.path.insert(0, str(ROOT / directory))
    spec = importlib.util.spec_from_file_location(name, ROOT / directory / (name + ".py"))
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load frozen stage module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_environment() -> dict:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy
    import torch
    actual = dict(python=platform.python_version(), numpy=numpy.__version__, torch=torch.__version__)
    expected = dict(python="3.12.14", numpy="2.5.4", torch="2.3.1+cpu")
    if actual != expected:
        raise RuntimeError("Numerical environment differs from pinned reproduction environment: " + json.dumps(actual))
    return actual


def ensure_complete(stage: str) -> None:
    filename = "LEARNING_DIAGNOSTIC_COMPLETE.json" if stage == "LEARNING" else stage + "_RUN_COMPLETE.json"
    if not json.loads((ROOT / filename).read_text())["all_completed"]:
        raise ValueError("Previous stage was incomplete; do not resume or select partial results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["d1", "d2", "learning", "c", "aggregate", "aggregate-c", "figures", "environment"])
    parser.add_argument("command", nargs="?", choices=["check", "init", "run", "summarize"])
    args = parser.parse_args()
    if args.stage in ("aggregate", "aggregate-c", "figures", "environment") and args.command is not None:
        parser.error("This operation takes no extra command")
    if args.stage in STAGES and args.command is None:
        parser.error("A stage command is required")
    allowed = {"d1": {"check", "init", "run", "summarize"},
               "d2": {"check", "init", "run"}, "learning": {"init", "run"}, "c": {"init", "run"}}
    if args.stage in allowed and args.command not in allowed[args.stage]:
        parser.error("Unsupported stage command")
    if not (ROOT / "provenance/REPRODUCTION_CONTEXT.json").is_file():
        raise RuntimeError("Use only the workdir generated by prepare_reproduction_workdir.py")
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    verify_copied_runtime_files()
    env = check_environment()
    if args.stage == "environment":
        print(json.dumps(env)); return
    if args.stage == "aggregate":
        ensure_complete("D1"); ensure_complete("D2")
        if (ROOT / "DEVELOPMENT_AGGREGATE.json").exists():
            raise FileExistsError("Development aggregate already exists")
        module_from(".", "aggregate_development").main()
        return
    if args.stage == "aggregate-c":
        ensure_complete("CONFIRMATION")
        if not (ROOT / "aggregate_confirmation.py").is_file():
            raise FileNotFoundError("The source study did not include aggregate_confirmation.py")
        for target in ("CONFIRMATION_INFERENCE.json", "CONFIRMATION_SUMMARY.json",
                       "results/CONFIRMATION_FORECASTS_H1_H5_H20.csv", "results/CONFIRMATION_PRIMARY_INPUTS.npz"):
            if (ROOT / target).exists():
                raise FileExistsError("Confirmation aggregation output already exists")
        module_from(".", "aggregate_confirmation").main()
        return
    if args.stage == "figures":
        for stage in ("D1", "D2", "LEARNING", "CONFIRMATION"):
            ensure_complete(stage)
        for name in ("CONFIRMATION_SUMMARY.json", "CONFIRMATION_INFERENCE.json"):
            if not (ROOT / name).is_file():
                raise FileNotFoundError("Complete aggregate-c before figures: " + name)
        for target in FIGURE_OUTPUTS:
            path = ROOT / target
            if path.exists() or path.is_symlink():
                raise FileExistsError("Figure or learning-curve output already exists: " + target)
        # The author's root script exposes main(), with no CLI subcommand.
        # This performs postprocessing only; it never calls a market fitter.
        module_from(".", "make_figures").main()
        return
    directory, name, plan_name, started_name = STAGES[args.stage]
    if args.stage == "d2" and args.command in ("init", "run"):
        ensure_complete("D1")
    if args.stage == "learning":
        ensure_complete("D1"); ensure_complete("D2")
    if args.stage == "c":
        ensure_complete("D1"); ensure_complete("D2")
        if not (ROOT / "DEVELOPMENT_AGGREGATE.json").is_file():
            raise FileNotFoundError("Complete development aggregation before confirmation planning")
    if args.command in ("init", "run") and (ROOT / started_name).exists():
        raise FileExistsError("This stage has already started; preserve its original attempt")
    module = module_from(directory, name)
    if args.stage == "c":
        # Change only this invocation's path binding, never the frozen file.
        module.DEFAULT_SEALED = Path("local_only/data/confirmation_sealed/SHANGHAI_D4.csv")
    if args.command == "init":
        module.init()
        if args.stage == "c":
            path = ROOT / plan_name
            plan = json.loads(path.read_text())
            # A reproduction is not a new blind confirmation. The installer can
            # parse prices while rebuilding exact CSV bytes before model init.
            plan["prices_opened_before_freeze"] = True
            plan["reproduction_context"] = dict(new_independent_confirmation_claim=False,
                prices_opened_field_is_conservative=True,
                explanation="Published-data replay; data preparation may parse confirmation prices, without fitting or scoring.")
            path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        plan = verify_plan(args.stage)
        dump_new(ROOT / "provenance" / (args.stage.upper() + "_REPLAY_PLAN_CHECK.json"), dict(
            checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            task_config_seed_order_matches_historical=True,
            plan_sha256=hashlib.sha256((ROOT / plan_name).read_bytes()).hexdigest(),
            historical_plan_sha256=hashlib.sha256((ROOT / "historical_protocols" / plan_name).read_bytes()).hexdigest(),
            environment=env, tasks=len(plan["tasks"]), market_fit=False))
    elif args.command == "run":
        verify_plan(args.stage)
        check = json.loads((ROOT / "provenance" / (args.stage.upper() + "_REPLAY_PLAN_CHECK.json")).read_text())
        if hashlib.sha256((ROOT / plan_name).read_bytes()).hexdigest() != check["plan_sha256"]:
            raise ValueError("Plan changed after replay verification")
        module.run()
    elif args.command == "summarize":
        ensure_complete("D1")
        if (ROOT / "D1_SUMMARY.json").exists():
            raise FileExistsError("D1 summary already exists")
        module.summarize()
    else:
        if args.stage == "d1":
            result = dict(data=module.boundary_checks(), moments=module.moment_implementation_check())
            target = "IMPLEMENTATION_CHECK.json"
        else:
            result = module.algebra_check(); target = "D2_ALGEBRA_CHECK.json"
        dump_new(ROOT / target, result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
