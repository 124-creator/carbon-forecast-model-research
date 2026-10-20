"""Frozen Stage A: official N/DLinear classes and two unchanged approved RNN fits.

No unapproved data; no historical weights or predictions; exposed-development only.
Run commands from this directory with the approved NumPy/PyTorch environment:
  python -B stage_a.py check
  python -B stage_a.py timing
  python -B stage_a.py run
  python -B stage_a.py summarize
The run command writes STAGE_A_PRE_RUN.json before starting the first formal fit.
An existing run cannot be restarted and the original deadline cannot be reset.
"""
import os
for _thread_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_key] = "1"

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
HD_ROOT = ROOT.parent / "recurrent_response_20261014" / "hindsight_diagnostic_20261017"
FAIR_ROOT = HD_ROOT.parent / "fair_recurrent_tuning_20261016"
sys.path.insert(0, str(HD_ROOT))
sys.path.insert(0, str(FAIR_ROOT))
import hd_fit
from fair_core import configure, reference
from fair_data import load_prefix, support

AUTHOR_COMMIT = "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
HORIZON = 20
LOOKBACK = 20
CUTS = (459, 703)
SEEDS = (17, 29, 43)
MODELS = ("NLinear", "DLinear", "c003_GRU", "c025_LSTM")
FORMAL_BUDGET = 600
CONFIGS = {
    "c003_GRU": dict(id="c003", family="gru", group="price", length=20,
                     anchor="stat", shrink=1.0, hidden=16, lr=.003,
                     weight_decay=.001, slice="factorial"),
    "c025_LSTM": dict(id="c025", family="lstm", group="price", length=20,
                      anchor="rw", shrink=1.0, hidden=16, lr=.003,
                      weight_decay=.001, slice="factorial"),
}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(net):
    raw = b"".join(name.encode() + value.detach().cpu().numpy().tobytes()
                   for name, value in sorted(net.state_dict().items()))
    return hashlib.sha256(raw).hexdigest()


def deadline_check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Stage A original 600-second deadline reached; no reset")


def official_class(name):
    if name not in ("NLinear", "DLinear"):
        raise ValueError("Not an official linear class")
    path = ROOT / "vendor" / "ltsf_linear" / (name + ".py")
    spec = importlib.util.spec_from_file_location("author_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def observed_windows(price, origins):
    price = np.asarray(price)
    origins = np.asarray(origins, dtype=np.int64)
    if (price.ndim != 1 or origins.ndim != 1 or not len(origins)
            or origins.min() < LOOKBACK - 1 or origins.max() >= len(price)
            or np.any(np.diff(origins) <= 0)):
        raise ValueError("Invalid strictly observed linear input windows")
    return price[origins[:, None] - np.arange(LOOKBACK - 1, -1, -1)][..., None]


def linear_predict(model, price, origins):
    origins = np.asarray(origins, dtype=np.int64)
    # Select only already observed input positions. No target construction here.
    raw = observed_windows(price[:int(origins.max()) + 1], origins)
    tensor = torch.as_tensor((raw - model.mean) / model.sd)
    model.net.eval()
    with torch.no_grad():
        predicted_level = model.net(tensor).squeeze(-1).cpu().numpy() * model.sd + model.mean
    return predicted_level - np.asarray(price)[origins, None]


def linear_fit(price, dates, end, name, seed, max_epochs=60, deadline=None):
    del dates  # Date metadata is not used as a learned market feature.
    configure()
    torch.manual_seed(int(seed))
    started = time.perf_counter()
    prefix = np.array(price[:end], dtype=np.float64, copy=True)
    ids, train, validation, increment_scale, increment_target = support(prefix, end)
    raw = observed_windows(prefix, ids)
    # One global scale estimated from the repeated training input windows only.
    mean = float(raw[train].mean())
    sd = max(float(raw[train].std()), 1e-6)
    levels = increment_target + prefix[ids, None]
    x = (raw - mean) / sd
    y = (levels - mean) / sd
    tensor = torch.as_tensor(x)
    label = torch.as_tensor(y[..., None])
    net = official_class(name)(SimpleNamespace(seq_len=LOOKBACK, pred_len=HORIZON,
                                              enc_in=1, individual=False))
    optimizer = torch.optim.Adam(net.parameters(), lr=.003)

    def validation_loss():
        net.eval()
        with torch.no_grad():
            return float((net(tensor[validation]) - label[validation]).square().mean())

    initial = best = validation_loss()
    saved, selected, stale, trace = copy.deepcopy(net.state_dict()), 0, 0, []
    for epoch in range(1, int(max_epochs) + 1):
        deadline_check(deadline)
        net.train()
        order = torch.as_tensor(train)[torch.randperm(len(train))]
        sum_objective, count, max_gradient = 0., 0, 0.
        for offset in range(0, len(order), 32):
            deadline_check(deadline)
            batch = order[offset:offset + 32]
            optimizer.zero_grad(set_to_none=True)
            loss = (net(tensor[batch]) - label[batch]).square().mean()
            loss.backward()
            gradient = float(torch.sqrt(sum(parameter.grad.detach().square().sum()
                                 for parameter in net.parameters() if parameter.grad is not None)))
            if not np.isfinite([float(loss.detach()), gradient]).all():
                raise FloatingPointError("Nonfinite linear fit; no retry")
            # No clipping, added regularizer, warm start, or post-hoc shrinkage.
            optimizer.step()
            sum_objective += float(loss.detach()) * len(batch)
            count += len(batch)
            max_gradient = max(max_gradient, gradient)
        value = validation_loss()
        trace.append([epoch, sum_objective / count, value, max_gradient])
        if not np.isfinite(trace[-1]).all():
            raise FloatingPointError("Nonfinite epoch summary")
        if value < best - 1e-6:
            best, selected, stale = value, epoch, 0
            saved = copy.deepcopy(net.state_dict())
        else:
            stale += 1
        if stale >= 12:
            break
    net.load_state_dict(saved)
    net.eval()
    model = SimpleNamespace(net=net, mean=mean, sd=sd, name=name)
    all_increment_predictions = linear_predict(model, prefix, ids)
    predicted_normalized = ((all_increment_predictions + prefix[ids, None] - mean) / sd)
    hidx = np.array([0, 4, 19])
    val_errors = (all_increment_predictions[validation][:, hidx]
                  - increment_target[validation][:, hidx]) ** 2
    validation_primary = float(np.mean(np.mean(val_errors, axis=0)
                            / np.mean(increment_target[validation][:, hidx] ** 2, axis=0)))
    receipt = dict(
        model=name, author_commit=AUTHOR_COMMIT, seed=int(seed), end=int(end),
        train_n=len(train), validation_n=len(validation), pool_start=int(ids[0]),
        train_max_origin=int(ids[train[-1]]),
        train_max_target=int(ids[train[-1]] + HORIZON),
        validation_first_origin=int(ids[validation[0]]),
        validation_max_target=int(ids[-1] + HORIZON),
        input_global_mean=mean, input_global_sd=sd,
        normalization_fit_source="raw[train] input windows only, global scalar mean/std",
        fair_support_increment_scale=increment_scale,
        training_target="20 future standardized price levels",
        returned_prediction="20 cumulative price increments relative to current observed price",
        parameter_count=sum(parameter.numel() for parameter in net.parameters()),
        state_sha256=state_digest(net), epochs=len(trace), selected_epoch=selected,
        initial_validation=initial, best_validation=best,
        validation_primary=validation_primary,
        train_mse=float(np.mean((predicted_normalized[train] - y[train]) ** 2)),
        seconds=time.perf_counter() - started, trace=trace,
        trace_columns=["epoch", "minibatch_training_MSE_weighted_mean",
                       "postepoch_validation_MSE", "maximum_unclipped_batch_gradient_norm"],
        optimizer=dict(name="Adam", lr=.003, batch_size=32, max_epochs=max_epochs,
                       patience=12, minimum_validation_improvement=1e-6,
                       weight_decay=0., gradient_clipping=False),
        config=dict(seq_len=LOOKBACK, pred_len=HORIZON, enc_in=1, individual=False),
    )
    deadline_check(deadline)
    return model, receipt


def fresh_fit(name, price, dates, cut, seed, max_epochs, deadline):
    if name in ("NLinear", "DLinear"):
        return linear_fit(price, dates, cut, name, seed, max_epochs, deadline)
    model, receipt = hd_fit.fit(price, dates, cut, CONFIGS[name], seed=seed,
                                max_epochs=max_epochs, deadline=deadline)
    receipt.update(model=name, baseline_source="unchanged approved hd_fit.fit")
    return model, receipt


def fresh_predict(name, model, price, dates, origins):
    if name in ("NLinear", "DLinear"):
        return linear_predict(model, price, origins)
    return hd_fit.predict(model, price, dates, origins)


def planned_tasks():
    # Every model is paired on both annual cuts before moving to the next seed.
    return [dict(model=name, seed=seed, cut=cut)
            for seed in SEEDS for name in MODELS for cut in CUTS]


def task_stem(task):
    return "{model}_seed{seed}_cut{cut}".format(**task)


def source_files():
    return [ROOT / "stage_a.py", ROOT / "STAGE_A_PROTOCOL.md",
            ROOT / "vendor/ltsf_linear/NLinear.py", ROOT / "vendor/ltsf_linear/DLinear.py",
            ROOT / "vendor/ltsf_linear/LICENSE", HD_ROOT / "hd_fit.py",
            HD_ROOT / "hd_core.py", FAIR_ROOT / "fair_core.py", FAIR_ROOT / "fair_data.py"]


def source_hashes():
    return {str(path.relative_to(WORKSPACE)): digest(path) for path in source_files()}


def synthetic_check():
    configure()
    result = dict(scope="synthetic inputs only; no market data loaded", passed=True, cases=[])
    h = torch.arange(1, HORIZON + 1)
    price = 100. + .125 * np.arange(300)
    origins = np.array([99, 150, 220])
    raw = observed_windows(price, origins)
    tensor = torch.as_tensor(raw)
    truth = torch.as_tensor(price[origins, None] + .125 * np.arange(1, HORIZON + 1)[None, :])
    for name in ("NLinear", "DLinear"):
        net = official_class(name)(SimpleNamespace(seq_len=LOOKBACK, pred_len=HORIZON,
                                                  enc_in=1, individual=False))
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
            if name == "NLinear":
                net.Linear.weight[:, -2] = -h
            else:
                for layer in (net.Linear_Seasonal, net.Linear_Trend):
                    layer.weight[:, -1] = 1 + h
                    layer.weight[:, -2] = -h
        prediction = net(tensor).squeeze(-1)
        max_error = float((prediction - truth).abs().max())
        loss = prediction.square().mean()
        loss.backward()
        finite_gradients = all(bool(torch.isfinite(parameter.grad).all())
                               for parameter in net.parameters())
        model = SimpleNamespace(net=net, mean=0., sd=1., name=name)
        changed = price.copy()
        changed[221:] += 5000. + 10. * np.arange(len(price) - 221)
        original_boundary = linear_predict(model, price, np.array([220]))
        changed_boundary = linear_predict(model, changed, np.array([220]))
        boundary_error = float(np.max(np.abs(original_boundary - changed_boundary)))
        row = dict(model=name, output_shape=list(prediction.shape),
                   exact_ramp_max_error=max_error, future_suffix_prediction_difference=boundary_error,
                   finite_gradients=finite_gradients)
        result["cases"].append(row)
        result["passed"] &= (prediction.shape == truth.shape and max_error < 1e-9
                              and boundary_error == 0. and finite_gradients)
    ref_a = reference(price)[0]
    changed = price.copy()
    changed[221:] += 5000.
    ref_b = reference(changed)[0]
    result["reference_prefix_max_difference"] = float(np.max(np.abs(ref_a[:221] - ref_b[:221])))
    result["passed"] &= result["reference_prefix_max_difference"] == 0.
    ids, train, val, scale, target = support(price[:250], 250)
    result["support"] = dict(train_n=len(train), validation_n=len(val),
                            train_max_target=int(ids[train[-1]] + HORIZON),
                            validation_first_origin=int(ids[val[0]]),
                            validation_max_target=int(ids[-1] + HORIZON))
    result["passed"] &= (ids[train[-1]] + HORIZON < ids[val[0]]
                          and ids[-1] + HORIZON < 250 and target.shape[1] == 20 and scale > 0)
    if not result["passed"]:
        raise AssertionError(result)
    write_json(ROOT / "STAGE_A_SYNTHETIC_CHECK.json", result)
    print(json.dumps(result), flush=True)


def timing():
    if not read_json(ROOT / "STAGE_A_SYNTHETIC_CHECK.json")["passed"]:
        raise ValueError("Synthetic check required")
    if (ROOT / "STAGE_A_TIMING_PLAN.json").exists():
        raise FileExistsError("Timing already started; no implicit repeats")
    names = ("NLinear", "DLinear", "c003_GRU")
    write_json(ROOT / "STAGE_A_TIMING_PLAN.json",
               dict(timing_fits=3, models=list(names), cut=459, seed=113, max_epochs=3,
                    source_sha256=source_hashes(), starts_at_utc=datetime.now(timezone.utc).isoformat(),
                    score_2020_2021=False, formal_fits_started=False))
    price, dates, data = load_prefix()
    records = []
    (ROOT / "timing_receipts").mkdir()
    for name in names:
        model, receipt = fresh_fit(name, price, dates, 459, 113, 3, None)
        write_json(ROOT / "timing_receipts" / (name + ".json"), receipt)
        records.append(dict(model=name, epochs=receipt["epochs"], seconds=receipt["seconds"],
                            state_sha256=receipt["state_sha256"]))
        print("TIMING " + json.dumps(records[-1]), flush=True)
        del model
    write_json(ROOT / "STAGE_A_TIMING.json",
               dict(timing_fits=3, records=records, data=data, formal_fits_started=False,
                    score_2020_2021_computed=False, source_sha256=source_hashes()))


def run():
    configure()
    if (ROOT / "STAGE_A_PRE_RUN.json").exists():
        raise FileExistsError("Existing formal run cannot restart or reset its deadline")
    timing_record = read_json(ROOT / "STAGE_A_TIMING.json")
    if timing_record["timing_fits"] != 3 or timing_record["source_sha256"] != source_hashes():
        raise ValueError("Missing timing or changed frozen implementation")
    price, dates, data = load_prefix()
    if tuple(int((dates < year).sum()) for year in ("2020-01-01", "2021-01-01")) != CUTS:
        raise ValueError("Annual cuts differ")
    paths = ROOT / "local_only" / "stage_a"
    paths.mkdir(parents=True)
    (paths / "fits").mkdir()
    (ROOT / "stage_a_receipts").mkdir()
    started = time.monotonic()
    deadline = started + FORMAL_BUDGET
    before = dict(
        stage="A", formal_expected=24, tasks=planned_tasks(), cuts=list(CUTS), seeds=list(SEEDS),
        models=list(MODELS), max_epochs=60, budget_seconds=FORMAL_BUDGET,
        workers=1, device="cpu", math_threads=1, dtype="float64",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        started_monotonic=started, deadline_monotonic=deadline,
        source_sha256=source_hashes(), data=data, configs=CONFIGS,
        linear_training=dict(lookback=20, horizon=20, channels=1, individual=False,
                             global_input_train_window_standardization=True,
                             optimizer="Adam", learning_rate=.003, batch_size=32,
                             max_epochs=60, patience=12, validation_metric="all 20 outputs MSE"),
        score=dict(origin_count=451, annual_counts=[224, 227], horizons=[1, 5, 20],
                   weighting="equal weight over six year x horizon relative-MSE cells",
                   forecast_refresh="annual frozen neural model; reference updates per approved source"),
        environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
        author_model_retraining=True, unexposed_test=False, novelty_established=False,
        diagnostic_only=True, historical_predictions_or_weights_used=False,
    )
    write_json(ROOT / "STAGE_A_PRE_RUN.json", before)
    print("FROZEN " + json.dumps(dict(fits=24, seconds=FORMAL_BUDGET, deadline=deadline)), flush=True)
    with (paths / "INPUTS.npz").open("xb") as stream:
        np.savez(stream, price=price, dates=dates, reference=reference(price)[0])
    completed = []
    for index, task in enumerate(planned_tasks()):
        name, cut, seed = task["model"], task["cut"], task["seed"]
        stem = task_stem(task)
        attempt = dict(task=task, ordinal=index + 1, status="started",
                       monotonic=time.monotonic(), original_deadline_monotonic=deadline)
        with (ROOT / "STAGE_A_ATTEMPTS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(attempt) + "\n")
        try:
            deadline_check(deadline)
            model, receipt = fresh_fit(name, price, dates, cut, seed, 60, deadline)
            origins = np.arange(cut, 703 if cut == 459 else 930)
            pool_ids, train, validation, _, _ = support(price[:cut], cut)
            training_ids, validation_ids = pool_ids[train], pool_ids[validation]
            predictions = fresh_predict(name, model, price, dates, origins)
            training = fresh_predict(name, model, price, dates, training_ids)
            valid = fresh_predict(name, model, price, dates, validation_ids)
            boundary = fresh_predict(name, model, price, dates, np.array([cut - 1]))
            changed = price.copy()
            changed[cut:] += 10000. + 10. * np.arange(len(price) - cut)
            boundary_changed = fresh_predict(name, model, changed, dates, np.array([cut - 1]))
            boundary_difference = float(np.max(np.abs(boundary - boundary_changed)))
            if boundary_difference > 1e-10:
                raise AssertionError("Future suffix changed boundary forecast")
            if not all(np.isfinite(array).all() for array in (predictions, training, valid, boundary)):
                raise FloatingPointError("Nonfinite formal forecast; no fill or retry")
            receipt.update(
                task=task, formal_stage="A", ordinal=index + 1,
                first_origin=int(origins[0]), last_origin=int(origins[-1]),
                boundary_origin=cut - 1, boundary_forecast_sha256=hashlib.sha256(boundary.tobytes()).hexdigest(),
                frozen_fit_suffix_prediction_difference=boundary_difference,
                boundary_check_scope="frozen model prediction only; no extra market refit",
                completed_monotonic=time.monotonic(),
            )
            deadline_check(deadline)
            # All point predictions, input arrays and learned parameters stay local-only.
            states = {"state_" + key: value.detach().cpu().numpy()
                      for key, value in model.net.state_dict().items()}
            with (paths / "fits" / (stem + ".npz")).open("xb") as stream:
                np.savez(stream, origins=origins, predictions=predictions,
                         training_origins=training_ids, training_predictions=training,
                         validation_origins=validation_ids, validation_predictions=valid,
                         boundary_prediction=boundary, **states)
            write_json(ROOT / "stage_a_receipts" / (stem + ".json"), receipt)
            completed.append(stem)
            print("FIT " + json.dumps(dict(index=index + 1, case=stem,
                         epochs=receipt["epochs"], selected_epoch=receipt["selected_epoch"],
                         seconds=receipt["seconds"], elapsed=time.monotonic() - started)), flush=True)
            del model
        except Exception as error:
            failure = dict(task=task, status="failed", error_type=type(error).__name__,
                           error=str(error), monotonic=time.monotonic(),
                           completed_cases=completed, no_retry=True,
                           partial_ranking_forbidden=True, original_deadline_monotonic=deadline)
            write_json(ROOT / "STAGE_A_FAILED.json", failure)
            print("FAILED " + json.dumps(failure), flush=True)
            raise
    if source_hashes() != before["source_sha256"]:
        raise ValueError("Frozen source changed during run")
    elapsed = time.monotonic() - started
    deadline_check(deadline)
    write_json(ROOT / "STAGE_A_RUN_COMPLETE.json",
               dict(formal_fits=24, completed_cases=completed, elapsed_seconds=elapsed,
                    budget_seconds=FORMAL_BUDGET, completed_within_deadline=True,
                    completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    exposed_development_only=True, novelty_established=False))
    print("COMPLETE " + json.dumps(dict(fits=24, elapsed_seconds=elapsed)), flush=True)


def summarize():
    before = read_json(ROOT / "STAGE_A_PRE_RUN.json")
    complete = read_json(ROOT / "STAGE_A_RUN_COMPLETE.json")
    if complete["formal_fits"] != 24 or source_hashes() != before["source_sha256"]:
        raise ValueError("Incomplete run or changed implementation")
    if (ROOT / "STAGE_A_RESULTS.json").exists():
        raise FileExistsError("Existing aggregate cannot be overwritten")
    paths = ROOT / "local_only" / "stage_a"
    with np.load(paths / "INPUTS.npz", allow_pickle=False) as data:
        price, dates, stat = np.array(data["price"]), np.array(data["dates"]), np.array(data["reference"])
    expected = {task_stem(task) for task in planned_tasks()}
    if ({path.stem for path in (paths / "fits").glob("*.npz")} != expected
            or {path.stem for path in (ROOT / "stage_a_receipts").glob("*.json")} != expected):
        raise ValueError("Missing or unexpected fits")
    per_cell, per_seed, fit_summaries = [], [], []
    for seed in SEEDS:
        for name in MODELS + ("RW", "stat"):
            values, validations = [], []
            for cut, year in zip(CUTS, ("2020", "2021")):
                origins = np.arange(cut, 703 if cut == 459 else 930)
                if name in MODELS:
                    stem = task_stem(dict(model=name, seed=seed, cut=cut))
                    receipt = read_json(ROOT / "stage_a_receipts" / (stem + ".json"))
                    if receipt["completed_monotonic"] > before["deadline_monotonic"]:
                        raise ValueError("A formal fit exceeded the original deadline")
                    with np.load(paths / "fits" / (stem + ".npz"), allow_pickle=False) as data:
                        if not np.array_equal(data["origins"], origins):
                            raise ValueError("Forecast origins differ")
                        prediction = np.array(data["predictions"])
                    validations.append(receipt["validation_primary"])
                    fit_summaries.append({key: value for key, value in receipt.items() if key != "trace"})
                else:
                    prediction = np.zeros((len(origins), 20)) if name == "RW" else stat[origins]
                mask = np.char.startswith(dates[origins], year) & np.char.startswith(dates[origins + 20], year)
                scored = origins[mask]
                if len(scored) != (224 if cut == 459 else 227):
                    raise ValueError("451-origin annual support differs")
                for h in (1, 5, 20):
                    target = price[scored + h] - price[scored]
                    error = prediction[mask, h - 1] - target
                    mse, rw_mse = float(np.mean(error ** 2)), float(np.mean(target ** 2))
                    risk = mse / rw_mse
                    row = dict(model=name, seed=seed, year=year, horizon=h, origins=len(scored),
                               mse=mse, rw_mse=rw_mse, relative_mse=risk,
                               mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(mse)))
                    per_cell.append(row)
                    values.append(risk)
            per_seed.append(dict(model=name, seed=seed, score=float(np.mean(values)),
                                 cells=values, prior_validation=validations,
                                 unexposed_test=False, diagnostic_only=True))
    per_model = []
    for name in MODELS + ("RW", "stat"):
        rows = [row for row in per_seed if row["model"] == name]
        scores = np.array([row["score"] for row in rows])
        per_model.append(dict(model=name, seeds=list(SEEDS), mean_score=float(scores.mean()),
                              sd_score=float(scores.std(ddof=1)), min_score=float(scores.min()),
                              max_score=float(scores.max()),
                              cells_mean=np.mean([row["cells"] for row in rows], axis=0).tolist(),
                              cells_sd=np.std([row["cells"] for row in rows], axis=0, ddof=1).tolist()))
    result = dict(formal_fits=24, models=list(MODELS), controls=["RW", "stat"],
                  seeds=list(SEEDS), score_origins=451, annual_origin_counts=[224, 227],
                  horizons=[1, 5, 20], cell_order=["2020_h1", "2020_h5", "2020_h20",
                                                "2021_h1", "2021_h5", "2021_h20"],
                  rows=per_model, per_seed=per_seed, per_cell=per_cell,
                  unexposed_test=False, novelty_established=False, diagnostic_only=True,
                  no_selection_of_new_hyperparameters=True)
    write_json(ROOT / "STAGE_A_RESULTS.json", result)
    write_json(ROOT / "STAGE_A_TRAINING_SUMMARY.json",
               dict(formal_fits=24, records=fit_summaries,
                    all_full_traces="stage_a_receipts/*.json", point_predictions="local_only/stage_a/fits"))
    with (ROOT / "STAGE_A_PER_CELL.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_cell[0]))
        writer.writeheader()
        writer.writerows(per_cell)
    with (ROOT / "STAGE_A_PER_SEED.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "seed", "score", *result["cell_order"]])
        for row in per_seed:
            writer.writerow([row["model"], row["seed"], row["score"], *row["cells"]])
    print(json.dumps(dict(rows=per_model, formal_fits=24)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "timing", "run", "summarize"))
    args = parser.parse_args()
    {"check": synthetic_check, "timing": timing, "run": run, "summarize": summarize}[args.action]()
