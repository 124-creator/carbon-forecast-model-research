"""Bounded PROCEED protocol adaptation. See STAGE_B_PROTOCOL.md before use.

Actions: check (synthetic only), timing (3 approved fits), run (18 fresh fits
and 4239 mature online updates under one 900-second deadline), summarize.
No unexposed test, benchmark reproduction, or original-algorithm claim.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _thread_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_key] = "1"
os.environ["HF_HOME"] = str(ROOT / "local_only" / "huggingface_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import transformers

WORKSPACE = ROOT.parents[1]
FAIR_ROOT = ROOT.parent / "recurrent_response_20261014" / "fair_recurrent_tuning_20261016"
VENDOR = ROOT / "vendor" / "proceed"
sys.path.insert(0, str(FAIR_ROOT))
sys.path.insert(0, str(VENDOR))
from fair_core import configure, reference
from fair_data import load_prefix, support
from models.NLinear import Model as AuthorNLinear
from adapter.proceed import Proceed

AUTHOR_COMMIT = "0a8646199f6dcbf5346f933508b33c5e35b5708b"
LOOKBACK = HORIZON = 20
CUTS = (459, 703)
SEEDS = (17, 29, 43)
ARMS = ("NLinear_OGD", "PROCEED", "PROCEED_no_drift")
FORMAL_BUDGET = 900
ONLINE_LR = .0003
EXPECTED_UPDATES = 4239


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(items):
    value = hashlib.sha256()
    for name, tensor in sorted(items):
        value.update(name.encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def state_digest(net):
    return tensor_hash(net.state_dict().items())


def snapshot(net):
    return {name: parameter.detach().clone() for name, parameter in net.named_parameters()}


def deadline_check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Stage B original 900-second deadline reached; no reset")


def source_files():
    manifest = read_json(VENDOR / "SOURCE_MANIFEST.json")
    return [ROOT / "stage_b.py", ROOT / "STAGE_B_PROTOCOL.md",
            VENDOR / "SOURCE_MANIFEST.json", VENDOR / "LICENSE_NOTICE.md",
            FAIR_ROOT / "fair_core.py", FAIR_ROOT / "fair_data.py",
            *[VENDOR / row["path"] for row in manifest["files"]]]


def source_hashes():
    return {str(path.relative_to(WORKSPACE)): digest(path) for path in source_files()}


def verify_vendor():
    manifest = read_json(VENDOR / "SOURCE_MANIFEST.json")
    assert manifest["commit"] == AUTHOR_COMMIT and len(manifest["files"]) == 16
    for row in manifest["files"]:
        raw = (VENDOR / row["path"]).read_bytes()
        assert len(raw) == row["bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == row["github_blob_sha"]
    expected = {"models.NLinear": "models/NLinear.py", "adapter.proceed": "adapter/proceed.py",
                "adapter.module.generator": "adapter/module/generator.py",
                "adapter.module.base": "adapter/module/base.py",
                "adapter.module.ssf": "adapter/module/ssf.py", "adapter.module.up": "adapter/module/up.py",
                "adapter.module.down_up": "adapter/module/down_up.py", "layers.RevIN": "layers/RevIN.py"}
    for name, path in expected.items():
        assert Path(sys.modules[name].__file__).resolve() == (VENDOR / path).resolve()
    return {"commit": AUTHOR_COMMIT, "verified_files": 16, "import_closure": expected,
            "upstream_runner_executed": False}


def environment():
    return dict(python=platform.python_version(), executable=sys.executable,
                numpy=np.__version__, numpy_path=np.__file__, torch=torch.__version__,
                torch_path=torch.__file__, transformers=transformers.__version__,
                transformers_path=transformers.__file__, device="cpu", dtype="float64",
                math_threads=torch.get_num_threads(), deterministic_algorithms=True,
                matches_historical_environment=False)


class NoDriftProceed(Proceed):
    """Independently trained structural ablation; upstream modules are untouched."""
    def generate_adaptation(self, x):
        assert self.ema == 0
        concept = self.mlp1(x).mean(-2)
        recent_concept = self.mlp2(self.recent_batch).mean(-2).mean(
            list(range(0, self.recent_batch.dim() - 2)))
        # Retain both encoders and the entire generator. Do not use flag_basic.
        # Detached zero avoids undefined backward through the author's clip at 0.
        drift = torch.zeros_like(concept - recent_concept)
        return self.generator(drift, need_clip=not self.args.wo_clip)


def model_config():
    return SimpleNamespace(seq_len=LOOKBACK, pred_len=HORIZON, enc_in=1,
                           individual=False, freeze=False, concept_dim=16,
                           bottleneck_dim=4, act="sigmoid", individual_generator=False,
                           ema=0., tune_mode="down_up", merge_weights=True,
                           do_predict=True, wo_clip=False)


def make_net(arm, seed):
    if arm not in ARMS:
        raise ValueError("Unregistered arm")
    torch.manual_seed(int(seed))
    config = model_config()
    backbone = AuthorNLinear(config)
    if arm == "NLinear_OGD":
        return backbone
    return (Proceed if arm == "PROCEED" else NoDriftProceed)(backbone, config)


def parameter_groups(net):
    grouped = {"backbone_weights": [], "backbone_bias": [],
               "dynamic_adapter": [], "fixed_adapter_bias": []}
    fixed = ({id(module.biases[-1]) for module in net.generator.bottlenecks.values()}
             if isinstance(net, Proceed) else set())
    for name, parameter in net.named_parameters():
        if id(parameter) in fixed:
            group = "fixed_adapter_bias"
        elif isinstance(net, Proceed) and not name.startswith("backbone."):
            group = "dynamic_adapter"
        elif name.endswith("bias"):
            group = "backbone_bias"
        else:
            group = "backbone_weights"
        grouped[group].append((name, parameter))
    return grouped


def parameter_audit(net, before=None):
    records = {}
    for group, values in parameter_groups(net).items():
        row = dict(names=[name for name, _ in values],
                   parameter_count=sum(parameter.numel() for _, parameter in values),
                   trainable_count=sum(parameter.numel() for _, parameter in values if parameter.requires_grad),
                   sha256=tensor_hash(values))
        if before is not None:
            row.update(initial_sha256=tensor_hash([(name, before[name]) for name, _ in values]),
                       maximum_absolute_change=max([float((parameter.detach() - before[name]).abs().max())
                                                    for name, parameter in values] or [0.]),
                       bitwise_unchanged=all(torch.equal(parameter.detach(), before[name])
                                             for name, parameter in values))
        records[group] = row
    return records


def observed_windows(observed, origins):
    observed = np.asarray(observed)
    origins = np.asarray(origins, dtype=np.int64)
    if (observed.ndim != 1 or origins.ndim != 1 or not len(origins)
            or origins.min() < LOOKBACK - 1 or origins.max() >= len(observed)):
        raise ValueError("Invalid observed input support")
    return observed[origins[:, None] - np.arange(LOOKBACK - 1, -1, -1)][..., None]


def complete_sample(observed, origin, mean, sd):
    """Construct labels only when the entire H-step sample has matured."""
    origin = int(origin)
    if origin + HORIZON >= len(observed):
        raise ValueError("Requested incomplete/unmatured sample")
    raw = observed_windows(observed, np.array([origin]))
    labels = np.asarray(observed)[origin + 1:origin + HORIZON + 1][None, :, None]
    return torch.as_tensor((raw - mean) / sd), torch.as_tensor((labels - mean) / sd)


def set_recent(net, x, y):
    if isinstance(net, Proceed):
        net.recent_batch = torch.cat([x, y], dim=-2).detach().clone()


def inference(model, observed, origin):
    if len(observed) != origin + 1:
        raise ValueError("Inference must receive an exact observed prefix")
    x = torch.as_tensor((observed_windows(observed, np.array([origin])) - model.mean) / model.sd)
    model.net.eval()
    with torch.no_grad():
        levels = model.net(x)[0, :, 0].cpu().numpy() * model.sd + model.mean
    prediction = levels - observed[origin]
    if prediction.shape != (20,) or not np.isfinite(prediction).all():
        raise FloatingPointError("Invalid issued forecast")
    return prediction


def validation_predictions(net, prefix, ids, mean, sd, deadline=None):
    """Frozen validation; labels for each issued forecast remain scoring-only."""
    recent = net.recent_batch.detach().clone() if isinstance(net, Proceed) else None
    model = SimpleNamespace(net=net, mean=mean, sd=sd)
    output = []
    try:
        for origin in ids:
            deadline_check(deadline)
            observed = prefix[:int(origin) + 1]
            x, y = complete_sample(observed, int(origin) - HORIZON, mean, sd)
            set_recent(net, x, y)
            output.append(inference(model, observed, int(origin)))
    finally:
        if recent is not None:
            net.recent_batch = recent
    return np.stack(output)


def gradient_norm(net):
    gradients = [parameter.grad for parameter in net.parameters() if parameter.grad is not None]
    value = float(torch.sqrt(sum(gradient.detach().square().sum() for gradient in gradients)))
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite gradient; no retry")
    return value


def fit(arm, price, end, seed, max_epochs=60, deadline=None):
    configure()
    started = time.perf_counter()
    prefix = np.array(price[:end], dtype=np.float64, copy=True)
    ids, train, validation, scale, increment_target = support(prefix, end)
    raw = observed_windows(prefix, ids)
    mean, sd = float(raw[train].mean()), max(float(raw[train].std()), 1e-6)
    labels = increment_target + prefix[ids, None]
    x = torch.as_tensor((raw - mean) / sd)
    y = torch.as_tensor(((labels - mean) / sd)[..., None])
    net = make_net(arm, seed)
    initial_state = snapshot(net)
    initial_audit = parameter_audit(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=.003)
    sampler = torch.Generator().manual_seed(int(seed))

    def validate():
        prediction = validation_predictions(net, prefix, ids[validation], mean, sd, deadline)
        return float(np.mean(((prediction - increment_target[validation]) / sd) ** 2)), prediction

    initial = best = validate()[0]
    saved = copy.deepcopy(net.state_dict())
    saved_recent = net.recent_batch.detach().clone() if isinstance(net, Proceed) else None
    selected, stale, trace, nonzero_gradients = 0, 0, [], set()
    for epoch in range(1, int(max_epochs) + 1):
        deadline_check(deadline)
        order = torch.as_tensor(train)[torch.randperm(len(train), generator=sampler)]
        net.train()
        objective, count, maximum_gradient = 0., 0, 0.
        for offset in range(0, len(order), 32):
            deadline_check(deadline)
            batch = order[offset:offset + 32]
            optimizer.zero_grad(set_to_none=True)
            if isinstance(net, Proceed):
                net.flag_update = True
            loss = (net(x[batch]) - y[batch]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite pretraining loss; no retry")
            loss.backward()
            gradient = gradient_norm(net)
            for name, parameter in net.named_parameters():
                if parameter.grad is not None and bool(torch.any(parameter.grad != 0)):
                    nonzero_gradients.add(name)
            optimizer.step()
            set_recent(net, x[batch], y[batch])
            if isinstance(net, Proceed):
                net.flag_update = False
            objective += float(loss.detach()) * len(batch)
            count += len(batch)
            maximum_gradient = max(maximum_gradient, gradient)
        value, _ = validate()
        trace.append([epoch, objective / count, value, maximum_gradient])
        if not np.isfinite(trace[-1]).all():
            raise FloatingPointError("Nonfinite pretraining trace")
        if value < best - 1e-6:
            best, selected, stale = value, epoch, 0
            saved = copy.deepcopy(net.state_dict())
            saved_recent = net.recent_batch.detach().clone() if isinstance(net, Proceed) else None
        else:
            stale += 1
        if stale >= 12:
            break
    net.load_state_dict(saved)
    if saved_recent is not None:
        net.recent_batch = saved_recent
    net.eval()
    valid = validation_predictions(net, prefix, ids[validation], mean, sd, deadline)
    hidx = np.array([0, 4, 19])
    validation_primary = float(np.mean(np.mean((valid[:, hidx] - increment_target[validation][:, hidx]) ** 2, axis=0)
                                      / np.mean(increment_target[validation][:, hidx] ** 2, axis=0)))
    receipt = dict(arm=arm, author_commit=AUTHOR_COMMIT, cut=int(end), seed=int(seed),
                   train_n=len(train), validation_n=len(validation), pool_start=int(ids[0]),
                   train_first_origin=int(ids[train[0]]), train_max_origin=int(ids[train[-1]]),
                   train_max_target=int(ids[train[-1]] + HORIZON),
                   validation_first_origin=int(ids[validation[0]]),
                   validation_max_target=int(ids[validation[-1]] + HORIZON),
                   input_global_mean=mean, input_global_sd=sd,
                   fair_support_increment_scale=scale,
                   normalization_fit_source="raw[train] input windows only; one mean/std",
                   target="20 standardized future price levels",
                   validation="frozen parameters; per-origin mature context; all 20 normalized outputs MSE",
                   pretraining_recent_batch="previous shuffled training minibatch; zeros initially",
                   sampler="separate torch.Generator seeded identically across arms",
                   parameter_count=sum(parameter.numel() for parameter in net.parameters()),
                   initial_parameter_audit=initial_audit,
                   selected_pretraining_parameter_audit=parameter_audit(net, initial_state),
                   parameters_with_nonzero_gradient_in_any_attempted_pretraining_batch=sorted(nonzero_gradients),
                   state_sha256=state_digest(net), epochs=len(trace), selected_epoch=selected,
                   initial_validation=initial, best_validation=best, validation_primary=validation_primary,
                   seconds=time.perf_counter() - started, trace=trace,
                   trace_columns=["epoch", "training_batch_weighted_mean_MSE", "frozen_validation_MSE",
                                  "maximum_unclipped_batch_gradient_norm"],
                   optimizer=dict(name="Adam", lr=.003, batch_size=32, max_epochs=max_epochs,
                                  patience=12, minimum_validation_improvement=1e-6,
                                  weight_decay=0., gradient_clipping=False),
                   config=vars(model_config()), no_drift=(arm == "PROCEED_no_drift"),
                   historical_predictions_or_weights_used=False)
    model = SimpleNamespace(net=net, mean=mean, sd=sd, arm=arm)
    deadline_check(deadline)
    return model, receipt, dict(validation_origins=ids[validation], validation_predictions=valid,
                                training_origins=ids[train])


def prepare_online(model, prefix):
    if isinstance(model.net, Proceed):
        model.net.freeze_adapter(True)
        model.net.freeze_bias(False)
    # The previous complete sample's last label is cut-1, before first stream t=cut.
    last = len(prefix) - HORIZON - 1
    set_recent(model.net, *complete_sample(prefix, last, model.mean, model.sd))
    model.recent_max_label = len(prefix) - 1
    return torch.optim.Adam([parameter for parameter in model.net.parameters() if parameter.requires_grad],
                            lr=ONLINE_LR)


def online_one(model, optimizer, observed, origin, deadline=None):
    deadline_check(deadline)
    origin = int(origin)
    if len(observed) != origin + 1 or model.recent_max_label != origin - 1:
        raise ValueError("Online data must be an exact consecutive observed prefix")
    mature_origin = origin - HORIZON
    x, y = complete_sample(observed, mature_origin, model.mean, model.sd)
    net = model.net
    net.train()
    optimizer.zero_grad(set_to_none=True)
    if isinstance(net, Proceed):
        net.flag_online_learning, net.flag_current = True, False
    # Appendix D: retain the adapter in the mature sample's forward computation.
    loss = (net(x) - y).square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite mature online loss; no retry")
    loss.backward()
    gradient = gradient_norm(net)
    optimizer.step()
    set_recent(net, x, y)
    if isinstance(net, Proceed):
        net.flag_online_learning, net.flag_current = False, True
    prior_recent = model.recent_max_label
    model.recent_max_label = origin
    prediction = inference(model, observed, origin)
    trace = [origin, mature_origin, mature_origin - LOOKBACK + 1, mature_origin,
             mature_origin + 1, mature_origin + HORIZON, origin - LOOKBACK + 1,
             origin, prior_recent, model.recent_max_label, float(loss.detach()), gradient]
    deadline_check(deadline)
    return prediction, trace


def online_stream(model, price, cut, stop, deadline=None):
    started = time.perf_counter()
    optimizer = prepare_online(model, price[:cut])
    before = snapshot(model.net)
    initial = parameter_audit(model.net)
    initial_state_hash = state_digest(model.net)
    predictions, traces = [], []
    frozen_names = [name for name, parameter in model.net.named_parameters() if not parameter.requires_grad]
    for origin in range(cut, stop):
        # No function called within online_one receives any future suffix.
        prediction, trace = online_one(model, optimizer, price[:origin + 1], origin, deadline)
        for name, parameter in model.net.named_parameters():
            if name in frozen_names and not torch.equal(parameter.detach(), before[name]):
                raise AssertionError("A frozen adapter parameter changed online: " + name)
        predictions.append(prediction)
        traces.append(trace)
    audit = parameter_audit(model.net, before)
    for group in ("dynamic_adapter", "backbone_bias") if isinstance(model.net, Proceed) else ():
        if not audit[group]["bitwise_unchanged"]:
            raise AssertionError("Frozen parameter group changed online")
    trace_array = np.asarray(traces, dtype=np.float64)
    if (not np.isfinite(trace_array).all() or not np.all(trace_array[:, 5] == trace_array[:, 0])
            or not np.all(trace_array[:, 7] == trace_array[:, 0])
            or not np.all(trace_array[:, 8] == trace_array[:, 0] - 1)):
        raise AssertionError("Online maturity trace failed")
    report = dict(updates=stop - cut, first_origin=cut, last_origin=stop - 1,
                  first_mature_origin=cut - HORIZON, last_mature_origin=stop - 1 - HORIZON,
                  all_label_maxima_equal_issue_origin=True, all_input_maxima_equal_issue_origin=True,
                  all_prior_context_label_maxima_equal_issue_origin_minus_one=True,
                  frozen_parameter_checks=(stop - cut) * len(frozen_names),
                  frozen_parameters_bitwise_unchanged=True,
                  initial_state_sha256=initial_state_hash,
                  final_state_sha256=state_digest(model.net),
                  initial_parameter_audit=initial, final_parameter_audit=audit,
                  loss_mean=float(trace_array[:, 10].mean()), loss_max=float(trace_array[:, 10].max()),
                  gradient_norm_mean=float(trace_array[:, 11].mean()),
                  gradient_norm_max=float(trace_array[:, 11].max()),
                  trace_sha256=hashlib.sha256(trace_array.tobytes()).hexdigest(),
                  trace_columns=["issue_origin", "mature_origin", "mature_input_min", "mature_input_max",
                                 "mature_target_min", "mature_target_max", "forecast_input_min",
                                 "forecast_input_max", "previous_recent_label_max", "new_recent_label_max",
                                 "normalized_mature_MSE", "unclipped_gradient_norm"],
                  trace_location="local_only/stage_b/fits/*.npz; pointwise not published",
                  seconds=time.perf_counter() - started, optimizer=dict(name="Adam", lr=ONLINE_LR,
                        batch_size=1, steps_per_origin=1, fresh_online_optimizer=True,
                        generator_frozen=True, fixed_adapter_bias_trainable=True))
    return np.stack(predictions), trace_array, report


def planned_tasks():
    return [dict(arm=arm, seed=seed, cut=cut) for seed in SEEDS for arm in ARMS for cut in CUTS]


def task_stem(task):
    return "{arm}_seed{seed}_cut{cut}".format(**task)


def require_stage_a_complete():
    completed = read_json(ROOT / "STAGE_A_RUN_COMPLETE.json")
    assert completed["formal_fits"] == 24 and completed["completed_within_deadline"]


def synthetic_check():
    configure()
    verify = verify_vendor()
    result = dict(scope="synthetic data only; no market prices loaded", passed=True,
                  vendor=verify, environment=environment(), cases=[])
    synthetic = 100 + np.cumsum(.08 + .12 * np.sin(np.arange(950) * .13))
    observed = synthetic[:151]
    try:
        complete_sample(observed, 131, 100., 2.)
    except ValueError:
        result["unmatured_sample_rejected"] = True
    else:
        raise AssertionError("Future incomplete labels accepted")
    reference_outputs = None
    for arm in ARMS:
        net = make_net(arm, 113)
        x, y = complete_sample(synthetic[:140], 100, 100., 2.)
        bx, by = torch.cat([x, x + .3]), torch.cat([y, y + .2])
        output = net(bx)
        if reference_outputs is None:
            reference_outputs = output.detach().clone()
        identity_difference = float((output.detach() - reference_outputs).abs().max())
        pre = snapshot(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=.003)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = (net(bx) - by).square().mean()
            loss.backward()
            gradient_norm(net)
            optimizer.step()
            set_recent(net, bx, by)
        shape = list(output.shape)
        # Test no-drift independence after nontrivial synthetic training.
        context_difference = None
        if isinstance(net, Proceed):
            net.eval()
            with torch.no_grad():
                net.recent_batch = torch.zeros(1, 40, 1)
                first = net(bx).detach().clone()
                net.recent_batch = torch.ones(1, 40, 1) * 5
                second = net(bx).detach().clone()
            context_difference = float((first - second).abs().max())
            if arm == "PROCEED_no_drift" and context_difference != 0.:
                raise AssertionError("No-drift control depends on recent context")
            if arm == "PROCEED" and context_difference <= 0.:
                raise AssertionError("Full generator context has no effect after synthetic training")
        saved_state = copy.deepcopy(net.state_dict())
        outputs, post_states, freeze_reports = [], [], []
        future_changed = synthetic.copy()
        future_changed[151:] += 5000. + np.arange(len(synthetic) - 151)
        for values in (synthetic, future_changed):
            replica = make_net(arm, 113)
            replica.load_state_dict(saved_state)
            model = SimpleNamespace(net=replica, mean=100., sd=2., arm=arm)
            opt = prepare_online(model, values[:150])
            before = snapshot(replica)
            pred, trace = online_one(model, opt, values[:151], 150)
            outputs.append(pred)
            post_states.append(state_digest(replica))
            freeze_reports.append(parameter_audit(replica, before))
            assert trace[1] == 130 and trace[5] == 150 and trace[7] == 150 and trace[8] == 149
        future_difference = float(np.max(np.abs(outputs[0] - outputs[1])))
        audit = freeze_reports[0]
        if isinstance(net, Proceed):
            assert audit["dynamic_adapter"]["bitwise_unchanged"]
            assert audit["backbone_bias"]["bitwise_unchanged"]
            assert audit["fixed_adapter_bias"]["maximum_absolute_change"] > 0.
        assert audit["backbone_weights"]["maximum_absolute_change"] > 0.
        assert shape == [2, 20, 1] and identity_difference < 1e-12 and future_difference == 0.
        assert post_states[0] == post_states[1]
        result["cases"].append(dict(arm=arm, output_shape=shape,
               identity_initialization_difference=identity_difference,
               context_dependence_after_synthetic_training=context_difference,
               future_suffix_prediction_difference=future_difference,
               future_suffix_postupdate_state_identical=post_states[0] == post_states[1],
               finite_forward_backward=True, synthetic_online_parameter_audit=audit,
               synthetic_pretraining_parameter_audit=parameter_audit(net, pre)))
    result["support"] = []
    for cut in CUTS:
        ids, train, val, _, _ = support(synthetic[:cut], cut)
        assert ids[train[-1]] + 20 < ids[val[0]] and ids[val[-1]] + 20 == cut - 1
        result["support"].append(dict(cut=cut, train_n=len(train), validation_n=len(val),
                                  train_max_target=int(ids[train[-1]] + 20),
                                  validation_first_origin=int(ids[val[0]]),
                                  validation_max_target=int(ids[val[-1]] + 20)))
    write_json(ROOT / "STAGE_B_SYNTHETIC_CHECK.json", result)
    print("CHECK " + json.dumps(dict(passed=True, arms=list(ARMS), environment=environment())), flush=True)


def timing():
    configure()
    require_stage_a_complete()
    if not read_json(ROOT / "STAGE_B_SYNTHETIC_CHECK.json")["passed"]:
        raise ValueError("Synthetic checks required")
    if (ROOT / "STAGE_B_TIMING_PLAN.json").exists():
        raise FileExistsError("Timing already started; no implicit repeats")
    write_json(ROOT / "STAGE_B_TIMING_PLAN.json", dict(timing_fits=3, arms=list(ARMS),
               cut=459, seed=113, max_epochs=3, source_sha256=source_hashes(),
               started_at_utc=datetime.now(timezone.utc).isoformat(),
               market_online_updates=0, development_year_scores_computed=False))
    (ROOT / "stage_b_timing_receipts").mkdir()
    price, dates, data = load_prefix()
    del dates
    records = []
    for arm in ARMS:
        model, receipt, _ = fit(arm, price, 459, 113, max_epochs=3)
        write_json(ROOT / "stage_b_timing_receipts" / (arm + ".json"), receipt)
        records.append(dict(arm=arm, epochs=receipt["epochs"], seconds=receipt["seconds"],
                            state_sha256=receipt["state_sha256"]))
        print("TIMING " + json.dumps(records[-1]), flush=True)
        del model
    write_json(ROOT / "STAGE_B_TIMING.json", dict(timing_fits=3, records=records,
               data=data, source_sha256=source_hashes(), environment=environment(),
               market_online_updates=0, development_year_scores_computed=False))


def run():
    configure()
    require_stage_a_complete()
    verify = verify_vendor()
    if (ROOT / "STAGE_B_PRE_RUN.json").exists():
        raise FileExistsError("Existing formal Stage B cannot restart or reset deadline")
    timing_record = read_json(ROOT / "STAGE_B_TIMING.json")
    if timing_record["timing_fits"] != 3 or timing_record["source_sha256"] != source_hashes():
        raise ValueError("Missing timing or implementation changed after timing")
    if timing_record["environment"] != environment():
        raise ValueError("Environment changed after timing")
    price, dates, data = load_prefix()
    assert tuple(int((dates < year).sum()) for year in ("2020-01-01", "2021-01-01")) == CUTS
    paths = ROOT / "local_only" / "stage_b"
    paths.mkdir(parents=True)
    (paths / "fits").mkdir()
    (ROOT / "stage_b_receipts").mkdir()
    started = time.monotonic()
    deadline = started + FORMAL_BUDGET
    before = dict(stage="B", formal_expected=18, tasks=planned_tasks(), cuts=list(CUTS),
                   seeds=list(SEEDS), arms=list(ARMS), online_expected=EXPECTED_UPDATES,
                   stream_ranges_inclusive=[[459, 702], [703, 929]],
                   stream_counts_per_cut=[244, 227], max_epochs=60,
                   budget_seconds=FORMAL_BUDGET, workers=1, device="cpu", math_threads=1,
                   started_monotonic=started, deadline_monotonic=deadline,
                   started_at_utc=datetime.now(timezone.utc).isoformat(),
                   source_sha256=source_hashes(), data=data, environment=environment(), vendor=verify,
                   config=vars(model_config()), pretraining_learning_rate=.003,
                   online_learning_rate=ONLINE_LR, online_minibatch=1, steps_per_origin=1,
                   score=dict(origin_count=451, annual_counts=[224, 227], horizons=[1, 5, 20],
                              weighting="equal six year x horizon MSE / RW MSE cells",
                              refresh="all three arms update once at every observed origin before forecasting"),
                   no_test_per_epoch=True, historical_predictions_or_weights_used=False,
                   protocol_adaptation=True, original_benchmark_reproduction=False,
                   unexposed_test=False, novelty_established=False)
    write_json(ROOT / "STAGE_B_PRE_RUN.json", before)
    print("FROZEN " + json.dumps(dict(fits=18, updates=EXPECTED_UPDATES, seconds=FORMAL_BUDGET,
                                      deadline=deadline)), flush=True)
    with (paths / "INPUTS.npz").open("xb") as stream:
        np.savez(stream, price=price, dates=dates, reference=reference(price)[0])
    completed, update_count = [], 0
    for ordinal, task in enumerate(planned_tasks(), 1):
        arm, cut, seed = task["arm"], task["cut"], task["seed"]
        stem = task_stem(task)
        with (ROOT / "STAGE_B_ATTEMPTS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(task=task, ordinal=ordinal, status="started",
                         monotonic=time.monotonic(), original_deadline_monotonic=deadline)) + "\n")
        try:
            deadline_check(deadline)
            model, receipt, local = fit(arm, price, cut, seed, 60, deadline)
            selected_state = {"pretrained_" + name: value.detach().cpu().numpy().copy()
                              for name, value in model.net.state_dict().items()}
            pretrained_recent = (model.net.recent_batch.detach().cpu().numpy().copy()
                                 if isinstance(model.net, Proceed) else np.empty((0, 40, 1)))
            # Prediction-only boundary check; no additional market gradient update.
            set_recent(model.net, *complete_sample(price[:cut], cut - HORIZON - 1, model.mean, model.sd))
            initial_recent = (model.net.recent_batch.detach().cpu().numpy().copy()
                              if isinstance(model.net, Proceed) else np.empty((0, 40, 1)))
            boundary = inference(model, price[:cut], cut - 1)
            changed = price.copy()
            changed[cut:] += 10000. + np.arange(len(price) - cut)
            boundary_changed = inference(model, changed[:cut], cut - 1)
            difference = float(np.max(np.abs(boundary - boundary_changed)))
            assert difference == 0.
            stop = 703 if cut == 459 else 930
            predictions, trace, online = online_stream(model, price, cut, stop, deadline)
            update_count += online["updates"]
            states = {"final_" + name: value.detach().cpu().numpy().copy()
                      for name, value in model.net.state_dict().items()}
            final_recent = (model.net.recent_batch.detach().cpu().numpy().copy()
                            if isinstance(model.net, Proceed) else np.empty((0, 40, 1)))
            receipt.update(task=task, formal_stage="B", ordinal=ordinal, online=online,
                           final_state_sha256=state_digest(model.net),
                           boundary_origin=cut - 1, frozen_boundary_suffix_difference=difference,
                           boundary_forecast_sha256=hashlib.sha256(boundary.tobytes()).hexdigest(),
                           completed_monotonic=time.monotonic())
            deadline_check(deadline)
            with (paths / "fits" / (stem + ".npz")).open("xb") as stream:
                np.savez(stream, origins=np.arange(cut, stop), predictions=predictions,
                         online_trace=trace, boundary_prediction=boundary,
                         pretrained_recent_batch=pretrained_recent,
                         online_initial_recent_batch=initial_recent,
                         final_recent_batch=final_recent,
                         input_global_mean=np.array(model.mean), input_global_sd=np.array(model.sd),
                         **local, **selected_state, **states)
            write_json(ROOT / "stage_b_receipts" / (stem + ".json"), receipt)
            completed.append(stem)
            print("FIT " + json.dumps(dict(ordinal=ordinal, case=stem, epochs=receipt["epochs"],
                  selected_epoch=receipt["selected_epoch"], fit_seconds=receipt["seconds"],
                  online_seconds=online["seconds"], online_updates=online["updates"],
                  total_updates=update_count, elapsed=time.monotonic() - started)), flush=True)
            del model
        except Exception as error:
            failure = dict(task=task, status="failed", error_type=type(error).__name__, error=str(error),
                           monotonic=time.monotonic(), completed_cases=completed,
                           online_updates_in_completed_streams=update_count,
                           no_retry=True, partial_ranking_forbidden=True,
                           original_deadline_monotonic=deadline)
            write_json(ROOT / "STAGE_B_FAILED.json", failure)
            print("FAILED " + json.dumps(failure), flush=True)
            raise
    assert update_count == EXPECTED_UPDATES and len(completed) == 18
    assert source_hashes() == before["source_sha256"]
    deadline_check(deadline)
    write_json(ROOT / "STAGE_B_RUN_COMPLETE.json", dict(formal_fits=18, completed_cases=completed,
               online_updates=update_count, elapsed_seconds=time.monotonic() - started,
               budget_seconds=FORMAL_BUDGET, completed_within_deadline=True,
               completed_at_utc=datetime.now(timezone.utc).isoformat(),
               exposed_development_only=True, novelty_established=False))
    print("COMPLETE " + json.dumps(dict(fits=18, updates=update_count,
                                        elapsed_seconds=time.monotonic() - started)), flush=True)


def summarize():
    before, complete = read_json(ROOT / "STAGE_B_PRE_RUN.json"), read_json(ROOT / "STAGE_B_RUN_COMPLETE.json")
    assert complete["formal_fits"] == 18 and complete["online_updates"] == EXPECTED_UPDATES
    assert source_hashes() == before["source_sha256"]
    if (ROOT / "STAGE_B_RESULTS.json").exists():
        raise FileExistsError("Existing aggregate cannot be overwritten")
    paths = ROOT / "local_only" / "stage_b"
    with np.load(paths / "INPUTS.npz", allow_pickle=False) as inputs:
        price, dates, stat = np.array(inputs["price"]), np.array(inputs["dates"]), np.array(inputs["reference"])
    expected = {task_stem(task) for task in planned_tasks()}
    assert {path.stem for path in (paths / "fits").glob("*.npz")} == expected
    assert {path.stem for path in (ROOT / "stage_b_receipts").glob("*.json")} == expected
    per_cell, per_seed, receipts = [], [], []
    for seed in SEEDS:
        for arm in ARMS + ("RW", "stat"):
            values, validation_scores = [], []
            for cut, year in zip(CUTS, ("2020", "2021")):
                origins = np.arange(cut, 703 if cut == 459 else 930)
                if arm in ARMS:
                    stem = task_stem(dict(arm=arm, seed=seed, cut=cut))
                    receipt = read_json(ROOT / "stage_b_receipts" / (stem + ".json"))
                    assert receipt["completed_monotonic"] <= before["deadline_monotonic"]
                    with np.load(paths / "fits" / (stem + ".npz"), allow_pickle=False) as stored:
                        assert np.array_equal(stored["origins"], origins)
                        prediction = np.array(stored["predictions"])
                        trace = np.array(stored["online_trace"])
                    assert hashlib.sha256(trace.tobytes()).hexdigest() == receipt["online"]["trace_sha256"]
                    validation_scores.append(receipt["validation_primary"])
                    receipts.append({key: value for key, value in receipt.items() if key != "trace"})
                else:
                    prediction = np.zeros((len(origins), 20)) if arm == "RW" else stat[origins]
                mask = np.char.startswith(dates[origins], year) & np.char.startswith(dates[origins + 20], year)
                scored = origins[mask]
                assert len(scored) == (224 if cut == 459 else 227)
                for horizon in (1, 5, 20):
                    target = price[scored + horizon] - price[scored]
                    error = prediction[mask, horizon - 1] - target
                    mse, rw_mse = float(np.mean(error ** 2)), float(np.mean(target ** 2))
                    risk = mse / rw_mse
                    per_cell.append(dict(arm=arm, seed=seed, year=year, horizon=horizon,
                            origins=len(scored), mse=mse, rw_mse=rw_mse, relative_mse=risk,
                            mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(mse))))
                    values.append(risk)
            per_seed.append(dict(arm=arm, seed=seed, score=float(np.mean(values)), cells=values,
                                 prior_validation=validation_scores))
    rows = []
    for arm in ARMS + ("RW", "stat"):
        selected = [row for row in per_seed if row["arm"] == arm]
        scores = np.array([row["score"] for row in selected])
        rows.append(dict(arm=arm, seeds=list(SEEDS), mean_score=float(scores.mean()),
                    sd_score=float(scores.std(ddof=1)), min_score=float(scores.min()), max_score=float(scores.max()),
                    cells_mean=np.mean([row["cells"] for row in selected], axis=0).tolist(),
                    cells_sd=np.std([row["cells"] for row in selected], axis=0, ddof=1).tolist()))
    keyed = {row["arm"]: row for row in rows}
    full_minus_nodrift = [next(row["score"] for row in per_seed if row["arm"] == "PROCEED" and row["seed"] == seed)
                          - next(row["score"] for row in per_seed if row["arm"] == "PROCEED_no_drift" and row["seed"] == seed)
                          for seed in SEEDS]
    result = dict(formal_fits=18, online_updates=EXPECTED_UPDATES, arms=list(ARMS),
              seeds=list(SEEDS), score_origins=451, annual_origin_counts=[224, 227], horizons=[1, 5, 20],
              cell_order=["2020_h1", "2020_h5", "2020_h20", "2021_h1", "2021_h5", "2021_h20"],
              rows=rows, per_seed=per_seed, per_cell=per_cell,
              mechanism_diagnostic=dict(full_minus_no_drift_by_seed=full_minus_nodrift,
                  full_beats_no_drift_mean=keyed["PROCEED"]["mean_score"] < keyed["PROCEED_no_drift"]["mean_score"],
                  full_beats_plain_mean=keyed["PROCEED"]["mean_score"] < keyed["NLinear_OGD"]["mean_score"],
                  full_beats_RW_mean=keyed["PROCEED"]["mean_score"] < 1.,
                  full_beats_stat_mean=keyed["PROCEED"]["mean_score"] < keyed["stat"]["mean_score"]),
              refresh="all arms use identical mature single-sample online update frequency",
              stage_a_comparison="annual frozen models are references only; refresh strategies differ",
              unexposed_test=False, novelty_established=False, original_benchmark_reproduction=False,
              protocol_adaptation=True, no_selection_of_new_hyperparameters=True)
    write_json(ROOT / "STAGE_B_RESULTS.json", result)
    write_json(ROOT / "STAGE_B_TRAINING_SUMMARY.json", dict(formal_fits=18, online_updates=EXPECTED_UPDATES,
               records=receipts, full_pretraining_traces="stage_b_receipts/*.json",
               point_predictions_and_online_trace="local_only/stage_b/fits/*.npz"))
    with (ROOT / "STAGE_B_PER_CELL.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_cell[0]))
        writer.writeheader()
        writer.writerows(per_cell)
    with (ROOT / "STAGE_B_PER_SEED.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "seed", "score", *result["cell_order"]])
        for row in per_seed:
            writer.writerow([row["arm"], row["seed"], row["score"], *row["cells"]])
    print(json.dumps(dict(rows=rows, mechanism_diagnostic=result["mechanism_diagnostic"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "timing", "run", "summarize"))
    args = parser.parse_args()
    {"check": synthetic_check, "timing": timing, "run": run, "summarize": summarize}[args.action]()
