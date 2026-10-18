"""Paired profiled-transfer learners; independent implementation, 2026-10-18.

No data loading, file writing, download, optimizer momentum, clipping, stochastic
minibatch selection, or author benchmark driver is contained here.  The caller
provides already causal support/meta/validation/test tensors and one common
source-episode schedule.  Source ridge can execute the separately audited R2D2
numerical kernel; target heads support a general precision matrix.

The inner objective is mean((Y-XW)**2) + trace(W.T @ P @ W)/H.  Consequently
W=solve(X.T@X/n+P, X.T@Y/n).  Prediction units are the caller's standardized
cumulative changes.  ``sigma`` converts them back to raw cumulative changes.

``aux_one`` / ``aux_five`` are warm-started target HEAD-only approximations,
not the original Auto-Lambda joint virtual update or an author reproduction.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import math
import time
from typing import Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.func import functional_call


MODEL_NAMES = (
    "local_profile", "uniform_profile", "aux_one", "aux_five", "profile",
    "transport", "transport_uniform", "covariance_profile", "diagonal_profile",
)


class NumericalFailure(RuntimeError):
    """An explicit failed numerical condition; no regularization retry follows."""


class Encoder(nn.Module):
    """Causal 20*C -> 32(tanh) -> 8 features; no separate head intercept."""

    def __init__(self, input_dim=3, hidden_size=32, feature_dim=8, context_length=20):
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_length = int(context_length)
        self.hidden = nn.Linear(self.input_dim * self.context_length, hidden_size, dtype=torch.float64)
        self.feature = nn.Linear(hidden_size, feature_dim, dtype=torch.float64)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.context_length, self.input_dim):
            raise ValueError("Encoder requires [B, context_length, input_dim]")
        return self.feature(torch.tanh(self.hidden(x.reshape(len(x), -1))))


def _sym(x):
    return 0.5 * (x + x.T)


def _finite(x, name):
    if not bool(torch.isfinite(x).all()):
        raise NumericalFailure(name + " is not finite")


def covariance(x):
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("Nonempty [n,d] feature matrix required")
    return _sym(x.T @ x / len(x))


def ridge_head(x, y, precision=0.1):
    """Differentiable primal ridge solve, with no jitter, clipping or retry."""
    if y.ndim != 2 or len(y) != len(x):
        raise ValueError("Ridge targets must be [n,H]")
    if not isinstance(precision, torch.Tensor):
        precision = torch.eye(x.shape[1], dtype=x.dtype, device=x.device) * precision
    if precision.shape != (x.shape[1], x.shape[1]):
        raise ValueError("Precision must have shape [d,d]")
    answer = torch.linalg.solve(_sym(covariance(x) + precision), x.T @ y / len(x))
    _finite(answer, "ridge head")
    return answer


def profile_objective(x, y, precision):
    head = ridge_head(x, y, precision)
    mse = (x @ head - y).square().mean()
    penalty = (head * (precision @ head)).sum() / y.shape[1]
    return mse + penalty, head


def rank_diagnostics(x, rank_rtol=None):
    with torch.no_grad():
        singular = torch.linalg.svdvals(x)
        largest = float(singular[0])
        smallest = float(singular[-1])
        relative = (max(x.shape) * torch.finfo(x.dtype).eps * 10
                    if rank_rtol is None else float(rank_rtol))
        tolerance = relative * largest
        full_rank = len(x) >= x.shape[1] and smallest > tolerance and largest > 0
    return {"full_column_rank": full_rank, "smallest_singular": smallest,
            "largest_singular": largest, "rank_tolerance": tolerance,
            "condition": largest / smallest if smallest > 0 else float("inf")}


def transport_precision(x, x_new, precision, residual_penalty=0.1,
                        rank_rtol=None, return_aux=False):
    """A=(X'X)^-1 X'X_new via algebraically equivalent reduced QR.

    All operations, including QR, A, E and the transported precision, remain in
    the autograd graph. Rank diagnostics use detached values only for failure
    detection. A rank failure is reported; no pseudoinverse/jitter is substituted.
    """
    if x.shape != x_new.shape:
        raise ValueError("Transport requires the same target support rows")
    diagnostics = rank_diagnostics(x, rank_rtol)
    if not diagnostics["full_column_rank"]:
        raise NumericalFailure("Transport support is not full column rank: " + str(diagnostics))
    new_diagnostics = rank_diagnostics(x_new, rank_rtol)
    if not new_diagnostics["full_column_rank"]:
        raise NumericalFailure("New transport support is not full column rank: " + str(new_diagnostics))
    with torch.no_grad():
        if not bool((torch.linalg.eigvalsh(_sym(precision)).min() > 0)):
            raise NumericalFailure("Old transport precision is not positive definite")
    q, r = torch.linalg.qr(x, mode="reduced")
    a = torch.linalg.solve_triangular(r, q.T @ x_new, upper=True)
    e = x_new - x @ a
    transported = _sym(a.T @ precision @ a + residual_penalty * (e.T @ e) / len(x))
    _finite(transported, "transported precision")
    with torch.no_grad():
        if not bool((torch.linalg.eigvalsh(transported).min() > 0)):
            raise NumericalFailure("Transported precision is not positive definite")
    if not return_aux:
        return transported
    return transported, {"A": a, "E": e, "old_rank": diagnostics, "new_rank": new_diagnostics}


def covariance_calibration(x, lambda0=0.1):
    """Match initial hat-matrix trace to ordinary lambda0 ridge, at full rank."""
    diag = rank_diagnostics(x)
    if not diag["full_column_rank"]:
        raise NumericalFailure("Covariance control needs full column rank")
    c = covariance(x)
    identity = torch.eye(c.shape[0], dtype=c.dtype, device=c.device)
    df0 = torch.trace(torch.linalg.solve(c + lambda0 * identity, c))
    mu = (c.shape[0] / df0 - 1.0).detach()
    _finite(mu, "covariance coefficient")
    if float(mu) < 0:
        raise NumericalFailure("Negative covariance coefficient")
    return mu, {"initial_df": float(df0.detach()), "mu": float(mu), **diag}


def covariance_precision(x, mu):
    return mu * covariance(x)


def head_gradient_steps(x_new, y, warm_head, old_covariance, precision,
                        steps=1, return_aux=False):
    """1/5 head-only GD steps with a conservative detached OLD Hessian rate.

    Actual inner loss averages both samples and horizons, so its Hessian is
    2*(C+P)/H. rate=H/(2*lambda_max(C_old+P)) gives the code update below.
    There is no claim this old-matrix rate stays contractive for every proposal.
    """
    maximum = torch.linalg.eigvalsh(_sym(old_covariance + precision)).max().detach()
    if float(maximum) <= 0:
        raise NumericalFailure("Nonpositive head Hessian eigenvalue")
    c = covariance(x_new)
    z = x_new.T @ y / len(x_new)
    matrix = _sym(c + precision)
    head = warm_head
    for _ in range(int(steps)):
        head = head - (matrix @ head - z) / maximum
    _finite(head, "approximate head")
    residual = torch.linalg.vector_norm(matrix @ head - z)
    denominator = torch.linalg.vector_norm(z).clamp_min(torch.finfo(x_new.dtype).tiny)
    if not return_aux:
        return head
    return head, {"head_steps": int(steps),
                  "head_learning_rate": y.shape[1] / (2 * float(maximum)),
                  "relative_normal_equation_residual": float((residual / denominator).detach()),
                  "new_to_old_max_hessian_ratio": float(
                      (torch.linalg.eigvalsh(matrix).max() / maximum).detach())}


def state_hash(state):
    """Canonical tensor identity independent of torch.save container bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _clone_state(model, precision, logits, r, mu):
    output = {"encoder." + k: v.detach().cpu().clone()
              for k, v in model.state_dict().items()}
    output["precision"] = precision.detach().cpu().clone()
    output["logits"] = logits.detach().cpu().clone()
    output["diagonal_log_precision"] = r.detach().cpu().clone()
    output["covariance_mu"] = mu.detach().cpu().clone()
    return output


def _tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(device="cpu", dtype=torch.float64)
    return torch.as_tensor(x, dtype=torch.float64, device="cpu")


def _batch(batch, need_y=True):
    answer = {"x": _tensor(batch["x"])}
    if need_y:
        answer["y"] = _tensor(batch["y"])
        if answer["y"].ndim != 2 or len(answer["y"]) != len(answer["x"]):
            raise ValueError("Batch y must be [n,H]")
        _finite(answer["y"], "batch y")
    _finite(answer["x"], "batch x")
    for name in ("sigma", "delta"):
        if name in batch:
            answer[name] = _tensor(batch[name])
    return answer


def _sigma(batch, shape):
    sigma = batch.get("sigma")
    if sigma is None:
        raise ValueError("sigma is required to restore prediction units")
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    if sigma.shape not in ((shape[0], 1), shape):
        raise ValueError("sigma must be [n], [n,1] or [n,H]")
    if not bool(torch.isfinite(sigma).all()) or not bool((sigma > 0).all()):
        raise ValueError("sigma must be finite and positive")
    return sigma


def validation_score(prediction, batch, horizons=(1, 5, 20)):
    """Equal selected-horizon raw-delta MSE/RW-MSE; not training-unit MSE."""
    delta = batch.get("delta")
    if delta is None or delta.shape != prediction.shape:
        raise ValueError("Validation delta [n,H] is required")
    raw = prediction * _sigma(batch, prediction.shape)
    indices = torch.tensor([h - 1 for h in horizons], dtype=torch.long)
    denominator = delta[:, indices].square().mean(dim=0)
    if not bool((denominator > 0).all()):
        raise NumericalFailure("Zero validation RW denominator")
    ratios = (raw[:, indices] - delta[:, indices]).square().mean(dim=0) / denominator
    _finite(ratios, "validation ratios")
    return float(ratios.mean()), ratios.detach().cpu().tolist()


def _norm(gradients):
    return float(torch.sqrt(sum(g.detach().square().sum() for g in gradients.values())))


def _gradient(loss, parameters):
    values = torch.autograd.grad(loss, tuple(parameters.values()))
    output = OrderedDict((name, value.detach()) for name, value in zip(parameters, values))
    for name, value in output.items():
        _finite(value, "gradient " + name)
    return output


def _proposal(parameters, primary, sources, weights, lr):
    output = OrderedDict()
    for name, p in parameters.items():
        g = primary[name]
        for weight, source in zip(weights, sources):
            g = g + weight * source[name]
        output[name] = p.detach() - lr * g
    return output


def _precision_for_virtual(model_name, old_x, new_x, precision, lambda0, mu, r):
    if model_name in ("transport", "transport_uniform"):
        return transport_precision(old_x, new_x, precision, lambda0, return_aux=True)
    if model_name == "covariance_profile":
        if not rank_diagnostics(new_x)["full_column_rank"]:
            raise NumericalFailure("Covariance support is not full column rank")
        return covariance_precision(new_x, mu), {}
    if model_name == "diagonal_profile":
        return lambda0 * torch.diag(torch.exp(r)), {}
    identity = torch.eye(new_x.shape[1], dtype=new_x.dtype, device=new_x.device)
    return lambda0 * identity, {}


def _source_indices(case, step, source_names, horizon):
    episodes = case["source_episodes"]
    episode = episodes[step] if not isinstance(episodes, Mapping) else episodes[step]
    if set(episode) != set(source_names):
        raise ValueError("Every scheduled step must contain exactly all source names")
    result = {}
    for name in source_names:
        source = case["sources"][name]
        si = torch.as_tensor(episode[name]["support"], dtype=torch.long)
        qi = torch.as_tensor(episode[name]["query"], dtype=torch.long)
        for idx in (si, qi):
            if idx.ndim != 1 or len(idx) == 0 or int(idx.min()) < 0 or int(idx.max()) >= len(source["x"]):
                raise ValueError("Invalid source-episode indices")
            if len(torch.unique(idx)) != len(idx):
                raise ValueError("Source-episode indices must be unique")
        origin_key = "origin" if "origin" in source else "origins"
        if origin_key in source:
            origins = torch.as_tensor(source[origin_key], dtype=torch.long)
            if int(origins[si].max()) + horizon >= int(origins[qi].min()):
                raise ValueError("Source support labels are not purged before query")
        elif bool(torch.isin(si, qi).any()):
            raise ValueError("Source support/query overlap")
        result[name] = (si, qi)
    return result


def fit(case, model_name, seed, config=None, *, source_ridge_fn=None):
    """Run one bounded fit on caller-supplied tensors only.

    Required case entries: target_support/meta/val/test and sources; each batch
    has x[n,20,3], y[n,20] (test y is never read by this routine). target_val must
    also provide raw cumulative delta[n,20] and sigma[n,20]. target_test provides
    sigma. source_episodes[step][name] has support/query index vectors.

    Return dictionary: predictions (normalized test deltas, NumPy), raw_predictions,
    val_predictions (normalized), traces, state (best encoder/precision/logits/r/mu/head),
    receipt. A deadline/numerical failure returns eligible=False and no test
    prediction; it must not be silently included in selection or retried.
    """
    config = dict(config or {})
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown model: " + str(model_name))
    torch.set_num_threads(1)
    started = time.monotonic()
    deadline = float(config.get("deadline", math.inf))
    max_steps = int(config.get("max_steps", config.get("maxsteps", 600)))
    def setting(primary, alias, default):
        if primary in config and alias in config and config[primary] != config[alias]:
            raise ValueError("Conflicting configuration aliases: " + primary + "/" + alias)
        return config.get(primary, config.get(alias, default))
    interval = int(setting("validation_interval", "val_interval", 10))
    patience = int(config.get("patience", 10))
    min_delta = float(config.get("min_delta", 1e-6))
    lr = float(setting("learning_rate", "theta_lr", 0.01))
    meta_lr = float(setting("meta_learning_rate", "meta_lr", 0.1))
    lambda0 = float(setting("ridge_lambda", "lambda0", 0.1))
    if min(max_steps, interval, patience) <= 0 or min(lr, meta_lr, lambda0) <= 0:
        raise ValueError("Positive fixed optimizer settings required")
    support = _batch(case["target_support"])
    meta = _batch(case["target_meta"])
    val = _batch(case["target_val"])
    horizon = support["y"].shape[1]
    if horizon != 20 or support["x"].shape[1:] != (20, 3):
        raise ValueError("This study freezes horizon=20 and input support=20x3")
    sources = {name: _batch(batch) for name, batch in sorted(case["sources"].items())}
    names = list(sources)
    if not names:
        raise ValueError("At least one named source is required for paired scheduling")
    if len(case["source_episodes"]) < max_steps:
        raise ValueError("Source episode schedule shorter than maximum fit budget")

    # Author counter is observational only; own fallback is explicitly labelled.
    author_counter = None
    if source_ridge_fn is None:
        try:
            from author_kernel import author_ridge, call_count
            source_ridge_fn, author_counter = author_ridge, call_count
        except ImportError as error:
            raise ImportError("Default source backend requires the audited author_kernel module; "
                              "pass source_ridge_fn=ridge_head explicitly for an own-kernel synthetic check") from error
    else:
        import sys
        module = sys.modules.get(getattr(source_ridge_fn, "__module__", ""))
        author_counter = getattr(module, "call_count", None)
    author_before = int(author_counter()) if author_counter is not None else 0
    backend_name = getattr(source_ridge_fn, "__module__", "") + "." + getattr(source_ridge_fn, "__name__", "callable")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = Encoder().to(device="cpu", dtype=torch.float64)
    identity = torch.eye(8, dtype=torch.float64)
    precision = lambda0 * identity
    logits = torch.log(torch.tensor([0.5] + [0.5 / len(names)] * len(names), dtype=torch.float64))
    r = torch.zeros(8, dtype=torch.float64)
    mu = torch.zeros((), dtype=torch.float64)
    df_receipt = None
    initial_state = _clone_state(model, precision, logits, r, mu)
    initial_encoder_hash = state_hash({k: v for k, v in initial_state.items() if k.startswith("encoder.")})
    adaptive = model_name not in ("local_profile", "uniform_profile", "transport_uniform")
    fixed_weights = torch.full((len(names),), 0.5 / len(names), dtype=torch.float64)
    if model_name == "local_profile":
        fixed_weights.zero_()

    traces = []
    best_state = None
    best_score = math.inf
    best_step = 0
    bad_checks = 0
    completed_steps = 0
    status = "max_steps"
    failure = None

    def evaluate():
        with torch.no_grad():
            xs = model(support["x"])
            head = ridge_head(xs, support["y"], precision)
            prediction = model(val["x"]) @ head
            score, ratios = validation_score(prediction, val)
        return score, ratios

    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("Deadline before initial validation")
        if model_name == "covariance_profile":
            with torch.no_grad():
                initial_x = model(support["x"])
                mu, df_receipt = covariance_calibration(initial_x, lambda0)
                precision = covariance_precision(initial_x, mu)
                initial_state = _clone_state(model, precision, logits, r, mu)
        score, ratios = evaluate()
        best_score, best_state = score, _clone_state(model, precision, logits, r, mu)
        traces.append({"step": 0, "validation_score": score, "validation_horizon_ratios": ratios,
                       "selected": True, "source_weights": fixed_weights.tolist() if not adaptive else torch.softmax(logits, 0)[1:].tolist(),
                       "seconds": time.monotonic() - started})
        for step in range(max_steps):
            if time.monotonic() >= deadline:
                raise TimeoutError("Cooperative deadline before next complete step")
            index_pairs = _source_indices(case, step, names, horizon)
            parameters = OrderedDict(model.named_parameters())
            xs = model(support["x"])
            # Transported/diagonal priors are state values during this gradient.
            # Covariance is a function of the CURRENT encoder and its derivative
            # is part of the known covariance-profile control, not detached.
            base_precision = (covariance_precision(xs, mu) if model_name == "covariance_profile"
                              else precision.detach())
            base_loss, current_head = profile_objective(xs, support["y"], base_precision)
            old_x, warm_head = xs.detach(), current_head.detach()
            old_c = covariance(old_x)
            primary_grad = _gradient(base_loss, parameters)
            source_grads = []
            source_losses = []
            for name in names:
                si, qi = index_pairs[name]
                sb = sources[name]
                sx = model(sb["x"][si])
                sw = source_ridge_fn(sx, sb["y"][si], lambda0)
                sq = model(sb["x"][qi])
                loss = (sq @ sw - sb["y"][qi]).square().mean()
                source_grads.append(_gradient(loss, parameters))
                source_losses.append(float(loss.detach()))

            head_info = {}
            meta_value = None
            meta_gradient_norm = None
            if adaptive:
                a_variable = logits.detach().requires_grad_(True)
                r_variable = r.detach().requires_grad_(model_name == "diagonal_profile")
                old_weights = torch.softmax(a_variable, 0)[1:]
                virtual_parameters = _proposal(parameters, primary_grad, source_grads, old_weights, lr)
                new_x = functional_call(model, virtual_parameters, (support["x"],), strict=True)
                new_q = functional_call(model, virtual_parameters, (meta["x"],), strict=True)
                virtual_precision, transport_info = _precision_for_virtual(
                    model_name, old_x, new_x, precision.detach(), lambda0, mu, r_variable)
                if model_name in ("aux_one", "aux_five"):
                    virtual_head, head_info = head_gradient_steps(
                        new_x, support["y"], warm_head, old_c, virtual_precision,
                        1 if model_name == "aux_one" else 5, return_aux=True)
                else:
                    virtual_head = ridge_head(new_x, support["y"], virtual_precision)
                meta_loss = (new_q @ virtual_head - meta["y"]).square().mean()
                variables = (a_variable, r_variable) if model_name == "diagonal_profile" else (a_variable,)
                outer_grads = torch.autograd.grad(meta_loss, variables)
                for gradient in outer_grads:
                    _finite(gradient, "meta gradient")
                logits = (a_variable - meta_lr * outer_grads[0]).detach()
                logits = logits - logits.mean()  # exact softmax gauge; no clipping.
                if model_name == "diagonal_profile":
                    r = (r_variable - meta_lr * outer_grads[1]).detach().clamp(-4.0, 4.0)
                weights = torch.softmax(logits, 0)[1:].detach()
                meta_value = float(meta_loss.detach())
                meta_gradient_norm = float(torch.sqrt(sum(g.square().sum() for g in outer_grads)))
            else:
                weights = fixed_weights
            actual_parameters = _proposal(parameters, primary_grad, source_grads, weights, lr)
            combined_grad_norm = float(torch.sqrt(sum(
                ((parameters[k].detach() - actual_parameters[k]) / lr).square().sum()
                for k in parameters)))
            with torch.no_grad():
                for name, p in model.named_parameters():
                    _finite(actual_parameters[name], "updated parameter " + name)
                    p.copy_(actual_parameters[name])
                actual_x = model(support["x"])
                precision, actual_transport = _precision_for_virtual(
                    model_name, old_x, actual_x, precision, lambda0, mu, r)
                precision = precision.detach()
                eigen = torch.linalg.eigvalsh(precision)
                _finite(eigen, "precision eigenvalues")
                if not bool((eigen.min() > 0)):
                    raise NumericalFailure("Updated target precision is not positive definite")
            completed_steps = step + 1
            row = {"step": completed_steps, "target_profile_objective": float(base_loss.detach()),
                   "source_query_losses": dict(zip(names, source_losses)),
                   "source_gradient_norms": dict(zip(names, [_norm(g) for g in source_grads])),
                   "target_gradient_norm": _norm(primary_grad), "combined_gradient_norm": combined_grad_norm,
                   "meta_objective_before_logit_update": meta_value, "meta_gradient_norm": meta_gradient_norm,
                   "source_weights": weights.tolist(), "null_weight": float(1.0 - weights.sum()),
                   "precision_min_eigenvalue": float(eigen.min()), "precision_max_eigenvalue": float(eigen.max()),
                   "precision_trace": float(eigen.sum()), "diagonal_log_precision": r.tolist(),
                   "validation_score": None, "selected": False, **head_info,
                   "seconds": time.monotonic() - started}
            if actual_transport:
                e = actual_transport["E"]
                row["transport_relative_residual"] = float(torch.linalg.vector_norm(e) /
                    torch.linalg.vector_norm(actual_x).clamp_min(torch.finfo(actual_x.dtype).tiny))
                row["transport_old_feature_condition"] = actual_transport["old_rank"]["condition"]
            if completed_steps % interval == 0 or completed_steps == max_steps:
                score, ratios = evaluate()
                row["validation_score"], row["validation_horizon_ratios"] = score, ratios
                if score < best_score - min_delta:
                    best_score, best_step = score, completed_steps
                    best_state = _clone_state(model, precision, logits, r, mu)
                    bad_checks = 0
                    row["selected"] = True
                else:
                    bad_checks += 1
            traces.append(row)
            if time.monotonic() >= deadline:
                raise TimeoutError("Cooperative deadline after complete step")
            if bad_checks >= patience:
                status = "early_stopping"
                break
    except (NumericalFailure, torch.linalg.LinAlgError, TimeoutError) as error:
        status = "deadline" if isinstance(error, TimeoutError) else "numerical_failure"
        failure = str(error)

    eligible = status in ("max_steps", "early_stopping") and best_state is not None
    receipt = {"model_name": model_name, "head_control": {"aux_one": "head1", "aux_five": "head5"}.get(model_name),
               "seed": int(seed), "eligible": eligible, "status": status, "failure": failure,
               "steps": completed_steps, "selected_step": best_step,
               "selected_validation_score": best_score if math.isfinite(best_score) else None,
               "seconds": time.monotonic() - started, "initial_encoder_hash": initial_encoder_hash,
               "initial_state_hash": state_hash(initial_state), "source_names": names,
               "source_ridge_backend": backend_name,
               "author_ridge_calls": (int(author_counter()) - author_before) if author_counter else 0,
               "source_ridge_calls_expected_completed_steps": completed_steps * len(names),
               "covariance_calibration": df_receipt, "encoder_parameters": sum(p.numel() for p in model.parameters()),
               "feature_dimension": 8, "horizon": horizon, "gradient_clipping": False,
               "theta_lr": lr, "meta_lr": meta_lr, "lambda0": lambda0,
               "max_steps": max_steps, "val_interval": interval, "patience": patience, "min_delta": min_delta,
               "base_target_precision_is_fixed_during_gradient": model_name != "covariance_profile",
               "test_forward_calls": 0, "test_target_values_used_in_fit": False,
               "prediction_units": "predictions and val_predictions are normalized caller y units; raw_predictions multiply sigma exactly once",
               "scope": "Own paired learning protocol; source may execute audited author ridge numerical kernel only"}
    output = {"predictions": None, "normalized_predictions": None, "raw_predictions": None,
              "val_predictions": None, "validation_predictions": None, "raw_val_predictions": None,
              "traces": traces, "state": best_state, "receipt": receipt}
    if not eligible:
        return output
    model.load_state_dict({k.removeprefix("encoder."): v for k, v in best_state.items() if k.startswith("encoder.")})
    precision, logits, r, mu = (best_state[k] for k in ("precision", "logits", "diagonal_log_precision", "covariance_mu"))
    # The test dictionary is deliberately not materialized earlier. Its y/delta
    # entries are not read at all, and it cannot affect selection or gradients.
    test = {"x": _tensor(case["target_test"]["x"]), "sigma": _tensor(case["target_test"]["sigma"])}
    with torch.no_grad():
        head = ridge_head(model(support["x"]), support["y"], precision)
        val_pred = model(val["x"]) @ head
        normalized = model(test["x"]) @ head
        _finite(normalized, "selected test prediction")
        raw = normalized * _sigma(test, normalized.shape)
    output["state"]["target_head"] = head.detach().cpu().clone()
    output["predictions"] = normalized.detach().cpu().numpy().copy()
    output["normalized_predictions"] = output["predictions"]
    output["raw_predictions"] = raw.detach().cpu().numpy().copy()
    output["val_predictions"] = val_pred.detach().cpu().numpy().copy()
    output["validation_predictions"] = output["val_predictions"]
    output["raw_val_predictions"] = (val_pred * _sigma(val, val_pred.shape)).detach().cpu().numpy().copy()
    receipt["selected_state_hash"] = state_hash(output["state"])
    receipt["test_forward_calls"] = 1
    receipt["seconds"] = time.monotonic() - started
    return output
