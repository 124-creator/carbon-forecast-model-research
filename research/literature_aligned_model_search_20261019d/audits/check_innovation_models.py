"""Synthetic-only implementation checks; no market reading or model selection."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "source"))
from innovation_models import ShockAllocationForecaster


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def independent_numpy(model, x):
    """Reimplement the recurrence using exported arrays, no model helpers."""
    p = {name: value.detach().cpu().numpy().copy() for name, value in model.named_parameters()}
    fractions = sigmoid(p["raw_rates"])
    rho1 = model.rho_max * fractions[0]
    rho = np.array([rho1, rho1 + (model.rho_max - rho1) * fractions[1]])

    def log_w(logits):
        a = np.concatenate((np.zeros((*logits.shape[:-1], 1)), logits), axis=-1)
        max_a = np.max(a, axis=-1, keepdims=True)
        normalizer = max_a + np.log(np.exp(a - max_a).sum(axis=-1, keepdims=True))
        return np.log(1.0 - model.permanent_floor) + a[..., 1:] - normalizer

    base_log_w = log_w(p["base_logits"])
    base_w = np.exp(base_log_w)
    state = np.zeros((len(x), 2))
    previous_r = np.zeros((len(x), 1))
    permanent = np.zeros(len(x))
    states, reconstructions, innovations = [], [], []
    multiplier = np.ones_like(state)
    for t in range(x.shape[1]):
        r = x[:, t, 0]
        u = x[:, t, 1:1 + model.n_exog]
        extra = x[:, t, 1 + model.n_exog:]
        exogenous = np.einsum("bd,d->b", u, p["beta"])
        innovation = r - np.einsum("bk,k->b", state, rho - 1.0) - exogenous
        if model.variant == "constant_allocation":
            proposed_log_w = np.broadcast_to(base_log_w, state.shape)
        else:
            inputs = [previous_r, state, u, extra]
            if model.innovation_feature == "magnitude":
                inputs.append(np.log1p(np.abs(innovation))[:, None])
            features = np.concatenate(inputs, axis=-1)
            hidden = np.tanh(features @ p["gate.0.weight"].T + p["gate.0.bias"])
            raw = hidden @ p["gate.2.weight"].T + p["gate.2.bias"]
            adjustment = model.gate_logit_bound * np.tanh(raw)
            proposed_log_w = log_w(p["base_logits"] + adjustment)
        allocation = np.exp(proposed_log_w) if model.variant == "full" else base_w
        state = rho * state + allocation * innovation[:, None]
        permanent += exogenous + (1.0 - allocation.sum(axis=-1)) * innovation
        if model.variant == "readout_gate" and t == x.shape[1] - 1:
            multiplier = np.exp(proposed_log_w - base_log_w)
        states.append(state.copy())
        reconstructions.append(permanent + state.sum(-1))
        innovations.append(innovation)
        previous_r = r[:, None]
    response = rho[None, :] ** np.arange(1, model.horizon + 1)[:, None] - 1.0
    return {
        "forecast": (state * multiplier) @ response.T,
        "transient_state": np.stack(states, axis=1),
        "reconstructed_level": np.stack(reconstructions, axis=1),
        "innovation": np.stack(innovations, axis=1),
    }


def set_nonconstant_gate(model, rng):
    with torch.no_grad():
        if model.gate is not None:
            model.gate[2].weight.copy_(torch.as_tensor(
                rng.normal(0.0, 0.18, tuple(model.gate[2].weight.shape)), dtype=torch.float64
            ))
            model.gate[2].bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float64))
        model.beta.copy_(torch.tensor([0.2, -0.12, 0.08], dtype=torch.float64))


def generative_allocation(state, exog, innovation):
    """A known, even-in-current-innovation allocation, independent of torch."""
    logits = np.array([
        -1.1 + 0.5 * exog[0] - 0.9 * np.log1p(abs(innovation)) + 0.2 * state[0],
        -1.6 - 0.3 * exog[1] - 0.5 * np.log1p(abs(innovation)) - 0.1 * state[1],
    ])
    weights = np.exp(np.r_[0.0, logits] - max(0.0, *logits))
    return 0.95 * (weights / weights.sum())[1:]


def synthetic_windows(rng, n=48, length=24, horizon=20):
    rho, beta = np.array([0.5, 0.9]), np.array([0.2, -0.12, 0.08])
    x, target, oracle = [], [], []
    for _ in range(n):
        state = np.zeros(2)
        observations = []
        for t in range(length + horizon):
            exog = rng.normal(size=3)
            innovation = rng.normal()
            r = state @ (rho - 1.0) + beta @ exog + innovation
            allocation = generative_allocation(state, exog, innovation)
            state = rho * state + allocation * innovation
            observations.append(np.r_[r, exog])
            if t == length - 1:
                oracle.append((rho[None, :] ** np.arange(1, horizon + 1)[:, None] - 1.0) @ state)
        observations = np.asarray(observations)
        x.append(observations[:length])
        target.append(np.cumsum(observations[length:, 0]))
    return np.asarray(x), np.asarray(target), np.asarray(oracle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=STUDY / "audits/INNOVATION_MODEL_CHECK.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing receipt: {args.output}")
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261019)
    output = {
        "scope": "Synthetic-only implementation qualification; no market data or performance gate",
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "source_sha256": hashlib.sha256((STUDY / "source/innovation_models.py").read_bytes()).hexdigest(),
        "checks": {},
    }
    models = {}
    for variant in ShockAllocationForecaster.VARIANTS:
        torch.manual_seed(1729)
        models[variant] = ShockAllocationForecaster(variant=variant).double()
    x = torch.tensor(rng.normal(size=(5, 80, 4)), dtype=torch.float64)
    initial = {name: model(x[:, :64]).detach().numpy() for name, model in models.items()}
    initial_gap = max(float(np.max(np.abs(value - initial["full"]))) for value in initial.values())
    require(initial_gap == 0.0, "The three variants must have exactly equal initial forecasts")
    output["checks"]["initialization_and_capacity"] = {
        "max_initial_forecast_gap": initial_gap,
        "trainable_parameters": {name: m.parameter_count() for name, m in models.items()},
        "constant_has_no_unused_neural_parameters": models["constant_allocation"].gate is None,
        "output_shape": list(initial["full"].shape),
        "note": "Equal initial forecasts are not generally the random-walk forecast",
    }
    for model in models.values():
        set_nonconstant_gate(model, rng)
    # Match the learned parameters exactly to isolate location of the gate.
    models["readout_gate"].load_state_dict(models["full"].state_dict())
    for name in ("raw_rates", "beta", "base_logits"):
        with torch.no_grad():
            getattr(models["constant_allocation"], name).copy_(getattr(models["full"], name))
    replay = {}
    for name, model in models.items():
        pred, diagnostics = model.forecast_with_diagnostics(x)
        independent = independent_numpy(model, x.numpy())
        errors = {"forecast": float(np.max(np.abs(pred.detach().numpy() - independent["forecast"])))}
        for field in ("transient_state", "innovation", "reconstructed_level"):
            errors[field] = float(np.max(np.abs(diagnostics[field].detach().numpy() - independent[field])))
        conservation = float((diagnostics["reconstructed_level"] - x[..., 0].cumsum(1)).abs().max())
        errors["level_conservation"] = conservation
        require(max(errors.values()) < 2e-12, f"Independent forward mismatch: {name}")
        allocation = diagnostics["allocation"]
        require(bool((allocation >= 0).all()), "Negative allocation")
        require(bool((allocation[..., 0] >= 0.05 - 1e-14).all()), "Permanent allocation floor violated")
        require(float((allocation.sum(-1) - 1.0).abs().max()) < 1e-14, "Allocation not conservative")
        perturbed = x.clone()
        perturbed[:, 64:] += torch.tensor(rng.normal(0, 20, size=(5, 16, 4)), dtype=torch.float64)
        _, other = model.forecast_with_diagnostics(perturbed)
        prefix_gap = max(float((diagnostics[field][:, :64] - other[field][:, :64]).abs().max())
                         for field in ("transient_state", "innovation", "allocation"))
        require(prefix_gap == 0.0, "Historical state depends on future input")
        require(torch.equal(model(x[:, :, 0], x[:, :, 1:]), pred), "Packed and separate APIs differ")
        replay[name] = {"max_errors": errors, "future_input_prefix_gap": prefix_gap}
    output["checks"]["independent_numpy_conservation_and_prefix"] = replay

    full = models["full"]
    readout = models["readout_gate"]
    structural_gap = float((full(x[:, :64]) - readout(x[:, :64])).abs().max())
    one_observation_gap = float((full(x[:, :1]) - readout(x[:, :1])).abs().max())
    require(structural_gap > 1e-6, "Gate-placement variants accidentally equivalent")
    require(one_observation_gap < 2e-14, "One-observation equivalence is broken")
    z = torch.zeros_like(x[:, :64])
    superposition_gap = float((full(x[:, :64]) + full(0.4 * x[:, :64]) - full(1.4 * x[:, :64]) - full(z)).abs().max())
    require(superposition_gap > 1e-6, "Full model has unintentionally collapsed to a linear filter")
    output["checks"]["mechanism_not_only_a_linear_ema"] = {
        "full_vs_readout_same_parameters_max_gap": structural_gap,
        "one_observation_equivalence_gap": one_observation_gap,
        "superposition_violation": superposition_gap,
        "interpretation": "Functional non-equivalence only, not novelty or predictive superiority",
    }

    # Directional finite differences independently verify the full gradient.
    gx = x[:2, :7].clone()
    gy = torch.tensor(rng.normal(size=(2, 20)), dtype=torch.float64)
    full.zero_grad(set_to_none=True)
    loss = (full(gx) - gy).square().mean()
    loss.backward()
    differences = {}
    for name, parameter in full.named_parameters():
        direction = torch.tensor(rng.normal(size=tuple(parameter.shape)), dtype=torch.float64)
        direction = direction / direction.norm()
        automatic = float((parameter.grad * direction).sum())
        original = parameter.detach().clone()
        step = 1e-6
        values = []
        for sign in (1.0, -1.0):
            with torch.no_grad():
                parameter.copy_(original + sign * step * direction)
                values.append(float((full(gx) - gy).square().mean()))
        with torch.no_grad():
            parameter.copy_(original)
        numerical = (values[0] - values[1]) / (2 * step)
        error = abs(automatic - numerical)
        require(np.isclose(automatic, numerical, rtol=5e-5, atol=2e-8), f"Gradient check failed for {name}")
        differences[name] = {"autograd": automatic, "finite_difference": numerical, "absolute_error": error}
    output["checks"]["directional_gradient"] = differences

    # Enumerate a future law outside the model implementation.  Conditional
    # sign symmetry leaves E[w(e)e|past,u]=0 even with magnitude-dependent w.
    rho = np.array([0.5, 0.9])
    beta = np.array([0.2, -0.12, 0.08])
    initial_state = np.array([0.4, -0.1])
    mean_path = np.zeros(3)
    paths = 0
    for signs in itertools.product((-1.0, 1.0), repeat=6):
        state = initial_state.copy()
        cumulative = 0.0
        for t in range(3):
            exog = signs[2*t] * np.array([0.7, -0.4, 0.2])
            innovation = signs[2*t+1] * 0.8
            r = state @ (rho - 1.0) + beta @ exog + innovation
            allocation = generative_allocation(state, exog, innovation)
            state = rho * state + allocation * innovation
            cumulative += r
            mean_path[t] += cumulative
        paths += 1
    mean_path /= paths
    analytic = (rho[None, :] ** np.arange(1, 4)[:, None] - 1.0) @ initial_state
    expectation_error = float(np.max(np.abs(mean_path - analytic)))
    require(expectation_error < 2e-14, "Conditional-mean identity failed under its stated law")
    response_matrix = rho[None, :] ** np.array([1, 5, 20])[:, None] - 1.0
    singular_values = np.linalg.svd(response_matrix, compute_uv=False)
    output["checks"]["known_law_conditional_mean"] = {
        "enumerated_paths": paths,
        "analytic_mean": analytic.tolist(),
        "enumerated_mean": mean_path.tolist(),
        "maximum_error": expectation_error,
        "known_rate_state_observability_singular_values": singular_values.tolist(),
        "interpretation": "State observability for known distinct rates, not general neural-parameter identification",
    }

    train_x, train_y, oracle_y = synthetic_windows(rng)
    train_x = torch.tensor(train_x, dtype=torch.float64)
    train_y = torch.tensor(train_y, dtype=torch.float64)
    training = {}
    for variant in ShockAllocationForecaster.VARIANTS:
        torch.manual_seed(1729)
        model = ShockAllocationForecaster(variant=variant).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        initial_loss = float((model(train_x) - train_y).square().mean())
        maximum_gradient_norm = {name: 0.0 for name in before}
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(train_x) - train_y).square().mean()
            require(bool(torch.isfinite(loss)), "Non-finite synthetic training loss")
            loss.backward()
            for name, parameter in model.named_parameters():
                require(parameter.grad is not None, f"No gradient for {variant}:{name}")
                maximum_gradient_norm[name] = max(maximum_gradient_norm[name], float(parameter.grad.norm()))
            optimizer.step()
        changes = {name: float((parameter - before[name]).abs().max()) for name, parameter in model.named_parameters()}
        require(all(value > 0 for value in maximum_gradient_norm.values()), "A parameter group never receives a gradient")
        require(all(value > 0 for value in changes.values()), "A parameter group was never updated")
        training[variant] = {
            "optimizer_steps": 12,
            "initial_training_mse": initial_loss,
            "final_training_mse": float((model(train_x) - train_y).square().mean()),
            "maximum_gradient_norm_by_parameter": maximum_gradient_norm,
            "maximum_parameter_change": changes,
        }
    output["checks"]["real_gradient_training_smoke"] = {
        "independent_synthetic_windows": 48,
        "history_length": 24,
        "future_length": 20,
        "labels": "Realized independently generated future cumulative changes; oracle means not used for training",
        "oracle_mean_square_diagnostic": float(np.mean(oracle_y ** 2)),
        "variants": training,
        "formal_market_fits": 0,
        "synthetic_smoke_fits": 3,
        "total_optimizer_steps": 36,
        "selection_or_gain_gate": "none",
    }
    output["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": output["status"],
        "output": str(args.output),
        "parameters": output["checks"]["initialization_and_capacity"]["trainable_parameters"],
        "max_gradient_error": max(item["absolute_error"] for item in differences.values()),
        "known_mean_error": expectation_error,
        "synthetic_smoke_fits": 3,
        "market_fits": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
