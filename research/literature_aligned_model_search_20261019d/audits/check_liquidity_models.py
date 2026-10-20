"""Synthetic implementation checks for the separately recorded volume model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "source"))
from liquidity_models import LiquidityAllocationForecaster


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def number(value):
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def independent_numpy(model, x):
    p = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
    fractions = 1 / (1 + np.exp(-p["raw_rates"]))
    first = 0.98 * fractions[0]
    rho = np.array([first, first + (0.98 - first) * fractions[1]])
    b = p["base_logits"]

    def log_allocation(logits):
        joined = np.concatenate((np.zeros((*logits.shape[:-1], 1)), logits), axis=-1)
        maximum = joined.max(-1, keepdims=True)
        normalization = maximum + np.log(np.exp(joined - maximum).sum(-1, keepdims=True))
        return np.log(0.95) + joined[..., 1:] - normalization

    base_log = log_allocation(b)
    base_w = np.exp(base_log)
    if model.gate is not None:
        def coefficients(name):
            raw = p[name]
            if model.gate.constrained:
                return np.logaddexp(0, raw)
            reference = p["gate.coefficient_reference"]
            return np.logaddexp(0, reference) + (raw - reference)

        gamma = coefficients("gate.gamma_raw")
        delta = coefficients("gate.delta_raw") if "gate.delta_raw" in p else np.zeros(2)
    state = np.zeros((len(x), 2))
    permanent = np.zeros(len(x))
    multiplier = np.ones_like(state)
    reconstruction, states = [], []
    for t in range(x.shape[1]):
        r, u = x[:, t, 0], x[:, t, 1:4]
        exogenous = u @ p["beta"]
        innovation = r - state @ (rho - 1) - exogenous
        if model.gate is None:
            proposed_log = np.broadcast_to(base_log, state.shape)
        else:
            surprise = (x[:, t, 4] * x[:, t, 5])[:, None]
            raw = np.log1p(np.abs(innovation))[:, None] * gamma - surprise * delta
            proposed_log = log_allocation(b + 2 * np.tanh(raw))
        w = np.exp(proposed_log) if model.variant == "full" else base_w
        state = rho * state + w * innovation[:, None]
        permanent += exogenous + (1 - w.sum(-1)) * innovation
        states.append(state.copy())
        reconstruction.append(permanent + state.sum(-1))
        if model.variant == "readout_gate" and t == x.shape[1] - 1:
            multiplier = np.exp(proposed_log - base_log)
    response = rho[None] ** np.arange(1, 21)[:, None] - 1
    return ((state * multiplier) @ response.T,
            np.stack(states, 1), np.stack(reconstruction, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=STUDY / "audits/LIQUIDITY_MODEL_CHECK.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing receipt: {args.output}")
    torch.set_num_threads(1)
    rng = np.random.default_rng(202610192)
    output = {
        "stage": "New development extension after partial v1 results; synthetic implementation checks only",
        "environment": {"python": sys.executable, "python_version": sys.version.split()[0],
                        "numpy": np.__version__, "torch": torch.__version__},
        "source_sha256": {
            name: hashlib.sha256((STUDY / "source" / name).read_bytes()).hexdigest()
            for name in ("innovation_models.py", "liquidity_models.py")
        },
        "checks": {},
    }
    require(output["source_sha256"]["innovation_models.py"] ==
            "cca83f45474c5110f180ae8b7763131f33596a664939c4ebd34c32db24b4dc1e",
            "The inherited v1 model source changed")
    aliases = ["liquidity_full", "liquidity_readout", "shock_size_only",
               "liquidity_unconstrained", "constant_allocation"]
    models = {}
    for name in aliases:
        torch.manual_seed(17)
        models[name] = LiquidityAllocationForecaster(variant=name).double()
    x = torch.tensor(rng.normal(size=(6, 72, 6)), dtype=torch.float64)
    x[..., 5] = torch.tensor(rng.integers(0, 2, size=(6, 72)), dtype=torch.float64)
    initial = {name: model(x[:, :64]).detach() for name, model in models.items()}
    constrained_gap = number((initial["liquidity_full"] - initial["liquidity_unconstrained"]).abs().max())
    require(constrained_gap == 0, "Constrained and unconstrained initial forecasts differ")
    output["checks"]["capacity_and_initialization"] = {
        "parameters": {name: model.parameter_count() for name, model in models.items()},
        "constrained_vs_unconstrained_initial_gap": constrained_gap,
        "full_vs_readout_initial_gap": number((initial["liquidity_full"] - initial["liquidity_readout"]).abs().max()),
        "full_vs_constant_initial_gap": number((initial["liquidity_full"] - initial["constant_allocation"]).abs().max()),
        "output_shape": list(initial["liquidity_full"].shape),
        "note": "Full/readout share initial parameters, not initial forecasts; no unused MLP remains",
    }
    require([model.parameter_count() for model in models.values()] == [11, 11, 9, 11, 7], "Wrong capacity")
    forward_checks = {}
    for name, model in models.items():
        with torch.no_grad():
            model.beta.copy_(torch.tensor([0.15, -0.1, 0.07], dtype=torch.float64))
        prediction, diagnostics = model.forecast_with_diagnostics(x)
        np_pred, np_state, np_reconstruction = independent_numpy(model, x.numpy())
        errors = {
            "forecast": float(np.max(np.abs(prediction.detach().numpy() - np_pred))),
            "state": float(np.max(np.abs(diagnostics["transient_state"].detach().numpy() - np_state))),
            "reconstruction": float(np.max(np.abs(diagnostics["reconstructed_level"].detach().numpy() - np_reconstruction))),
            "observed_price_conservation": number((diagnostics["reconstructed_level"] - x[..., 0].cumsum(1)).abs().max()),
        }
        require(max(errors.values()) < 2e-12, f"Independent recurrence mismatch: {name}")
        missing = x.clone()
        missing[..., 5] = 0
        changed = missing.clone()
        changed[..., 4] += 100
        mask_gap = number((model(missing) - model(changed)).abs().max())
        require(mask_gap == 0, "Missing volume placeholder affects forecasts")
        altered_future = x.clone()
        altered_future[:, 64:, :5] += 10
        altered_future[:, 64:, 5] = 1 - altered_future[:, 64:, 5]
        _, other = model.forecast_with_diagnostics(altered_future)
        prefix_gap = number((diagnostics["transient_state"][:, :64] - other["transient_state"][:, :64]).abs().max())
        require(prefix_gap == 0, "Historical state depends on later features")
        require(torch.equal(model(x[..., 0], x[..., 1:4], x[..., 4:]), prediction), "Packed API differs")
        forward_checks[name] = {"errors": errors, "invalid_volume_placeholder_gap": mask_gap,
                                "future_feature_prefix_gap": prefix_gap}
    output["checks"]["independent_forward_mask_and_prefix"] = forward_checks

    full = models["liquidity_full"]
    gate_features = torch.zeros((3, full.gate.input_dimension), dtype=torch.float64)
    gate_features[:, full.gate.shock_index] = torch.tensor([0.2, 1.5, 1.5], dtype=torch.float64)
    gate_features[:, full.gate.volume_index] = torch.tensor([-2.0, -2.0, 2.0], dtype=torch.float64)
    gate_features[:, full.gate.valid_index] = 1
    totals = full._log_transient_allocation(
        full.base_logits + 2 * torch.tanh(full.gate(gate_features))
    ).exp().sum(-1)
    require(totals[1] > totals[0] and totals[1] > totals[2], "Economic monotonicity hypothesis encoded incorrectly")
    free = models["liquidity_unconstrained"]
    with torch.no_grad():
        free.gate.gamma_raw.copy_(free.gate.coefficient_reference - 1)
        free.gate.delta_raw.copy_(free.gate.coefficient_reference - 1)
    gamma, delta = free.gate.coefficients()
    require(bool((gamma < 0).all() & (delta < 0).all()), "Unconstrained coefficients cannot change sign")
    require(bool(torch.isfinite(free(x)).all()), "Free signs cause nonfinite forecasts")
    shock = models["shock_size_only"]
    changed_volume = x.clone()
    changed_volume[..., 4] += 50
    changed_volume[..., 5] = 1 - changed_volume[..., 5]
    shock_volume_gap = number((shock(x) - shock(changed_volume)).abs().max())
    require(shock_volume_gap == 0, "Shock-only model still depends on volume")
    output["checks"]["specified_sign_hypotheses"] = {
        "total_transitory_mass_small_shock_low_volume": number(totals[0]),
        "total_transitory_mass_large_shock_low_volume": number(totals[1]),
        "total_transitory_mass_large_shock_high_volume": number(totals[2]),
        "unconstrained_negative_gamma_example": gamma.detach().tolist(),
        "unconstrained_negative_delta_example": delta.detach().tolist(),
        "shock_only_volume_gap": shock_volume_gap,
        "interpretation": "Tests the imposed hypothesis, not its empirical truth or individual-mode monotonicity",
    }

    gx = x[:3, :8]
    gy = torch.tensor(rng.normal(size=(3, 20)), dtype=torch.float64)
    derivatives = {}
    for model_name in ("liquidity_full", "liquidity_unconstrained"):
        model = models[model_name]
        model.zero_grad(set_to_none=True)
        (model(gx) - gy).square().mean().backward()
        for name, parameter in model.named_parameters():
            direction = torch.tensor(rng.normal(size=tuple(parameter.shape)), dtype=torch.float64)
            direction /= direction.norm()
            automatic = number((parameter.grad * direction).sum())
            original = parameter.detach().clone()
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    parameter.copy_(original + sign * 1e-6 * direction)
                    values.append(number((model(gx) - gy).square().mean()))
            with torch.no_grad():
                parameter.copy_(original)
            numerical = (values[0] - values[1]) / 2e-6
            require(np.isclose(automatic, numerical, rtol=5e-5, atol=2e-8), f"Gradient mismatch {model_name}:{name}")
            derivatives[f"{model_name}:{name}"] = {
                "autograd": automatic, "finite_difference": numerical,
                "absolute_error": abs(automatic - numerical),
            }
    output["checks"]["directional_gradients"] = derivatives

    tx = torch.tensor(rng.normal(size=(24, 16, 6)), dtype=torch.float64)
    tx[..., 5] = torch.tensor(rng.integers(0, 2, size=(24, 16)), dtype=torch.float64)
    ty = torch.tensor(np.cumsum(rng.normal(size=(24, 20)), axis=1), dtype=torch.float64)
    training = {}
    for name in aliases[:4]:
        torch.manual_seed(17)
        model = LiquidityAllocationForecaster(variant=name).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = {key: p.detach().clone() for key, p in model.named_parameters()}
        gradient_norms = {key: 0.0 for key in before}
        first_loss = number((model(tx) - ty).square().mean())
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(tx) - ty).square().mean()
            require(bool(torch.isfinite(loss)), "Nonfinite synthetic optimization")
            loss.backward()
            for key, p in model.named_parameters():
                require(p.grad is not None, f"No gradient: {name}:{key}")
                gradient_norms[key] = max(gradient_norms[key], number(p.grad.norm()))
            optimizer.step()
        changes = {key: number((p - before[key]).abs().max()) for key, p in model.named_parameters()}
        require(all(value > 0 for value in gradient_norms.values()), "Unused parameter group")
        require(all(value > 0 for value in changes.values()), "Unchanged parameter group")
        training[name] = {"optimizer_steps": 10, "initial_mse": first_loss,
                          "final_mse": number((model(tx) - ty).square().mean()),
                          "maximum_gradient_norm": gradient_norms, "maximum_parameter_change": changes}
    output["checks"]["gradient_training_smoke"] = {
        "synthetic_smoke_fits": 4, "total_optimizer_steps": 40,
        "formal_market_fits": 0, "gain_gate": "none",
        "labels": "Fixed independent random future cumulative changes; this is not a performance experiment",
        "variants": training,
    }
    output["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      "parameters": output["checks"]["capacity_and_initialization"]["parameters"],
                      "max_gradient_error": max(v["absolute_error"] for v in derivatives.values()),
                      "market_fits": 0, "synthetic_smoke_fits": 4}, ensure_ascii=False))


if __name__ == "__main__":
    main()
