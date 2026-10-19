"""Compact conditional signed responses, stage e (no imports from frozen stage d).

This is an origin-conditioned multihorizon regression, not a generative SSM.
The two historical exponential filters are standard linear features. Context
changes either future response duration or amplitude. Signed linear readouts
are solved exactly by training-only weighted ridge (variable projection).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

DTYPE = torch.float64
HORIZONS = (1, 5, 20)
LOOKBACK = 64
RATE_CAP = .98
FAMILIES = ("fixed_response", "static_response", "amplitude_context",
            "signed_context_response", "free_horizon_context", "signed_shock_only")


def ordered_rates(raw):
    first = RATE_CAP * torch.sigmoid(raw[..., 0])
    second = first + (RATE_CAP-first) * torch.sigmoid(raw[..., 1])
    return torch.stack((first, second), dim=-1)


def initial_raw_rates():
    return torch.tensor([math.log(.5/(RATE_CAP-.5)),
                         math.log((.9-.5)/(RATE_CAP-.9))], dtype=DTYPE)


class SignedResponse(nn.Module):
    """Input (B,64,3): normalized arithmetic change, volume surprise, valid.

    ``basis(x)`` returns (B,3,P), in a shared global coefficient coordinate.
    ``forward(x)`` requires coefficients previously fitted to training labels.
    All six arms start with rates (.5,.9), zero gates and zero coefficients.
    Fitted epoch0 is a trained linear model, not this zero-coefficient state.
    """
    def __init__(self, family="signed_context_response", horizons=HORIZONS):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"Unknown family {family}")
        self.family = family
        self.horizons = tuple(int(h) for h in horizons)
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("Positive horizons required")
        raw = initial_raw_rates()
        if family == "fixed_response":
            self.register_buffer("rate_raw", raw)
        else:
            self.rate_raw = nn.Parameter(raw)
        if family in ("amplitude_context", "signed_context_response", "free_horizon_context", "signed_shock_only"):
            self.gate = nn.Parameter(torch.zeros((2, 1 if family == "signed_shock_only" else 2), dtype=DTYPE))
        else:
            self.register_buffer("gate", torch.zeros((2, 2), dtype=DTYPE))
        p = 3*len(self.horizons) if self.family == "free_horizon_context" else 3
        self.register_buffer("coefficients", torch.zeros(p, dtype=DTYPE))
        self.register_buffer("head_column_scale", torch.ones(p, dtype=DTYPE))

    def parameter_counts(self):
        nonlinear = sum(p.numel() for p in self.parameters())
        linear = self.coefficients.numel()
        return {"nonlinear_optimizer_parameters": nonlinear,
                "exactly_fitted_linear_coefficients": linear,
                "total_fitted_scalars": nonlinear+linear}

    def states_and_context(self, x):
        if x.ndim != 3 or x.shape[1:] != (LOOKBACK, 3):
            raise ValueError(f"Expected B x {LOOKBACK} x 3, got {tuple(x.shape)}")
        rho = ordered_rates(self.rate_raw)
        lags = torch.arange(LOOKBACK-1, -1, -1, dtype=x.dtype, device=x.device)
        weights = (1-rho)[None] * rho[None]**lags[:, None]
        states = x[..., 0] @ weights
        z = torch.stack((torch.log1p(torch.abs(x[:, -1, 0])),
                         x[:, -1, 1]*x[:, -1, 2]), dim=-1)
        return states, z, rho

    def components(self, x, horizons=None):
        hs = self.horizons if horizons is None else tuple(int(h) for h in horizons)
        states, z, rho = self.states_and_context(x)
        gate_input = z[:, :1] if self.family == "signed_shock_only" else z
        q = 2*torch.tanh(gate_input @ self.gate.T)
        if self.family in ("signed_context_response", "free_horizon_context", "signed_shock_only"):
            future_rates = ordered_rates(self.rate_raw[None]+q)
        else:
            future_rates = rho[None].expand(len(x), -1)
        amplitude = torch.exp(q) if self.family == "amplitude_context" else torch.ones_like(q)
        # A finite sum is stable even when the learned rate is close to one.
        steps = torch.arange(1, max(hs)+1, dtype=x.dtype, device=x.device)
        cumulative = torch.cumsum(future_rates[:, None, :]**steps[None, :, None], dim=1)
        index = torch.as_tensor(hs, device=x.device)-1
        response = cumulative[:, index, :] * (states*amplitude)[:, None, :]
        h = torch.as_tensor(hs, dtype=x.dtype, device=x.device)
        base = torch.cat((h[None, :, None].expand(len(x), -1, -1), response), dim=-1)
        return base, {"states": states, "context": z, "historical_rates": rho,
                      "future_rates": future_rates, "amplitude": amplitude, "gate_shift": q}

    def basis(self, x):
        base, _ = self.components(x)
        if self.family != "free_horizon_context":
            return base
        eye = torch.eye(len(self.horizons), dtype=x.dtype, device=x.device)
        return (base[:, :, None, :] * eye[None, :, :, None]).reshape(len(x), len(self.horizons), -1)

    def forward(self, x):
        return self.basis(x) @ self.coefficients

    def path(self, x, horizon=20):
        if self.family == "free_horizon_context":
            raise ValueError("Free heads are fitted only at the declared horizons")
        return self.components(x, tuple(range(1, horizon+1)))[0] @ self.coefficients

    def set_coefficients(self, value, column_scale=None):
        if value.shape != self.coefficients.shape:
            raise ValueError("Linear coefficient shape mismatch")
        with torch.no_grad():
            self.coefficients.copy_(value.detach())
            if column_scale is not None:
                self.head_column_scale.copy_(column_scale.detach())


def solve_readout(basis, y, variance, alpha):
    """Minimize mean((Phi beta-y)^2 / variance) + alpha * ||s beta||^2.

    Mean is over both origins and trained horizons. All three types of
    coefficients, including shared drift, have the same stated penalty after
    training-only weighted column RMS scaling s. Scaling remains in the graph.
    No validation labels, adaptive jitter, pseudoinverse cutoff or clipping.
    """
    if alpha <= 0:
        raise ValueError("A strictly positive ridge penalty is required")
    if basis.shape[:2] != y.shape or variance.shape != y.shape[1:]:
        raise ValueError("Incompatible ridge inputs")
    weight = torch.rsqrt(variance)
    a = (basis*weight[None, :, None]).reshape(-1, basis.shape[-1])
    b = (y*weight[None]).reshape(-1)
    column_scale = torch.sqrt(torch.mean(a*a, dim=0)).clamp_min(1e-8)
    a = a / column_scale
    n = a.shape[0]
    gram = a.T @ a / n
    rhs = a.T @ b / n
    standardized = torch.linalg.solve(gram + alpha*torch.eye(a.shape[1], dtype=a.dtype, device=a.device), rhs)
    coefficients = standardized / column_scale
    predicted = basis @ coefficients
    data_loss = torch.mean((predicted-y)**2/variance)
    objective = data_loss + alpha*torch.sum(standardized**2)
    residual = (gram+alpha*torch.eye(a.shape[1], dtype=a.dtype, device=a.device))@standardized-rhs
    return coefficients, objective, data_loss, residual, column_scale


@dataclass(frozen=True)
class FitConfig:
    learning_rate: float = .01
    alpha: float = .001
    maximum_epochs: int = 80
    minimum_epochs: int = 15
    patience: int = 12
    selection_tolerance: float = 1e-7
    gradient_clip: float = 5.


def fit_response(family, x_train, y_train, variance, config, *,
                 x_val=None, y_val=None, fixed_epochs=None):
    """Deterministic full-batch VP; validation can select an epoch, never beta.

    A refit is independently initialized on all mature pretest labels and runs
    exactly the chosen number of updates. Epoch0 still solves its linear beta.
    No random seed repeats are reported for this deterministic algorithm.
    """
    if (x_val is None) != (y_val is None):
        raise ValueError("Provide both validation inputs and labels")
    model = SignedResponse(family)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate) if params else None
    max_epochs = config.maximum_epochs if fixed_epochs is None else int(fixed_epochs)
    if not params:
        max_epochs = 0
    best_value, best_epoch, best_state = float("inf"), 0, None
    history, stale, positive_steps = [], 0, 0
    initial_parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    last_parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    for epoch in range(max_epochs+1):
        coefficients, objective, training_loss, residual, column_scale = solve_readout(
            model.basis(x_train), y_train, variance, config.alpha)
        model.set_coefficients(coefficients, column_scale)
        with torch.no_grad():
            selection = (torch.mean((model(x_val)-y_val)**2/variance)
                         if x_val is not None else training_loss)
        value = float(selection.detach())
        if not np.isfinite(value) or not bool(torch.isfinite(objective)):
            raise FloatingPointError("Nonfinite profiled objective or selection loss")
        record = {"epoch": epoch, "training_data_loss": float(training_loss.detach()),
                  "training_profiled_objective": float(objective.detach()),
                  "validation_loss": float(selection.detach()) if x_val is not None else None,
                  "linear_normal_equation_max_residual": float(residual.detach().abs().max()),
                  "historical_rates": ordered_rates(model.rate_raw).detach().tolist()}
        history.append(record)
        if value < best_value-config.selection_tolerance:
            best_value, best_epoch = value, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == max_epochs:
            break
        if fixed_epochs is None and x_val is not None and epoch >= config.minimum_epochs and stale >= config.patience:
            break
        if optimizer is None:
            break
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradients = [p.grad for p in params if p.grad is not None]
        if len(gradients) != len(params) or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError("Missing or nonfinite nonlinear gradient")
        norm = float(torch.sqrt(sum(torch.sum(g*g) for g in gradients)))
        if norm > 0:
            positive_steps += 1
        record["gradient_norm_before_clip"] = norm
        torch.nn.utils.clip_grad_norm_(params, config.gradient_clip)
        optimizer.step()
        last_parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    # Fixed-epoch final refits retain the last state, regardless of training
    # nonmonotonicity. Selection fits restore the validation-chosen state.
    if fixed_epochs is None:
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
    else:
        selected_epoch = history[-1]["epoch"]
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_gap = {k: float(torch.max(torch.abs(v-initial_parameters[k])).detach())
                   for k, v in model.named_parameters()}
    attempted_gap = {k: float(torch.max(torch.abs(last_parameters[k]-initial_parameters[k])))
                     for k in initial_parameters}
    receipt = {"family": family, "config": asdict(config), "deterministic": True,
               "parameter_counts": model.parameter_counts(),
               "selected_epoch": selected_epoch, "epochs_executed": history[-1]["epoch"],
               "positive_gradient_steps": positive_steps,
               "validation_loss": best_value if x_val is not None else None,
               "selected_nonlinear_parameter_max_change": initial_gap,
               "attempted_last_nonlinear_parameter_max_change": attempted_gap,
               "linear_readout_fitted_at_epoch0": True,
               "state_dict": {k: v.tolist() for k, v in state.items()}, "history": history}
    return model, receipt
