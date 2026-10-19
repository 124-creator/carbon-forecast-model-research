"""Small, differentiable price-innovation allocation models.

This module does not read data, fit scalers, select epochs, or score experiments.
All inputs must already be available at the forecast origin and expressed in
the units fixed by the experiment runner.  Output h-1 is the cumulative change
over the next h observed target records, in the target input's units.

The full model is a restricted innovations state-space predictor, not a new
state-space framework or an assertion that economic permanent shocks are
observed.  See methods/SHOCK_ALLOCATION_DESIGN_ZH.md for its mean assumptions.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class ShockAllocationForecaster(nn.Module):
    """Learn where each observed innovation enters a permanent/transient split.

    Variants:
        full: context changes the allocation at every historical observation.
        readout_gate: a constant allocation builds the states; the same neural
            gate changes only their final forecast amplitudes.
        constant_allocation: only the constant allocation and common scalar
            parameters are learned.  This nested model has fewer parameters.

    Inputs can be either ``forward(r, u, context)`` or ``forward(x)``.  The
    former uses r:(B,L), u:(B,L,n_exog), context:(B,L,n_context).  Packed x has
    columns [target_change, exogenous_changes..., additional_context...].
    There is no assumption that an exogenous change represents a causal shock.

    By default the gate sees log1p(abs(current innovation)), but never its
    sign.  ``innovation_feature='none'`` removes this feature by a design-time
    choice; it is not an additional automatically selected configuration.
    """

    VARIANTS = ("full", "readout_gate", "constant_allocation")

    def __init__(
        self,
        n_exog: int = 3,
        horizon: int = 20,
        hidden: int = 16,
        modes: int = 2,
        variant: str = "full",
        n_context: int = 0,
        innovation_feature: str = "magnitude",
        rho_max: float = 0.98,
        permanent_floor: float = 0.05,
        gate_logit_bound: float = 2.0,
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        if innovation_feature not in ("magnitude", "none"):
            raise ValueError("innovation_feature must be 'magnitude' or 'none'")
        if modes != 2:
            raise ValueError("This fixed candidate specifies exactly two modes")
        if min(n_exog, n_context) < 0 or min(horizon, hidden) < 1:
            raise ValueError("Invalid dimension")
        if not 0.9 < rho_max < 1.0:
            raise ValueError("rho_max must exceed the initial slow rate 0.9")
        if not 0.0 < permanent_floor < 0.8:
            raise ValueError("Invalid permanent allocation floor")
        if not math.isfinite(gate_logit_bound) or gate_logit_bound <= 0:
            raise ValueError("gate_logit_bound must be finite and positive")
        self.n_exog = n_exog
        self.n_context = n_context
        self.horizon = horizon
        self.hidden = hidden
        self.modes = modes
        self.variant = variant
        self.innovation_feature = innovation_feature
        self.rho_max = float(rho_max)
        self.permanent_floor = float(permanent_floor)
        self.gate_logit_bound = float(gate_logit_bound)

        # Global rates start at .5 and .9, and remain strictly ordered.
        first_fraction = 0.5 / rho_max
        second_fraction = (0.9 - 0.5) / (rho_max - 0.5)
        self.raw_rates = nn.Parameter(torch.tensor([
            math.log(first_fraction / (1.0 - first_fraction)),
            math.log(second_fraction / (1.0 - second_fraction)),
        ]))
        self.beta = nn.Parameter(torch.zeros(n_exog))
        # Each transient allocation starts at .05, so .90 is permanent.
        transient_each = 0.05
        softmax_permanent_mass = 1.0 - permanent_floor - modes * transient_each
        initial_logit = math.log(transient_each / softmax_permanent_mass)
        self.base_logits = nn.Parameter(torch.full((modes,), initial_logit))
        self.gate_input_dim = (
            1 + modes + n_exog + n_context
            + int(innovation_feature == "magnitude")
        )
        if variant == "constant_allocation":
            self.gate = None
        else:
            self.gate = nn.Sequential(
                nn.Linear(self.gate_input_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, modes),
            )
            # A paired seed gives full/readout the same initial model.  The
            # first gradient updates the final gate layer; earlier layers
            # receive gradients once that layer is nonzero.
            nn.init.zeros_(self.gate[2].weight)
            nn.init.zeros_(self.gate[2].bias)

    def rates(self) -> Tensor:
        fractions = torch.sigmoid(self.raw_rates)
        first = self.rho_max * fractions[0]
        second = first + (self.rho_max - first) * fractions[1]
        return torch.stack((first, second))

    def _log_transient_allocation(self, logits: Tensor) -> Tensor:
        # Fix the permanent category's logit at zero to remove the softmax
        # translation redundancy.  The permanent floor is part of the model,
        # not an after-the-fact numerical clipping operation.
        permanent_logit = logits.new_zeros((*logits.shape[:-1], 1))
        probabilities = torch.log_softmax(
            torch.cat((permanent_logit, logits), dim=-1), dim=-1
        )
        return math.log(1.0 - self.permanent_floor) + probabilities[..., 1:]

    def _unpack(
        self, target_changes: Tensor, exog_changes: Optional[Tensor],
        context: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if exog_changes is None and target_changes.ndim == 3:
            if context is not None:
                raise ValueError("Packed input already includes additional context")
            expected = 1 + self.n_exog + self.n_context
            if target_changes.shape[-1] != expected:
                raise ValueError(f"Packed input requires {expected} channels")
            x = target_changes
            target_changes = x[..., 0]
            exog_changes = x[..., 1:1 + self.n_exog]
            context = x[..., 1 + self.n_exog:]
        if target_changes.ndim != 2 or target_changes.shape[1] < 1:
            raise ValueError("Target changes must be a nonempty (batch, history) tensor")
        batch, length = target_changes.shape
        if exog_changes is None:
            if self.n_exog:
                raise ValueError("Exogenous input is required")
            exog_changes = target_changes.new_empty((batch, length, 0))
        if context is None:
            if self.n_context:
                raise ValueError("Additional historical context is required")
            context = target_changes.new_empty((batch, length, 0))
        for value, dimension, name in (
            (exog_changes, self.n_exog, "exogenous"),
            (context, self.n_context, "context"),
        ):
            if value.shape != (batch, length, dimension):
                raise ValueError(f"Unexpected {name} input shape")
            if value.dtype != target_changes.dtype or value.device != target_changes.device:
                raise ValueError("All inputs must share dtype and device")
        if not target_changes.is_floating_point():
            raise TypeError("Model inputs must be floating point")
        if target_changes.dtype != self.beta.dtype or target_changes.device != self.beta.device:
            raise ValueError("Input dtype/device must match the model")
        if any(not bool(torch.isfinite(value).all()) for value in (
            target_changes, exog_changes, context,
        )):
            raise ValueError("Non-finite model input")
        return target_changes, exog_changes, context

    def _compute(
        self, target_changes: Tensor, exog_changes: Optional[Tensor],
        context: Optional[Tensor], collect: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        r, u, context_values = self._unpack(target_changes, exog_changes, context)
        batch, length = r.shape
        rho = self.rates()
        base_log_w = self._log_transient_allocation(self.base_logits)
        base_w = base_log_w.exp()
        state = r.new_zeros((batch, self.modes))
        previous_r = r.new_zeros((batch, 1))
        permanent_level = r.new_zeros(batch)
        readout_multiplier = torch.ones_like(state)
        records: Dict[str, list] = {
            key: [] for key in (
                "innovation", "transient_state", "allocation", "proposed_allocation",
                "permanent_increment", "reconstructed_level", "gate_adjustment",
            )
        }
        for t in range(length):
            known_exog_effect = (u[:, t] * self.beta).sum(-1)
            previous_expected_change = (state * (rho - 1.0)).sum(-1)
            innovation = r[:, t] - previous_expected_change - known_exog_effect
            if self.gate is None:
                adjustment = torch.zeros_like(state)
                proposed_log_w = base_log_w.expand(batch, -1)
            else:
                gate_inputs = [previous_r, state, u[:, t], context_values[:, t]]
                if self.innovation_feature == "magnitude":
                    gate_inputs.append(torch.log1p(innovation.abs()).unsqueeze(-1))
                raw_adjustment = self.gate(torch.cat(gate_inputs, dim=-1))
                adjustment = self.gate_logit_bound * torch.tanh(raw_adjustment)
                proposed_log_w = self._log_transient_allocation(
                    self.base_logits + adjustment
                )
            proposed_w = proposed_log_w.exp()
            w = proposed_w if self.variant == "full" else base_w.expand(batch, -1)
            state = rho * state + w * innovation.unsqueeze(-1)
            permanent_increment = known_exog_effect + (1.0 - w.sum(-1)) * innovation
            permanent_level = permanent_level + permanent_increment
            if self.variant == "readout_gate" and t == length - 1:
                # Log-ratio remains stable for very small base allocations.
                # With bounded adjustments it is bounded independently of
                # base logits; there is no division by a small probability.
                readout_multiplier = torch.exp(proposed_log_w - base_log_w)
            if collect:
                records["innovation"].append(innovation)
                records["transient_state"].append(state)
                records["allocation"].append(
                    torch.cat(((1.0 - w.sum(-1, keepdim=True)), w), dim=-1)
                )
                records["proposed_allocation"].append(
                    torch.cat(((1.0 - proposed_w.sum(-1, keepdim=True)), proposed_w), dim=-1)
                )
                records["permanent_increment"].append(permanent_increment)
                records["reconstructed_level"].append(permanent_level + state.sum(-1))
                records["gate_adjustment"].append(adjustment)
            previous_r = r[:, t:t + 1]
        horizons = torch.arange(1, self.horizon + 1, dtype=r.dtype, device=r.device)
        response = rho.unsqueeze(0).pow(horizons.unsqueeze(-1)) - 1.0
        forecast = (state * readout_multiplier) @ response.T
        if not bool(torch.isfinite(forecast).all()):
            raise FloatingPointError("Non-finite innovation model forecast")
        diagnostics: Dict[str, Tensor] = {}
        if collect:
            diagnostics = {key: torch.stack(value, dim=1) for key, value in records.items()}
            diagnostics.update({
                "rates": rho,
                "beta": self.beta,
                "base_allocation": torch.cat(((1.0 - base_w.sum()).unsqueeze(0), base_w)),
                "readout_multiplier": readout_multiplier,
            })
        return forecast, diagnostics

    def forward(
        self, target_changes: Tensor, exog_changes: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """Return (batch, horizon) cumulative forecasts, without future inputs."""
        return self._compute(target_changes, exog_changes, context, False)[0]

    def forecast_with_diagnostics(
        self, target_changes: Tensor, exog_changes: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return forecasts and differentiable historical-state diagnostics."""
        return self._compute(target_changes, exog_changes, context, True)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

