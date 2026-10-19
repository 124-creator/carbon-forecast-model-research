"""Post-v1-development compact volume-conditioned innovation allocation.

The price recurrence is imported unchanged from innovation_models.py.  This
file specifies a different, explicitly recorded model candidate, not a repair
or silent continuation of v1.  Volume surprise and its validity mask must be
constructed causally by the experiment runner.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from innovation_models import ShockAllocationForecaster


class VolumeInnovationGate(nn.Module):
    """Return raw gate logits; the inherited recurrence applies 2*tanh once.

    gamma * log1p(abs(e)) - delta * volume_surprise * valid

    Coefficients are positive for constrained models.  The unconstrained
    parameterization is an affine extension sharing exactly the constrained
    model's initial effective coefficients, for the same tensor dtype.
    """

    def __init__(
        self, n_exog: int = 3, modes: int = 2, use_volume: bool = True,
        constrained: bool = True, initial_coefficient: float = 0.1,
    ) -> None:
        super().__init__()
        if initial_coefficient <= 0 or not math.isfinite(initial_coefficient):
            raise ValueError("initial_coefficient must be finite and positive")
        self.n_exog = n_exog
        self.modes = modes
        self.use_volume = use_volume
        self.constrained = constrained
        self.volume_index = 1 + modes + n_exog
        self.valid_index = self.volume_index + 1
        self.shock_index = self.volume_index + 2
        self.input_dimension = self.shock_index + 1
        raw_initial = math.log(math.expm1(initial_coefficient))
        self.gamma_raw = nn.Parameter(torch.full((modes,), raw_initial))
        if use_volume:
            self.delta_raw = nn.Parameter(torch.full((modes,), raw_initial))
        else:
            self.register_parameter("delta_raw", None)
        self.register_buffer("coefficient_reference", torch.full((modes,), raw_initial))

    def coefficients(self) -> Tuple[Tensor, Tensor]:
        def transform(raw):
            if self.constrained:
                return F.softplus(raw)
            reference = self.coefficient_reference
            return F.softplus(reference) + (raw - reference)

        gamma = transform(self.gamma_raw)
        delta = torch.zeros_like(gamma) if self.delta_raw is None else transform(self.delta_raw)
        return gamma, delta

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dimension:
            raise ValueError("Unexpected inherited gate feature layout")
        gamma, delta = self.coefficients()
        magnitude = features[:, self.shock_index:self.shock_index + 1]
        valid_volume = (
            features[:, self.volume_index:self.volume_index + 1]
            * features[:, self.valid_index:self.valid_index + 1]
        )
        return magnitude * gamma - valid_volume * delta


class LiquidityAllocationForecaster(ShockAllocationForecaster):
    """Compact volume model with the unchanged v1 state and forecast equations.

    Default packed input x:(B,L,6) has columns:
        [target_change, energy_1, energy_2, energy_3, volume_surprise, valid].
    Separate input API:
        forward(r:(B,L), u:(B,L,3), context:(B,L,2)).
    Output:(B,20) cumulative changes in the normalized target units.

    Variants full/readout_gate use constrained 11-parameter models;
    shock_size_only has 9 parameters; unconstrained has 11 parameters and
    permits either economic sign; constant_allocation has the same 7 learned
    scalar parameters as the original constant model.
    """

    LIQUIDITY_VARIANTS = (
        "full", "readout_gate", "shock_size_only", "unconstrained", "constant_allocation",
    )

    def __init__(
        self, n_exog: int = 3, horizon: int = 20, variant: str = "full",
        initial_coefficient: float = 0.1,
    ) -> None:
        variant = {
            "liquidity_full": "full",
            "liquidity_readout": "readout_gate",
            "liquidity_unconstrained": "unconstrained",
        }.get(variant, variant)
        if variant not in self.LIQUIDITY_VARIANTS:
            raise ValueError(f"Unknown liquidity variant: {variant}")
        base_variant = "readout_gate" if variant == "readout_gate" else (
            "constant_allocation" if variant == "constant_allocation" else "full"
        )
        super().__init__(
            n_exog=n_exog, horizon=horizon, modes=2, hidden=16,
            variant=base_variant, n_context=2, innovation_feature="magnitude",
        )
        self.liquidity_variant = variant
        self.initial_coefficient = float(initial_coefficient)
        if variant != "constant_allocation":
            # The temporary MLP allocated by the base constructor is replaced,
            # so none of its unused parameters remain in this module.
            self.gate = VolumeInnovationGate(
                n_exog=n_exog, modes=2,
                use_volume=(variant != "shock_size_only"),
                constrained=(variant != "unconstrained"),
                initial_coefficient=initial_coefficient,
            )

    def _unpack(
        self, target_changes: Tensor, exog_changes: Optional[Tensor],
        context: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r, u, volume = super()._unpack(target_changes, exog_changes, context)
        valid = volume[..., 1]
        if not bool(((valid == 0.0) | (valid == 1.0)).all()):
            raise ValueError("Volume validity mask must contain only zero or one")
        return r, u, volume

