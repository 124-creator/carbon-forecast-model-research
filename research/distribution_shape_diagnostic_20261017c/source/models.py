"""Six diagnostic distributions, all with exactly zero conditional mean.

No family in this module is claimed as a novel algorithm. The main conditional
families share GRU32. The two DeepNPTS arms execute an unmodified author MLP and
probability head in an explicitly different direct-horizon task adapter.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn


FAMILIES = (
    "global_shape", "conditional_shape", "scale_only", "shape_scale",
    "deepnpts_rps", "deepnpts_crps",
)
AUTHOR_COMMIT = "d60556ca71167a3ec1ea807b491dd982d1de2112"
AUTHOR_PROBABILITY_FLOOR_MIXTURE = 0.03


@lru_cache(maxsize=1)
def author_core():
    path = Path(__file__).resolve().parents[1] / "vendor" / "deep_npts" / "runtime.py"
    spec = importlib.util.spec_from_file_location("_shape_deepnpts_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_author()


def exact_crps(atoms: torch.Tensor, probs: torch.Tensor,
               target: torch.Tensor) -> torch.Tensor:
    """Exact finite-distribution CRPS, not a quantile-grid approximation."""
    first = (probs * (atoms - target.unsqueeze(-1)).abs()).sum(dim=-1)
    pairwise = (atoms.unsqueeze(-1) - atoms.unsqueeze(-2)).abs()
    second = 0.5 * (probs.unsqueeze(-1) * probs.unsqueeze(-2) * pairwise).sum(dim=(-1, -2))
    return first - second


def author_rps(atoms: torch.Tensor, probs: torch.Tensor,
               target: torch.Tensor) -> torch.Tensor:
    """Execute the fixed author's actual score on the issued distribution.

Upstream calls this RPS and exposes its negative as log_prob; we do not label
it log likelihood or exact CRPS. No substitute formula is used here.
"""
    shape, k = atoms.shape[:-1], atoms.shape[-1]
    distribution = author_core().distribution.DiscreteDistribution(
        values=atoms.reshape(-1, k), probs=probs.reshape(-1, k), validate_args=False,
    )
    return distribution.rps(target.reshape(-1, 1)).reshape(shape)


def standardize_measure(atoms: torch.Tensor,
                        probs: torch.Tensor) -> torch.Tensor:
    """Probability-weighted center and variance; gradients remain connected."""
    center = (atoms * probs).sum(dim=-1, keepdim=True)
    centered = atoms - center
    variance = (centered.square() * probs).sum(dim=-1, keepdim=True)
    if not bool(torch.isfinite(variance).all()) or not bool((variance > 0).all()):
        raise ValueError("Cannot standardize a degenerate/nonfinite distribution")
    return centered / variance.sqrt()


class Model(nn.Module):
    def __init__(self, family: str, input_channels: int = 3,
                 H: int = 20, K: int = 31):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError("Unknown diagnostic family: " + family)
        if input_channels < 1 or H < 1 or K < 2:
            raise ValueError("Invalid input/H/K dimensions")
        self.family, self.input_channels, self.H, self.K = family, input_channels, H, K
        self.context_length = 20
        self.is_author = family.startswith("deepnpts_")

        if self.is_author:
            # This is not the author's recursive forecasting task. The source
            # support is a fixed training-prefix FHS distribution for each h.
            # The original past_time_feat slots carry the current observed x,
            # flattened and zero-padded. No future times or targets are supplied.
            self.author_feature_channels = math.ceil(self.context_length * input_channels / K)
            self.author_net = author_core().network.DeepNPTSNetworkDiscrete(
                context_length=K, num_hidden_nodes=[32, 32], cardinality=[1],
                embedding_dimension=[1], num_time_features=self.author_feature_channels,
                batch_norm=False, input_scaling=None, dropout_rate=0.0, use_softmax=False,
            )
            # Upstream keeps embeddings in a plain list. With the one-category
            # task adapter this is a constant feature. Register the same object
            # on this wrapper for dtype/device/state completeness, but freeze it
            # to retain the upstream optimizer's non-updating behavior. Do not
            # reinitialize or alter its numerical value.
            self.author_dummy_embedding = self.author_net.embedder.embedders[0]
            self.author_dummy_embedding.requires_grad_(False)
        elif family == "global_shape":
            self.global_delta = nn.Parameter(torch.zeros(H, K-1))
        else:
            self.encoder = nn.GRU(input_channels, 32, batch_first=True)
            if family in ("conditional_shape", "shape_scale"):
                self.shape_head = nn.Linear(32, H * (K-1))
                nn.init.zeros_(self.shape_head.weight)
                nn.init.zeros_(self.shape_head.bias)
            if family in ("scale_only", "shape_scale"):
                self.scale_head = nn.Linear(32, H)
                nn.init.zeros_(self.scale_head.weight)
                nn.init.zeros_(self.scale_head.bias)

    def _check(self, x, base_atoms):
        if x.ndim != 3 or x.shape[1:] != (self.context_length, self.input_channels):
            raise ValueError("x must have shape [B,20,input_channels]")
        if base_atoms.shape != (x.shape[0], self.H, self.K):
            raise ValueError("base_atoms must have shape [B,H,K]")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(base_atoms).all()):
            raise ValueError("Nonfinite model inputs")
        if not bool((torch.diff(base_atoms, dim=-1) > 0).all()):
            raise ValueError("FHS atoms must have strictly positive frozen gaps")
        if x.dtype != base_atoms.dtype or x.device != base_atoms.device:
            raise ValueError("Input dtype/device mismatch")

    def forward(self, x: torch.Tensor, base_atoms: torch.Tensor):
        self._check(x, base_atoms)
        b = x.shape[0]
        if self.is_author:
            flat = x.reshape(b, -1)
            width = self.K * self.author_feature_channels
            padding = flat.new_zeros(b, width-flat.shape[-1])
            feature = torch.cat([flat, padding], dim=-1)
            feature = feature[:, None, :].expand(b, self.H, width)
            feature = feature.reshape(b*self.H, self.K, self.author_feature_channels)
            horizon = torch.arange(1, self.H+1, device=x.device, dtype=x.dtype) / self.H
            horizon = horizon[None, :, None].expand(b, self.H, 1).reshape(b*self.H, 1)
            distribution = self.author_net(
                feat_static_cat=torch.zeros(b*self.H, 1, device=x.device, dtype=torch.int64),
                feat_static_real=horizon,
                past_target=base_atoms.reshape(b*self.H, self.K),
                past_observed_values=base_atoms.new_ones(b*self.H, self.K),
                past_time_feat=feature,
            )
            weights = distribution.probs.reshape(b, self.H, self.K)
            # Explicit task adaptation shared by BOTH author arms. A fixed
            # positive floor prevents a tiny atom weight hiding enormous
            # standardized support after probability-weighted normalization.
            floor = AUTHOR_PROBABILITY_FLOOR_MIXTURE
            probs = floor/self.K + (1-floor)*weights
            atoms = standardize_measure(base_atoms, probs)
            return atoms, probs

        probs = torch.full_like(base_atoms, 1.0/self.K)
        hidden = None
        if self.family != "global_shape":
            _, state = self.encoder(x)
            hidden = state[-1]
        if self.family == "global_shape":
            delta = self.global_delta[None].expand(b, -1, -1)
        elif self.family in ("conditional_shape", "shape_scale"):
            delta = self.shape_head(hidden).reshape(b, self.H, self.K-1)
        else:
            delta = None

        if delta is None:
            atoms = standardize_measure(base_atoms, probs)
        else:
            # Every main family starts at exactly the same FHS measure up to
            # floating-point recentering. Positive source gaps stay ordered;
            # the zero-initialized head makes their multiplier exactly one.
            gaps = torch.diff(base_atoms, dim=-1) * torch.exp(2*torch.tanh(delta))
            atoms = torch.cat([base_atoms[..., :1],
                               base_atoms[..., :1] + torch.cumsum(gaps, dim=-1)], dim=-1)
            atoms = standardize_measure(atoms, probs)
        if self.family in ("scale_only", "shape_scale"):
            multiplier = torch.exp(torch.tanh(self.scale_head(hidden))).unsqueeze(-1)
            atoms = atoms * multiplier
        return atoms, probs

    def training_loss(self, atoms: torch.Tensor, probs: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
        if atoms.shape != probs.shape or target.shape != atoms.shape[:-1]:
            raise ValueError("training_loss shape mismatch")
        if self.family == "deepnpts_rps":
            return author_rps(atoms, probs, target)
        return exact_crps(atoms, probs, target)
