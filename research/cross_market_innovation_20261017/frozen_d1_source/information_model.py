"""Shared GRU + rectified, blocked kernel conditional-moment regularization.

Research candidate, 2026-10-17. This is NOT a new tower identity or new KCM
principle. See the accompanying method note for the 2026 MCSSL and 2020 KCM
antecedents. No conditional imputer and no market data are needed by this file.

Core tokens: [price change / training-only scale, log1p(observation-index gap),
              log1p(calendar-day gap)]. The caller may append market one-hot to
all neural arms. Validity MUST be supplied by lengths or mask.
Output: 20 cumulative price changes in the SAME units used by the supplied y.
Only differences BETWEEN retained vertices become tokens: the predecessor of a
removed vertex must never leak into a coarse view. A zero price change remains
a valid observed token. Targets always use the original observation index.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class SharedInformationGRU(nn.Module):
    """An ordinary shared GRU; the proposed intervention is in the loss.

    No learned market embedding, attention, added GRU gate, or learned kernel.
    Explicit market one-hot may be appended by the caller to every valid token.
    `mask` may contain holes ONLY if the retained token changes/intervals have
    already been recomputed. Use `make_price_view`, not simple raw-token dropout.
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 16,
                 horizon: int = 20) -> None:
        super().__init__()
        if hidden_size not in (16, 32):
            raise ValueError("The bounded candidate uses hidden_size 16 or 32.")
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.gru.input_size:
            raise ValueError("x must have shape [batch, length, input_size].")
        b, length, _ = x.shape
        if mask is None:
            if lengths is None:
                raise ValueError("Supply lengths or mask explicitly; feature channels are not validity flags.")
            else:
                mask = torch.arange(length, device=x.device)[None, :] < lengths.to(x.device)[:, None]
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)
        if mask.shape != (b, length):
            raise ValueError("mask shape must match x's first two dimensions.")
        actual_lengths = mask.sum(1)
        if torch.any(actual_lengths < 1):
            raise ValueError("Every history must contain at least one observed token.")
        if lengths is not None and not torch.equal(actual_lengths.cpu(), lengths.long().cpu()):
            raise ValueError("lengths and mask disagree.")
        # Every view in the current protocol has equal-length prefix masks.
        # This fast path is the same GRU recurrence without B Python copies or
        # packing. General variable-length/holed masks still use the fallback.
        n = int(actual_lengths[0])
        prefix = torch.arange(length, device=x.device)[None, :] < n
        if bool((actual_lengths == n).all()) and torch.equal(mask, prefix.expand(b, -1)):
            _, state = self.gru(x[:, :n])
        else:
            # Compaction is only padding management; it does not aggregate increments.
            compact = x.new_zeros(x.shape)
            for row in range(b):
                n = int(actual_lengths[row])
                compact[row, :n] = x[row, mask[row]]
            packed = pack_padded_sequence(compact, actual_lengths.cpu(),
                                          batch_first=True, enforce_sorted=False)
            _, state = self.gru(packed)
        return self.head(state[-1])


def make_price_view(prices: Tensor, days: Tensor, keep: Tensor,
                    increment_scale: Tensor | float = 1.0,
                    pad_to: int = 20) -> tuple[Tensor, Tensor]:
    """Create a compact view using ONLY retained prices and retained dates.

    prices/days/keep: [B,L+1] vertices, days numeric calendar days.
    keep must retain the current (last) point. All scales must be fit before the
    training cutoff by the caller. No statistics are computed from these inputs.
    Full 21 vertices give 20 increments; retained 11 vertices give 10 increments.
    Returns right-padded tokens [B,pad_to,3] and interval lengths [B].
    """
    if prices.ndim != 2 or days.shape != prices.shape or keep.shape != prices.shape:
        raise ValueError("prices, days and keep must all have shape [B,L].")
    keep = keep.to(device=prices.device, dtype=torch.bool)
    if not bool(keep[:, -1].all()):
        raise ValueError("Coarse and full views must retain the same current price.")
    if not bool(torch.isfinite(prices).all() and torch.isfinite(days).all()):
        raise ValueError("Nonfinite price/date input.")
    if bool((days[:, 1:] <= days[:, :-1]).any()):
        raise ValueError("Original observation dates must be strictly increasing.")
    lengths = keep.sum(1) - 1
    if bool((lengths < 1).any()) or int(lengths.max()) > pad_to:
        raise ValueError("Need at least two retained vertices and sufficient pad_to.")
    scale = torch.as_tensor(increment_scale, dtype=prices.dtype, device=prices.device)
    if scale.ndim == 0:
        scale = scale.expand(prices.shape[0])
    scale = scale.reshape(-1)
    if scale.numel() != prices.shape[0] or bool((scale <= 0).any()):
        raise ValueError("increment_scale must be positive scalar or [B].")
    tokens = prices.new_zeros((prices.shape[0], pad_to, 3))
    for row in range(prices.shape[0]):
        p = prices[row, keep[row]]
        t = days[row, keep[row]].to(prices.dtype)
        indices = torch.nonzero(keep[row], as_tuple=False).flatten().to(prices.dtype)
        n = p.numel() - 1
        tokens[row, :n, 0] = (p[1:] - p[:-1]) / scale[row]
        tokens[row, :n, 1] = torch.log1p(indices[1:] - indices[:-1])
        tokens[row, :n, 2] = torch.log1p(t[1:] - t[:-1])
    return tokens, lengths


def nested_keep_masks(batch_size: int, length: int = 21, coarse_length: int = 11,
                      mode: str = "recent", generator: Optional[torch.Generator] = None,
                      device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
    """Two prespecified interventions; never infer missing trades from prices.

    recent: most recent 11 of 21 vertices (10 most recent increments).
    sparse: the 11 vertices at indices 0,2,...,20 (10 aggregate increments).
    random: optional intervention retaining endpoints plus random interiors.
    Randomness uses a CPU generator so the mask sequence is device-independent.
    """
    if not 2 <= coarse_length <= length:
        raise ValueError("Require 2 <= coarse_length <= length.")
    full = torch.ones((batch_size, length), dtype=torch.bool)
    coarse = torch.zeros_like(full)
    if mode == "recent":
        coarse[:, -coarse_length:] = True
    elif mode == "sparse":
        if length != 21 or coarse_length != 11:
            raise ValueError("Prespecified sparse view requires 21/11 vertices.")
        coarse[:, ::2] = True
    elif mode == "random":
        coarse[:, 0] = coarse[:, -1] = True
        for row in range(batch_size):
            ids = torch.randperm(length - 2, generator=generator)[:coarse_length - 2] + 1
            coarse[row, ids] = True
    else:
        raise ValueError("mode must be recent, sparse, or random.")
    return full.to(device), coarse.to(device)


def fixed_coarse_features(coarse_x: Tensor, pad_to: int = 20) -> Tensor:
    """Injective padded raw-token coordinates; detached from learned parameters.

    Different coarse-view families must be separate groups in the moment loss.
    A universal RBF kernel concerns these supplied coordinates, not omitted raw
    absolute price/date information. Do not claim full-information calibration.
    """
    if coarse_x.ndim != 3 or coarse_x.shape[1] > pad_to:
        raise ValueError("coarse_x must be [B,L,C] with L <= pad_to.")
    z = coarse_x.detach().new_zeros((coarse_x.shape[0], pad_to, coarse_x.shape[-1]))
    z[:, :coarse_x.shape[1]] = coarse_x.detach()
    return z.flatten(1)


def rbf_mixture(features: Tensor,
                bandwidths: Sequence[float] = (0.5, 1.0, 2.0)) -> Tensor:
    """Fixed PSD kernel in [0,1]; mean squared coordinate distance is used.

    Removing pairs later DOES NOT preserve PSD. The optimization safeguard is
    rectification of each group/horizon scalar, not a claim that the mask is PSD.
    """
    z = features.detach()
    if z.ndim != 2 or not bandwidths or any(float(s) <= 0 for s in bandwidths):
        raise ValueError("Invalid features/bandwidths.")
    # Algebraically the same squared distance; no B x B x D temporary tensor.
    norms = z.square().sum(1)
    distance = (norms[:, None] + norms[None, :] - 2.0 * (z @ z.T))
    distance = distance.clamp_min_(0.0).div_(z.shape[1])
    kernel = torch.zeros_like(distance)
    for s in bandwidths:
        kernel.add_(torch.exp(-distance / (2.0 * float(s) ** 2)))
    return kernel.div_(len(bandwidths))


def blocked_kernel_moment_penalty(
        fine_prediction: Tensor, coarse_prediction: Tensor, coarse_features: Tensor,
        support_start: Tensor, support_end: Tensor, groups: Optional[Tensor] = None,
        horizon_scale: Optional[Tensor] = None,
        bandwidths: Sequence[float] = (0.5, 1.0, 2.0),
        gap: int = 0) -> tuple[Tensor, dict]:
    """Return mean_g mean_h max(0, U[g,h]) plus detached diagnostics.

    U = sum_{ordered eligible i,j} k(c_i,c_j) d_i,h d_j,h / #eligible pairs.
    d is optionally divided by a strictly positive TRAINING-only horizon scale.
    A group should encode market AND coarse-view type (optionally fixed regime).
    Supports are inclusive local observation indices, preferably [t-20,t+20].
    No self pair, no overlapping-support pair, and no cross-group pair is used.
    Nonoverlap does not imply independence for an observed time series. This is
    not advertised as an unbiased finite-series conditional-moment estimator.

    Rectification makes the penalty >= 0 and removes negative-unbounded loss.
    It introduces finite-sample upward bias and has zero gradient when U < 0.
    A group with zero pairs is skipped and MUST be monitored in the receipt.
    """
    if fine_prediction.ndim != 2 or coarse_prediction.shape != fine_prediction.shape:
        raise ValueError("Predictions must have matching [B,H] shapes.")
    b, _ = fine_prediction.shape
    device = fine_prediction.device
    start, end = support_start.to(device).reshape(-1), support_end.to(device).reshape(-1)
    if start.numel() != b or end.numel() != b or bool((start > end).any()):
        raise ValueError("Invalid support intervals.")
    if groups is None:
        groups = torch.zeros(b, device=device, dtype=torch.long)
    else:
        groups = groups.to(device).reshape(-1)
    if groups.numel() != b or coarse_features.shape[0] != b or gap < 0:
        raise ValueError("Invalid group/features/gap.")
    d = fine_prediction - coarse_prediction
    if horizon_scale is not None:
        hs = horizon_scale.to(device=device, dtype=d.dtype)
        if bool((hs <= 0).any()) or not bool(torch.isfinite(hs).all()):
            raise ValueError("horizon_scale must be finite and strictly positive.")
        d = d / hs
    k = rbf_mixture(coarse_features.to(device=device, dtype=d.dtype), bandwidths)
    separated = (end[:, None] + gap < start[None, :]) | (end[None, :] + gap < start[:, None])
    terms, records = [], []
    for group in torch.unique(groups):
        members = groups == group
        eligible = separated & members[:, None] & members[None, :]
        count = int(eligible.sum())
        if count == 0:
            records.append({"group": int(group), "ordered_pair_count": 0,
                            "skipped": True})
            continue
        weights = k * eligible.to(k.dtype)
        u = torch.einsum("ij,ih,jh->h", weights, d, d) / count
        rectified = torch.clamp_min(u, 0.0)
        terms.append(rectified.mean())
        records.append({"group": int(group), "ordered_pair_count": count,
                        "unordered_pair_count": count // 2,
                        "kernel_weight_sum": float(weights.detach().sum()),
                        "raw_u": u.detach().cpu().tolist(),
                        "rectified_u": rectified.detach().cpu().tolist(),
                        "positive_horizon_fraction": float((u.detach() > 0).float().mean()),
                        "skipped": False})
    penalty = torch.stack(terms).mean() if terms else d.sum() * 0.0
    return penalty, {"estimator": "group_horizon_positive_part_blocked_U",
                     "value": float(penalty.detach()), "groups": records,
                     "all_groups_skipped": not bool(terms),
                     "independence_claimed": False,
                     "finite_sample_unbiased_claimed": False}


def information_objective(fine_prediction: Tensor, coarse_prediction: Tensor,
                          target: Tensor, penalty: Tensor, strength: float = 0.1,
                          horizon_scale: Optional[Tensor] = None,
                          groups: Optional[Tensor] = None) -> tuple[Tensor, dict]:
    """Equal group, equal horizon, equal full/coarse squared error + lambda R.

    Set strength=0 for the mandatory otherwise-identical masking-only ablation.
    Default 0.1 is a prespecifiable starting value, not empirically optimized.
    horizon_scale is applied ONCE here and ONCE to d by the moment function;
    do not also pre-divide predictions when passing the same horizon_scale.
    """
    if strength < 0 or not math.isfinite(strength):
        raise ValueError("strength must be finite and nonnegative.")
    if target.shape != fine_prediction.shape or coarse_prediction.shape != target.shape:
        raise ValueError("All predictions/targets must share [B,H].")
    scale = 1.0 if horizon_scale is None else horizon_scale.to(target)
    if isinstance(scale, Tensor) and bool((scale <= 0).any()):
        raise ValueError("horizon_scale must be positive.")
    losses = 0.5 * (((fine_prediction - target) / scale).square()
                    + ((coarse_prediction - target) / scale).square()).mean(1)
    if groups is None:
        supervised = losses.mean()
    else:
        groups = groups.to(losses.device)
        supervised = torch.stack([losses[groups == g].mean()
                                  for g in torch.unique(groups)]).mean()
    total = supervised + float(strength) * penalty
    return total, {"supervised": float(supervised.detach()),
                   "moment_penalty": float(penalty.detach()),
                   "lambda": float(strength), "total": float(total.detach())}
