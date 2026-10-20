"""Paper-style MCSSL completion control, NOT author-code reproduction.

Source: https://arxiv.org/html/2605.11846v1, sections 3.4 and F.2/F.4/F.6.
Author-inspired elements: independent Gaussian missing-coordinate noise (.25),
MLP 128/64/32 with GELU/dropout .1, masked reconstruction MSE, AdamW .001/.0001.
Declared adaptations: GRU forecaster is owned by caller; price-vertex rather
than coordinate masking; shared market one-hot; separate imputer pretraining;
coarse-only deterministic date completion; caller's group/horizon positive-part
stabilization of Eq.(7). No claim of oracle conditional sampling or unbiasedness
for the TRUE conditional price law.

`fit_imputer` is an explicit training entrypoint. This module has no import-time
fit or market-data access. `sample_tokens` NEVER reads data['full'], future y,
raw_y, dates, or source provenance. Only coarse tokens and market IDs are used.
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


VERTICES = {"short": tuple(range(10, 21)), "sparse": tuple(range(0, 21, 2))}


class PriceCompletionMLP(nn.Module):
    def __init__(self, n_markets: int = 3, widths: Sequence[int] = (128, 64, 32),
                 noise_scale: float = 0.25) -> None:
        super().__init__()
        self.n_markets = n_markets
        self.noise_scale = float(noise_scale)
        self.widths = tuple(widths)
        layers = []
        width = 21 * 3 + n_markets
        for nxt in widths:
            layers += [nn.Linear(width, nxt), nn.GELU(), nn.Dropout(0.1)]
            width = nxt
        layers.append(nn.Linear(width, 21))
        self.net = nn.Sequential(*layers)

    def forward(self, observed_price: Tensor, observed_day: Tensor, mask: Tensor,
                market: Tensor, standard_normal_noise: Tensor | None = None) -> Tensor:
        if observed_price.shape != observed_day.shape or mask.shape != observed_price.shape:
            raise ValueError("Observed vertices and mask must match [B,21].")
        if observed_price.shape[1] != 21:
            raise ValueError("The fixed protocol uses 21 vertices.")
        mask = mask.to(dtype=torch.bool, device=observed_price.device)
        if standard_normal_noise is None:
            standard_normal_noise = torch.zeros_like(observed_price)
        if standard_normal_noise.shape != observed_price.shape:
            raise ValueError("Noise must have shape [B,21].")
        # Unknown true values are absent from these arguments. Force masked slots
        # to zero even if a caller accidentally leaves arbitrary filler there.
        p = observed_price * mask + self.noise_scale * standard_normal_noise * (~mask)
        # Relative dates can be large; log scale is deterministic and uses no fit.
        day = torch.sign(observed_day) * torch.log1p(observed_day.abs()) * mask
        market = market.to(device=p.device, dtype=torch.long)
        if bool(((market < 0) | (market >= self.n_markets)).any()):
            raise ValueError("Market ID outside the fixed one-hot vocabulary.")
        one_hot = torch.nn.functional.one_hot(market, self.n_markets).to(p.dtype)
        features = torch.cat([p, day, mask.to(p.dtype), one_hot], dim=1)
        return self.net(features)


def _tensor(value, device, dtype) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def observed_vertices(data: Mapping, view: str, *, device="cpu",
                      dtype=torch.float64) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Recover ONLY retained relative prices/dates and generated full date grid.

    Returned p/day arrays have zero placeholders at unknown positions. The
    returned calendar completion uses only known vertices: linear bridges for
    sparse and backwards extrapolation at the mean observed gap for short.
    These completed dates are synthetic floating values, not observed dates.
    """
    if view not in VERTICES:
        raise ValueError("view must be short or sparse.")
    x = _tensor(data[view], device, dtype)
    if x.ndim != 3 or x.shape[1] != 20 or x.shape[2] < 3:
        raise ValueError("Coarse tokens must have shape [B,20,C>=3].")
    n = 10
    if view + "_mask" in data:
        valid = _tensor(data[view + "_mask"], device, torch.bool)
        expected = torch.arange(20, device=device)[None, :] < n
        if not torch.equal(valid, expected.expand(x.shape[0], -1)):
            raise ValueError("This control requires the prespecified ten-interval coarse view.")
    delta = x[:, :n, 0]
    day_gap = torch.expm1(x[:, :n, 2])
    if not bool(torch.isfinite(delta).all() and torch.isfinite(day_gap).all()) or bool((day_gap <= 0).any()):
        raise ValueError("Nonfinite or nonpositive retained date interval.")
    zero = torch.zeros((x.shape[0], 1), device=device, dtype=dtype)
    cp = torch.cat([zero, delta.cumsum(1)], 1)
    cd = torch.cat([zero, day_gap.cumsum(1)], 1)
    cp, cd = cp - cp[:, -1:], cd - cd[:, -1:]
    indices = torch.tensor(VERTICES[view], device=device)
    p, d = torch.zeros((x.shape[0], 21), device=device, dtype=dtype), torch.zeros((x.shape[0], 21), device=device, dtype=dtype)
    mask = torch.zeros_like(p, dtype=torch.bool)
    p[:, indices], d[:, indices], mask[:, indices] = cp, cd, True
    completed_days = d.clone()
    if view == "sparse":
        completed_days[:, 1::2] = 0.5 * (cd[:, :-1] + cd[:, 1:])
    else:
        distance = torch.arange(10, 0, -1, device=device, dtype=dtype)[None, :]
        completed_days[:, :10] = cd[:, :1] - distance * day_gap.mean(1, keepdim=True)
    if bool((completed_days[:, 1:] <= completed_days[:, :-1]).any()):
        raise ValueError("Generated dates must remain strictly ordered.")
    market = _tensor(data["market"], device, torch.long).reshape(-1)
    if market.numel() != x.shape[0]:
        raise ValueError("Market vector length mismatch.")
    return p, d, mask, market, completed_days


def _full_price_target(data: Mapping, *, device="cpu", dtype=torch.float64) -> Tensor:
    """Only training reconstruction labels; not called by sample_tokens."""
    x = _tensor(data["full"], device, dtype)
    if x.ndim != 3 or x.shape[1] != 20 or x.shape[2] < 3:
        raise ValueError("full training tokens must be [B,20,C>=3].")
    p = torch.cat([torch.zeros((x.shape[0], 1), device=device, dtype=dtype),
                   x[:, :, 0].cumsum(1)], 1)
    return p - p[:, -1:]


def _noise(shape, generator, device, dtype):
    # Dedicated CPU generator keeps the exact noise stream explicit/reproducible.
    return torch.randn(shape, generator=generator, dtype=dtype, device="cpu").to(device)


def sample_tokens(data: Mapping, view: str, q: PriceCompletionMLP,
                  noise_gen: torch.Generator, *, return_receipt: bool = False):
    """Return a full-shape [B,20,3] completion, conditioned only on coarse data.

    Call twice sequentially using the same dedicated generator for independent
    noise draws. Do not reseed to the same seed between the two calls. This
    function freezes q and returns detached inputs for the caller's forecaster.
    """
    parameter = next(q.parameters())
    p, day, mask, market, completed_day = observed_vertices(
        data, view, device=parameter.device, dtype=parameter.dtype)
    old_mode = q.training
    q.eval()
    with torch.no_grad():
        noise = _noise(p.shape, noise_gen, parameter.device, parameter.dtype)
        imputed_price = q(p, day, mask, market, noise)
        completed_price = torch.where(mask, p, imputed_price)
        price_gap = completed_price[:, 1:] - completed_price[:, :-1]
        calendar_gap = completed_day[:, 1:] - completed_day[:, :-1]
        index_gap = torch.full_like(price_gap, np.log(2.0))
        tokens = torch.stack([price_gap, index_gap, torch.log1p(calendar_gap)], -1)
    q.train(old_mode)
    if not return_receipt:
        return tokens
    return tokens, {"view": view, "shape": list(tokens.shape),
                    "real_missing_price_read": False, "real_missing_date_read": False,
                    "future_target_read": False,
                    "date_completion": "coarse-only linear bridge or observed-mean-gap backwards extrapolation",
                    "date_values_are_synthetic": True,
                    "noise_scale": q.noise_scale,
                    "max_retained_price_error": float((completed_price[mask] - p[mask]).abs().max()),
                    "minimum_generated_day_gap": float(calendar_gap.min())}


def fit_imputer(train: Mapping, *, maxsteps: int = 500, batch_size: int = 256,
                seed: int = 17, n_markets: int | None = None, device="cpu",
                dtype=torch.float64, widths: Sequence[int] = (128,64,32),
                learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                views: Sequence[str] = ("short", "sparse")) -> tuple[PriceCompletionMLP, dict]:
    """Fit only missing historical price coordinates, with no future Y access.

    Caller must pass TRAINING origins only. This function does not split data,
    select by validation performance, read files, or refit the GRU forecaster.
    Training masks alternate view families by independently drawn row labels;
    row sampling gives each market equal expected probability.
    """
    if not (0 <= maxsteps <= 500) or batch_size < 1:
        raise ValueError("Explicit control budget: 0 <= maxsteps <= 500, batch_size >= 1.")
    if not views or any(v not in VERTICES for v in views):
        raise ValueError("Invalid completion view family.")
    mids = np.asarray(train["market"], dtype=np.int64)
    if not len(mids):
        raise ValueError("No training origins.")
    if n_markets is None:
        n_markets = int(mids.max()) + 1
    unique = np.unique(mids)
    sampling_weight = np.array([1.0 / (len(unique) * np.sum(mids == m)) for m in mids])
    tensors = [observed_vertices(train, v, device=device, dtype=dtype) for v in views]
    target = _full_price_target(train, device=device, dtype=dtype)
    if target.shape[0] != len(mids):
        raise ValueError("Training target row mismatch.")
    # Mask construction and reconstruction targets must agree at observed nodes.
    for p, _, mask, _, _ in tensors:
        if not torch.allclose(p[mask], target[mask], atol=1e-8, rtol=1e-8):
            raise ValueError("Full/coarse training views disagree on observed price vertices.")
    dedicated = torch.Generator(device="cpu").manual_seed(int(seed) + 1079)
    devices = [] if str(device) == "cpu" else [torch.device(device).index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        q = PriceCompletionMLP(n_markets, widths).to(device=device, dtype=dtype)
        optimizer = torch.optim.AdamW(q.parameters(), lr=learning_rate, weight_decay=weight_decay)
        weights = torch.as_tensor(sampling_weight, dtype=torch.float64)
        losses = []
        for step in range(maxsteps):
            index_cpu = torch.multinomial(weights, batch_size, replacement=True, generator=dedicated)
            view_cpu = torch.randint(len(views), (batch_size,), generator=dedicated)
            index = index_cpu.to(device)
            p = torch.stack([t[0] for t in tensors])[view_cpu.to(device), index]
            day = torch.stack([t[1] for t in tensors])[view_cpu.to(device), index]
            mask = torch.stack([t[2] for t in tensors])[view_cpu.to(device), index]
            market = tensors[0][3][index]
            label = target[index]
            noise = _noise(p.shape, dedicated, device, dtype)
            output = q(p, day, mask, market, noise)
            missing = (~mask).to(dtype)
            per_row = ((output - label).square() * missing).sum(1) / missing.sum(1)
            loss = per_row.mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite historical imputation loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in q.parameters()):
                raise FloatingPointError("Nonfinite imputer gradient.")
            optimizer.step()
            losses.append(float(loss.detach()))
    q.eval()
    state_hash = hashlib.sha256()
    for name, value in q.state_dict().items():
        state_hash.update(name.encode())
        state_hash.update(value.detach().cpu().numpy().tobytes())
    support = []
    if "origin" in train:
        origins = np.asarray(train["origin"], dtype=np.int64)
        support = [{"market_id": int(m), "origins": int(np.sum(mids==m)),
                    "minimum_origin": int(origins[mids==m].min()),
                    "maximum_origin": int(origins[mids==m].max())} for m in unique]
    receipt = {"implementation": "paper-style MCSSL price-completion control; not author reproduction",
               "source": "https://arxiv.org/html/2605.11846v1",
               "seed": int(seed), "maxsteps": maxsteps, "optimizer_steps": len(losses),
               "batch_size": batch_size, "training_origins": len(mids),
               "support": support, "widths": list(widths),
               "parameter_count": sum(p.numel() for p in q.parameters()),
               "learning_rate": learning_rate, "weight_decay": weight_decay,
               "noise_scale": q.noise_scale, "views": list(views),
               "sampling": "equal-market probability; uniform view probability; replacement",
               "loss": "MSE on missing historical price vertices only",
               "future_target_access": False, "raw_y_access": False,
               "real_missing_date_access": False,
               "separate_imputer_pretraining": True, "validation_selection": False,
               "training_loss_trace": losses, "state_sha256": state_hash.hexdigest(),
               "conditional_law_correctness_claimed": False,
               "adaptations": ["price-vertex masking and GRU forecaster",
                               "market one-hot", "separate imputer pretraining",
                               "coarse-only deterministic synthetic date completion",
                               "caller may rectify group/horizon mean twoMC; this is not original Eq7"]}
    return q, receipt
