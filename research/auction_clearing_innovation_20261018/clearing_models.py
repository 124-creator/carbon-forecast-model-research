"""Small forecasting models for the predeclared auction-clearing experiment.

The Gamma demand measure and its price distribution are working statistical
models. They do not identify the confidential order book or cancellation rule.
All inputs to forward are known at the forecast origin; future bids are labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _GammaCDF(torch.autograd.Function):
    """Regularized lower Gamma CDF with a checked numerical shape derivative.

    PyTorch 2.5.1 does not implement the shape derivative of gammainc. Forward
    and central shape differences are evaluated in float64. The x derivative
    is analytic. This is first-order optimization, not exact symbolic AD.
    """
    @staticmethod
    def forward(ctx, a, x):
        aa, xx = a.double(), x.double().clamp_min(1e-12)
        ctx.save_for_backward(aa, xx)
        ctx.input_dtype = a.dtype
        return torch.special.gammainc(aa, xx).to(a.dtype)

    @staticmethod
    def backward(ctx, grad):
        a, x = ctx.saved_tensors
        eps = torch.minimum(1e-3 * (1 + a), 0.2 * a)
        da = (torch.special.gammainc(a + eps, x)
              - torch.special.gammainc(a - eps, x)) / (2 * eps)
        dx = torch.exp((a - 1) * x.log() - x - torch.lgamma(a))
        return ((grad.double() * da).to(ctx.input_dtype),
                (grad.double() * dx).to(ctx.input_dtype))


def gamma_cdf(a, x):
    return _GammaCDF.apply(a.clamp_min(1e-6), x.clamp_min(1e-12))


class _GammaSurvival(torch.autograd.Function):
    """Upper Gamma probability avoids catastrophic 1-CDF cancellation.

    Probabilities remain float64 through conditional normalization. Gradients
    return in the network parameter dtype. The derivative is checked separately.
    """
    @staticmethod
    def forward(ctx, a, x):
        aa, xx = a.double(), x.double().clamp_min(1e-12)
        ctx.save_for_backward(aa, xx)
        ctx.input_dtype = a.dtype
        return torch.special.gammaincc(aa, xx)

    @staticmethod
    def backward(ctx, grad):
        a, x = ctx.saved_tensors
        eps = torch.minimum(1e-3*(1+a), .2*a)
        da = (torch.special.gammaincc(a+eps, x)-torch.special.gammaincc(a-eps, x))/(2*eps)
        dx = -torch.exp((a-1)*x.log()-x-torch.lgamma(a))
        return ((grad.double()*da).to(ctx.input_dtype),
                (grad.double()*dx).to(ctx.input_dtype))


def clearing_masses(fractions, concentration, mean_volume, supply):
    """Unconditional price-atom masses and the mechanical failure probability.

    Shapes: fractions [N,C,K], concentration/mean_volume [N,C], supply [N].
    Atom j clears iff demand STRICTLY ABOVE j < S <= demand AT OR ABOVE j.
    The Gamma variables have a common scale within each component.
    """
    alpha = fractions * concentration[..., None]
    tail_inclusive = torch.flip(torch.cumsum(torch.flip(alpha, [-1]), -1), [-1])
    x = (supply[:, None] * concentration / mean_volume.clamp_min(1e-7))
    survival_inclusive = _GammaSurvival.apply(tail_inclusive.clamp_min(1e-6),
                                              x[..., None].expand_as(alpha))
    survival_exclusive = torch.cat([survival_inclusive[..., 1:],
                                    torch.zeros_like(survival_inclusive[..., :1])], -1)
    masses = (survival_inclusive-survival_exclusive).clamp_min(0)
    fail = 1-survival_inclusive[..., 0]
    return masses, fail


def atom_crps(atoms, weights, y):
    """Exact CRPS of a finite predictive distribution, including unequal grids."""
    xx, order = torch.sort(atoms, dim=-1)
    ww = torch.gather(weights, -1, order)
    cw = ww.cumsum(-1) - ww
    cwx = (ww * xx).cumsum(-1) - ww * xx
    half_pair = (ww * (xx * cw - cwx)).sum(-1)
    return (weights * (atoms - y[:, None]).abs()).sum(-1) - half_pair


def weighted_quantile(atoms, weights, q):
    xx, ix = torch.sort(atoms, -1)
    ww = torch.gather(weights, -1, ix)
    cc = ww.cumsum(-1)
    jj = (cc < q).sum(-1).clamp(max=atoms.shape[-1] - 1)
    return torch.gather(xx, -1, jj[:, None]).squeeze(-1)


class Encoder(nn.Module):
    def __init__(self, n_features, n_context, hidden=32, kind="gru"):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.history = nn.GRU(n_features, hidden, batch_first=True)
        else:
            self.history = nn.Sequential(nn.Linear(n_features * 20, hidden),
                                         nn.SiLU(), nn.Linear(hidden, hidden))
        self.combine = nn.Sequential(nn.Linear(hidden + n_context, hidden),
                                     nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())

    def forward(self, x, context):
        if self.kind == "gru":
            _, hidden = self.history(x)
            h = hidden[-1]
        else:
            h = self.history(x.flatten(1))
        return self.combine(torch.cat([h, context], -1))


class PointNetwork(nn.Module):
    def __init__(self, n_features, n_context, hidden=32, kind="gru"):
        super().__init__()
        self.encoder = Encoder(n_features, n_context, hidden, kind)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, context, base):
        mean = self.head(self.encoder(x, context)).squeeze(-1)
        return {"mean": mean}


class DemandNetwork(nn.Module):
    """Mixture of Gamma random demand measures and matched ablations.

    Modes:
      stochastic: integrate random demand through the exact grid clearing law;
      direct: same encoder and demand auxiliary task, direct price probabilities;
      plugin: clear the mean demand measure and mix its point outcomes;
      no_supply: stochastic operator receives last observed supply; the encoder
                 still receives all the same information (operator ablation).
    The K prices per component are centered against the last known auction.
    """
    def __init__(self, n_features, n_context, hidden=32, components=3, knots=17,
                 mode="stochastic", kind="gru"):
        super().__init__()
        self.encoder = Encoder(n_features, n_context, hidden, kind)
        self.components = components
        self.knots = knots
        self.mode = mode
        # mixture logits, mean bid ratio, concentration, location, span, K weights
        self.head = nn.Linear(hidden, components * (5 + knots))
        self.supply_to_bid_volume = nn.Linear(2, components, bias=False)
        nn.init.normal_(self.head.weight, std=0.005)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.supply_to_bid_volume.weight)
        self.register_buffer("grid", torch.linspace(-3, 3, knots))

    def forward(self, x, context, base):
        n = len(x)
        # Demand price shares, mixture weights and grid locations cannot freely
        # re-encode the announced supply. Supply reaches these price forecasts
        # through the clearing operator and a separately supervised bid-volume
        # head. The matched direct head sees all of the same available inputs.
        demand_context = context.clone()
        if self.mode != "direct":
            demand_context[:, :2] = 0
        h = self.encoder(x, demand_context)
        raw = self.head(h).reshape(n, self.components, 5 + self.knots)
        mix = F.softmax(raw[..., 0], -1)
        mean_bid_ratio = torch.exp((raw[..., 1]+self.supply_to_bid_volume(context[:, :2])).clamp(-3, 3))
        concentration = 1.0 + 39.0 * torch.sigmoid(raw[..., 2] - 0.9)
        location = raw[..., 3]
        span = 0.2 + F.softplus(raw[..., 4] + 0.3)
        fraction = F.softmax(raw[..., 5:], -1)
        # base columns: last S / last B, planned S / last B, known forecast
        # price scale / current price. No future B or other future result.
        supply0 = base[:, 0].clamp_min(1e-5)
        supply1 = (base[:, 0] if self.mode == "no_supply" else base[:, 1]).clamp_min(1e-5)
        reference_m, reference_fail = clearing_masses(
            fraction, concentration, torch.ones_like(mean_bid_ratio), supply0)
        reference_w = reference_m / reference_m.sum(-1, keepdim=True).clamp_min(1e-300)
        relative_scale = base[:, 2].clamp_min(1e-5)
        scaled_grid = relative_scale[:, None, None]*span[..., None]*self.grid
        reference_log_moment = torch.logsumexp(
            reference_w.clamp_min(1e-30).log()+scaled_grid, -1)
        # Positive price atoms, with exact RW anchoring in the numerical interior:
        # E_ref[P/P0] = exp(r*location), hence location=0 gives E_ref[P]=P0.
        # Return atoms in the standardized residual units used by every model.
        log_ratio = (relative_scale[:, None, None]*location[..., None]
                     + scaled_grid - reference_log_moment[..., None])
        atoms = torch.expm1(log_ratio.clamp(-15, 15))/relative_scale[:, None, None]
        masses, mechanical_fail = clearing_masses(
            fraction, concentration, mean_bid_ratio, supply1)
        component_success = masses.sum(-1)
        if self.mode == "direct":
            # Identical bid-volume head and objective; the price law bypasses
            # random-demand clearing. Direct locations use the same center map.
            masses = fraction * component_success[..., None]
        elif self.mode == "plugin":
            # Piecewise-linear inverse of the MEAN bid measure. This is an
            # explicitly smooth approximation to a discrete boundary rule.
            u = (1 - supply1[:, None] / mean_bid_ratio).clamp(0, 1 - 1e-7)
            cumulative = fraction.cumsum(-1)
            j = (cumulative <= u[..., None]).sum(-1).clamp(max=self.knots-1)
            j0 = (j - 1).clamp_min(0)
            x0 = torch.gather(atoms, -1, j0[..., None]).squeeze(-1)
            x1 = torch.gather(atoms, -1, j[..., None]).squeeze(-1)
            c0 = torch.gather(cumulative, -1, j0[..., None]).squeeze(-1)
            c0 = torch.where(j == 0, torch.zeros_like(c0), c0)
            cj = torch.gather(cumulative, -1, j[..., None]).squeeze(-1)
            zz = ((u - c0) / (cj - c0).clamp_min(1e-7)).clamp(0, 1)
            atoms = (x0 + zz * (x1-x0))[..., None]
            masses = component_success[..., None]
        joint = mix[..., None] * masses
        normalizer = joint.sum((-1, -2), keepdim=True)
        if bool((normalizer <= 1e-290).any()):
            raise FloatingPointError("Mechanical success probability underflow; conditional price is undefined")
        success = normalizer.flatten().clamp(0, 1)
        weights = (joint / normalizer).flatten(1)
        flat_atoms = atoms.flatten(1)
        mean = (weights * flat_atoms).sum(-1)
        return {"mean": mean, "atoms": flat_atoms, "weights": weights,
                "mechanical_failure": 1-success, "mix": mix,
                "mean_bid_ratio": mean_bid_ratio, "concentration": concentration}


def gamma_mixture_nll(output, future_bid_ratio):
    a = output["concentration"]
    mu = output["mean_bid_ratio"]
    theta = mu / a
    y = future_bid_ratio[:, None].clamp_min(1e-7)
    logp = (a-1)*y.log() - y/theta - torch.lgamma(a) - a*theta.log()
    return -torch.logsumexp(output["mix"].clamp_min(1e-12).log()+logp, -1)


def composite_loss(output, target, completed, future_bid_ratio, has_bid,
                   bid_weight=0.02, crps_weight=0.2):
    # Price/status absence is never converted to a numerical zero price.
    good = completed.bool()
    safe_target = torch.where(good, target, torch.zeros_like(target))
    mse = (output["mean"]-safe_target).square()
    price = mse[good].mean() if good.any() else output["mean"].sum()*0
    components = {"mse": float(price.detach())}
    if "atoms" in output:
        crps = atom_crps(output["atoms"], output["weights"], safe_target)
        c = crps[good].mean() if good.any() else crps.sum()*0
        # B density is a marginal auxiliary likelihood, not a joint (B,P) likelihood.
        safe_bid_ratio = torch.where(has_bid.bool(), future_bid_ratio,
                                     torch.ones_like(future_bid_ratio))
        nll = gamma_mixture_nll(output, safe_bid_ratio)
        b = nll[has_bid.bool()].mean() if has_bid.bool().any() else nll.sum()*0
        price = price + crps_weight*c + bid_weight*b
        components.update(crps=float(c.detach()), bid_nll=float(b.detach()))
    return price, components


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
