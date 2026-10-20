"""One post-v1, pre-result hypothesis: separate bid location from auction basis.

The reported median is an observable auxiliary label, not an identified market
price or a volume-weighted quantile. A conditional independence working model
combines a lognormal future median with a Gamma-clearing price/median ratio.
Its price distribution is a finite lognormal mixture. Losses are composite
marginal/conditional scores, not an asserted exact joint order-book likelihood.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clearing_models import Encoder, PointNetwork, clearing_masses, atom_crps, gamma_mixture_nll


def lognormal_crps(log_mu, sigma, weights, y):
    """Exact CRPS of a finite lognormal mixture, in the supplied positive units.

    Evaluates E|X-y| - E|X-X'|/2 using truncated lognormal moments.
    Float64 calculations avoid subtractive loss of precision near narrow laws.
    """
    mu, sig, w = log_mu.double(), sigma.double(), weights.double()
    sig = torch.broadcast_to(sig, mu.shape).clamp_min(1e-7)
    yy = y.double().clamp_min(1e-12)[:, None]
    mm = torch.exp(mu + .5*sig.square())
    z = (yy.log()-mu)/sig
    first = mm*(1-2*torch.special.ndtr(z-sig)) + yy*(2*torch.special.ndtr(z)-1)
    mi, mj = mu[:, :, None], mu[:, None, :]
    si, sj = sig[:, :, None], sig[:, None, :]
    den = (si.square()+sj.square()).sqrt()
    ei, ej = mm[:, :, None], mm[:, None, :]
    emin = (ei*torch.special.ndtr((mj-mi-si.square())/den)
            + ej*torch.special.ndtr((mi-mj-sj.square())/den))
    distance = ei+ej-2*emin
    return (w*first).sum(-1)-.5*(w[:, :, None]*w[:, None, :]*distance).sum((-1, -2))


class FactorizedDemandNetwork(nn.Module):
    """Same-capacity factorial model and three mechanistic ablations.

    base: S0/B0, plannedS/B0, forecast_scale/P0, micro_scale/M0, M0/P0.
    S is excluded from the structured location/shape encoder and reaches price
    through clearing and the separately supervised demand-volume head.
    The direct counterpart may use all known context, and has no Gamma law in
    its price probabilities or in its reference-centering calculation.
    """
    def __init__(self, n_features, n_context, hidden=32, components=3, knots=17,
                 mode="stochastic"):
        super().__init__()
        self.encoder = Encoder(n_features, n_context, hidden, "gru")
        self.components, self.knots, self.mode = components, knots, mode
        self.macro_head = nn.Linear(hidden, 2)
        self.micro_head = nn.Linear(hidden, components*(5+knots))
        self.supply_to_bid_volume = nn.Linear(2, components, bias=False)
        nn.init.zeros_(self.macro_head.weight)
        nn.init.zeros_(self.macro_head.bias)
        nn.init.normal_(self.micro_head.weight, std=.005)
        nn.init.zeros_(self.micro_head.bias)
        nn.init.zeros_(self.supply_to_bid_volume.weight)
        self.register_buffer("grid", torch.linspace(-3, 3, knots))

    def forward(self, x, context, base):
        cc = context.clone()
        if self.mode != "direct":
            cc[:, :2] = 0
        h = self.encoder(x, cc)
        macro = self.macro_head(h)
        raw = self.micro_head(h).reshape(len(x), self.components, 5+self.knots)
        mix = F.softmax(raw[..., 0], -1)
        mean_bid_ratio = torch.exp((raw[..., 1]+self.supply_to_bid_volume(context[:, :2])).clamp(-3, 3))
        concentration = 1+39*torch.sigmoid(raw[..., 2]-.9)
        location = 4*torch.tanh(raw[..., 3]/4)
        span = .25+3.75*torch.sigmoid(raw[..., 4]-1.38629436112)
        fractions = F.softmax(raw[..., 5:], -1)
        supply0 = base[:, 0].clamp_min(1e-5)
        supply1 = (base[:, 0] if self.mode == "no_supply" else base[:, 1]).clamp_min(1e-5)
        rprice = base[:, 2].clamp_min(1e-6).double()
        micro_r = base[:, 3].clamp_min(1e-6).double()
        median_over_price = base[:, 4].clamp_min(1e-6).double()
        macro_r = rprice/median_over_price
        # The scale ablation retains all labels and objectives; only the
        # clearing grid uses macro scale, reproducing the suspected coupling.
        grid_r = macro_r if self.mode == "unseparated" else micro_r
        if self.mode == "direct":
            reference_w = fractions.double()
            weights = (mix[..., None]*fractions).flatten(1).double()
            weights = weights/weights.sum(-1, keepdim=True)
            mechanical_failure = torch.full((len(x),), float("nan"), device=x.device)
        else:
            reference_m, _ = clearing_masses(fractions, concentration, torch.ones_like(mean_bid_ratio), supply0)
            reference_w = reference_m/reference_m.sum(-1, keepdim=True).clamp_min(1e-300)
            masses, _ = clearing_masses(fractions, concentration, mean_bid_ratio, supply1)
            joint = mix[..., None]*masses
            normalizer = joint.sum((-1, -2), keepdim=True)
            if bool((normalizer <= 1e-290).any()):
                raise FloatingPointError("Undefined conditional clearing distribution")
            weights = (joint/normalizer).flatten(1)
            mechanical_failure = 1-normalizer.flatten().clamp(0, 1)
        shifted_grid = grid_r[:, None, None]*span.double()[..., None]*self.grid.double()
        center = torch.logsumexp(reference_w.clamp_min(1e-300).log()+shifted_grid, -1)
        # E_ref[price/median] = P0/M0 at zero micro location.
        log_q = (-median_over_price.log()[:, None, None]
                 + grid_r[:, None, None]*location.double()[..., None]
                 + shifted_grid-center[..., None])
        q_atoms = torch.exp(log_q.clamp(-15, 15))
        if self.mode == "plugin":
            u = (1-supply1[:, None]/mean_bid_ratio).clamp(0, 1-1e-7)
            cum = fractions.cumsum(-1)
            j = (cum <= u[..., None]).sum(-1).clamp(max=self.knots-1)
            j0 = (j-1).clamp_min(0)
            c0 = torch.gather(cum, -1, j0[..., None]).squeeze(-1)
            c0 = torch.where(j == 0, torch.zeros_like(c0), c0)
            c1 = torch.gather(cum, -1, j[..., None]).squeeze(-1)
            q0 = torch.gather(q_atoms, -1, j0[..., None]).squeeze(-1)
            q1 = torch.gather(q_atoms, -1, j[..., None]).squeeze(-1)
            frac = ((u-c0)/(c1-c0).clamp_min(1e-7)).clamp(0, 1)
            q_atoms = (q0+frac*(q1-q0))[..., None]
            weights = weights.reshape(len(x), self.components, self.knots).sum(-1)
        else:
            q_atoms = q_atoms.flatten(1)
        q_atoms = q_atoms.reshape(len(x), -1)
        # Macro median mean is M0*exp(r*mu). The -variance/2 term keeps
        # its arithmetic mean anchored when macro location equals zero.
        macro_sigma = (macro_r*(.05+F.softplus(macro[:, 1].double()+.46))).clamp(1e-5, 1.5)
        median_log_mu_ratio = macro_r*macro[:, 0].double()-.5*macro_sigma.square()
        price_log_mu_ratio = (median_over_price.log()[:, None]
                              + median_log_mu_ratio[:, None]+q_atoms.log())
        price_mean_ratio = (weights*torch.exp(price_log_mu_ratio+.5*macro_sigma[:, None].square())).sum(-1)
        mean = (price_mean_ratio-1)/rprice
        median_mean_z = median_over_price*torch.expm1(macro_r*macro[:, 0].double())/rprice
        basis_atoms_z = (q_atoms-1/median_over_price[:, None])/micro_r[:, None]
        return {"mean": mean, "lognormal_mu_ratio": price_log_mu_ratio,
                "lognormal_sigma": macro_sigma[:, None], "weights": weights,
                "median_log_mu_ratio": median_log_mu_ratio[:, None],
                "median_mean_z": median_mean_z, "basis_atoms_z": basis_atoms_z,
                "basis_mean_z": (weights*basis_atoms_z).sum(-1),
                "mechanical_failure": mechanical_failure,
                "mean_bid_ratio": mean_bid_ratio, "concentration": concentration,
                "mix": mix, "rprice": rprice, "median_over_price": median_over_price}


def _masked_mean(values, mask):
    return values[mask.bool()].mean() if bool(mask.bool().any()) else values.sum()*0


def factorized_loss(output, target, completed, future_bid_ratio, has_bid,
                    median_z, median_ratio, has_median, basis_z, has_basis, cfg):
    good = completed.bool()
    safe_target = torch.where(good, target, torch.zeros_like(target))
    mse = _masked_mean((output["mean"]-safe_target).square(), good)
    if "lognormal_mu_ratio" not in output:
        return mse, {"price_mse": float(mse.detach())}
    yy = 1+output["rprice"]*safe_target.double()
    crps = _masked_mean(lognormal_crps(output["lognormal_mu_ratio"], output["lognormal_sigma"],
                                      output["weights"], yy)/output["rprice"], good)
    mz = torch.where(has_median.bool(), median_z, torch.zeros_like(median_z))
    mr = torch.where(has_median.bool(), median_ratio, torch.ones_like(median_ratio))
    median_mse = _masked_mean((output["median_mean_z"]-mz).square(), has_median)
    median_crps = _masked_mean(lognormal_crps(output["median_log_mu_ratio"], output["lognormal_sigma"],
          torch.ones_like(output["median_log_mu_ratio"]), mr)
          *output["median_over_price"]/output["rprice"], has_median)
    bz = torch.where(has_basis.bool(), basis_z, torch.zeros_like(basis_z))
    basis_mse = _masked_mean((output["basis_mean_z"]-bz).square(), has_basis)
    basis_crps = _masked_mean(atom_crps(output["basis_atoms_z"], output["weights"], bz), has_basis)
    safe_bid = torch.where(has_bid.bool(), future_bid_ratio, torch.ones_like(future_bid_ratio))
    bid_nll = _masked_mean(gamma_mixture_nll(output, safe_bid), has_bid)
    total = (mse+cfg["crps_weight"]*crps
             +cfg["median_auxiliary_weight"]*(median_mse+cfg["crps_weight"]*median_crps)
             +cfg["basis_auxiliary_weight"]*(basis_mse+cfg["crps_weight"]*basis_crps)
             +cfg["future_bid_marginal_nll_weight"]*bid_nll)
    pieces = {"price_mse": mse, "price_crps": crps, "median_mse": median_mse,
              "median_crps": median_crps, "basis_mse": basis_mse,
              "basis_crps": basis_crps, "bid_nll": bid_nll}
    return total, {k: float(v.detach()) for k, v in pieces.items()}
