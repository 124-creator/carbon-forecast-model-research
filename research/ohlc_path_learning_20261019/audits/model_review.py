"""Independent mathematical/numerical audit. Synthetic inputs only; no fit/data.

This does not certify economic identification or forecasting improvement.
Run with the project's Python, from any working directory.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reviewed_model", ROOT / "model.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(429)
RESULTS = {}


def close(name, got, expected, atol=1e-10, rtol=1e-10):
    torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
    RESULTS[name] = {"passed": True, "maximum_absolute_difference": float((got - expected).detach().abs().max())}


def inputs():
    return (torch.randn(2, 20, 8), torch.randn(2, 2), torch.tensor([20., 65.]),
            torch.tensor([.02, .04]), torch.tensor([.1, -.2]), torch.tensor([.7, 1.2]))


def audit_state():
    model = M.make_model("summary_state")
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor([.3, -.25, .4, .2, -.5, .2, -.3, .5, -.3, .2, .3, -.4, .7]))
    args = inputs()
    x, context, p0, scale, basis0, radius0 = args
    eps = M.sobol_normal(128, 52)
    got = model(*args, eps)
    v = model.head(model.encoder(x, context))
    drift = .5 * torch.tanh(v[:, 0]); z = 2 * torch.tanh(v[:, 1])
    rho = .98 * torch.sigmoid(v[:, 2] + 1); loading = .5 * torch.tanh(v[:, 3] + .2)
    sigma = .05 + 1.95 * torch.sigmoid(v[:, 4])
    zmean = z.clone(); zpath = z[:, None].expand(-1, len(eps))
    summean = torch.zeros_like(z); sumstate = torch.zeros_like(zpath)
    varz = torch.zeros_like(z); varsum = torch.zeros_like(z); cov_z_sum = torch.zeros_like(z)
    price_noise = torch.zeros_like(zpath)
    refmeanlog, refvar, refprice = [], [], []
    for j in range(20):
        zmean = rho * zmean
        zpath = rho[:, None] * zpath + torch.sqrt(1 - rho.square())[:, None] * eps[None, :, j, 0]
        summean = summean + zmean; sumstate = sumstate + zpath
        cov_new_oldsum = rho * cov_z_sum
        varz = rho.square() * varz + 1 - rho.square()
        varsum = varsum + varz + 2 * cov_new_oldsum
        cov_z_sum = cov_new_oldsum + varz
        ml = scale * (drift * (j + 1) + loading * summean)
        vv = scale.square() * (loading.square() * varsum + sigma.square() * (j + 1))
        price_noise = price_noise + eps[None, :, j, 1]
        lp = scale[:, None] * (drift[:, None] * (j + 1) + loading[:, None] * sumstate + sigma[:, None] * price_noise) - .5 * vv[:, None]
        refmeanlog.append(ml); refvar.append(vv); refprice.append(p0[:, None] * lp.exp())
    close("state_meanlog_independent_recurrence", got["meanlog"], torch.stack(refmeanlog, 1))
    close("state_variance_independent_recurrence", got["variance"], torch.stack(refvar, 1))
    close("state_price_independent_recurrence", got["daily"][0], torch.stack(refprice, 2))
    close("state_analytic_mean", got["mean"], p0[:, None] * torch.stack(refmeanlog, 1)[:, [0, 4, 19]].exp())
    assert int(got["clip_count"]) == 0
    return model, args


def audit_free():
    model = M.make_model("free_path")
    with torch.no_grad():
        model.head.bias.copy_(torch.linspace(-.4, .6, 180))
    args = inputs(); x, context, p0, scale, *_ = args
    eps = M.sobol_normal(128, 52); got = model(*args, eps)
    v = model.head(model.encoder(x, context)).reshape(2, 20, 9)
    mu = .5 * v[:, :, 0].tanh(); loading = .5 * (v[:, :, 1] + .2).tanh()
    sigma = .05 + 1.95 * v[:, :, 2].sigmoid()
    ml = torch.zeros(2); load_sum = torch.zeros(2); independent_var = torch.zeros(2)
    noise = torch.zeros(2, len(eps)); refmeanlog, refvar, refprice = [], [], []
    for j in range(20):
        ml = ml + scale * mu[:, j]
        load_sum = load_sum + loading[:, j]
        independent_var = independent_var + sigma[:, j].square()
        noise = noise + scale[:, None] * (loading[:, j, None] * eps[None, :, 0, 0] + sigma[:, j, None] * eps[None, :, j, 1])
        vv = scale.square() * (load_sum.square() + independent_var)
        refmeanlog.append(ml); refvar.append(vv); refprice.append(p0[:, None] * (ml[:, None] + noise - .5 * vv[:, None]).exp())
    close("free_meanlog_independent_recurrence", got["meanlog"], torch.stack(refmeanlog, 1))
    close("free_variance_independent_recurrence", got["variance"], torch.stack(refvar, 1))
    close("free_price_independent_recurrence", got["daily"][0], torch.stack(refprice, 2))
    assert int(got["clip_count"]) == 0
    return model, args


def audit_summary_and_score():
    p = torch.arange(1., 21.).reshape(1, 1, 20)
    hi = torch.tensor([9., 11., 7., 3., 6., 21., 5., 7., 11., 10., 9., 8., 7., 12., 7., 6., 5., 4., 3., 2.]).reshape(1, 1, 20)
    lo = hi - 1
    expected = torch.tensor([[[[1., 9., 8.], [5., 11., 2.], [20., 21., 1.]]]])
    close("future_only_windows_1_5_20", M.summaries(p, hi, lo), expected)
    samples = torch.tensor([[[1., 2., 3.], [2., 0., 4.], [1., 2., 3.], [5., 4., 0.]]], requires_grad=True)
    truth = torch.tensor([[2., 100., 1.]])
    mask = torch.tensor([[1., 0., 1.]])
    projected = [[r[0], r[2]] for r in samples.detach()[0].tolist()]
    distance = lambda a, b: math.sqrt(sum((x-y)**2 for x, y in zip(a,b)))
    expected_es = (sum(distance(r, [2., 1.]) for r in projected) / 4 - .5 * sum(distance(a,b) for a in projected for b in projected) / 16) / math.sqrt(2)
    score = M.empirical_energy(samples, truth, mask)
    close("energy_exact_equal_weight_V_statistic", score, torch.tensor([expected_es]))
    score.sum().backward(); assert torch.isfinite(samples.grad).all()
    RESULTS["energy_duplicate_support_gradient_finite"] = {"passed": True}
    alternative = truth.clone(); alternative[:, 1] = -100000.
    close("masked_finite_coordinate_invariance", score.detach(), M.empirical_energy(samples, alternative, mask).detach())
    truth[:, 1] = float("nan")
    RESULTS["masked_nan_requires_runner_finite_placeholder"] = {"nan_propagates": bool(torch.isnan(M.empirical_energy(samples, truth, mask)).any()), "contract": "All unavailable targets require finite placeholders before objective; mask multiplication alone is insufficient."}


def audit_ablation_and_clip(model, args):
    eps = M.sobol_normal(32, 51); out = model(*args, eps)
    y = out["samples"][:, 0].detach().clone(); valid = torch.ones_like(y); denominator = torch.ones(3, 3)
    y2 = y.clone(); y2[:, 1:] += 20
    close("step1_ignores_h5_h20_targets", M.objective(out,y,valid,denominator,"state_step1")[0], M.objective(out,y2,valid,denominator,"state_step1")[0])
    y2 = y.clone(); y2[:, :, 1:] += 20
    close("state_price_ignores_future_range_targets", M.objective(out,y,valid,denominator,"state_price")[0], M.objective(out,y2,valid,denominator,"state_price")[0])
    for p, h, l in [out["daily"]]:
        assert (p > 0).all() and (h >= l).all() and (l > 0).all()
    assert (out["samples"][:, :, 1:, 1] >= out["samples"][:, :, :-1, 1]).all()
    assert (out["samples"][:, :, 1:, 2] <= out["samples"][:, :, :-1, 2]).all()
    RESULTS["positivity_range_order_and_nested_windows"] = {"passed": True}
    stressed = list(args); stressed[3] = torch.full((2,), 4.)
    extreme = model(*stressed, eps)
    assert int(extreme["clip_count"]) > 0
    RESULTS["clip_monitor_fires_under_stress"] = {"passed": True, "count": int(extreme["clip_count"]), "contract": "Any production clip invalidates analytic/sample agreement; abort or invalidate run, do not silently accept clipped sampling."}


def audit_price_context():
    model = M.make_model("gru_price")
    with torch.no_grad():
        model.head.weight.normal_(0, .2)
    args = list(inputs()); changed = [a.clone() for a in args]
    changed[0][:, :, 1:] = torch.randn_like(changed[0][:, :, 1:]) * 50
    close("gru_price_masks_nonprice_history_features", model(*args)["mean"], model(*changed)["mean"])
    changed = [a.clone() for a in args]; changed[1] += 2
    delta = float((model(*args)["mean"] - model(*changed)["mean"]).detach().abs().max())
    assert delta > 1e-8
    RESULTS["gru_price_context_requires_price_only_features"] = {"context_changes_predictions": True, "maximum_difference": delta, "contract": "Runner must construct both context columns solely from historical price; model forwards context unchanged."}
    with torch.no_grad():
        model.head.weight.zero_(); model.head.bias.fill_(10000.)
        clipped = model(*args)
    assert int(clipped["clip_count"]) == 6
    RESULTS["gru_numerical_clip_monitor_fires"] = {"passed": True, "count": int(clipped["clip_count"])}
    positive_radius = torch.tensor([1e-4, .7, 30., 100., 1000.])
    close("inverse_softplus_has_no_upper_cap", torch.nn.functional.softplus(M.inverse_softplus(positive_radius)), positive_radius)
    close("zero_radius_has_declared_finite_proxy", torch.nn.functional.softplus(M.inverse_softplus(torch.zeros(1))), torch.full((1,),1e-5))


def main():
    state, stateargs = audit_state(); free, freeargs = audit_free()
    audit_summary_and_score(); audit_ablation_and_clip(state, stateargs); audit_price_context()
    for name, model, args in [("state", state, stateargs), ("free", free, freeargs)]:
        with torch.no_grad():
            large = model(*args, M.sobol_normal(32768, 139))
        relative_error = ((large["samples"][:, :, :, 0].mean(1) - large["mean"]) / large["mean"]).abs().max().item()
        assert relative_error < .0005, (name, relative_error)
        RESULTS[name + "_independent_Sobol_mean_check"] = {"passed": True, "maximum_relative_error": relative_error, "nodes": 32768, "note": "Numerical integration check, not an unbiasedness claim or a stochastic hypothesis test."}
    report = {"model_sha256": hashlib.sha256((ROOT / "model.py").read_bytes()).hexdigest(), "synthetic_only": True, "results": RESULTS}
    target = Path(__file__).with_suffix(".json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
