"""Bounded independent numerical checks using synthetic values only.

No market CSV, confirmation data, fitting, optimizer, or model selection is
used. NumPy draws independent Gamma atom volumes and implements descending
clearing directly. Public receipts identify the exact checked model file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import scipy
from scipy import special
import torch

import clearing_models as model

ROOT = Path(__file__).resolve().parent
N_SIM = 300_000
SEED = 2026101821
torch.set_num_threads(2)


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def tensor(x, dtype=torch.float64):
    return torch.as_tensor(x, dtype=dtype)


def scipy_mass(fraction, concentration, mean_volume, supply):
    shape = np.asarray(fraction) * np.asarray(concentration)[..., None]
    tail = np.cumsum(shape[..., ::-1], axis=-1)[..., ::-1]
    x = np.asarray(supply) * np.asarray(concentration) / np.asarray(mean_volume)
    survival_inclusive = special.gammaincc(tail, x[..., None])
    survival_exclusive = np.concatenate([survival_inclusive[..., 1:], np.zeros_like(survival_inclusive[..., :1])], axis=-1)
    return survival_inclusive - survival_exclusive, 1-survival_inclusive[..., 0]


def simulate_counts(rng, fraction, concentration, mean_volume, supply, n):
    """Independent volume draws; choose first descending cumulative >= S."""
    volumes = rng.gamma(np.asarray(fraction) * concentration, mean_volume / concentration,
                        size=(n, len(fraction)))
    descending = np.cumsum(volumes[:, ::-1], axis=-1)
    success = descending[:, -1] >= supply
    first = np.argmax(descending >= supply, axis=-1)
    price_index = len(fraction) - 1 - first
    counts = np.bincount(price_index[success], minlength=len(fraction))
    return counts, int((~success).sum())


def compare_sampling(observed, expected, n, label):
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    error = np.abs(observed - expected)
    allowance = 6 * np.sqrt(expected * (1 - expected) / n) + 5 / n
    require(np.all(error <= allowance), f"{label}: Monte Carlo discrepancy exceeds fixed six-SE + five-count allowance")
    return {"max_absolute_probability_error": float(error.max()),
            "max_fraction_of_fixed_allowance": float((error / allowance).max()),
            "observed": observed.tolist(), "formula": expected.tolist(),
            "draws": int(n), "criterion": "Each cell: abs error <= 6*sqrt(p*(1-p)/n)+5/n; fixed before simulation."}


def normalization_and_scipy():
    rng = np.random.default_rng(SEED + 1)
    fractions = rng.dirichlet(np.ones(17) * 0.6, size=(12, 3))
    concentration = rng.uniform(1, 40, size=(12, 3))
    mean_volume = np.exp(rng.uniform(-2.8, 2.8, size=(12, 3)))
    supply = np.exp(rng.uniform(-2.5, 1.2, size=12))
    reference_m, reference_f = scipy_mass(fractions, concentration, mean_volume, supply[:, None])
    results = {}
    # Revised before the second run by root: the initial 2e-11 float64
    # criterion failed on a verified 3.62e-10 torch.special.gammainc error.
    # The initial receipt is retained; this is a disclosed numerical tolerance,
    # not a changed mass formula or a claim of bitwise special-function identity.
    for dtype, tolerance in ((torch.float64, 1e-8), (torch.float32, 2e-6)):
        m, f = model.clearing_masses(tensor(fractions, dtype), tensor(concentration, dtype),
                                     tensor(mean_volume, dtype), tensor(supply, dtype))
        mm, ff = m.detach().numpy(), f.detach().numpy()
        normalization = float(np.max(np.abs(mm.sum(-1) + ff - 1)))
        error = max(float(np.max(np.abs(mm - reference_m))), float(np.max(np.abs(ff - reference_f))))
        require(np.isfinite(mm).all() and np.isfinite(ff).all(), "Nonfinite masses/failure")
        require((mm >= 0).all() and (ff >= 0).all() and (ff <= 1).all(), "Invalid probabilities")
        require(normalization <= tolerance and error <= tolerance, "Gamma mass normalization or SciPy disagreement")
        results[str(dtype)] = {"max_normalization_error": normalization, "max_scipy_error": error, "tolerance": tolerance}
    return {"shape": [12, 3, 17], "checks": results}


def deterministic_strict_boundary():
    prices = np.array([10., 20., 30.])
    volumes = np.ones(3)
    cases = []
    for supply, expected in ((1., 30.), (2., 20.), (3., 10.), (3.1, None)):
        crossed = np.cumsum(volumes[::-1]) >= supply
        direct = float(prices[::-1][np.argmax(crossed)]) if crossed.any() else None
        require(direct == expected, "Independent descending boundary failed")
        # Structural verification of production inclusive/exclusive tail wiring:
        # deterministic demand D has Pr(D < S)=1{D<S}; Gamma continuity does
        # not apply at atoms. This is explicitly a substituted CDF, not a claim
        # that the stochastic Gamma law equals a deterministic limit at a tie.
        if hasattr(model,'_GammaSurvival'):
            substituted = patch.object(model._GammaSurvival,'apply',side_effect=lambda a,x:(a>=x).double())
        else:
            substituted = patch.object(model,'gamma_cdf',side_effect=lambda a,x:(a<x).to(a.dtype))
        with substituted:
            mass, fail = model.clearing_masses(tensor([[[1/3, 1/3, 1/3]]]), tensor([[3.]]), tensor([[3.]]), tensor([supply]))
        observed = mass.numpy().ravel()
        desired = np.zeros(3)
        if expected is not None:
            desired[np.where(prices == expected)[0][0]] = 1
        require(np.array_equal(observed, desired), "Production tail indexing violates strict deterministic boundary")
        require(float(fail.item()) == float(expected is None), "Failure boundary mismatch")
        cases.append({"supply": supply, "expected_clearing_price": expected, "price_masses": observed.tolist(), "failure": float(fail.item())})
    return {"prices": prices.tolist(), "volumes": volumes.tolist(), "cases": cases,
            "scope": "Direct deterministic descending clearing plus production tail-index structure under substituted deterministic CDF. The Gamma model itself is continuous."}


def single_gamma_simulation():
    fraction = np.array([.10, .20, .30, .40])
    concentration, mean_volume, supply = 2.7, 1.3, .9
    counts, fail_count = simulate_counts(np.random.default_rng(SEED + 2), fraction, concentration, mean_volume, supply, N_SIM)
    mass, fail = model.clearing_masses(tensor(fraction[None, None]), tensor([[concentration]]), tensor([[mean_volume]]), tensor([supply]))
    expected = np.r_[mass.numpy().ravel(), fail.item()]
    return compare_sampling(np.r_[counts, fail_count] / N_SIM, expected, N_SIM, "single Gamma")


def mixed_gamma_simulation():
    fractions = np.array([[.10, .20, .30, .40], [.40, .30, .20, .10], [.15, .25, .35, .25]])
    concentrations = np.array([.9, 3.2, 11.])
    means = np.array([.32, 1.0, 2.0])
    mix = np.array([.55, .30, .15])
    supply = 1.15
    rng = np.random.default_rng(SEED + 3)
    labels = rng.choice(3, N_SIM, p=mix)
    cell_counts = np.zeros((3, 4), dtype=int)
    failure_counts = np.zeros(3, dtype=int)
    for c in range(3):
        cell_counts[c], failure_counts[c] = simulate_counts(rng, fractions[c], concentrations[c], means[c], supply, int((labels == c).sum()))
    masses, failure = model.clearing_masses(tensor(fractions[None]), tensor(concentrations[None]), tensor(means[None]), tensor([supply]))
    mm, ff = masses.numpy()[0], failure.numpy()[0]
    joint = mix[:, None] * mm
    mixture_failure = float(mix @ ff)
    unconditional = compare_sampling(np.r_[cell_counts.ravel(), failure_counts.sum()] / N_SIM,
                                     np.r_[joint.ravel(), mixture_failure], N_SIM, "mixed Gamma unconditional")
    success_count = int(cell_counts.sum())
    posterior = joint / joint.sum()
    conditional = compare_sampling(cell_counts.ravel() / success_count, posterior.ravel(), success_count, "mixed Gamma conditioned after mixing")
    wrong = mix[:, None] * (mm / mm.sum(-1, keepdims=True))
    wrong_gap = float(np.abs(wrong - posterior).sum())
    require(wrong_gap > .1, "Chosen synthetic case does not distinguish conditioning order")
    observed_wrong_gap = float(np.abs(cell_counts / success_count - wrong).sum())
    require(observed_wrong_gap > .1, "Simulation does not reject equal component weights after separate conditioning")
    # Verify actual DemandNetwork weighting with a synthetic fixed head.
    net = model.DemandNetwork(2, 2, hidden=4, components=3, knots=4).double()
    with torch.no_grad():
        net.head.weight.zero_()
        raw = net.head.bias.reshape(3, 9)
        raw.zero_()
        raw[:, 0] = tensor(np.log(mix))
        raw[:, 1] = tensor(np.log(means))
        # Network concentrations are restricted to (1,40); use a second exact
        # network case for weighting, retaining the unrestricted operator MC.
        target_a = np.array([1.2, 3.2, 11.])
        prob = (target_a - 1) / 39
        raw[:, 2] = tensor(np.log(prob / (1-prob)) + .9)
        raw[:, 5:] = tensor(np.log(fractions))
    output = net(torch.zeros(1,20,2,dtype=torch.float64), torch.zeros(1,2,dtype=torch.float64), tensor([[.4,supply,.02]]))
    nm, nf = scipy_mass(fractions, target_a, means, supply)
    expected_network = mix[:,None]*nm
    expected_network /= expected_network.sum()
    network_error = float(np.max(np.abs(output['weights'].detach().numpy().reshape(3,4)-expected_network)))
    require(network_error < 1e-8, "DemandNetwork conditions components before unconditional mixing")
    return {"unconditional": unconditional, "conditional": conditional,
            "component_success_posterior": posterior.sum(-1).tolist(), "prior_component_mix": mix.tolist(),
            "L1_wrong_conditioning_order_gap": wrong_gap,
            "L1_empirical_to_wrong_conditioning": observed_wrong_gap,
            "network_weight_max_error": network_error}


def reference_center():
    records = {}
    for dtype, tolerance in ((torch.float64, 2e-11), (torch.float32, 1e-4)):
        net = model.DemandNetwork(3, 2, hidden=5, components=3, knots=17).to(dtype)
        with torch.no_grad():
            net.head.weight.zero_()
            raw = net.head.bias.reshape(3,22)
            raw.zero_()
            raw[:,0] = tensor([-.6,.4,.1], dtype)
            raw[:,2] = tensor([-2.,0.,2.], dtype)
            raw[:,5:] = tensor(np.vstack([np.linspace(-1,1,17),np.linspace(1,-1,17),np.sin(np.arange(17))]),dtype)
        supply = tensor([.05,.2,.5,1.,1.5], dtype)
        relative_scale=tensor([.002,.005,.02,.05,.1],dtype)
        out = net(torch.zeros(5,20,3,dtype=dtype),torch.zeros(5,2,dtype=dtype),torch.stack([supply,supply,relative_scale],-1))
        error = float(out['mean'].abs().max().item())
        relative_price_error=float((out['mean']*relative_scale).abs().max().item())
        require(error < tolerance, "Reference center does not return zero normalized price change")
        require(relative_price_error < (2e-12 if dtype==torch.float64 else 5e-7), "Reference mean relative-price error exceeds precision allowance")
        require(torch.allclose(out['mean_bid_ratio'],torch.ones_like(out['mean_bid_ratio'])), "Reference test meanB differs from lastB")
        records[str(dtype)] = {"max_absolute_normalized_mean": error,"tolerance":tolerance,"max_relative_price_mean_error":relative_price_error}
    return {"condition": "meanB/lastB=1, planned S=last S, location=0; positive log-price grid normalized by reference exponential moment; all synthetic cases retained; law is conditional on mechanical success.","relative_scale_values":[.002,.005,.02,.05,.1],"precision_note":"Small relative_scale magnifies float32 residual-unit roundoff; both normalized and relative-price errors are checked.","checks":records}


def gamma_shape_gradient():
    a, x = np.meshgrid(np.array([1e-5,.001,.03,.2,1.,5.,15.,40.]),
                      np.array([1e-5,.01,.1,.3,1.,3.,10.,30.,60.,160.]),indexing='ij')
    aa=tensor(a.ravel()).requires_grad_(True)
    xx=tensor(x.ravel()).requires_grad_(True)
    value=model._GammaCDF.apply(aa,xx)
    ga,gx=torch.autograd.grad(value.sum(),(aa,xx))
    av,xv=a.ravel(),x.ravel()
    h=np.minimum(1e-5*(1+av),.01*av)
    fine=(special.gammainc(av-2*h,xv)-8*special.gammainc(av-h,xv)+8*special.gammainc(av+h,xv)-special.gammainc(av+2*h,xv))/(12*h)
    h2=h/2
    finer=(special.gammainc(av-2*h2,xv)-8*special.gammainc(av-h2,xv)+8*special.gammainc(av+h2,xv)-special.gammainc(av+2*h2,xv))/(12*h2)
    torch_fine=((torch.special.gammainc(tensor(av+h),tensor(xv))-torch.special.gammainc(tensor(av-h),tensor(xv)))/(2*tensor(h))).numpy()
    actual=ga.detach().numpy()
    tolerance=2e-5+2e-3*np.abs(finer)
    absolute=np.abs(actual-finer)
    require(np.isfinite(actual).all() and np.isfinite(gx.detach().numpy()).all(), "Nonfinite Gamma derivative")
    require(np.all(absolute<=tolerance), "Shape finite-difference derivative exceeds fixed atol 2e-5 + rtol .002")
    dx=np.exp((av-1)*np.log(xv)-xv-special.gammaln(av))
    require(np.allclose(gx.detach().numpy(),dx,atol=1e-10,rtol=2e-11),"Analytic Gamma x derivative mismatch")
    significant=np.abs(finer)>1e-5
    worst=int(np.argmax(absolute/tolerance))
    return {"points":len(av),"shape_range":[float(av.min()),float(av.max())],"x_range":[float(xv.min()),float(xv.max())],
            "max_absolute_shape_error":float(absolute.max()),
            "max_relative_shape_error_where_reference_above_1e_minus5":float((absolute[significant]/np.abs(finer[significant])).max()),
            "max_fraction_of_fixed_allowance":float((absolute/tolerance).max()),
            "fine_vs_finer_scipy_max_difference":float(np.abs(fine-finer).max()),
            "torch64_fine_vs_scipy_finer_max_difference":float(np.abs(torch_fine-finer).max()),
            "worst_allowance_case":{"shape":float(av[worst]),"x":float(xv[worst]),"actual":float(actual[worst]),"fine_reference":float(finer[worst])},
            "criterion":"All 80 shape derivatives: abs error <= 2e-5 + .002*abs(reference); x derivative rtol 2e-11, atol 1e-10.",
            "limitation":"Checked first derivatives on a finite operational/stress grid; no claim of exact shape AD or higher-order gradients."}


def crps_pairwise():
    rng=np.random.default_rng(SEED+4)
    atoms=rng.normal(size=(9,13))*rng.uniform(.2,3,size=(9,1))
    atoms[:,3]=atoms[:,2]
    weights=rng.dirichlet(np.ones(13)*.6,size=9)
    weights[0,:]=0;weights[0,5]=1
    y=rng.normal(size=9)
    reference=np.sum(weights*np.abs(atoms-y[:,None]),axis=-1)-.5*np.sum(weights[:,:,None]*weights[:,None,:]*np.abs(atoms[:,:,None]-atoms[:,None,:]),axis=(-1,-2))
    actual=model.atom_crps(tensor(atoms),tensor(weights),tensor(y)).numpy()
    error=float(np.max(np.abs(actual-reference)))
    require(error<2e-12 and (actual>=-2e-12).all(),"Finite-atom CRPS differs from independent O(K^2) definition")
    return {"cases":9,"atoms_per_case":13,"unsorted_unequal_grid":True,"duplicate_atoms":True,"point_mass_case":True,"max_absolute_error":error}


def gamma_survival_shape_gradient():
    aa,xx=np.meshgrid(np.array([1e-5,.001,.03,.2,1.,5.,15.,40.]),
                      np.array([1e-5,.01,.1,.3,1.,3.,10.,30.,60.,160.]),indexing='ij')
    av=np.r_[aa.ravel(),40.];xv=np.r_[xx.ravel(),400.]
    a=tensor(av).requires_grad_(True);x=tensor(xv).requires_grad_(True)
    value=model._GammaSurvival.apply(a,x)
    ga,gx=torch.autograd.grad(value.sum(),(a,x))
    h=np.minimum(1e-5*(1+av),.01*av)
    def five_point(step):
        return (special.gammaincc(av-2*step,xv)-8*special.gammaincc(av-step,xv)+8*special.gammaincc(av+step,xv)-special.gammaincc(av+2*step,xv))/(12*step)
    fine,finer=five_point(h),five_point(h/2)
    torch_fine=((torch.special.gammaincc(tensor(av+h),tensor(xv))-torch.special.gammaincc(tensor(av-h),tensor(xv)))/(2*tensor(h))).numpy()
    actual=ga.detach().numpy();absolute=np.abs(actual-finer)
    require(np.isfinite(actual).all() and np.isfinite(gx.detach().numpy()).all(),'Nonfinite Gamma survival derivative')
    require(np.all(absolute<=2e-5+.002*np.abs(finer)),'SF shape error exceeds fixed absolute/relative allowance')
    sf=special.gammaincc(av,xv)
    identifiable=(np.abs(finer)>1e-5)|((sf<.1)&(np.abs(finer)>1e-200))
    relative=absolute[identifiable]/np.abs(finer[identifiable])
    require((relative<=.002).all(),'SF shape derivative relative error exceeds .002 in resolved tails')
    dx=-np.exp((av-1)*np.log(xv)-xv-special.gammaln(av))
    require(np.allclose(gx.detach().numpy(),dx,atol=1e-10,rtol=2e-11),'SF analytic x derivative mismatch')
    return {'points':len(av),'max_absolute_shape_error':float(absolute.max()),'max_relative_shape_error_resolved':float(relative.max()),
            'fine_vs_finer_scipy_max_difference':float(np.abs(fine-finer).max()),'torch64_fine_vs_scipy_finer_max_difference':float(np.abs(torch_fine-finer).max()),
            'tiny_tail_case':{'shape':40.,'x':400.,'survival':float(sf[-1]),'actual_shape_derivative':float(actual[-1]),'fine_reference':float(finer[-1]),'relative_error':float(absolute[-1]/abs(finer[-1]))},
            'criterion':'Same 80-point grid plus the requested tiny-tail point; absolute 2e-5+.002*|reference| and relative .002 where derivative is resolved (including small SF tails).'}


def tiny_tail_survival():
    records={}
    for dtype in (torch.float64,torch.float32):
        fraction=tensor(np.full((1,3,17),1/17),dtype)
        concentration=tensor([[40.]*3],dtype);mean=tensor([[.3]*3],dtype);supply=tensor([3.],dtype)
        mass,fail=model.clearing_masses(fraction,concentration,mean,supply)
        require(mass.dtype==torch.float64,'Tiny masses are not retained in float64')
        success=mass.sum(-1)
        require(torch.isfinite(mass).all() and (success>0).all(),'Tiny mechanical success was silently rounded to zero')
        weights=mass/success[...,None]
        require(torch.allclose(weights.sum(-1),torch.ones_like(success),atol=2e-12,rtol=0),'Tiny conditional weights are not normalized')
        net=model.DemandNetwork(3,2,hidden=4,components=3,knots=17).to(dtype)
        with torch.no_grad():
            net.head.weight.zero_();raw=net.head.bias.reshape(3,22);raw.zero_()
            raw[:,1]=np.log(.3);raw[:,2]=100
            net.supply_to_bid_volume.weight.zero_()
        out=net(torch.zeros(1,20,3,dtype=dtype),torch.zeros(1,2,dtype=dtype),tensor([[.3,3.,.02]],dtype))
        require(torch.isfinite(out['weights']).all() and abs(out['weights'].sum().item()-1)<2e-12,'Network tiny-tail conditioning failed')
        score=model.atom_crps(out['atoms'],out['weights'],torch.zeros(1,dtype=dtype))
        require(torch.isfinite(score).all() and score.item()>0,'Tiny-tail case produced false zero/nonfinite CRPS')
        score.sum().backward()
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()),'Tiny-tail CRPS backward is nonfinite')
        records[str(dtype)]={'success_mass':float(success[0,0]),'network_weight_sum':float(out['weights'].sum()),'synthetic_crps':float(score.item()),'first_backward_finite':True}
    return {'requested_case':{'supply_over_lastB':3.,'concentration':40.,'meanB_over_lastB':.3},'checks':records,
            'interpretation':'Failure can round to 1; price weights must still normalize direct float64 survival masses, never 1-failure.'}


def mode_shapes_and_finiteness():
    torch.manual_seed(SEED)
    records={}
    for mode in ('stochastic','direct','plugin','no_supply'):
        net=model.DemandNetwork(5,3,hidden=8,components=3,knots=7,mode=mode).double()
        x=torch.randn(4,20,5,dtype=torch.float64)*.2
        ctx=torch.randn(4,3,dtype=torch.float64)*.2
        base=tensor([[.1,.2,.005],[.4,.6,.02],[.8,1.8,.05],[1.2,.9,.1]])
        out=net(x,ctx,base)
        for key,value in out.items():
            require(torch.isfinite(value).all(),f'{mode}: nonfinite {key}')
        size=3 if mode=='plugin' else 21
        require(out['mean'].shape==(4,) and out['atoms'].shape==(4,size) and out['weights'].shape==(4,size),f'{mode}: wrong output dimensions')
        require((1+base[:,2,None]*out['atoms']>0).all(),f'{mode}: nonpositive implied price atom')
        require(torch.allclose(out['weights'].sum(-1),torch.ones(4,dtype=torch.float64),atol=2e-12,rtol=0),f'{mode}: conditional weights not normalized')
        loss,_=model.composite_loss(out,tensor([.1,-.2,float('nan'),.4]),tensor([1,1,0,1]),tensor([.8,1.1,.3,1.6]),tensor([1,1,1,1]))
        require(torch.isfinite(loss),f'{mode}: nonfinite synthetic objective')
        loss.backward()
        grads=[p.grad for p in net.parameters() if p.grad is not None]
        require(all(torch.isfinite(g).all() for g in grads),f'{mode}: nonfinite first backward')
        records[mode]={'mean_shape':list(out['mean'].shape),'atom_shape':list(out['atoms'].shape),'weight_shape':list(out['weights'].shape),'synthetic_first_backward_finite':True,'optimizer_steps':0}
    return {'modes':records,'plugin_scope':'Plugin is the explicitly smooth inverse of mean demand with stochastic success weights; not exact discrete deterministic clearing.'}


def missing_bid_mask_backward():
    outcomes=[]
    for dtype in (torch.float64,torch.float32):
        for all_missing in (False,True):
            torch.manual_seed(SEED+5)
            net=model.DemandNetwork(5,3,hidden=8,components=3,knots=7).to(dtype)
            out=net(torch.zeros(3,20,5,dtype=dtype),torch.zeros(3,3,dtype=dtype),tensor([[.3,.4,.02]]*3,dtype))
            target=tensor([.1,float('nan'),.2],dtype)
            completed=torch.tensor([True,False,True])
            bid=tensor([float('nan')]*3 if all_missing else [.8,float('nan'),1.2],dtype)
            has_bid=torch.tensor([False]*3 if all_missing else [True,False,True])
            loss,_=model.composite_loss(out,target,completed,bid,has_bid)
            require(torch.isfinite(loss),'Missing-bid masking produced a nonfinite synthetic loss')
            loss.backward()
            bad=[name for name,p in net.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            require(not bad,'Missing bid marked has_bid=False still creates nonfinite gradients: '+','.join(bad))
            outcomes.append({'dtype':str(dtype),'all_bid_labels_missing':all_missing,'finite_loss':True,'all_parameter_gradients_finite':True,'optimizer_steps':0})
    return {'cases':outcomes,'initial_failure_receipt':'results/MISSING_BID_MASK_INITIAL.json',
            'remediation':'Replace unavailable B labels with a finite dummy through where(has_bid, label, 1) BEFORE evaluating Gamma log density; then mask loss.',
            'scope':'Synthetic missing-label boundary. Discovered and corrected before real training or market performance scoring.'}


def main():
    started=time.perf_counter()
    source=ROOT/'clearing_models.py'
    original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    checks={}
    for name,func in [('gamma_normalization_and_scipy',normalization_and_scipy),('strict_three_bid_boundary',deterministic_strict_boundary),
                      ('single_gamma_monte_carlo',single_gamma_simulation),('mixed_gamma_monte_carlo_and_conditioning',mixed_gamma_simulation),
                      ('reference_center_rw',reference_center),('gamma_shape_gradient',gamma_shape_gradient),
                      ('atom_crps_quadratic_definition',crps_pairwise),('mode_dimensions_and_finiteness',mode_shapes_and_finiteness),
                      ('missing_bid_mask_finite_backward',missing_bid_mask_backward),
                      ('gamma_survival_shape_gradient',gamma_survival_shape_gradient),('tiny_tail_nonzero_conditional_weights',tiny_tail_survival)]:
        try:
            checks[name]={'status':'PASS',**func()}
        except Exception as exc:
            checks[name]={'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)}
    after_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    overall=all(x['status']=='PASS' for x in checks.values()) and original_hash==after_hash
    output={'overall_status':'PASS' if overall else 'FAIL','clearing_models_sha256':original_hash,'model_file_unchanged_during_check':original_hash==after_hash,
            'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__},
            'torch_threads':torch.get_num_threads(),'rng_seed_base':SEED,'gamma_simulations_single':N_SIM,'gamma_simulations_mixture':N_SIM,
            'scope':{'synthetic_only':True,'market_rows_read':0,'confirmation_rows_read':0,'optimizer_steps':0,'real_training_runs':0},
            'checks':checks,'elapsed_seconds':time.perf_counter()-started,'current_operator_form':'Upper Gamma survival differences; float64 masses retained through mixture conditioning.',
            'initial_attempt':{'receipt':'results/OPERATOR_CHECK_INITIAL.json','retained':(ROOT/'results/OPERATOR_CHECK_INITIAL.json').exists(),
                               'reason_for_repeat':'Initial old two-column synthetic base no longer matched the updated three-column positive-price interface. Initial float64 CDF tolerance 2e-11 was stricter than the verified torch kernel accuracy; root set 1e-8 float64 and 2e-6 float32 before rerun.',
                               'mpmath70_worst_point_check':{'shape':25.00486062035667,'x':25.82209242451278,'torch_absolute_error':3.6235521467764525e-10,'scipy_absolute_error':4.223332811711944e-17}}}
    dest=ROOT/'results/OPERATOR_CHECK.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'overall_status':output['overall_status'],'model_sha256':original_hash,'checks':{k:v['status'] for k,v in checks.items()},'failures':{k:v for k,v in checks.items() if v['status']!='PASS'},'elapsed_seconds':output['elapsed_seconds']},ensure_ascii=False))
    if not overall: raise SystemExit(2)


if __name__=='__main__': main()
