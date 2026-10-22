"""R21 positive variance Markov state with full shape/radial Gamma inference.

Default execution is guarded pure synthetic checks. No real fitting or 2026
confirmation is part of this delivery. A later explicit --run-development fits
only reused2025 development. Any reused2026 diagnostic is NOT a fresh holdout;
a new sample is required for future confirmation. Effective M, nominal S and
volume/risk independence are working hypotheses, not true N/IV identification.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sampling_density_gamma_reference as REF
import round20_observation_operator_gru as R20
import holdout_2026_range_builder as HB

S17, B, ASSETS, DTYPE = R20.S17, R20.B, R20.ASSETS, R20.DTYPE
HERE = Path(__file__).resolve().parent
BASES = ('global_dense', 'shape_dense', 'fixed_mixture')
MODES = BASES + ('neural_density',)
SEEDS = (11, 22, 33)
GRID_SIZE, GH_SIZE = 256, 16
MAX_EPOCHS, PATIENCE, MIN_EPOCHS = 80, 15, 20
LR = .01


def raw_proxy(D):
    """Real runner requires original rawP, including unresolved/zero ranges."""
    P = np.asarray(D['proxy_raw'], dtype=float)
    assert P.shape == (D['n'],) and np.isfinite(P).all() and (P >= 0).all()
    assert (P[D['observed']] > 0).all()
    return P


def fit_hurdle(volume, zero):
    """Pre25 current labels only. Half a zero/positive pseudo-observation at v0.

    Minimize mean augmented Bernoulli NLL + .01*slope^2 by damped Newton.
    Pseudocounts anchor the intercept; ridge bounds separation along volume.
    This is fixed preprocessing, never re-estimated from validation or test.
    """
    v = np.asarray(volume, float); y = np.asarray(zero, float)
    assert v.ndim == 1 and y.shape == v.shape and len(v) > 2
    assert np.isfinite(v).all() and np.isin(y, [0, 1]).all()
    X = np.column_stack([np.ones(len(v)+1), np.r_[v, 0.]])
    target = np.r_[y, .5]; weight = np.ones(len(v)+1); denom = len(v)+1
    theta = np.array([math.log((y.sum()+.5)/(len(y)-y.sum()+.5)), 0.])
    def value(t):
        z = X @ t
        return float(np.sum(weight * (np.logaddexp(0,z) - target*z))/denom + .01*t[1]**2)
    steps = 0
    for steps in range(1, 101):
        z = X @ theta; p = np.exp(-np.logaddexp(0,-z))
        grad = X.T @ ((p-target)*weight)/denom + [0., .02*theta[1]]
        H = (X.T * (p*(1-p)*weight)) @ X / denom + np.diag([0., .02])
        step = np.linalg.solve(H, grad)
        rate = 1.; old = value(theta)
        while rate > 2**-30 and value(theta-rate*step) > old:
            rate *= .5
        theta = theta-rate*step
        if np.max(np.abs(rate*step)) < 1e-10:
            break
    assert np.isfinite(theta).all()
    return dict(intercept=float(theta[0]), slope=float(theta[1]), n=len(v), zeros=int(y.sum()),
        ridge=.01, pseudocount_each=.5, pseudocount_volume=0., objective=value(theta), newton_steps=steps,
        meaning='no resolvable range; not zero true volatility or no trading')


def preprocessing(datasets):
    fixed = {}
    ar = R20.fit_volume_ar(datasets)
    for a, D in datasets.items():
        P = raw_proxy(D); cut = D['cut']; scale = float(P[:cut].mean())
        assert scale > 0 and (np.asarray(D['bank_index']) < cut).all()
        hurdle = fit_hurdle(D['F17'][:cut, 2], ~np.asarray(D['observed'][:cut]))
        fixed[a] = dict(scale=scale, scale_n=cut, scale_includes_zero=True, hurdle=hurdle, volume_AR=ar[a])
    return fixed


class VarianceProcess(nn.Module):
    def __init__(self, bounded=None):
        super().__init__()
        self.frozen = bounded is not None
        if bounded is not None:
            for name in ('mu', 'phi', 'q', 'kappa'):
                value = torch.as_tensor(bounded[name], dtype=DTYPE).detach().clone()
                self.register_buffer(name, value)
        else:
            values = dict(rawmu=0., rawphi=math.log((.98-.8)/(.9995-.98)),
                rawq=math.log((.03-.005)/(.25-.03)), rawk=0.)
            for name, value in values.items():
                self.register_parameter(name, nn.Parameter(torch.tensor(value, dtype=DTYPE)))

    def theta(self):
        if self.frozen:
            return self.mu, self.phi, self.q, self.kappa
        return torch.exp(3*torch.tanh(self.rawmu)), .8+.1995*torch.sigmoid(self.rawphi), \
            .005+.245*torch.sigmoid(self.rawq), torch.exp(1.5*torch.tanh(self.rawk))

    def bounded(self):
        return {k: float(v.detach()) for k,v in zip(('mu','phi','q','kappa'), self.theta())}


def initial_ig(grid, mu):
    """IG(shape3,scale2mu) density times Jacobian on equal log-S cells."""
    return torch.softmax(-3*torch.log(grid)-2*mu/grid, dim=0)


def transition_matrix(grid, mu, phi, q, nodes, weights):
    """Positive mass interpolation in S, not log S, with explicit endpoint clip."""
    shock = torch.exp(torch.sqrt(2*q)*nodes-q/2)
    moved = (1-phi)*mu+phi*grid[:,None]*shock[None,:]
    outside = (moved < grid[0]) | (moved > grid[-1])
    clipped = moved.clamp(min=grid[0], max=grid[-1])
    lo = (torch.searchsorted(grid, clipped.contiguous(), right=True)-1).clamp(0,len(grid)-2)
    hi = lo+1
    upper = (clipped-grid[lo])/(grid[hi]-grid[lo])
    assert bool(((upper >= 0)&(upper <= 1)).all()), 'Negative interpolation probability'
    row = torch.arange(len(grid), device=grid.device)[:,None] * len(grid)
    indices = torch.cat([(row+lo).reshape(-1),(row+hi).reshape(-1)])
    mass = torch.cat([((1-upper)*weights).reshape(-1),(upper*weights).reshape(-1)])
    matrix = grid.new_zeros(len(grid)*len(grid)).scatter_add(0,indices,mass).reshape(len(grid),len(grid))
    ideal = mu+phi*(grid-mu)
    actual = matrix @ grid
    quadrature = (moved*weights).sum(1)
    diag = dict(row_clip_mass=(outside.to(grid)*weights).sum(1), row_mean_error=actual-ideal,
        row_quadrature_error=quadrature-ideal, row_interpolation_error=actual-(clipped*weights).sum(1),
        interior_rows=~outside.any(1), lower_shock_mask=moved<grid[0], upper_shock_mask=moved>grid[-1])
    assert bool(torch.isfinite(matrix).all() and (matrix>=0).all())
    torch.testing.assert_close(matrix.sum(1), torch.ones_like(grid), rtol=1e-13, atol=1e-14)
    return matrix, diag


class GammaVarianceFilter(nn.Module):
    def __init__(self, datasets, fixed, reference, mode, seed=11, frozen_theta=None, assets=ASSETS, grid_size=GRID_SIZE):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        if mode=='neural_density' and frozen_theta is None:
            raise ValueError('NN requires independent selected fixed_mixture theta')
        self.mode, self.assets = mode, tuple(assets)
        self.reference = copy.deepcopy(reference)
        self.processes = nn.ModuleDict({a:VarianceProcess(None if frozen_theta is None else frozen_theta[a]) for a in self.assets})
        self.register_buffer('grid',torch.exp(torch.linspace(-8,8,grid_size,dtype=DTYPE)))
        nodes,weights=np.polynomial.hermite.hermgauss(GH_SIZE);weights=weights/weights.sum()
        self.register_buffer('gh_nodes',torch.tensor(nodes,dtype=DTYPE))
        self.register_buffer('gh_weights',torch.tensor(weights,dtype=DTYPE))
        self.register_buffer('logM',torch.log(torch.tensor(REF.MS,dtype=DTYPE)))
        for a in self.assets:
            D=datasets[a]; bank=np.asarray(D['bank_index'])
            assert len(bank) and (bank<D['cut']).all() and D['observed'][bank].all()
            self.register_buffer(f'bank_v_{a}',torch.tensor(D['F17'][bank,2].copy(),dtype=DTYPE))
            self.register_buffer(f'bank_index_{a}',torch.tensor(bank,dtype=torch.long))
            self.register_buffer(f'scale_{a}',torch.tensor(fixed[a]['scale'],dtype=DTYPE))
            self.register_buffer(f'hurdle_{a}',torch.tensor([fixed[a]['hurdle'][k] for k in ('intercept','slope')],dtype=DTYPE))
            self.register_buffer(f'ar_{a}',torch.tensor([fixed[a]['volume_AR'][k] for k in ('mu','rho','residual_std')],dtype=DTYPE))
        torch.manual_seed(seed)
        if mode=='neural_density':
            self.prior_net=nn.Sequential(nn.Linear(1,4),nn.Tanh(),nn.Linear(4,7)).to(dtype=DTYPE)
            nn.init.zeros_(self.prior_net[-1].weight);nn.init.zeros_(self.prior_net[-1].bias)
            assert sum(p.numel() for p in self.parameters())==43

    def positive_log_prior(self, volume):
        if self.mode in ('global_dense','shape_dense'):
            logpi=volume.new_full((*volume.shape,7),-torch.inf);logpi[...,-1]=0
            return logpi
        correction=self.prior_net(volume[...,None]) if self.mode=='neural_density' else torch.zeros((*volume.shape,7),dtype=volume.dtype,device=volume.device)
        return torch.log_softmax(self.logM+correction,dim=-1)

    def hurdle_logs(self,a,volume):
        intercept,slope=getattr(self,f'hurdle_{a}');logit=intercept+slope*volume
        return F.logsigmoid(logit),F.logsigmoid(-logit)

    def effective_c(self,a,volume):
        _,logpositive=self.hurdle_logs(a,volume)
        return torch.exp(logpositive)*(torch.exp(self.positive_log_prior(volume))*self.reference.c_global).sum(-1)

    def c_ref(self,a):
        return self.effective_c(a,getattr(self,f'bank_v_{a}')).mean()

    def filter_asset(self,a,D,T,stop):
        assert 0<stop<=D['n']
        mu,phi,q,kappa=self.processes[a].theta();grid=self.grid
        matrix,transition=transition_matrix(grid,mu,phi,q,self.gh_nodes,self.gh_weights)
        cref=self.c_ref(a);v=T['F17'][:stop,2]
        logzero,logpositive=self.hurdle_logs(a,v);logpi=self.positive_log_prior(v)
        observed=torch.as_tensor(D['observed'][:stop],dtype=torch.bool,device=grid.device)
        ids=torch.nonzero(observed,as_tuple=False).flatten()
        y=T['proxy_raw'][:stop]/getattr(self,f'scale_{a}')
        cells=R20.geometry31(T['F17'][:stop],observed)
        positive_component=grid.new_zeros((len(ids),len(grid),7))
        if len(ids):
            if self.mode=='global_dense':
                positive_component=self.reference.global_log_prob(y[ids],grid,cref,kappa)
            else:
                positive_component=self.reference.joint_log_prob(y[ids],grid,cref,cells[ids],kappa)
            positive_component=positive_component+logpi[ids,None,:]
        lookup=torch.full((stop,),-1,dtype=torch.long,device=grid.device);lookup[ids]=torch.arange(len(ids),device=grid.device)
        posterior=initial_ig(grid,mu)
        means,variances,entropies,posts,priors,marginal,Mpost,clipmass,meanerror,boundary=[],[],[],[],[],[],[],[],[],[]
        for t in range(stop):
            clipmass.append(posterior@transition['row_clip_mass'])
            meanerror.append(posterior@transition['row_mean_error'])
            prior=posterior@matrix
            if bool(observed[t]):
                component=positive_component[lookup[t]]
                logprior=torch.log(prior.clamp_min(1e-300))  # only a logarithm safeguard
                logL=logpositive[t]+torch.logsumexp(component,dim=-1)
                evidence=torch.logsumexp(logprior+logL,dim=0)
                posterior=torch.softmax(logprior+logL,dim=0)
                # Integrate PRIOR(S)*Gamma radial evidence, not shape-only weights.
                mposterior=torch.softmax(torch.logsumexp(logprior[:,None]+component,dim=0),dim=0)
            else:
                evidence=logzero[t]
                posterior=prior  # zero branch advances state but adds no S information
                mposterior=grid.new_zeros(7)  # undefined M for hurdle zeros; use validity mask
            m=posterior@grid;variance=posterior@(grid-m).square()
            means.append(m);variances.append(variance)
            entropies.append(-(posterior*torch.log(posterior.clamp_min(1e-300))).sum())
            priors.append(prior);posts.append(posterior);marginal.append(evidence);Mpost.append(mposterior)
            boundary.append(posterior[0]+posterior[-1])
        return dict(mean=torch.stack(means),variance=torch.stack(variances),entropy=torch.stack(entropies),
            posterior=torch.stack(posts),prior=torch.stack(priors),log_marginal=torch.stack(marginal),
            M_posterior=torch.stack(Mpost),M_posterior_valid=observed,
            posterior_weighted_clip_mass=torch.stack(clipmass),posterior_weighted_mean_error=torch.stack(meanerror),
            boundary_mass=torch.stack(boundary),transition=transition,c_ref=cref,log_prior_M=logpi)

    def future_factor(self,a,current_volume,cref):
        muV,rho,sigma=getattr(self,f'ar_{a}');k=torch.arange(1,23,dtype=DTYPE,device=current_volume.device)
        mean=muV+rho.pow(k)*(current_volume[:,None]-muV)
        variance=sigma.square()*(1-rho.pow(2*k))/(1-rho.square())
        volume=mean[...,None]+torch.sqrt(2*variance)[...,None]*self.gh_nodes
        ceff=self.effective_c(a,volume)
        # Ratio-centered GH makes constant ceff/cref return exact1 (up to ratio roundoff).
        return 1+((ceff/cref-1)*self.gh_weights).sum(-1)

    def forecast(self,a,T,origins,filtered):
        mu,phi,_,_=self.processes[a].theta();k=torch.arange(1,23,dtype=DTYPE,device=self.grid.device)
        meanS=mu+phi.pow(k)*(filtered['mean'][origins,None]-mu)
        factor=self.future_factor(a,T['F17'][origins,2],filtered['c_ref'])
        daily=meanS*factor*getattr(self,f'scale_{a}')
        cumulative=daily.cumsum(1)
        value=torch.stack([cumulative[:,h-1]/h for h in HORIZONS],1)
        assert bool(torch.isfinite(value).all() and (value>0).all())
        return value,daily,factor


HORIZONS=B.HORIZONS


def make_tensors(datasets):
    T=S17.make_tensors(datasets)
    for a,D in datasets.items():
        T[a]['proxy_raw']=torch.tensor(raw_proxy(D),dtype=DTYPE)
    return T


def objective(model,datasets,T):
    losses,sensors,kpenalties={},{},{}
    for a in model.assets:
        D=datasets[a];tr=D['tr'];origins=D['origin'][tr]
        assert len(tr) and (origins+22<D['cut']).all()
        filtered=model.filter_asset(a,D,T[a],D['cut'])
        pred,_,_=model.forecast(a,T[a],origins,filtered)
        losses[a]=B.qlike(pred,T[a]['y'][tr],T[a]['ewma'][tr]).mean()
        sensors[a]=-filtered['log_marginal'].mean()
        kpenalties[a]=torch.log(model.processes[a].theta()[3]).square()
    n=len(model.assets);data=sum(losses.values())/n;sensor=sum(sensors.values())/n;regularizer=.005*sum(kpenalties.values())/n
    return data+.01*sensor+regularizer,dict(data=data,sensor=sensor,kappa_penalty=regularizer,asset_losses=losses)


def train_epoch(model,datasets,T,optimizer):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss,info=objective(model,datasets,T);assert torch.isfinite(loss)
    loss.backward();norm=float(nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],2))
    assert math.isfinite(norm);optimizer.step()
    return dict(train_joint_QLIKE=float(info['data'].detach()),sensor_normalized_y_NLL=float(info['sensor'].detach()),
        kappa_penalty=float(info['kappa_penalty'].detach()),gradient_norm_before_clip=norm,optimizer_steps=1,backward_calls=1)


@torch.no_grad()
def evaluate(model,datasets,T):
    model.eval();pred,scores,aux={}, {}, {}
    for a in model.assets:
        D=datasets[a];origins=D['origin'][D['va']]
        filtered=model.filter_asset(a,D,T[a],D['n'])
        value,daily,factor=model.forecast(a,T[a],origins,filtered)
        pred[a]=value.numpy();scores[a]=float(B.qlike(value,T[a]['y'][D['va']],T[a]['ewma'][D['va']]).mean())
        aux[a]={key:filtered[key].numpy() for key in ('mean','variance','entropy','M_posterior','M_posterior_valid',
            'posterior_weighted_clip_mass','posterior_weighted_mean_error','boundary_mass')}
        aux[a].update(c_ref=float(filtered['c_ref']),daily=daily.numpy(),future_factor=factor.numpy(),
            max_boundary_mass=float(filtered['boundary_mass'].max()),
            mean_clip_mass=float(filtered['posterior_weighted_clip_mass'].mean()),
            max_clip_mass=float(filtered['posterior_weighted_clip_mass'].max()),
            max_abs_transition_mean_error=float(filtered['posterior_weighted_mean_error'].abs().max()))
    return pred,sum(scores.values())/len(scores),scores,aux


def train_fit(datasets,fixed,reference,mode,seed=11,frozen_theta=None,assets=ASSETS):
    T=make_tensors(datasets);model=GammaVarianceFilter(datasets,fixed,reference,mode,seed,frozen_theta,assets)
    immutable={k:v.clone() for k,v in model.named_buffers()}
    _,best,scores,_=evaluate(model,datasets,T)
    history=[dict(epoch=0,validation_joint_QLIKE=best,validation_asset_QLIKE=scores)]
    best_state=copy.deepcopy(model.state_dict());best_epoch=bad=0
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    for epoch in range(1,MAX_EPOCHS+1):
        info=train_epoch(model,datasets,T,optimizer);_,score,scores,_=evaluate(model,datasets,T)
        history.append(dict(epoch=epoch,validation_joint_QLIKE=score,validation_asset_QLIKE=scores,**info))
        if score<best-1e-5:
            best,best_epoch,bad=score,epoch,0;best_state=copy.deepcopy(model.state_dict())
        else:bad+=1
        if bad>=PATIENCE and epoch>=MIN_EPOCHS:break
    model.load_state_dict(best_state);pred,score,scores,aux=evaluate(model,datasets,T)
    for k,v in model.named_buffers():assert torch.equal(v,immutable[k]),f'Frozen buffer changed: {k}'
    info=dict(mode=mode,seed=seed,best_joint_epoch=best_epoch,epochs_run=len(history)-1,
        best_joint_validation_QLIKE=best,returned_joint_validation_QLIKE=score,asset_scores=scores,
        theta={a:model.processes[a].bounded() for a in model.assets},
        trainable_parameters=sum(p.numel() for p in model.parameters()),history=history,
        grid_size=len(model.grid),grid_log_bounds=[-8,8],GH=16)
    return model,pred,aux,info


def synthetic_reference():
    support=REF.LOGREF.structural_support();p=support.astype(float);p/=p.sum(1,keepdims=True)
    c=np.linspace(.35,.95,7)[:,None]*(.8+.4*np.linspace(0,1,31)[None,:])
    k=np.tile(np.linspace(.5,3.,7)[:,None],(1,31));variance=c*c/k
    globalc=np.linspace(.35,.95,7);globalk=np.linspace(.5,3.,7)
    return REF.GammaObservationKernel(dict(M=list(REF.MS),supported=support.tolist(),p_shape=p.tolist(),
        c_cell=c.tolist(),k_cell=k.tolist(),var_cell=variance.tolist(),c_global=globalc.tolist(),
        k_global=globalk.tolist(),var_global=(globalc**2/globalk).tolist()))


def synthetic_pair():
    D,_=S17.synthetic_pair()
    for a in ASSETS:
        P=np.zeros(D[a]['n']);P[D[a]['observed']]=np.exp(D[a]['log_proxy'][D[a]['observed']])
        D[a]['proxy_raw']=P
    return D


def _synthetic_checks():
    D=synthetic_pair();fixed=preprocessing(D);T=make_tensors(D);reference=synthetic_reference()
    process=VarianceProcess();mu,phi,q,kappa=process.theta()
    torch.testing.assert_close(torch.stack([mu,phi,q,kappa]),torch.tensor([1.,.98,.03,1.],dtype=DTYPE))
    grid=torch.exp(torch.linspace(-8,8,256,dtype=DTYPE));nodes,weights=np.polynomial.hermite.hermgauss(16)
    nodes=torch.tensor(nodes,dtype=DTYPE);weights=torch.tensor(weights/weights.sum(),dtype=DTYPE)
    matrix,diag=transition_matrix(grid,mu,phi,q,nodes,weights)
    interior=diag['interior_rows'];ideal=mu+phi*(grid-mu)
    torch.testing.assert_close((matrix@grid)[interior],ideal[interior],rtol=1e-12,atol=1e-12)
    assert diag['row_clip_mass'][~interior].max()>0 and diag['upper_shock_mask'].any()
    assert torch.isfinite(matrix).all() and (matrix>=0).all()
    assert diag['row_interpolation_error'].abs().max()<1e-11
    # Strong drift and q produce explicit LOWER and UPPER boundary masks.
    _,bounds=transition_matrix(grid,mu*0+.05,phi*0+.9995,q*0+.25,nodes,weights)
    assert bounds['lower_shock_mask'].any() and bounds['upper_shock_mask'].any()
    post=initial_ig(grid,mu);assert (post>=0).all() and abs(float(post.detach().sum())-1)<1e-14
    frozen={a:process.bounded() for a in ASSETS}
    baseline=GammaVarianceFilter(D,fixed,reference,'fixed_mixture',frozen_theta=frozen)
    neural=GammaVarianceFilter(D,fixed,reference,'neural_density',11,frozen)
    assert sum(p.numel() for p in neural.parameters())==43
    largest_grid_comparison=0.;max_weighted_clip=0.;max_weighted_mean_error=0.
    for a in ASSETS:
        first=baseline.filter_asset(a,D[a],T[a],D[a]['n']);second=neural.filter_asset(a,D[a],T[a],D[a]['n'])
        max_weighted_clip=max(max_weighted_clip,float(first['posterior_weighted_clip_mass'].detach().max()))
        max_weighted_mean_error=max(max_weighted_mean_error,float(first['posterior_weighted_mean_error'].detach().abs().max()))
        for key in ('posterior','mean','variance','log_marginal','M_posterior'):
            torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        torch.testing.assert_close(first['posterior'][12],first['prior'][12],rtol=0,atol=0)
        assert not first['M_posterior_valid'][12] and first['M_posterior'][12].sum()==0
        torch.testing.assert_close(first['log_marginal'][12],baseline.hurdle_logs(a,T[a]['F17'][12:13,2])[0][0],rtol=0,atol=0)
        assert (first['posterior']>=0).all() and (first['mean']>0).all() and (first['variance']>=0).all()
        torch.testing.assert_close(first['posterior'].sum(1),torch.ones(D[a]['n'],dtype=DTYPE),rtol=1e-12,atol=1e-12)
        ids=D[a]['origin'][D[a]['va']];value,daily,factor=baseline.forecast(a,T[a],ids,first)
        torch.testing.assert_close(value,torch.stack([daily[:,:h].mean(1) for h in HORIZONS],1))
        assert (daily>0).all() and torch.isfinite(first['entropy']).all()
        # Shape posterior is not a full radial posterior; using different y changes M.
        cells=R20.geometry31(T[a]['F17'],torch.tensor(D[a]['observed']))
        t=int(np.flatnonzero((cells.numpy()>=6)&D[a]['observed'])[0]);cref=first['c_ref']
        logpi=baseline.positive_log_prior(T[a]['F17'][t:t+1,2])[0]
        shape_only=torch.softmax(logpi+reference.shape_log_prob(cells[t:t+1])[0],0)
        assert (first['M_posterior'][t]-shape_only).abs().max()>1e-6
        varied={k:v.clone() for k,v in T[a].items()};varied['proxy_raw'][t]*=8
        altered=baseline.filter_asset(a,D[a],varied,D[a]['n'])
        assert (altered['M_posterior'][t]-first['M_posterior'][t]).abs().max()>1e-6
        changed={k:v.clone() for k,v in T[a].items()};changed['F17'][40:,2]+=100;changed['proxy_raw'][40:]*=5
        causal=baseline.filter_asset(a,D[a],changed,D[a]['n'])
        torch.testing.assert_close(first['posterior'][:40],causal['posterior'][:40],rtol=0,atol=0)
        # Scale/unit equivariance: same normalized radial y and fixed means.
        unit=copy.deepcopy(D);fixedunit=copy.deepcopy(fixed);unitT={aa:{k:v.clone() for k,v in T[aa].items()} for aa in ASSETS}
        for aa in ASSETS:
            unit[aa]['proxy_raw']*=7;unit[aa]['y']*=7;fixedunit[aa]['scale']*=7
            unitT[aa]['proxy_raw']*=7;unitT[aa]['y']*=7;unitT[aa]['ewma']*=7
        scaled=GammaVarianceFilter(unit,fixedunit,reference,'fixed_mixture',frozen_theta=frozen)
        ss=scaled.filter_asset(a,unit[a],unitT[a],unit[a]['n'])
        torch.testing.assert_close(ss['posterior'],first['posterior'],rtol=1e-11,atol=1e-12)
        sv,_,_=scaled.forecast(a,unitT[a],ids,ss);torch.testing.assert_close(sv,7*value,rtol=1e-11,atol=1e-12)
        fine=GammaVarianceFilter(D,fixed,reference,'fixed_mixture',frozen_theta=frozen,grid_size=512)
        fine_state=fine.filter_asset(a,D[a],T[a],D[a]['n'])
        difference=float((fine_state['mean']-first['mean']).detach().abs().max())
        largest_grid_comparison=max(largest_grid_comparison,difference)
        # Grid sensitivity is reported, never tuned against any holdout or hidden.
    # M1 analytic density and Gamma parameterization against Torch distribution.
    y=torch.tensor([.03,.3,1.],dtype=DTYPE)
    analytic=torch.distributions.Gamma(torch.tensor(.5,dtype=DTYPE),torch.tensor(2*math.log(2),dtype=DTYPE)).log_prob(y)
    torch.testing.assert_close(REF.analytic_m1_log_prob(y),analytic,rtol=1e-14,atol=1e-14)
    cells=torch.tensor([0,6,30]);params=reference.observation_parameters(y,grid[:5],.7,cells,kappa=1.3)
    lp=reference(y,grid[:5],.7,cells,kappa=1.3);support=params['supported'][:,None,:].expand_as(lp)
    expected=torch.distributions.Gamma(params['shape'],1/params['scale']).log_prob(y[:,None,None])
    torch.testing.assert_close(lp[support],expected[support],rtol=1e-12,atol=1e-12)
    assert torch.isneginf(lp[~support]).all()
    # Constant prior and hurdle imply futurefactor1 under the same c_ref frame.
    const=GammaVarianceFilter(D,fixed,reference,'global_dense',frozen_theta=frozen)
    for a in ASSETS:
        with torch.no_grad():getattr(const,f'hurdle_{a}')[1]=0
        factor=const.future_factor(a,torch.tensor([-10.,0.,10.],dtype=DTYPE),const.c_ref(a))
        torch.testing.assert_close(factor,torch.ones_like(factor),rtol=1e-14,atol=1e-14)
    # Preprocessing and bank changes after cut cannot change scale/hurdle/AR.
    changed=copy.deepcopy(D)
    for a in ASSETS:
        changed[a]['proxy_raw'][D[a]['cut']:]*=100
        changed[a]['F17'][D[a]['cut']:,2]+=100
        changed[a]['observed'][D[a]['cut']:]=False
    assert preprocessing(changed)==fixed
    # Joint NN gradients contain no validation features or targets; no dead gate.
    gradT={a:{**T[a],'F17':T[a]['F17'].clone().requires_grad_(),'proxy_raw':T[a]['proxy_raw'].clone().requires_grad_(),
                  'y':T[a]['y'].clone().requires_grad_()} for a in ASSETS}
    loss,info=objective(neural,D,gradT);loss.backward()
    assert neural.prior_net[-1].weight.grad.abs().sum()>0
    for a in ASSETS:
        for key in ('F17','proxy_raw'):
            assert (gradT[a][key].grad[D[a]['cut']:] ==0).all()
        assert (gradT[a]['y'].grad[D[a]['va']]==0).all()
    buffers={k:v.clone() for k,v in neural.named_buffers()}
    optimizer=torch.optim.Adam(neural.parameters(),lr=LR)
    with patch('torch.autograd.backward',wraps=torch.autograd.backward) as backward:
        with patch.object(optimizer,'step',wraps=optimizer.step) as step:
            train_epoch(neural,D,T,optimizer)
    assert backward.call_count==step.call_count==1
    for k,v in neural.named_buffers():assert torch.equal(v,buffers[k])
    # Four process parameters also have finite train gradients on a synthetic base.
    base=GammaVarianceFilter(D,fixed,reference,'shape_dense',assets=(ASSETS[0],))
    ll,_=objective(base,D,T);ll.backward()
    assert len(list(base.parameters()))==4 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in base.parameters())
    return dict(nonnegative_stochastic_transition=True,interior_mean_preserving=True,boundary_masks_and_weighted_clip_diagnostics=True,
        Gamma_matches_torch_and_analytic_M1=True,unsupported_shape_stays_minus_inf=True,
        positive_normalized_posterior_mean_variance=True,full_radial_M_posterior_not_shape_only=True,
        NN0_exact_fixed_mixture_43params=True,zero_branch_advance_no_S_information=True,
        future_causal_and_no_validation_gradients=True,pre25_scale_hurdle_AR_bank_fixed=True,
        unit_change_equivariance=True,constant_hurdle_prior_futurefactor_one=True,
        prior_head_gradient_nonzero_at_epoch0=True,one_backward_one_step=True,frozen_NN_buffers_unchanged=True,
        process4parameter_gradients_finite=True,synthetic_grid256_vs512_max_mean_difference=largest_grid_comparison,
        synthetic_posterior_weighted_max_clip_mass=max_weighted_clip,
        synthetic_posterior_weighted_max_abs_transition_mean_error=max_weighted_mean_error,
        transition_max_abs_interior_mean_error=float(((matrix@grid)-ideal)[interior].detach().abs().max()),
        transition_max_abs_interpolation_error=float(diag['row_interpolation_error'].detach().abs().max()),
        transition_max_boundary_row_clip_mass=float(diag['row_clip_mass'].detach().max()),
        initial_IG_grid_mean=float((post@grid).detach()),
        grid_sensitivity_report_only_not_a_grid_selection=True,real_model_training_executed=False,real2026_used=False)


def synthetic_checks():
    with ExitStack() as stack:
        for name in ('pandas.read_csv','numpy.load','pathlib.Path.read_text','torch.load',
                     'holdout_2026_range_builder.build_pair','sampling_density_gamma_reference.load_gamma_reference'):
            stack.enter_context(patch(name,side_effect=AssertionError('real I/O forbidden in purechecks')))
        result=_synthetic_checks()
    result['real_input_readers_guarded']=True
    return result


def source_hashes():
    sources=dict(HB.source_paths());sources.update(runner=Path(__file__),R20=Path(R20.__file__),
        Gamma_reference=Path(REF.__file__),Gamma_JSON=HERE/'sampling_density_gamma_calibration.json',
        Gamma_NPZ=HERE/'sampling_density_gamma_calibration.npz',log_reference=Path(REF.LOGREF.__file__))
    return {k:dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for k,p in sources.items()}


def run_development(tag):
    """Future authorized development only. No2026 path is exposed by this script."""
    assert re.fullmatch(r'round21_gamma_[A-Za-z0-9_-]+',tag)
    paths=dict(results=HERE/f'{tag}_results.json',predictions=HERE/f'{tag}_validation_predictions.npz')
    checkpoint_paths={f'{mode}_{a}':HERE/f'{tag}_{mode}_{a}.pt' for mode in BASES for a in ASSETS}
    checkpoint_paths.update({f'neural_density_seed{s}':HERE/f'{tag}_neural_density_seed{s}.pt' for s in SEEDS})
    for path in [*paths.values(),*checkpoint_paths.values()]:
        if path.exists():raise FileExistsError(f'Existing output protected: {path}')
    hashes=source_hashes();started=time.time()
    datasets,_,audit=HB.build_pair(development_only=True,check_npz=True)
    assert all(len(D['te'])==0 and (D['date']<HB.TEST_START).all() for D in datasets.values())
    fixed=preprocessing(datasets);reference=REF.load_gamma_reference()
    arrays=dict(horizons=np.array(HORIZONS));predictions={a:{} for a in ASSETS};fits={};fixed_theta={}
    def save_model(key,model,info):
        target=checkpoint_paths[key]
        own=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            metadata=dict(format_version=1,fit=info,preprocessing=fixed,source_hashes=hashes,
                          inference='nominal positive variance and effective M, not trueN/IV'))
        with target.open('xb') as handle:torch.save(own,handle)
        torch.load(target,weights_only=True,map_location='cpu')
        return dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    checkpoints={}
    for mode in BASES:
        fits[mode]={}
        for a in ASSETS:
            model,pred,aux,info=train_fit(datasets,fixed,reference,mode,11,assets=(a,))
            fits[mode][a]=info;predictions[a][mode]=pred[a]
            if mode=='fixed_mixture':fixed_theta[a]=info['theta'][a]
            for key,value in aux[a].items():arrays[f'{a}_{mode}_{key}']=np.asarray(value)
            checkpoints[f'{mode}_{a}']=save_model(f'{mode}_{a}',model,info)
    fits['neural_density']=[]
    for seed in SEEDS:
        # Pre-training nesting gate on the independent selected fixed-mixture models.
        zero=GammaVarianceFilter(datasets,fixed,reference,'neural_density',seed,fixed_theta)
        T=make_tensors(datasets);nested,_,_,_=evaluate(zero,datasets,T)
        for a in ASSETS:np.testing.assert_array_equal(nested[a],predictions[a]['fixed_mixture'])
        model,pred,aux,info=train_fit(datasets,fixed,reference,'neural_density',seed,fixed_theta)
        fits['neural_density'].append(info)
        for a in ASSETS:
            predictions[a][f'neural_density_seed{seed}']=pred[a]
            for key,value in aux[a].items():arrays[f'{a}_neural_density_seed{seed}_{key}']=np.asarray(value)
        checkpoints[f'neural_density_seed{seed}']=save_model(f'neural_density_seed{seed}',model,info)
    for a in ASSETS:
        predictions[a]['neural_density_ensemble']=np.mean([predictions[a][f'neural_density_seed{s}'] for s in SEEDS],axis=0)
        D=datasets[a];q=D['origin'][D['va']]
        arrays.update({f'{a}_date':D['date'][q],f'{a}_origin':q,f'{a}_label_end':D['date'][q+22],f'{a}_target':D['y'][D['va']]})
        arrays.update({f'{a}_{key}':value for key,value in predictions[a].items()})
    report=dict(status='REUSED2025 DEVELOPMENT; any reused2026 later is diagnostic, not fresh confirmation',
        source_hashes=hashes,config=dict(grid=256,log_grid=[-8,8],GH=16,epochs=80,patience=15,min_epochs=20,
            lr=.01,sensor_NLL_weight=.01,kappa_penalty=.005,clip=2,seeds=SEEDS),preprocessing=fixed,
        builder=audit,fits=fits,checkpoints=checkpoints,
        scores={a:S17.metrics(predictions[a],datasets[a]) for a in ASSETS},
        fixed_mixture_theta_for_NN=fixed_theta,neural_epoch0_exact_fixed_mixture=True,
        working_assumption='independent future volume and latent-variance innovations',
        transition='GH16, nonnegative linear probability scatter in S; boundary clipping is measured, not hidden',
        sensor='normalized rawP Gamma radial + shape once + separate unresolved-range hurdle',
        no_true_N_IV_or_zero_volatility_identification=True,elapsed_seconds=time.time()-started)
    assert source_hashes()==hashes,'Source changed during development'
    with paths['predictions'].open('xb') as handle:np.savez_compressed(handle,**arrays)
    with paths['results'].open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    flags=ap.add_mutually_exclusive_group();flags.add_argument('--check-only',action='store_true');flags.add_argument('--run-development',action='store_true')
    ap.add_argument('--tag',default='round21_gamma_development')
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.run_development:run_development(args.tag)
    else:print(json.dumps(dict(checks=synthetic_checks(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),flush=True)


if __name__=='__main__':main()
