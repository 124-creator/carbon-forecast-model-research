"""R22 predictive risk/quote path tilts over frozen R21 global-dense risk.

Default is pure synthetic checks. No real fitting, GRN I/O, used2026 diagnostic,
or prospective confirmation is part of this delivery. Geometry predicts future
risk/quote paths; it is NOT used to de-bias a current observation-noise model.
Quote event means raw range proxy>0, not no-trade. Old2026 was already used;
any reuse is diagnostic, never fresh confirmation. The proposed series is fixed
by later development protocol, not by inspecting any reused-period outcome.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import re
import time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import round21_gamma_observation_filter as S21
from round21_fast_filter import FastGammaVarianceFilter
import round21_fast_filter as FAST

HB,R20,DTYPE=S21.HB,S21.R20,S21.DTYPE
HERE=Path(__file__).resolve().parent
ASSETS=S21.ASSETS
SEEDS=(11,22,33)
MODES=('coupled_path','independent_path','direct_marked_control')
BUCKET_LENGTHS=(1,4,17)
DAY_BUCKET=(0,)+(1,)*4+(2,)*17
MAX_EPOCHS,PATIENCE,MIN_EPOCHS=80,15,20
LR=.01
OUTPUT_PENALTY=.005


def state_digest(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def analytic_future_moments(mean,variance,mu,phi,q):
    """Positive Markov first/second moment recursion, no future observations."""
    means,variances=[],[]
    m,V=mean,variance
    qRel=torch.expm1(q)
    for _ in range(22):
        nextV=phi.square()*((1+qRel)*V+qRel*m.square())
        m=mu+phi*(m-mu);V=nextV
        means.append(m);variances.append(V)
    return torch.stack(means,1),torch.stack(variances,1)


@torch.no_grad()
def build_asset_cache(D,backbone,asset):
    """Observation-only inference; NEVER reads y or future event labels."""
    before=state_digest(backbone)
    if any(p.requires_grad for p in backbone.parameters()):raise ValueError('Backbone must be frozen')
    inputs={k:torch.tensor(D[k],dtype=DTYPE) for k in ('F17','proxy_raw')}
    # Filtering can see current/past observations only. Targets are not supplied.
    observed=np.asarray(D['observed'],dtype=bool)
    view=dict(n=D['n'],cut=D['cut'],observed=observed)
    f=backbone.filter_asset(asset,view,inputs,D['n'])
    mu,phi,q,kappa=backbone.processes[asset].theta()
    matrix,_=S21.transition_matrix(backbone.grid,mu,phi,q,backbone.gh_nodes,backbone.gh_weights)
    bank=torch.tensor(D['bank_index'],dtype=torch.long)
    assert len(bank) and (bank<D['cut']).all() and observed[bank.numpy()].all()
    torch.testing.assert_close(getattr(backbone,f'bank_index_{asset}'),bank,rtol=0,atol=0)
    torch.testing.assert_close(getattr(backbone,f'bank_v_{asset}'),inputs['F17'][bank,2],rtol=0,atol=0)
    risk=torch.stack([torch.log(f['mean']/mu),torch.log1p(f['variance']/f['mean'].square())],1)
    risk_mean=risk[bank].mean(0);risk_std=risk[bank].std(0,unbiased=False).clamp_min(1e-5)
    ctx=torch.cat([inputs['F17'],(risk-risk_mean)/risk_std,torch.tensor(observed,dtype=DTYPE)[:,None]],1)
    origins=np.arange(D['n']);_,daily,factor=backbone.forecast(asset,inputs,origins,f)
    cref=f['c_ref'];c512=backbone.reference.c_global[-1]
    active=factor*cref/c512
    assert torch.isfinite(active).all() and (active>0).all() and (active<1).all()
    analytic_mean,analytic_var=analytic_future_moments(f['mean'],f['variance'],mu,phi,q)
    # This discrete-chain mean is distinct from the physical analytic mean;
    # endpoint clipping/grid error must not corrupt the zero-anchor denominator.
    prior=f['posterior'];markov=[]
    for _ in range(22):
        prior=prior@matrix;markov.append(prior@backbone.grid)
    cache=dict(ctx=ctx,posterior=f['posterior'],transition=matrix,grid=backbone.grid,
        mu=mu,phi=phi,q=q,kappa=kappa,scale=getattr(backbone,f'scale_{asset}'),c512=c512,c_ref=cref,
        bank_index=bank,risk_mean=risk_mean,risk_std=risk_std,
        base_daily=daily,base_active=active,base_analytic_mean=analytic_mean,
        base_future_log1pCV2=torch.log1p(analytic_var/analytic_mean.square()),
        base_markov_mean=torch.stack(markov,1),posterior_mean=f['mean'],posterior_variance=f['variance'])
    assert state_digest(backbone)==before
    return {k:v.detach().clone() for k,v in cache.items()}


def build_caches(datasets,backbones,assets=None):
    """Generic in-memory adapter; a future G builder supplies its own datasets."""
    assets=tuple(datasets) if assets is None else tuple(assets)
    return {a:build_asset_cache(datasets[a],backbones[a],a) for a in assets}


def path_expectation(posterior,transition,grid,mu,active,a,b,c):
    """Pure agent4 kernel; full joint moments, never a product approximation."""
    kernel=importlib.import_module('risk_quote_path_tilt')
    psi=torch.tanh(torch.log(grid/mu))
    coefficients=torch.stack([a,b,c],dim=-1)
    result=kernel.risk_quote_path_tilt(posterior,transition,grid,psi,active,coefficients)
    return dict(EqS=result['E_S'],EqZ=result['E_Z'],EqSZ=result['E_SZ'],KL=result['KL'],
        covariance=result['Cov_SZ'],base_markov_mean=result['baseline_E_S'])


class PredictivePath(nn.Module):
    def __init__(self,caches,mode,seed,assets=None,kernel=None):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;self.assets=tuple(caches) if assets is None else tuple(assets)
        self.kernel=path_expectation if kernel is None else kernel
        for i,a in enumerate(self.assets):
            for key,value in caches[a].items():self.register_buffer(f'cache_{i}_{key}',value.detach().clone())
        self.register_buffer('day_bucket',torch.tensor(DAY_BUCKET,dtype=torch.long))
        self.register_buffer('lengths',torch.tensor(BUCKET_LENGTHS,dtype=DTYPE))
        torch.manual_seed(seed)
        self.channels=2 if mode=='independent_path' else 3
        self.head=nn.Sequential(nn.Linear(6,8),nn.Tanh(),nn.Linear(8,3*self.channels)).to(dtype=DTYPE)
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        assert sum(p.numel() for p in self.parameters())==(110 if self.channels==2 else 137)

    def cache(self,asset,key):return getattr(self,f'cache_{self.assets.index(asset)}_{key}')

    def raw(self,features):
        fields=torch.tanh(self.head(features)).reshape(-1,3,self.channels)
        a=.25*fields[:,:,0];b=fields[:,:,1]
        c=.25*fields[:,:,2] if self.channels==3 else torch.zeros_like(a)
        return torch.stack([a,b,c],-1)

    def fields(self,asset,origins):
        ctx=self.cache(asset,'ctx');bank=self.cache(asset,'bank_index')
        bank_raw=self.raw(ctx[bank]);center=bank_raw.mean(0)
        delta=self.raw(ctx[origins])-center
        reference=bank_raw-center
        daily=delta[:,self.day_bucket,:]
        # Only path potentials are integrated energies divided by bucket length.
        # Direct a/c are bucket log-mark corrections and retain their full size.
        daily_scale=(self.lengths[self.day_bucket][None,:] if self.mode!='direct_marked_control'
                     else daily.new_ones((1,22)))
        return (daily[:,:,0]/daily_scale,daily[:,:,1],daily[:,:,2]/daily_scale),reference,center,delta

    def predict(self,asset,origins):
        origins=torch.as_tensor(origins,dtype=torch.long,device=self.cache(asset,'ctx').device)
        (a,b,c),reference,center,delta=self.fields(asset,origins)
        active=self.cache(asset,'base_active')[origins]
        base=self.cache(asset,'base_daily')[origins]
        if self.mode=='direct_marked_control':
            uCV=self.cache(asset,'base_future_log1pCV2')[origins]
            mark_factor=torch.exp(a+c*uCV)
            # Algebraically sigmoid(logit(p)+b)/p; exact1 at b0 without a
            # zero-head branch, preserving every head's main-loss derivative.
            active_ratio=torch.exp(b)/(1+active*torch.expm1(b))
            EqZ=active*active_ratio
            EqS=self.cache(asset,'base_analytic_mean')[origins]*mark_factor
            EqSZ=EqS*EqZ
            daily=base*mark_factor*active_ratio
            covariance=torch.zeros_like(EqSZ)
            KL=EqSZ.new_zeros(len(origins));KL_defined=False
        else:
            out=self.kernel(self.cache(asset,'posterior')[origins],self.cache(asset,'transition'),
                self.cache(asset,'grid'),self.cache(asset,'mu'),active,a,b,c)
            EqS,EqZ,EqSZ=out['EqS'],out['EqZ'],out['EqSZ']
            KL=out['KL'];covariance=out['covariance'];KL_defined=True
            # Same Markov reference and reduction convention as the pure core.
            denominator=out['base_markov_mean']*active
            daily=base*(EqSZ/denominator)
        assert torch.isfinite(daily).all() and (daily>0).all()
        assert torch.isfinite(EqZ).all() and (EqZ>=0).all() and (EqZ<=1).all()
        cumulative=daily.cumsum(1)
        value=torch.stack([cumulative[:,h-1]/h for h in S21.HORIZONS],1)
        aux=dict(daily=daily,EqS=EqS,EqZ=EqZ,EqSZ=EqSZ,risk_resolution_covariance=covariance,
            KL=KL,KL_defined=KL_defined,delta_bucket=delta,delta_daily=torch.stack([a,b,c],-1),
            bank_center=center,bank_reference=reference)
        return value,aux


def future_event_labels(D,origins,stop):
    """Labels only. Call AFTER origin maturity gate; never called by caching."""
    q=np.asarray(origins,dtype=int)
    assert len(q) and (q+22<stop).all() and stop<=D['n']
    proxy=np.asarray(D['proxy_raw'])
    return torch.tensor(np.stack([proxy[t+1:t+23]>0 for t in q]),dtype=DTYPE)


def event_bce(probability,label):
    return F.binary_cross_entropy(probability.clamp(1e-12,1-1e-12),label,reduction='mean')


def objective(model,datasets):
    qlike,bce,KL,penalty={},{},{},{}
    for a in model.assets:
        D=datasets[a];ids=D['tr'];origins=D['origin'][ids]
        labels=future_event_labels(D,origins,D['cut'])
        pred,aux=model.predict(a,origins)
        y=torch.tensor(D['y'][ids],dtype=DTYPE);ewma=torch.tensor(D['ewma'][ids],dtype=DTYPE)
        qlike[a]=S21.B.qlike(pred,y,ewma).mean()
        bce[a]=event_bce(aux['EqZ'],labels)
        KL[a]=aux['KL'].mean()/22 if aux['KL_defined'] else pred.new_zeros(())
        ref=aux['bank_reference'][:,:,:model.channels]
        penalty[a]=ref.square().sum(dim=(1,2)).mean()
    n=len(model.assets);data=sum(qlike.values())/n;event=sum(bce.values())/n
    pathKL=sum(KL.values())/n;regularizer=OUTPUT_PENALTY*sum(penalty.values())/n
    return data+.05*event+.05*pathKL+regularizer,dict(QLIKE=data,BCE=event,KL_per_day=pathKL,
        output_penalty=regularizer,asset_QLIKE=qlike,asset_BCE=bce,asset_KL_per_day=KL)


def train_epoch(model,datasets,optimizer):
    model.train();optimizer.zero_grad(set_to_none=True);loss,info=objective(model,datasets)
    assert torch.isfinite(loss)
    loss.backward();norm=float(nn.utils.clip_grad_norm_(model.parameters(),2))
    assert np.isfinite(norm);optimizer.step()
    return {**{k:float(info[k].detach()) for k in ('QLIKE','BCE','KL_per_day','output_penalty')},
        **{k:{a:float(v.detach()) for a,v in info[k].items()} for k in ('asset_QLIKE','asset_BCE','asset_KL_per_day')},
        'KL_applies':model.mode!='direct_marked_control',
        'gradient_norm_before_clip':norm,'optimizer_steps':1,'backward_calls':1}


@torch.no_grad()
def evaluate(model,datasets):
    model.eval();pred,scores,auxout,decomposition={},{},{},{}
    for a in model.assets:
        D=datasets[a];ids=D['va'];origins=D['origin'][ids]
        predT,aux=model.predict(a,origins)
        scores[a]=float(S21.B.qlike(predT,torch.tensor(D['y'][ids],dtype=DTYPE),torch.tensor(D['ewma'][ids],dtype=DTYPE)).mean())
        bce=float(event_bce(aux['EqZ'],future_event_labels(D,origins,D['n'])))
        pred[a]=predT.numpy();auxout[a]={k:v.numpy() for k,v in aux.items() if isinstance(v,torch.Tensor)}
        decomposition[a]=dict(QLIKE=scores[a],BCE=bce,KL_per_day=float(aux['KL'].mean()/22) if aux['KL_defined'] else None,
            risk_resolution_covariance_mean=float(aux['risk_resolution_covariance'].mean()),
            KL_applies=aux['KL_defined'])
    return pred,sum(scores.values())/len(scores),scores,auxout,decomposition


def train_one(caches,datasets,mode,seed):
    model=PredictivePath(caches,mode,seed);immutable={k:v.clone() for k,v in model.named_buffers()}
    _,best,scores,_,decomp=evaluate(model,datasets)
    history=[dict(epoch=0,validation_joint_QLIKE=best,validation_asset_QLIKE=scores,decomposition=decomp)]
    best_epoch=bad=0;best_state=copy.deepcopy(model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=LR)
    for epoch in range(1,MAX_EPOCHS+1):
        info=train_epoch(model,datasets,opt);_,score,scores,_,decomp=evaluate(model,datasets)
        history.append(dict(epoch=epoch,validation_joint_QLIKE=score,validation_asset_QLIKE=scores,decomposition=decomp,training=info))
        if score<best-1e-5:
            best,best_epoch,bad=score,epoch,0;best_state=copy.deepcopy(model.state_dict())
        else:bad+=1
        if bad>=PATIENCE and epoch>=MIN_EPOCHS:break
    model.load_state_dict(best_state);pred,returned,scores,aux,decomp=evaluate(model,datasets)
    for k,v in model.named_buffers():assert torch.equal(v,immutable[k]),'Frozen cache changed: '+k
    return model,pred,aux,dict(mode=mode,seed=seed,best_joint_epoch=best_epoch,epochs_run=len(history)-1,
        best_joint_validation_QLIKE=best,returned_joint_QLIKE=returned,selected_asset_scores=scores,
        trainable_parameters=sum(p.numel() for p in model.parameters()),decomposition=decomp,
        selected_joint_decomposition=dict(QLIKE=returned,BCE=sum(v['BCE'] for v in decomp.values())/len(decomp),
            KL_per_day=sum(v['KL_per_day'] for v in decomp.values())/len(decomp) if mode!='direct_marked_control' else None,
            risk_resolution_covariance_mean=sum(v['risk_resolution_covariance_mean'] for v in decomp.values())/len(decomp)),
        history=history)


def synthetic_context():
    D=S21.synthetic_pair();fixed=S21.preprocessing(D);reference=S21.synthetic_reference();backbones={}
    for a in D:
        m=FastGammaVarianceFilter(D,fixed,reference,'global_dense',11,assets=(a,));m.eval();m.requires_grad_(False)
        backbones[a]=m
    return D,backbones,build_caches(D,backbones)


def _purechecks(modes=MODES):
    D,backbones,caches=synthetic_context();out={}
    expected={a:S21.evaluate(backbones[a],D,S21.make_tensors(D))[0][a] for a in D}
    for mode in modes:
        m=PredictivePath(caches,mode,11)
        assert sum(p.numel() for p in m.parameters())==(110 if mode=='independent_path' else 137)
        for a in D:
            ids=D[a]['origin'][D[a]['va']];p,aux=m.predict(a,ids)
            np.testing.assert_allclose(p.detach().numpy(),expected[a],rtol=1e-11,atol=1e-14)
            assert (aux['daily']>0).all()
            torch.testing.assert_close(p,torch.stack([aux['daily'][:,:h].mean(1) for h in S21.HORIZONS],1))
            fields,ref,center,delta=m.fields(a,ids)
            assert torch.all(ref.mean(0).abs()<1e-14)
            assert (delta[:,:,0].abs()<=.5).all() and (delta[:,:,1].abs()<=2).all() and (delta[:,:,2].abs()<=.5).all()
        # Every active output channel must influence QLIKE at zero head. No
        # if(head==0) bypass; direct c obtains a derivative through future CV.
        m.zero_grad(set_to_none=True);ql=0
        for a in D:
            ids=D[a]['tr'];p,_=m.predict(a,D[a]['origin'][ids])
            ql=ql+S21.B.qlike(p,torch.tensor(D[a]['y'][ids],dtype=DTYPE),torch.tensor(D[a]['ewma'][ids],dtype=DTYPE)).mean()/len(D)
        ql.backward();grad=m.head[-1].weight.grad.abs().sum(1)
        assert (grad>0).all(),(mode,grad)
        # Nonzero field for causal/frozen/penalty and val feature-gradient tests.
        with torch.no_grad():
            m.head[-1].weight.copy_(torch.linspace(-.05,.08,m.head[-1].weight.numel(),dtype=DTYPE).reshape_as(m.head[-1].weight))
        for a in D:
            origins=D[a]['origin'][D[a]['tr']]
            fields,reference,_,delta=m.fields(a,origins)
            for day,bucket in ((0,0),(1,1),(5,2)):
                L=1 if mode=='direct_marked_control' else BUCKET_LENGTHS[bucket]
                torch.testing.assert_close(fields[0][:,day],delta[:,bucket,0]/L,rtol=0,atol=0)
                torch.testing.assert_close(fields[1][:,day],delta[:,bucket,1],rtol=0,atol=0)
                torch.testing.assert_close(fields[2][:,day],delta[:,bucket,2]/L,rtol=0,atol=0)
            assert (delta[:,:,0].abs()<=.5).all() and (delta[:,:,1].abs()<=2).all() and (delta[:,:,2].abs()<=.5).all()
            _,nonzero_aux=m.predict(a,origins)
            torch.testing.assert_close(nonzero_aux['risk_resolution_covariance'],
                nonzero_aux['EqSZ']-nonzero_aux['EqS']*nonzero_aux['EqZ'],rtol=1e-10,atol=1e-12)
            if mode=='coupled_path':
                assert nonzero_aux['risk_resolution_covariance'].abs().max()>1e-10
            else:
                assert nonzero_aux['risk_resolution_covariance'].abs().max()<1e-10
            m.cache(a,'ctx').requires_grad_(True)
        loss,info=objective(m,D)
        explicit_penalty=sum(m.fields(a,D[a]['origin'][D[a]['tr']])[1][:,:,:m.channels].square().sum((1,2)).mean() for a in D)/len(D)*OUTPUT_PENALTY
        torch.testing.assert_close(info['output_penalty'],explicit_penalty,rtol=0,atol=0)
        m.zero_grad(set_to_none=True);loss.backward()
        for a in D:
            probe=m.cache(a,'ctx').grad
            assert probe is not None and (probe[D[a]['cut']:] ==0).all()
            m.cache(a,'ctx').requires_grad_(False)
        immutable={k:v.clone() for k,v in m.named_buffers()}
        opt=torch.optim.Adam(m.parameters(),lr=LR)
        with patch('torch.autograd.backward',wraps=torch.autograd.backward) as backward:
            with patch.object(opt,'step',wraps=opt.step) as step:train_epoch(m,D,opt)
        assert backward.call_count==step.call_count==1
        for k,v in m.named_buffers():assert torch.equal(v,immutable[k])
        out[mode]=dict(parameters=sum(p.numel() for p in m.parameters()),all_active_heads_have_main_loss_gradient=True,
            zero_head_nests_saved_global_synthetic=True,val_context_gradient_zero=True,one_backward_step=True,frozen_caches_unchanged=True)
    # Labels differ from the old thresholded observed flag: a tiny positive range
    # is a positive reported event, not an assumed zero/no-trade observation.
    DD=copy.deepcopy(D);a=tuple(D)[0];t=int(D[a]['origin'][D[a]['tr'][0]])
    DD[a]['proxy_raw'][t+1]=1e-30;DD[a]['observed'][t+1]=False
    assert future_event_labels(DD[a],[t],DD[a]['cut'])[0,0]==1
    try:future_event_labels(D[a],[D[a]['cut']-22],D[a]['cut'])
    except AssertionError:pass
    else:raise AssertionError('Unmatured future event labels accepted')
    changed=copy.deepcopy(D)
    for a in D:
        changed[a]['proxy_raw'][D[a]['cut']+4:]*=7
        changed[a]['F17'][D[a]['cut']+4:,2]+=4
        changed[a]['y']*=100
    altered=build_caches(changed,backbones)
    for a in D:
        boundary=D[a]['cut']+4
        for key in ('ctx','posterior','base_daily','base_active'):
            torch.testing.assert_close(caches[a][key][:boundary],altered[a][key][:boundary],rtol=0,atol=0)
        for key in ('risk_mean','risk_std','bank_index'):
            torch.testing.assert_close(caches[a][key],altered[a][key],rtol=0,atol=0)
    return dict(modes=out,cache_future_mutation_causality=True,cache_ignores_y=True,risk_bank_stats_trainonly=True,
        strict_h22_event_label_maturity=True,quote_event_rawP_positive_not_observed_or_no_trade=True,
        independent_and_direct_covariance_defined_by_their_controls=True,
        real_training_executed=False,real_future_labels_or_GRN_opened=False)


def purechecks():
    with ExitStack() as stack:
        for name in ('pandas.read_csv','numpy.load','pathlib.Path.read_text','torch.load',
                     'holdout_2026_range_builder.build_pair','sampling_density_gamma_reference.load_gamma_reference'):
            stack.enter_context(patch(name,side_effect=AssertionError('Real I/O forbidden in purechecks')))
        return _purechecks()


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def source_hashes():
    core=importlib.import_module('risk_quote_path_tilt')
    sources=dict(HB.source_paths());sources.update(runner=Path(__file__),R21=Path(S21.__file__),fast_filter=Path(FAST.__file__),
        path_tilt_kernel=Path(core.__file__),Gamma_reference=Path(S21.REF.__file__),
        Gamma_JSON=HERE/'sampling_density_gamma_calibration.json',Gamma_NPZ=HERE/'sampling_density_gamma_calibration.npz',
        R21_results=HERE/'round21_gamma_development_results.json',R21_predictions=HERE/'round21_gamma_development_validation_predictions.npz')
    for a in ASSETS:sources['global_dense_'+a]=HERE/f'round21_gamma_development_global_dense_{a}.pt'
    return {k:dict(path=str(Path(v).resolve()),sha256=digest(v)) for k,v in sources.items()}


def load_development():
    hashes=source_hashes();archive=read_json(HERE/'round21_gamma_development_results.json')
    for entry in archive['source_hashes'].values():assert digest(entry['path'])==entry['sha256']
    D,_,audit=HB.build_pair(development_only=True,check_npz=True)
    assert all(not len(d['te']) and (d['date']<HB.TEST_START).all() for d in D.values())
    if audit['preprocessing']!=archive['builder']['preprocessing']:raise PermissionError('Backbone preprocessing changed')
    reference=S21.REF.load_gamma_reference();backbones={};fixed=archive['preprocessing']
    for a in ASSETS:
        key='global_dense_'+a;row=archive['checkpoints'][key]
        assert Path(row['path']).resolve()==Path(hashes[key]['path']) and digest(row['path'])==row['sha256']
        own=torch.load(row['path'],weights_only=True,map_location='cpu')
        assert own['metadata']['preprocessing']==fixed and own['metadata']['fit']==archive['fits']['global_dense'][a]
        model=FastGammaVarianceFilter(D,fixed,reference,'global_dense',11,assets=(a,))
        model.load_state_dict(own['state_dict'],strict=True);model.eval();model.requires_grad_(False)
        backbones[a]=model
    caches=build_caches(D,backbones)
    # All three zero heads must reproduce archived global forecasts BEFORE fitting.
    gates={}
    with np.load(HERE/'round21_gamma_development_validation_predictions.npz',allow_pickle=False) as saved:
        for mode in MODES:
            m=PredictivePath(caches,mode,11);gates[mode]={}
            with torch.no_grad():
                for a in D:
                    ids=D[a]['origin'][D[a]['va']]
                    np.testing.assert_array_equal(saved[a+'_date'],D[a]['date'][ids])
                    np.testing.assert_array_equal(saved[a+'_target'],D[a]['y'][D[a]['va']])
                    pred=m.predict(a,ids)[0].numpy();expected=saved[a+'_global_dense']
                    np.testing.assert_allclose(pred,expected,rtol=1e-11,atol=1e-14)
                    gates[mode][a]=dict(max_absolute_error=float(np.abs(pred-expected).max()),rtol=1e-11,atol=1e-14)
    return D,caches,audit,hashes,gates


def run_development(tag):
    """Later authorized pre2026 development only; NOT run during code delivery."""
    assert re.fullmatch(r'round22_predictive_path_[A-Za-z0-9_-]+',tag)
    outputs=dict(results=HERE/f'{tag}_results.json',predictions=HERE/f'{tag}_validation_predictions.npz')
    checkpoints={(m,s):HERE/f'{tag}_{m}_seed{s}.pt' for m in MODES for s in SEEDS}
    for p in [*outputs.values(),*checkpoints.values()]:
        if p.exists():raise FileExistsError('Protected existing output: '+str(p))
    started=time.time();D,caches,audit,hashes,gates=load_development()
    pred={a:{} for a in D};arrays=dict(horizons=np.array(S21.HORIZONS));fits={};checkpoint_info={};ensemble_decomposition={}
    for mode in MODES:
        fits[mode]=[];component_aux={a:[] for a in D}
        for seed in SEEDS:
            model,values,aux,info=train_one(caches,D,mode,seed);fits[mode].append(info)
            for a in D:
                name=f'{mode}_seed{seed}';pred[a][name]=values[a];component_aux[a].append(aux[a])
                for key,value in aux[a].items():arrays[f'{a}_{name}_{key}']=value
            p=checkpoints[(mode,seed)]
            own=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                metadata=dict(mode=mode,seed=seed,selected_epoch=info['best_joint_epoch'],source_hashes=hashes,
                    original_backbone='R21 global_dense independently selected2025; no process/prior/preproc training'))
            with p.open('xb') as handle:torch.save(own,handle)
            torch.load(p,weights_only=True,map_location='cpu')
            checkpoint_info[f'{mode}_seed{seed}']=dict(path=str(p),sha256=digest(p))
        ensemble_decomposition[mode]={}
        for a in D:
            name=mode+'_ensemble';pred[a][name]=np.mean([pred[a][f'{mode}_seed{s}'] for s in SEEDS],axis=0)
            moments={key:np.mean([aux[key] for aux in component_aux[a]],axis=0) for key in ('EqS','EqZ','EqSZ','daily')}
            moments['risk_resolution_covariance']=moments['EqSZ']-moments['EqS']*moments['EqZ']
            arrays.update({f'{a}_{name}_{key}':value for key,value in moments.items()})
            ids=D[a]['va'];origins=D[a]['origin'][ids];value=pred[a][name];y=D[a]['y'][ids]
            labels=future_event_labels(D[a],origins,D[a]['n'])
            ensemble_decomposition[mode][a]=dict(
                QLIKE=float((np.log(value)+y/value-np.log(D[a]['ewma'][ids])[:,None]).mean()),
                BCE=float(event_bce(torch.tensor(moments['EqZ'],dtype=DTYPE),labels)),
                mean_component_KL_per_day=float(np.mean([aux['KL'].mean()/22 for aux in component_aux[a]])) if mode!='direct_marked_control' else None,
                KL_note='mean component path KL, not the KL of the ensemble mixture' if mode!='direct_marked_control' else 'no path KL for direct control',
                risk_resolution_covariance_mean=float(moments['risk_resolution_covariance'].mean()))
    for a in D:
        q=D[a]['origin'][D[a]['va']]
        arrays.update({a+'_date':D[a]['date'][q],a+'_origin':q,a+'_label_end':D[a]['date'][q+22],a+'_target':D[a]['y'][D[a]['va']]})
        arrays.update({a+'_'+key:value for key,value in pred[a].items()})
    joint_scores={}
    per_asset_scores={a:S21.S17.metrics(pred[a],D[a]) for a in D}
    for name in pred[tuple(D)[0]]:
        individual={}
        for a in D:
            ids=D[a]['va'];y=D[a]['y'][ids];v=pred[a][name]
            individual[a]=dict(known_scale_QLIKE=float((np.log(v)+y/v-np.log(D[a]['ewma'][ids])[:,None]).mean()),
                raw_QLIKE=float((np.log(v)+y/v).mean()),MSE=float(((v-y)**2).mean()))
        joint_scores[name]=dict(asset_scores=individual,
            **{key:sum(row[key] for row in individual.values())/len(D) for key in ('known_scale_QLIKE','raw_QLIKE','MSE')},
            per_h={str(h):{key:sum(per_asset_scores[a][name][str(h)][key] for a in D)/len(D)
                    for key in ('mean_QLIKE','loss_difference_vs_EWMA22','MSE')} for h in S21.HORIZONS})
    report=dict(status='REUSED2025 DEVELOPMENT; old2026 is already used, not fresh confirmation',
        sources=hashes,zero_head_global_gate=gates,config=dict(modes=MODES,seeds=SEEDS,lr=.01,clip=2,epochs=80,patience=15,min_epochs=20,
            daily_event_BCE_weight=.05,bank_output_penalty=.005,path_KL_per_day_weight=.05),
        output_penalty_rule='equal-asset mean_bank SUM over all active bucket/channel centered squares (9 or6 outputs)',
        regularization_note='Path KL is part of the method; direct is a marked control without path KL, not fully matched regularizers',
        head_bounds='raw a/c=.25tanh,b=tanh; centered bounds .5/2/.5; path daily a/c dividedL1/4/17; direct a/c NOT dividedL; b unchanged',
        ctx='F17 unchanged, log(postmean/mu), log1p(postvar/postmean²), observed flag; riskdims normalized asset pre25 bank only',
        quote_definition='rawP>0, not no-trade',prior_future_coupling='F-K returns joint E[SZ]; covariance saved, never replace by E[S]*E[Z]',
        zero_anchor='old global analytic daily * EqSZ/(discrete-T base Markov meanS * base active probability); no zero-head branch',
        frozen_backbone=True,cache_statistics={a:dict(bank_count=len(caches[a]['bank_index']),risk_mean=caches[a]['risk_mean'].tolist(),
            risk_std=caches[a]['risk_std'].tolist(),
            max_relative_Markov_vs_analytic_mean=float(((caches[a]['base_markov_mean']-caches[a]['base_analytic_mean'])/caches[a]['base_analytic_mean']).abs().max()),
            theta={k:float(caches[a][k]) for k in ('mu','phi','q','kappa')}) for a in D},
        fits=fits,ensemble_decomposition=ensemble_decomposition,seed_note='repeated initialization on the same datasets, not independent datasets',
        checkpoints=checkpoint_info,scores=per_asset_scores,joint_scores=joint_scores,
        builder=audit,elapsed_seconds=time.time()-started)
    assert source_hashes()==hashes
    with outputs['predictions'].open('xb') as handle:np.savez_compressed(handle,**arrays)
    with outputs['results'].open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    flags=ap.add_mutually_exclusive_group();flags.add_argument('--check-only',action='store_true');flags.add_argument('--run-development',action='store_true')
    ap.add_argument('--tag',default='round22_predictive_path_development');args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.run_development:run_development(args.tag)
    else:print(json.dumps(dict(checks=purechecks(),source_sha256=digest(__file__)),indent=2),flush=True)


if __name__=='__main__':main()
