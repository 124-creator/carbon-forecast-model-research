"""R21 performance-only subclass: replace repeated IndexBackward with one Unbind.

The original source, model, objective, parameters and data protocol are unchanged.
No import-time I/O. CLI runs synthetic equivalence checks and benchmarks only.
This file never trains a real candidate or changes/stops any existing job.
"""
from __future__ import annotations
import copy
import hashlib
import inspect
import json
from pathlib import Path
from statistics import median
import textwrap
import time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
import round21_gamma_observation_filter as S21

transition_matrix=S21.transition_matrix
initial_ig=S21.initial_ig
R20,ASSETS,DTYPE=S21.R20,S21.ASSETS,S21.DTYPE


class FastGammaVarianceFilter(S21.GammaVarianceFilter):
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
        components=iter(positive_component.unbind(0))
        posterior=initial_ig(grid,mu)
        means,variances,entropies,posts,priors,marginal,Mpost,clipmass,meanerror,boundary=[],[],[],[],[],[],[],[],[],[]
        for t in range(stop):
            clipmass.append(posterior@transition['row_clip_mass'])
            meanerror.append(posterior@transition['row_mean_error'])
            prior=posterior@matrix
            if bool(observed[t]):
                component=next(components)
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


def assert_only_row_access_changed():
    old=textwrap.dedent(inspect.getsource(S21.GammaVarianceFilter.filter_asset))
    new=textwrap.dedent(inspect.getsource(FastGammaVarianceFilter.filter_asset))
    old=old.replace("lookup=torch.full((stop,),-1,dtype=torch.long,device=grid.device);lookup[ids]=torch.arange(len(ids),device=grid.device)",
                    'components=iter(positive_component.unbind(0))')
    old=old.replace('component=positive_component[lookup[t]]','component=next(components)')
    assert old==new,'The override changed more than the two row-access lines'


def nonzero_pair(datasets,fixed,reference,mode):
    bounded={}
    for a in ASSETS:
        process=S21.VarianceProcess()
        with torch.no_grad():
            process.rawmu.add_(.11);process.rawphi.sub_(.07);process.rawq.add_(.13);process.rawk.sub_(.17)
        bounded[a]=process.bounded()
    original=S21.GammaVarianceFilter(datasets,fixed,reference,mode,22,
                frozen_theta=bounded if mode=='neural_density' else None)
    with torch.no_grad():
        if mode=='neural_density':
            original.prior_net[-1].weight.copy_(torch.linspace(-.12,.17,28,dtype=DTYPE).reshape(7,4))
            original.prior_net[-1].bias.copy_(torch.linspace(-.03,.04,7,dtype=DTYPE))
        else:
            for process in original.processes.values():
                process.rawmu.add_(.11);process.rawphi.sub_(.07);process.rawq.add_(.13);process.rawk.sub_(.17)
    fast=copy.deepcopy(original);fast.__class__=FastGammaVarianceFilter
    return original,fast


def equivalence_checks():
    assert_only_row_access_changed()
    D=S21.synthetic_pair();fixed=S21.preprocessing(D);reference=S21.synthetic_reference();T=S21.make_tensors(D)
    report={}
    for mode in S21.MODES:
        original,fast=nonzero_pair(D,fixed,reference,mode)
        original.zero_grad(set_to_none=True);fast.zero_grad(set_to_none=True)
        lo,io=S21.objective(original,D,T);lf,iff=S21.objective(fast,D,T)
        torch.testing.assert_close(lo,lf,rtol=0,atol=0)
        for key in ('data','sensor','kappa_penalty'):
            torch.testing.assert_close(io[key],iff[key],rtol=0,atol=0)
        lo.backward();lf.backward()
        gradients={}
        for (name,left),(other,right) in zip(original.named_parameters(),fast.named_parameters()):
            assert name==other and left.grad is not None and right.grad is not None
            torch.testing.assert_close(left.grad,right.grad,rtol=2e-10,atol=2e-12)
            gradients[name]=dict(max_absolute_error=float((left.grad-right.grad).abs().max()),
                relative_to_max_grad=float((left.grad-right.grad).abs().max()/left.grad.abs().max().clamp_min(1e-12)))
        with torch.no_grad():
            for a in ASSETS:
                left=original.filter_asset(a,D[a],T[a],D[a]['n']);right=fast.filter_asset(a,D[a],T[a],D[a]['n'])
                for name in left:
                    if isinstance(left[name],dict):
                        for key in left[name]:torch.testing.assert_close(left[name][key],right[name][key],rtol=0,atol=0)
                    else:torch.testing.assert_close(left[name],right[name],rtol=0,atol=0)
                torch.testing.assert_close(right['posterior'][12],right['prior'][12],rtol=0,atol=0)
                changed={key:value.clone() for key,value in T[a].items()}
                future=D[a]['cut']+3;changed['proxy_raw'][future:]*=4;changed['F17'][future:,2]+=6
                for model in (original,fast):
                    first=model.filter_asset(a,D[a],T[a],D[a]['n'])
                    second=model.filter_asset(a,D[a],changed,D[a]['n'])
                    torch.testing.assert_close(first['posterior'][:future],second['posterior'][:future],rtol=0,atol=0)
                    q=np.array([future-1]);p=model.forecast(a,T[a],q,first)[0];pp=model.forecast(a,changed,q,second)[0]
                    torch.testing.assert_close(p,pp,rtol=0,atol=0)
        report[mode]=dict(objective=float(lo.detach()),objective_absolute_error=float((lo-lf).detach().abs()),
            gradients=gradients,all_filter_outputs_exact=True,zero_rows_exact=True,future_causality_exact=True)
    return report


def long_synthetic_pair(cut):
    D=S21.synthetic_pair();rng=np.random.default_rng(2106)
    for j,a in enumerate(ASSETS):
        n=cut+34;old=D[a];observed=np.ones(n,dtype=bool);observed[12::101]=False
        volume=rng.normal(.1*j,.8,n);features=np.column_stack([rng.uniform(-.8,.8,n),rng.uniform(-.8,.8,n),volume])
        proxy=.0005*np.exp(.3*rng.normal(size=n));proxy[~observed]=0
        origins=np.r_[np.arange(100,cut-22,11),cut+2]
        D[a]=dict(old,n=n,cut=cut,date=np.r_[np.datetime64('2024-01-01')+np.arange(cut),
            np.datetime64('2025-01-01')+np.arange(n-cut)],observed=observed,proxy_raw=proxy,F17=features,
            log_proxy=np.log(np.maximum(proxy,1e-300)),even=np.zeros((n,3)),
            cell_b=np.zeros(n),cell_R=np.full(n,.3),bank_index=np.flatnonzero(observed&(np.arange(n)>=21)&(np.arange(n)<cut)),
            origin=origins,tr=np.arange(len(origins)-1),va=np.array([len(origins)-1]),
            y=np.full((len(origins),3),.0005),ewma=np.full(len(origins),.0005))
    return D


def full_objective_benchmark(cut=640,repetitions=2):
    D=long_synthetic_pair(cut);fixed=S21.preprocessing(D);reference=S21.synthetic_reference();T=S21.make_tensors(D)
    original,fast=nonzero_pair(D,fixed,reference,'shape_dense')
    # Single-asset full original objective, to reduce interference with the live job.
    original.assets=(ASSETS[0],);fast.assets=(ASSETS[0],)
    results={}
    for label,model in (('indexed',original),('unbind',fast)):
        times=[]
        for _ in range(repetitions):
            model.zero_grad(set_to_none=True);start=time.perf_counter()
            loss,_=S21.objective(model,D,T);forward=time.perf_counter()-start
            start=time.perf_counter();loss.backward();backward=time.perf_counter()-start
            times.append(dict(forward_seconds=forward,backward_seconds=backward,total_seconds=forward+backward))
        results[label]={key:median(row[key] for row in times) for key in times[0]}
    results.update(train_prefix=cut,grid=256,M=7,repetitions=repetitions,assets=1,
        backward_speedup=results['indexed']['backward_seconds']/results['unbind']['backward_seconds'],
        total_speedup=results['indexed']['total_seconds']/results['unbind']['total_seconds'],
        caveat='same host while original real job runs; timing is noisy; synthetic only, no optimizer step')
    return results


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source=Path(S21.__file__);before=hashlib.sha256(source.read_bytes()).hexdigest()
    with ExitStack() as stack:
        for name in ('pandas.read_csv','numpy.load','pathlib.Path.read_text','torch.load',
                     'sampling_density_gamma_reference.load_gamma_reference','holdout_2026_range_builder.build_pair'):
            stack.enter_context(patch(name,side_effect=AssertionError('real input forbidden in purebench')))
        equivalence=equivalence_checks();bench=full_objective_benchmark()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    print(json.dumps(dict(equivalence=equivalence,full_objective_benchmark=bench,
        original_source_sha256=before,fast_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_source_unchanged=True,main_job_untouched=True,real_model_training_executed=False),indent=2),flush=True)


if __name__=='__main__':main()
