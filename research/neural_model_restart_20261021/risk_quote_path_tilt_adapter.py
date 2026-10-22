"""Thin R22 call adapter for the frozen risk_quote_path_tilt core.

No extra daily/horizon scaling and no zero-coefficient output branch.
mu is a positive asset-level scalar. Z means reported future range>0.
"""
from pathlib import Path
import hashlib
import json
import torch
import risk_quote_path_tilt as CORE

baseline_markov_marginals=CORE.baseline_markov_marginals
EXPECTED_CORE_SHA256='30c24dfd921ff6a2f5d2fbfb916ed3ccf8f265d705a66f6e1f45bc171ed43b30'


def path_tilt(posterior,T,grid,mu,base_active_prob,a,b,c,*,baseline_cache=None):
    """a/b/c:[B,K] are ALREADY daily-scaled. No scaling is applied here.

    Returns EqS/EqZ/EqSZ/Cov:[B,K], KL/logZ:[B], state_marginals and
    qZ_given_state:[B,K,S], baseEqS/baseEqZ/baseEqSZ and base_state_marginals.
    baseline_cache is the frozen core helper's cache from identical p0/T/grid/K.
    """
    if not isinstance(posterior,torch.Tensor) or not posterior.is_floating_point():
        raise ValueError('posterior must be a floating Torch tensor')
    dtype,device=posterior.dtype,posterior.device
    aa,bb,cc=(torch.as_tensor(x,dtype=dtype,device=device) for x in (a,b,c))
    if aa.ndim!=2 or bb.shape!=aa.shape or cc.shape!=aa.shape:
        raise ValueError('a/b/c must share[B,K] shape')
    states=torch.as_tensor(grid,dtype=dtype,device=device)
    reference_mean=torch.as_tensor(mu,dtype=dtype,device=device)
    if reference_mean.numel()!=1 or not (torch.isfinite(reference_mean)&(reference_mean>0)).all():
        raise ValueError('mu must be a finite positive asset-level scalar')
    reference_mean=reference_mean.reshape(())
    psi=torch.tanh(torch.log(states/reference_mean))
    result=CORE.risk_quote_path_tilt(posterior,T,states,psi,base_active_prob,
        torch.stack([aa,bb,cc],dim=-1),baseline_cache=baseline_cache)
    return dict(EqS=result['E_S'],EqZ=result['E_Z'],EqSZ=result['E_SZ'],Cov=result['Cov_SZ'],
        KL=result['KL'],logZ=result['logZ'],state_marginals=result['state_marginals'],
        qZ_given_state=result['active_prob_given_state'],baseEqS=result['baseline_E_S'],
        baseEqZ=result['baseline_E_Z'],baseEqSZ=result['baseline_E_SZ'],
        base_state_marginals=result['baseline_state_marginals'],
        expected_log_potential=result['expected_log_potential'])


def pure_checks():
    dtype=torch.float64
    posterior=torch.tensor([[.7,.3],[.4,.6]],dtype=dtype)
    T=torch.tensor([[.8,.2],[.15,.85]],dtype=dtype);grid=torch.tensor([.4,1.8],dtype=dtype)
    mu=torch.tensor(1.2,dtype=dtype);p=torch.tensor([[.25,.6,.75],[.1,.9,.4]],dtype=dtype)
    values=torch.tensor([[[.3,-.2,.4],[-.1,.5,.25],[.2,.1,-.3]],
                         [[-.2,.3,.15],[.25,-.4,.5],[.1,.2,-.35]]],dtype=dtype)
    a,b,c=values.unbind(-1);result=path_tilt(posterior,T,grid,mu,p,a,b,c)
    raw=CORE.risk_quote_path_tilt(posterior,T,grid,torch.tanh(torch.log(grid/mu)),p,values)
    mapping=dict(EqS='E_S',EqZ='E_Z',EqSZ='E_SZ',Cov='Cov_SZ',KL='KL',logZ='logZ',
        state_marginals='state_marginals',qZ_given_state='active_prob_given_state',
        baseEqS='baseline_E_S',baseEqZ='baseline_E_Z',baseEqSZ='baseline_E_SZ',
        base_state_marginals='baseline_state_marginals',expected_log_potential='expected_log_potential')
    for new,old in mapping.items():torch.testing.assert_close(result[new],raw[old],rtol=0,atol=0)
    lhs=[x.detach().clone().requires_grad_(True) for x in (a,b,c)]
    rhs=[x.detach().clone().requires_grad_(True) for x in (a,b,c)]
    out=path_tilt(posterior,T,grid,mu,p,*lhs)
    direct=CORE.risk_quote_path_tilt(posterior,T,grid,torch.tanh(torch.log(grid/mu)),p,torch.stack(rhs,-1))
    g1=torch.autograd.grad(out['EqSZ'].sum()+.3*out['KL'].sum(),lhs)
    g2=torch.autograd.grad(direct['E_SZ'].sum()+.3*direct['KL'].sum(),rhs)
    gradient_error=max(float((u-v).abs().max()) for u,v in zip(g1,g2))
    for u,v in zip(g1,g2):
        assert torch.isfinite(u).all();torch.testing.assert_close(u,v,rtol=0,atol=0)
    zero=[torch.zeros_like(a,requires_grad=True) for _ in range(3)]
    untouched=path_tilt(posterior,T,grid,mu,p,*zero)
    torch.testing.assert_close(untouched['EqSZ'],untouched['baseEqSZ'],rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(untouched['EqS'],untouched['baseEqS'],rtol=1e-14,atol=1e-14)
    assert untouched['KL'].abs().max()<1e-14
    gradients=torch.autograd.grad(untouched['EqSZ'].sum(),zero)
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients)
    cache=baseline_markov_marginals(posterior,T,grid,3)
    cached=path_tilt(posterior,T,grid,mu,p,a,b,c,baseline_cache=cache)
    for key in result:torch.testing.assert_close(cached[key],result[key],rtol=0,atol=0)
    return dict(aliases_exact=True,already_daily_coefficients_unchanged=True,mu_psi_formula_exact=True,
        gradients_exact_to_core=True,max_gradient_difference=gradient_error,zero_branch_not_used=True,
        zero_initialization_anchor_error=float((untouched['EqSZ']-untouched['baseEqSZ']).detach().abs().max()),
        zero_each_channel_gradient_nonzero=True,zero_gradient_abs_sums=[float(g.abs().sum()) for g in gradients],
        cache_exact=True)


def main():
    source=Path(__file__).resolve();output=source.with_name('risk_quote_path_tilt_adapter_validation.json')
    if output.exists():raise FileExistsError(f'Protected validation: {output}')
    core=Path(CORE.__file__).resolve();before=hashlib.sha256(core.read_bytes()).hexdigest()
    if before!=EXPECTED_CORE_SHA256:raise ValueError('Frozen core source changed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    checks=pure_checks();assert hashlib.sha256(core.read_bytes()).hexdigest()==before
    report=dict(status='PURE R22 CALL ADAPTER VALIDATION',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        core_sha256=before,core_unchanged=True,checks=checks,
        signature='path_tilt(posterior,T,grid,mu,base_active_prob,a,b,c,*,baseline_cache=None)',
        output_keys=['EqS','EqZ','EqSZ','Cov','KL','logZ','state_marginals','qZ_given_state',
                     'baseEqS','baseEqZ','baseEqSZ','base_state_marginals','expected_log_potential'],
        no_extra_daily_scaling=True,no_zero_coefficient_branch=True,market_data_read=False,
        real2026_data_read=False,checkpoint_read=False,training_executed=False)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
