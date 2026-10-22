"""Pure Torch Gibbs tilt of fixed positive-risk and quote-resolution paths.

Future day k first advances S with T, then applies the potential at (S_k,Z_k).
Z=1 means future reported range>0, not physical trading. The reference assumes
independent Bernoulli Z_k with origin-conditioned probabilities p_k, independent
of the Markov state path. Coefficients are supplied by a causal caller, already
bounded to[-3,3]. No data readers, training, economic-IV or true-quote claims.
Feynman--Kac, Doob transforms and relative entropy are established mathematics.
"""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import torch
from torch import nn


def _reference_check(p0,transition,states):
    if p0.ndim!=2 or transition.shape!=(p0.shape[1],p0.shape[1]) or states.shape!=(p0.shape[1],):
        raise ValueError('Expected p0[B,S], transition[S,S], state_values[S]')
    if not (torch.isfinite(p0).all() and torch.isfinite(transition).all() and torch.isfinite(states).all()):
        raise ValueError('Reference must be finite')
    if (p0<0).any() or (transition<0).any() or (states<=0).any():
        raise ValueError('Reference masses nonnegative, states positive')
    tol=1e-12 if p0.dtype==torch.float64 else 2e-6
    if not torch.allclose(p0.sum(-1),torch.ones_like(p0[:,0]),rtol=0,atol=tol):
        raise ValueError('p0 must sum to one')
    if not torch.allclose(transition.sum(-1),torch.ones_like(states),rtol=0,atol=tol):
        raise ValueError('Transition must be row stochastic')


def baseline_markov_marginals(p0_state,transition,state_values,steps):
    """Cacheable future S1..SK baseline, never detached automatically.

    A cache must correspond to the same fixed reference and horizon. Reusing an
    autograd graph across backward calls is the caller's responsibility; the
    intended cache is a fixed reference with no learned tensors.
    """
    if not isinstance(steps,int) or steps<1:raise ValueError('steps must be positive integer')
    p0=torch.as_tensor(p0_state)
    if not p0.is_floating_point():raise ValueError('p0 must be floating')
    T=torch.as_tensor(transition,dtype=p0.dtype,device=p0.device)
    states=torch.as_tensor(state_values,dtype=p0.dtype,device=p0.device)
    _reference_check(p0,T,states)
    sequence=[];mass=p0
    for _ in range(steps):
        mass=mass@T;sequence.append(mass)
    marginals=torch.stack(sequence,dim=1)
    return dict(state_marginals=marginals,E_S=(marginals*states).sum(-1))


def risk_quote_path_tilt(p0_state,transition,state_values,psi,base_active_prob,coefficients,*,baseline_cache=None):
    """Return joint tilted future moments, marginals, logZ and KL[q||reference].

    Inputs: p0[B,S], T[S,S], positive states[S], psi[S] in[-1,1] nondecreasing
    in state, p[B,K] in[0,1], coefficients[B,K,3]=(a,b,c) in[-3,3]. Exact p0/1
    endpoints use centered Bernoulli algebra, never log/clamp probabilities.
    Backward messages follow beta=(g_next*beta_next)@T.T/c_next, with an extra
    common max rescaling for stability; that scalar cancels in alpha*beta.
    Zero-anchor equalities hold to reported floating-point precision without a
    branch that would destroy learning gradients at zero coefficients.
    """
    if not isinstance(coefficients,torch.Tensor) or not coefficients.is_floating_point():
        raise ValueError('coefficients must be a floating Torch tensor')
    if coefficients.ndim!=3 or coefficients.shape[-1]!=3 or coefficients.shape[0]<1 or coefficients.shape[1]<1:
        raise ValueError('coefficients must be[B,K,3], B/K positive')
    if not torch.isfinite(coefficients).all() or (coefficients.abs()>3).any():
        raise ValueError('Caller coefficients must be finite and bounded[-3,3]')
    B,K,_=coefficients.shape
    def tensor(value):return torch.as_tensor(value,dtype=coefficients.dtype,device=coefficients.device)
    p0,T,states,score,p=(tensor(value) for value in (p0_state,transition,state_values,psi,base_active_prob))
    _reference_check(p0,T,states)
    if p0.shape[0]!=B or score.shape!=states.shape or p.shape!=(B,K):
        raise ValueError('Reference/psi/p dimensions mismatch')
    if not torch.isfinite(score).all() or (score.abs()>1).any():raise ValueError('psi must be finite bounded[-1,1]')
    ordered=score[torch.argsort(states)]
    if (ordered[1:]<ordered[:-1]).any():raise ValueError('psi must be monotone in state_values')
    if not torch.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('base_active_prob must be in[0,1]')
    a,b,c=coefficients.unbind(-1)
    p3=p[...,None];d=b[...,None]+c[...,None]*score
    # E exp[d*(Z-p)] using two CENTERED terms. At p=0/1 this is exactly1,
    # with finite, correct one-sided p gradients and no endpoint branch/clamp.
    inactive=(1-p3)*torch.exp(-d*p3)
    active=p3*torch.exp(d*(1-p3))
    bernoulli_mgf=inactive+active
    qZ=active/bernoulli_mgf
    logg=a[...,None]*score+torch.log(bernoulli_mgf)
    shift=logg.amax(dim=-1)
    g=torch.exp(logg-shift[...,None])
    forward=[];normalizers=[];mass=p0
    for day in range(K):
        moved=mass@T
        raw=moved*g[:,day]
        normalizer=raw.sum(-1)
        if not (torch.isfinite(normalizer)&(normalizer>0)).all():
            raise FloatingPointError('Tilt normalization lost positive support')
        mass=raw/normalizer[:,None]
        forward.append(mass);normalizers.append(normalizer)
    alpha=torch.stack(forward,dim=1);day_c=torch.stack(normalizers,dim=1)
    logZ=(torch.log(day_c)+shift).sum(-1)
    beta=torch.ones_like(p0);posteriors=[None]*K;rescales=[None]*K
    for day in range(K-1,-1,-1):
        if day<K-1:
            beta=((g[:,day+1]*beta)@T.T)/day_c[:,day+1,None]
            rescale=beta.amax(dim=-1)
            if not (torch.isfinite(rescale)&(rescale>0)).all():
                raise FloatingPointError('Backward message lost finite support')
            beta=beta/rescale[:,None]
        else:rescale=torch.ones_like(day_c[:,day])
        raw=alpha[:,day]*beta;total=raw.sum(-1)
        if not (torch.isfinite(total)&(total>0)).all():
            raise FloatingPointError('Smoothing marginal lost support')
        posteriors[day]=raw/total[:,None];rescales[day]=rescale
    qS=torch.stack(posteriors,dim=1)
    ES=(qS*states).sum(-1)
    # Anchor the state-independent part algebraically, not with a zero-c branch.
    # This yields exact Cov0 at c0 and keeps its nonzero derivative with c.
    r0=qZ[...,:1];difference=qZ-r0
    EZ=r0[...,0]+(qS*difference).sum(-1)
    ESZ=(qS*states*qZ).sum(-1)  # joint expectation, NEVER ES*EZ substitution
    covariance=(qS*(states-ES[...,None])*difference).sum(-1)
    expected_daily=(qS*(a[...,None]*score+d*(qZ-p3))).sum(-1)
    expected_potential=expected_daily.sum(-1)
    KL=expected_potential-logZ  # no clamp; tiny negative roundoff is reported
    if baseline_cache is None:
        baseline=baseline_markov_marginals(p0,T,states,K)
    else:
        baseline={key:tensor(baseline_cache[key]) for key in ('state_marginals','E_S')}
        if baseline['state_marginals'].shape!=(B,K,len(states)) or baseline['E_S'].shape!=(B,K):
            raise ValueError('Cached baseline dimensions mismatch')
        if not all(torch.isfinite(value).all() for value in baseline.values()):
            raise ValueError('Cached baseline must be finite and from the same reference')
    return dict(E_S=ES,E_Z=EZ,E_SZ=ESZ,Cov_SZ=covariance,state_marginals=qS,
        active_prob_given_state=qZ,logZ=logZ,KL=KL,expected_log_potential=expected_potential,
        baseline_state_marginals=baseline['state_marginals'],baseline_E_S=baseline['E_S'],
        baseline_E_Z=p,baseline_E_SZ=baseline['E_S']*p,
        forward_state_marginals=alpha,shifted_day_normalizers=day_c,
        log_potential_shift=shift,backward_rescales=torch.stack(rescales,dim=1))


class RiskQuotePathTilt(nn.Module):
    """Stateless nn.Module facade; learned coefficients live in the caller."""
    def forward(self,p0_state,transition,state_values,psi,base_active_prob,coefficients,*,baseline_cache=None):
        return risk_quote_path_tilt(p0_state,transition,state_values,psi,base_active_prob,coefficients,
                                   baseline_cache=baseline_cache)


def _enumerate(p0,T,states,psi,p,coeff):
    """2states/K3 validation only: all S0..SK and Z1..ZK, including zero paths."""
    B,K,_=coeff.shape;S=len(states);weights=[];refs=[];potentials=[];s_paths=[];z_paths=[]
    for sp in itertools.product(range(S),repeat=K+1):
        for zp in itertools.product((0,1),repeat=K):
            ref=p0[:,sp[0]];h=coeff.new_zeros(B)
            for k in range(K):
                ref=ref*T[sp[k],sp[k+1]]*(p[:,k] if zp[k] else 1-p[:,k])
                a,b,c=coeff[:,k].unbind(-1);ps=psi[sp[k+1]]
                h=h+a*ps+(b+c*ps)*(zp[k]-p[:,k])
            refs.append(ref);potentials.append(h);weights.append(ref*torch.exp(h))
            s_paths.append(sp[1:]);z_paths.append(zp)
    raw=torch.stack(weights,1);reference=torch.stack(refs,1);h=torch.stack(potentials,1)
    Z=raw.sum(1);q=raw/Z[:,None]
    sp=torch.tensor(s_paths,device=coeff.device);zp=coeff.new_tensor(z_paths);sv=states[sp]
    qS=torch.stack([(q[:,:,None]*(sp==i)).sum(1) for i in range(S)],dim=-1)
    qZ_num=torch.stack([(q[:,:,None]*zp*(sp==i)).sum(1) for i in range(S)],dim=-1)
    qZ=qZ_num/torch.where(qS>0,qS,torch.ones_like(qS))
    ES=(q[:,:,None]*sv).sum(1);EZ=(q[:,:,None]*zp).sum(1);ESZ=(q[:,:,None]*sv*zp).sum(1)
    logZ=torch.log(Z);expected=(q*h).sum(1)
    return dict(E_S=ES,E_Z=EZ,E_SZ=ESZ,Cov_SZ=ESZ-ES*EZ,state_marginals=qS,
        active_prob_given_state=qZ,logZ=logZ,KL=expected-logZ,
        path_q=q,path_reference=reference,path_potential=h)


def _objective(out):
    return (out['E_S']*.13+out['E_Z']*.19+out['E_SZ']*.23+out['Cov_SZ']*.31).sum()+ \
        .17*out['logZ'].sum()+.29*out['KL'].sum()+.07*out['state_marginals'][...,0].sum()


def pure_checks():
    dtype=torch.float64
    p0=torch.tensor([[.7,.3],[.4,.6]],dtype=dtype)
    T=torch.tensor([[.8,.2],[.15,.85]],dtype=dtype)
    states=torch.tensor([.4,1.8],dtype=dtype);psi=torch.tanh(torch.log(states))
    p=torch.tensor([[.25,.6,.75],[.1,.9,.4]],dtype=dtype)
    coefficients=torch.tensor([[[.3,-.2,.4],[-.1,.5,.25],[.2,.1,-.3]],
                               [[-.2,.3,.15],[.25,-.4,.5],[.1,.2,-.35]]],dtype=dtype)
    functional=risk_quote_path_tilt
    fast=functional(p0,T,states,psi,p,coefficients)
    exact=_enumerate(p0,T,states,psi,p,coefficients)
    errors={}
    keys=('E_S','E_Z','E_SZ','Cov_SZ','state_marginals','active_prob_given_state','logZ','KL')
    for name in keys:
        errors[name]=float((fast[name]-exact[name]).abs().max())
        torch.testing.assert_close(fast[name],exact[name],rtol=1e-12,atol=1e-12)
    q_reconstructed=exact['path_reference']*torch.exp(exact['path_potential']-fast['logZ'][:,None])
    path_error=float((q_reconstructed-exact['path_q']).abs().max())
    torch.testing.assert_close(q_reconstructed,exact['path_q'],rtol=1e-12,atol=1e-12)
    direct_KL=(exact['path_q']*(torch.log(exact['path_q'])-torch.log(exact['path_reference']))).sum(1)
    torch.testing.assert_close(fast['KL'],direct_KL,rtol=1e-12,atol=1e-12)
    left=coefficients.clone().requires_grad_(True);right=coefficients.clone().requires_grad_(True)
    g1=torch.autograd.grad(_objective(functional(p0,T,states,psi,p,left)),left)[0]
    g2=torch.autograd.grad(_objective(_enumerate(p0,T,states,psi,p,right)),right)[0]
    gradient_error=float((g1-g2).abs().max())
    torch.testing.assert_close(g1,g2,rtol=2e-11,atol=2e-12)
    # Zero potentials: no branch overrides that could kill useful zero-head gradients.
    zero=torch.zeros_like(coefficients).requires_grad_(True)
    untouched=functional(p0,T,states,psi,p,zero)
    baseline=baseline_markov_marginals(p0,T,states,3)
    zero_errors=dict(state=float((untouched['state_marginals']-baseline['state_marginals']).detach().abs().max()),
        ES=float((untouched['E_S']-baseline['E_S']).detach().abs().max()),
        ESZ=float((untouched['E_SZ']-baseline['E_S']*p).detach().abs().max()),
        KL=float(untouched['KL'].detach().abs().max()),logZ=float(untouched['logZ'].detach().abs().max()))
    assert max(zero_errors.values())<2e-13
    assert torch.equal(untouched['Cov_SZ'],torch.zeros_like(p))
    zero_grad=torch.autograd.grad(untouched['E_SZ'].sum(),zero)[0]
    assert torch.isfinite(zero_grad).all() and (zero_grad.abs().sum((0,1))>0).all()
    b_only=torch.zeros_like(coefficients);b_only[:,:,1]=.7
    bvalue=functional(p0,T,states,psi,p,b_only)
    torch.testing.assert_close(bvalue['state_marginals'],baseline['state_marginals'],rtol=1e-13,atol=1e-13)
    assert (bvalue['E_Z']>p).all() and torch.equal(bvalue['Cov_SZ'],torch.zeros_like(p))
    a_only=torch.zeros_like(coefficients);a_only[:,:,0]=.6
    avalue=functional(p0,T,states,psi,p,a_only)
    assert torch.equal(avalue['E_Z'],p) and (avalue['E_S']>baseline['E_S']).all()
    assert torch.equal(avalue['Cov_SZ'],torch.zeros_like(p))
    c_only=torch.zeros_like(coefficients);c_only[:,:,2]=.8
    positive=functional(p0,T,states,psi,p,c_only);negative=functional(p0,T,states,psi,p,-c_only)
    assert (positive['Cov_SZ']>0).all() and (negative['Cov_SZ']<0).all()
    assert (positive['E_SZ']-positive['E_S']*positive['E_Z']).abs().min()>1e-6
    # Endpoint p: exact support, finite coefficient AND one-sided p derivatives.
    endpoint=torch.tensor([[0.,1.,.5],[1.,0.,1.]],dtype=dtype,requires_grad=True)
    epcoef=coefficients.clone().requires_grad_(True)
    ep=functional(p0,T,states,psi,endpoint,epcoef);ee=_enumerate(p0,T,states,psi,endpoint,epcoef)
    for name in ('E_S','E_Z','E_SZ','Cov_SZ','logZ','KL'):
        torch.testing.assert_close(ep[name],ee[name],rtol=1e-12,atol=1e-12)
    endpoints=(endpoint==0)|(endpoint==1)
    assert torch.equal(ep['active_prob_given_state'][endpoints],endpoint[endpoints,None].expand(-1,2))
    gp,gc=torch.autograd.grad(_objective(ep),(endpoint,epcoef),retain_graph=True)
    gp2,gc2=torch.autograd.grad(_objective(ee),(endpoint,epcoef))
    assert torch.isfinite(gp).all() and torch.isfinite(gc).all()
    torch.testing.assert_close(gp,gp2,rtol=2e-11,atol=2e-12)
    torch.testing.assert_close(gc,gc2,rtol=2e-11,atol=2e-12)
    assert torch.equal(gc[:,:,1:][endpoints],torch.zeros_like(gc[:,:,1:][endpoints]))
    # Exact state support zeros stay impossible; no log probability floors.
    deterministic=functional(torch.eye(2,dtype=dtype),torch.eye(2,dtype=dtype),states,psi,p,coefficients)
    assert torch.equal(deterministic['state_marginals'],torch.eye(2,dtype=dtype)[:,None,:].expand(-1,3,-1))
    scaled=functional(p0,T,states*7.3,psi,p,coefficients)
    for name in ('E_S','E_SZ','Cov_SZ'):
        torch.testing.assert_close(scaled[name],fast[name]*7.3,rtol=1e-12,atol=1e-12)
    for name in ('E_Z','state_marginals','logZ','KL'):
        torch.testing.assert_close(scaled[name],fast[name],rtol=1e-12,atol=1e-12)
    cached=functional(p0,T,states,psi,p,coefficients,baseline_cache=baseline)
    for name in keys:torch.testing.assert_close(cached[name],fast[name],rtol=0,atol=0)
    def fn(coef):
        result=functional(p0,T,states,psi,p,coef)
        return result['E_SZ'],result['Cov_SZ'],result['logZ'],result['KL']
    assert torch.autograd.gradcheck(fn,(coefficients.clone().requires_grad_(True),),eps=1e-6,atol=2e-5,rtol=2e-4)
    layer=RiskQuotePathTilt();assert sum(v.numel() for v in layer.parameters())==0
    return dict(enumerated_state_paths=16,enumerated_quote_paths=8,total_paths_per_origin=128,
        output_max_absolute_errors=errors,full_path_probability_error=path_error,coefficient_gradient_error=gradient_error,
        zero_anchor_errors=zero_errors,zero_head_each_channel_gradient_nonzero=True,
        c0_covariance_exact_zero=True,positive_c_positive_covariance=True,negative_c_negative_covariance=True,
        b_only_changes_observability=True,a_only_changes_risk=True,endpoints_exact=True,
        endpoint_probability_and_coefficient_gradients_match_enumeration=True,reference_support_zeros_exact=True,
        scale_units_equivariant_with_same_psi=True,baseline_cache_identical=True,coefficient_gradcheck=True,
        KL_matches_direct_relative_entropy=True,minimum_test_KL=float(fast['KL'].min()),
        stateless_no_parameters=True,positive_c_covariance=positive['Cov_SZ'].tolist(),
        zero_head_channel_gradient_abs_sums=zero_grad.abs().sum((0,1)).tolist())


def main():
    path=Path(__file__).resolve();output=path.with_name('risk_quote_path_tilt_validation.json')
    if output.exists():raise FileExistsError(f'Protected validation: {output}')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    checks=pure_checks()
    report=dict(status='PURE TORCH PATH-TILT MATH VALIDATION',source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checks=checks,reference='p(S0)*prod T(Sprevious,Sday); origin-conditional independent Bernoulli quote resolution byday',
        supported_domain='finite positive state values, monotone bounded psi[-1,1], nonnegative normalized p0/T, p in[0,1], coefficients[-3,3]; zero reference support remains zero',
        time_index='potential at S1..SK after each transition; no potential at S0',
        numerics='daywise logg shift and normalization; backward common max rescaling cancels in smoothed marginals; no probability log/clamp, no KL clamp',
        scale_note='unit scaling of state_values requires invariant psi, e.g. scaling mu in tanh(log(S/mu)) too',
        cache_note='caller supplies baseline cache only for identical fixed reference/horizon; no automatic detach',
        market_data_read=False,real2026_data_read=False,checkpoint_read=False,training_executed=False,old_files_modified=False,
        interpretation='Feynman-Kac/Doob/KL are known; no economic IV or true quote/trading mechanism identification')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
