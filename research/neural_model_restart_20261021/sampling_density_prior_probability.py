"""Probability-prior API over the already-frozen density moment library.

The old module/JSON/NPZ are never changed. Posterior q*likelihood/Z has no
log(q), clamp(q), or zero-boundary gradient cutoff. c_M comes only from the
frozen train_audit (not held diagnostics). This library models effective M,
not continuous activity A, real trade counts, or a market forecasting model.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import sampling_density_moments as FROZEN

HERE=Path(__file__).resolve().parent
CALIBRATION=HERE/'sampling_density_moments_calibration.json'


def training_c_M(payload):
    """Read only the training audit; leave any held fields completely unused."""
    values=[]
    for M in FROZEN.MS:
        row=payload['train_audit'][str(M)]
        if row['global_logP']['n']!=payload['config']['train_n']:
            raise ValueError('Train c_M sample provenance mismatch')
        values.append(float(row['global_mean_P']))
    values=np.asarray(values,dtype=float)
    if not (np.isfinite(values).all() and (values>0).all()):
        raise ValueError('Train c_M must be positive finite')
    return values


class ProbabilityDensityMomentDecoder(FROZEN.DensityMomentDecoder):
    """forward(cell_id,q,base_R); forward_logits keeps the original API.

    q is a normalized, nonnegative probability tensor and may have exact zeros.
    The posterior and moment gradients remain correct at a dense mixture gate
    alpha=0. Shannon entropy has an infinite directional derivative when mass
    enters a zero component: by default its VALUES are detached diagnostics.
    entropy_grad=True is allowed only for positive posterior on every possible
    component; a boundary request raises rather than inventing a finite gradient.
    """
    def __init__(self,calibration,train_c_M):
        super().__init__(calibration)
        c=torch.as_tensor(train_c_M,dtype=torch.float64)
        if c.shape!=(7,) or not (torch.isfinite(c)&(c>0)).all():
            raise ValueError('train_c_M must be seven finite positive train expectations')
        self.register_buffer('likelihood',torch.tensor(calibration['p_cell_given_M'],dtype=torch.float64))
        self.register_buffer('train_c_M',c.clone())

    @property
    def c_M(self):
        return self.train_c_M

    @property
    def c_M_relative_to_dense(self):
        # Exact dense1; does not silently replace an older model's proxy frame.
        return self.train_c_M/self.train_c_M[-1]

    def forward_logits(self,cell_id,prior_logits,base_R):
        return super().forward(cell_id,prior_logits,base_R)

    def forward(self,cell_id,q,base_R,*,entropy_grad=False):
        if not q.is_floating_point() or q.ndim<1 or q.shape[-1]!=7:
            raise ValueError('q must be floating with final dimension7')
        if not torch.isfinite(q).all() or (q<0).any():
            raise ValueError('q must be finite and nonnegative')
        tolerance=1e-12 if q.dtype==torch.float64 else 1e-6
        if not torch.allclose(q.sum(dim=-1),torch.ones_like(q[...,0]),rtol=0,atol=tolerance):
            raise ValueError('q must sum to one; not silently renormalized')
        cell=torch.as_tensor(cell_id,device=q.device)
        if cell.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):
            raise ValueError('cell_id must be integer')
        if tuple(cell.shape)!=tuple(q.shape[:-1]) or ((cell<0)|(cell>=31)).any():
            raise ValueError('cell_id shape/range invalid; zero ranges must be skipped')
        cell=cell.long()
        likelihood=self.likelihood.to(q)[:,cell].movedim(0,-1)
        product=q*likelihood  # NO log(q) or clamp: derivative at q_i=0 survives.
        total=product.sum(dim=-1,keepdim=True)
        if not (torch.isfinite(total)&(total>0)).all():
            raise ValueError('Prior probability gives no mass to possible densities')
        w=product/total
        if entropy_grad:
            possible=likelihood>0
            if ((w==0)&possible).any():
                raise ValueError('Shannon entropy boundary derivative is infinite; use diagnostic entropy')
            safe=torch.where(possible,w,torch.ones_like(w))
            entropy=-(w*torch.log(safe)).sum(dim=-1)
        else:
            with torch.no_grad():
                safe=torch.where(w>0,w,torch.ones_like(w))
                entropy=-(w*torch.log(safe)).sum(dim=-1)
        b=self.bias.to(q)[:,cell].movedim(0,-1)
        R=self.variance.to(q)[:,cell].movedim(0,-1)
        difference=b-b[...,-1:];ratio=R/R[...,-1:]
        delta=(w*difference).sum(dim=-1)
        relative=(w*ratio).sum(dim=-1)
        between=(w*(difference-delta[...,None]).square()).sum(dim=-1)
        base=torch.as_tensor(base_R,dtype=q.dtype,device=q.device)
        base=torch.broadcast_to(base,q.shape[:-1])
        if not (torch.isfinite(base)&(base>0)).all():raise ValueError('base_R must be positive finite')
        return dict(w=w,ElogM=(w*self.log_M.to(q)).sum(dim=-1),entropy=entropy,Delta_b=delta,
            relative_variance=relative,between_bias_variance=between,Rmix=base*relative+between)

    forward_probabilities=forward


def load_probability_decoder(json_path=CALIBRATION):
    path=Path(json_path)
    payload=json.loads(path.read_text(encoding='utf-8'))
    calibration=FROZEN.validate_artifact_payload(payload,path.parent/payload['npz_filename'])
    return ProbabilityDensityMomentDecoder(calibration,training_c_M(payload))


def pure_checks(decoder,payload):
    ids=torch.tensor([0,1,6,30]);dtype=torch.float64
    dense=torch.zeros((4,7),dtype=dtype);dense[:,-1]=1
    base=torch.tensor([.2,.3,.4,.5],dtype=dtype)
    pi=torch.tensor([.12,.11,.17,.14,.16,.13,.17],dtype=dtype).expand(4,7)
    def function(alpha):
        value=decoder(ids,(1-alpha)*dense+alpha*pi,base)
        return value['w'],value['Delta_b'],value['Rmix'],value['ElogM']
    alpha=torch.tensor(0.,dtype=dtype,requires_grad=True)
    output=function(alpha)
    derivative=torch.autograd.functional.jacobian(function,alpha)
    L=decoder.likelihood[:,ids].T
    b=decoder.bias[:,ids].T;R=decoder.variance[:,ids].T
    d=b-b[:,-1:];ratio=R/R[:,-1:]
    expected_dw=pi*L/L[:,-1:];expected_dw[:,-1]=0
    expected_dw[:,-1]=-expected_dw.sum(dim=-1)
    expected_delta=(expected_dw*d).sum(-1)
    expected_R=base*(expected_dw*ratio).sum(-1)+(expected_dw*d.square()).sum(-1)
    expected_logM=(expected_dw*decoder.log_M).sum(-1)
    for actual,expected in zip(derivative,(expected_dw,expected_delta,expected_R,expected_logM)):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-10)
    assert (derivative[0].abs().sum(-1)>0).all()
    assert derivative[1].abs().max()>1e-8 and derivative[2].abs().max()>1e-8
    assert torch.equal(output[1],torch.zeros(4,dtype=dtype)) and torch.equal(output[2],base)
    epsilon=1e-8;right=function(torch.tensor(epsilon,dtype=dtype))
    fd_errors={}
    for name,old,new,expected in zip(('w','Delta_b','Rmix','ElogM'),output,right,derivative):
        numerical=(new-old)/epsilon
        torch.testing.assert_close(numerical,expected,rtol=5e-5,atol=2e-5)
        fd_errors[name]=float((numerical-expected).abs().max().detach())
    # Interior positive probabilities agree with the frozen logits implementation.
    generator=torch.Generator().manual_seed(2026102224)
    all_ids=torch.arange(31);logits=torch.randn((31,7),dtype=dtype,generator=generator)
    q=torch.softmax(logits,dim=-1)
    pvalue=decoder(all_ids,q,.3,entropy_grad=True)
    lvalue=decoder.forward_logits(all_ids,torch.log(q),.3)
    for name in lvalue:
        torch.testing.assert_close(pvalue[name],lvalue[name],rtol=1e-12,atol=1e-12)
    q=dense.clone().requires_grad_(True)
    boundary=decoder(ids,q,base)
    loss=(boundary['Delta_b'].square()+boundary['Rmix'].log()+.1*boundary['ElogM']).sum()
    loss.backward();assert torch.isfinite(q.grad).all()
    assert not boundary['entropy'].requires_grad
    impossible=decoder.likelihood[:,ids].T==0
    assert torch.equal(q.grad[impossible],torch.zeros_like(q.grad[impossible]))
    try:decoder(ids,dense,base,entropy_grad=True)
    except ValueError:pass
    else:raise AssertionError('Boundary entropy gradient request did not reject')
    bad=torch.zeros((1,7),dtype=dtype);bad[0,0]=1
    try:decoder(torch.tensor([6]),bad,.3)
    except ValueError:pass
    else:raise AssertionError('Impossible probability prior accepted')
    for bad in (torch.ones((1,7),dtype=dtype),torch.tensor([[-.1,.1,0.,0.,0.,0.,1.]],dtype=dtype)):
        try:decoder(torch.tensor([0]),bad,.3)
        except ValueError:pass
        else:raise AssertionError('Invalid q accepted')
    changed=copy.deepcopy(payload);changed['held_diagnostics']={'not_used':1e300}
    np.testing.assert_array_equal(training_c_M(changed),training_c_M(payload))
    np.testing.assert_array_equal(decoder.c_M.numpy(),training_c_M(payload))
    assert decoder.c_M_relative_to_dense[-1]==1
    # Gradcheck remains inside the probability simplex, away from alpha boundary.
    small_logits=torch.randn((2,7),dtype=dtype,generator=generator,requires_grad=True)
    small_alpha=torch.tensor(.2,dtype=dtype,requires_grad=True)
    small_base=torch.tensor([.3,.6],dtype=dtype,requires_grad=True)
    small_dense=torch.zeros((2,7),dtype=dtype);small_dense[:,-1]=1
    def interior(logits,alpha,base):
        value=decoder(torch.tensor([0,6]),(1-alpha)*small_dense+alpha*torch.softmax(logits,dim=-1),base,entropy_grad=True)
        return value['Delta_b'],value['Rmix'],value['w'],value['entropy']
    assert torch.autograd.gradcheck(interior,(small_logits,small_alpha,small_base),eps=1e-6,atol=2e-5,rtol=2e-4)
    return dict(exact_zero_q_supported=True,dense_anchor_exact=True,alpha0_derivative_matches_analytic=True,
        alpha0_derivative_finite_nonzero=True,right_finite_difference=True,positive_q_matches_logits_API=True,
        direct_q_gradients_finite=True,impossible_q_errors=True,q_simplex_validation=True,
        entropy_boundary_explicit_diagnostic_only=True,train_cM_exposed_held_independent=True,
        interior_torch_gradcheck=True,
        alpha0=dict(cell_ids=ids.tolist(),dDelta_b_dalpha=derivative[1].tolist(),
                    dRmix_dalpha=derivative[2].tolist(),dw_dalpha=derivative[0].tolist(),
                    right_difference_epsilon=epsilon,max_absolute_FD_errors=fd_errors))


def main():
    output=HERE/'sampling_density_prior_probability_validation.json'
    if output.exists():raise FileExistsError(f'Protected validation: {output}')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    payload=json.loads(CALIBRATION.read_text(encoding='utf-8'))
    frozen=dict(module=Path(FROZEN.__file__).resolve(),JSON=CALIBRATION,NPZ=CALIBRATION.parent/payload['npz_filename'])
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in frozen.items()}
    decoder=load_probability_decoder();checks=pure_checks(decoder,payload)
    for name,path in frozen.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name]
    result=dict(status='PURE SYNTHETIC PROBABILITY API VALIDATION; not a market model',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frozen_inputs_unchanged=hashes,
        c_M=decoder.c_M.tolist(),c_M_relative_to_dense=decoder.c_M_relative_to_dense.tolist(),
        c_M_source='frozen train_audit.global_mean_P only;60000 train seed2026102221',checks=checks,
        entropy_note='correct Shannon values; finite differentiable moments/posterior at alpha0; entropy has no finite right derivative there, so detached diagnostic by default',
        scope='M reference only; continuous activity A and market forecasting remain outside this library',
        market_files_read=False,real2026_data_read=False,selection_created=False,old_files_modified=False)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
