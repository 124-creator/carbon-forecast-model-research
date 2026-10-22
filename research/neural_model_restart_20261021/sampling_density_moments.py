"""Frozen synthetic effective-sampling moments and a pure Torch decoder.

No import-time I/O. This module does not fit or select a forecasting model.
A future neural PRIOR may use liquidity/state context, not the same current
geometry already consumed by p(cell|M). Do not count that evidence twice.
Delta_b is a bias offset: corrected_x = baseline_x - Delta_b.
The reference is effective sampling M, never an identified real trade count.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
MS=(1,2,4,8,16,64,512)
EDGES=np.linspace(0.,1.,6)
CELLS=31
ALPHA=.5
SHRINK=50.
MIN_GROUP=20
R_FLOOR=1e-8
TRAIN_N,HELD_N=60000,20000
TRAIN_SEED,HELD_SEED=2026102221,2026102222


def structural_support():
    mask=np.ones((len(MS),CELLS),dtype=bool)
    mask[0,1:]=False
    mask[1,6:]=False
    return mask


def geometry_strata(u,l,r):
    """Pure Torch log-relative OHLC geometry: no extreme-time/order claim.

    u=log(H/O), l=log(L/O), r=log(C/O). Exact raw upper/lower wick
    zero defines the pattern. A zero range is INVALID (cell=-1), to be skipped
    by an observation filter rather than assigned a Brownian atom likelihood.
    """
    if u.shape!=l.shape or u.shape!=r.shape:
        raise ValueError('u/l/r must share shape')
    if not torch.isfinite(torch.stack([u,l,r])).all():
        raise ValueError('Geometry must be finite')
    zero=torch.zeros_like(r)
    upper=u-torch.maximum(r,zero);lower=torch.minimum(r,zero)-l
    if ((upper<0)|(lower<0)).any():raise ValueError('Inconsistent OHLC body/wick bounds')
    q=u-l;valid=q>0
    a=torch.abs(r)/torch.where(valid,q,torch.ones_like(q))
    wsum=upper+lower
    z=torch.abs(upper-lower)/torch.where(wsum>0,wsum,torch.ones_like(wsum))
    edges=torch.as_tensor(EDGES,dtype=u.dtype,device=u.device)
    ia=(torch.bucketize(a.contiguous(),edges,right=True)-1).clamp(0,4)
    iz=(torch.bucketize(z.contiguous(),edges,right=True)-1).clamp(0,4)
    atom=(upper==0)&(lower==0)&valid
    edge=((upper==0)^(lower==0))&valid
    cell=torch.where(atom,torch.zeros_like(ia),torch.where(edge,1+ia,6+5*ia+iz))
    cell=torch.where(valid,cell,torch.full_like(cell,-1))
    return dict(cell_id=cell,valid=valid,a=a,z=z,upper_wick=upper,lower_wick=lower)


def strata_from_sample(sample):
    ia=np.clip(np.searchsorted(EDGES,sample['a'],side='right')-1,0,4)
    iz=np.clip(np.searchsorted(EDGES,sample['z'],side='right')-1,0,4)
    return np.where(sample['phase']==0,0,np.where(sample['phase']==1,1+ia,6+5*ia+iz)).astype(np.int64)


def raw_stats(values):
    n=len(values)
    return dict(n=n,mean=float(values.mean()) if n else None,
                variance_ddof1=float(values.var(ddof=1)) if n>1 else None)


def fit_calibration(train):
    support=structural_support();counts=np.zeros((7,CELLS),dtype=np.int64)
    b=np.zeros((7,CELLS));R=np.zeros_like(b);audit={}
    for i,M in enumerate(MS):
        y=train[M]['y'];cell=strata_from_sample(train[M]);counts[i]=np.bincount(cell,minlength=CELLS)
        assert not counts[i,~support[i]].any()
        global_mean=float(y.mean());global_var=float(y.var(ddof=1))
        groups=[]
        for pattern in range(3):
            mask=(cell==0) if pattern==0 else ((cell>=1)&(cell<=5)) if pattern==1 else cell>=6
            values=y[mask];info=raw_stats(values)
            if len(values)>=MIN_GROUP:
                anchor_mean=info['mean'];anchor_var=info['variance_ddof1'];source='same_pattern'
            else:
                anchor_mean=global_mean;anchor_var=global_var;source='global_M'
            groups.append(dict(pattern=('atom','edge','interior')[pattern],raw=info,
                anchor_mean=anchor_mean,anchor_variance_ddof1=anchor_var,anchor_source=source))
        rows=[]
        for c in range(CELLS):
            pattern=0 if c==0 else 1 if c<=5 else 2
            group=groups[pattern];values=y[cell==c];raw=raw_stats(values);n=len(values)
            gm,gv=group['anchor_mean'],group['anchor_variance_ddof1']
            # Raw variances use ddof1. Shrink mean AND its variance+mean^2
            # moment; this keeps the between-mean term. Singleton variance is
            # unavailable and explicitly borrows the group, not invented support.
            weight=n/(n+SHRINK) if support[i,c] else 0.
            cm=raw['mean'] if n else gm
            cv=raw['variance_ddof1'] if n>1 else gv
            mean=weight*cm+(1-weight)*gm
            second=weight*(cv+cm*cm)+(1-weight)*(gv+gm*gm)
            variance=max(second-mean*mean,R_FLOOR)
            b[i,c],R[i,c]=mean,variance
            rows.append(dict(cell=c,pattern=group['pattern'],structurally_supported=bool(support[i,c]),
                train=raw,anchor_source=group['anchor_source'],anchor_group_n=group['raw']['n'],
                data_weight=weight,variance_source='cell_ddof1' if n>1 else 'anchor_no_cell_variance',
                b=mean,R=variance,floor_active=second-mean*mean<R_FLOOR))
        audit[str(M)]=dict(global_logP=raw_stats(y),global_mean_P=float(train[M]['P'].mean()),
            groups=groups,cells=rows,structural_cells=int(support[i].sum()))
    probability=np.zeros_like(b)
    for i in range(7):
        probability[i,support[i]]=(counts[i,support[i]]+ALPHA)/(TRAIN_N+ALPHA*support[i].sum())
    np.testing.assert_allclose(probability.sum(axis=1),1,rtol=0,atol=1e-14)
    assert np.isfinite(b).all() and np.isfinite(R).all() and (R>0).all()
    return dict(M=list(MS),edges=EDGES.tolist(),supported=support.tolist(),counts=counts.tolist(),
                p_cell_given_M=probability.tolist(),b=b.tolist(),R=R.tolist()),audit


class DensityMomentDecoder(nn.Module):
    """Posterior and relative moments. This module has ONLY frozen buffers.

    cell_id shape=batch; prior_logits shape=batch+[7], finite or -inf.
    base_R is positive and broadcastable to batch. Returns w, ElogM, entropy,
    Delta_b, relative_variance, between_bias_variance, Rmix (natural logs).
    -inf may encode a point prior; a prior with no possible support raises.
    """
    def __init__(self,calibration):
        super().__init__()
        if tuple(calibration['M'])!=MS:raise ValueError('Unexpected M ordering')
        p=np.asarray(calibration['p_cell_given_M'],dtype=float)
        b=np.asarray(calibration['b'],dtype=float);R=np.asarray(calibration['R'],dtype=float)
        support=np.asarray(calibration['supported'],dtype=bool)
        if p.shape!=(7,31) or b.shape!=p.shape or R.shape!=p.shape or support.shape!=p.shape:
            raise ValueError('Calibration dimensions must be7x31')
        if not np.array_equal(support,structural_support()):raise ValueError('Structural support changed')
        if not (np.isfinite(p).all() and np.isfinite(b).all() and np.isfinite(R).all() and (R>0).all()):
            raise ValueError('All stored fallback moments must be finite, R positive')
        if not ((p[~support]==0).all() and (p[support]>0).all()):raise ValueError('Probability support mismatch')
        np.testing.assert_allclose(p.sum(axis=1),1,rtol=0,atol=1e-12)
        logp=np.full_like(p,-np.inf);logp[support]=np.log(p[support])
        self.register_buffer('logp',torch.tensor(logp,dtype=torch.float64))
        self.register_buffer('bias',torch.tensor(b,dtype=torch.float64))
        self.register_buffer('variance',torch.tensor(R,dtype=torch.float64))
        self.register_buffer('log_M',torch.log(torch.tensor(MS,dtype=torch.float64)))

    def forward(self,cell_id,prior_logits,base_R):
        if not prior_logits.is_floating_point() or prior_logits.shape[-1]!=7:
            raise ValueError('prior_logits must have final dimension7 and floating dtype')
        if torch.isnan(prior_logits).any() or torch.isposinf(prior_logits).any():
            raise ValueError('Prior logits cannot contain NaN or +inf')
        cell=torch.as_tensor(cell_id,device=prior_logits.device)
        if cell.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):
            raise ValueError('cell_id must be integer')
        if tuple(cell.shape)!=tuple(prior_logits.shape[:-1]) or ((cell<0)|(cell>=31)).any():
            raise ValueError('cell_id shape/range invalid; zero-range observations must be skipped')
        cell=cell.long()
        logp=self.logp.to(prior_logits)[:,cell].movedim(0,-1)
        scores=prior_logits+logp
        possible=torch.isfinite(scores).any(dim=-1)
        if not possible.all():
            rows=torch.nonzero(~possible.reshape(-1),as_tuple=False).flatten().tolist()
            raise ValueError(f'Prior gives no mass to possible densities at rows {rows[:8]}')
        logw=scores-torch.logsumexp(scores,dim=-1,keepdim=True)
        w=torch.softmax(scores,dim=-1)
        safe_logw=torch.where(torch.isfinite(logw),logw,torch.zeros_like(logw))
        entropy=-(w*safe_logw).sum(dim=-1)
        b=self.bias.to(prior_logits)[:,cell].movedim(0,-1)
        R=self.variance.to(prior_logits)[:,cell].movedim(0,-1)
        difference=b-b[...,-1:]
        ratio=R/R[...,-1:]
        delta=(w*difference).sum(dim=-1)
        relative=(w*ratio).sum(dim=-1)
        between=(w*(difference-delta[...,None]).square()).sum(dim=-1)
        base=torch.as_tensor(base_R,dtype=prior_logits.dtype,device=prior_logits.device)
        base=torch.broadcast_to(base,prior_logits.shape[:-1])
        if not (torch.isfinite(base)&(base>0)).all():raise ValueError('base_R must be finite positive')
        mixed=base*relative+between
        return dict(w=w,ElogM=(w*self.log_M.to(prior_logits)).sum(dim=-1),entropy=entropy,
                    Delta_b=delta,relative_variance=relative,between_bias_variance=between,Rmix=mixed)


def validate_artifact_payload(payload,npz_path):
    expected=payload['npz_sha256']
    if hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()!=expected:
        raise ValueError('Frozen NPZ hash mismatch')
    cal=payload['calibration']
    with np.load(npz_path,allow_pickle=False) as arrays:
        for key in ('M','edges','supported','counts','p_cell_given_M','b','R'):
            np.testing.assert_array_equal(arrays[key],np.asarray(cal[key]))
    DensityMomentDecoder(cal)
    return cal


def load_frozen_calibration(json_path):
    path=Path(json_path)
    payload=json.loads(path.read_text(encoding='utf-8'))
    return validate_artifact_payload(payload,path.parent/payload['npz_filename'])


def pure_checks(calibration):
    decoder=DensityMomentDecoder(calibration)
    assert sum(p.numel() for p in decoder.parameters())==0
    ids=torch.arange(31);base=torch.linspace(.05,1.3,31,dtype=torch.float64)
    logits=torch.full((31,7),-torch.inf,dtype=torch.float64);logits[:,-1]=0
    dense=decoder(ids,logits,base)
    assert torch.equal(dense['Delta_b'],torch.zeros(31,dtype=torch.float64))
    assert torch.equal(dense['Rmix'],base)
    assert torch.equal(dense['relative_variance'],torch.ones_like(base))
    support=np.asarray(calibration['supported'])
    for i in range(7):
        cells=torch.tensor(np.flatnonzero(support[i]),dtype=torch.long)
        prior=torch.full((len(cells),7),-torch.inf,dtype=torch.float64);prior[:,i]=0
        out=decoder(cells,prior,base[cells])
        torch.testing.assert_close(out['Delta_b'],decoder.bias[i,cells]-decoder.bias[-1,cells],rtol=0,atol=0)
        torch.testing.assert_close(out['Rmix'],base[cells]*decoder.variance[i,cells]/decoder.variance[-1,cells],rtol=1e-14,atol=1e-14)
    generator=torch.Generator().manual_seed(2026102223)
    prior=torch.randn((31,7),dtype=torch.float64,generator=generator,requires_grad=True)
    base_R=decoder.variance[-1].clone().requires_grad_(True)
    out=decoder(ids,prior,base_R)
    b=decoder.bias.T;R=decoder.variance.T;mean=(out['w']*b).sum(-1)
    total=(out['w']*R).sum(-1)+(out['w']*(b-mean[:,None]).square()).sum(-1)
    torch.testing.assert_close(out['Rmix'],total,rtol=1e-12,atol=1e-12)
    assert torch.all(out['Rmix']>0)
    mask=torch.tensor(support.T)
    assert torch.equal(out['w'][~mask],torch.zeros_like(out['w'][~mask]))
    objective=(out['Delta_b'].square()+out['Rmix'].log()+.1*out['entropy']+.05*out['ElogM']).sum()
    objective.backward()
    assert torch.isfinite(prior.grad).all() and torch.isfinite(base_R.grad).all()
    assert torch.equal(prior.grad[~mask],torch.zeros_like(prior.grad[~mask]))
    bad=torch.full((1,7),-torch.inf,dtype=torch.float64);bad[:,0]=0
    try:decoder(torch.tensor([6]),bad,.3)
    except ValueError:pass
    else:raise AssertionError('Impossible M1 interior prior did not fail')
    try:decoder(torch.tensor([0]),torch.full((1,7),-torch.inf,dtype=torch.float64),.3)
    except ValueError:pass
    else:raise AssertionError('Empty prior did not fail')
    shifted=dict(calibration);shifted['b']=(np.asarray(calibration['b'])+.37).tolist()
    translated=DensityMomentDecoder(shifted)(ids,prior.detach(),base_R.detach())
    for key in ('Delta_b','Rmix'):
        torch.testing.assert_close(translated[key],out[key].detach(),rtol=1e-12,atol=1e-12)
    geometry=geometry_strata(torch.tensor([1.,0.,1.,.5,.8,.6,0.],dtype=torch.float64),
        torch.tensor([0.,-1.,-.5,-1.,-.6,-.8,0.],dtype=torch.float64),
        torch.tensor([1.,-1.,1.,-1.,.2,-.2,0.],dtype=torch.float64))
    torch.testing.assert_close(geometry['cell_id'],torch.tensor([0,0,4,4,6,6,-1]),rtol=0,atol=0)
    tiny_ids=torch.tensor([0,6]);tiny_logits=torch.randn((2,7),dtype=torch.float64,generator=generator,requires_grad=True)
    tiny_base=torch.tensor([.3,.6],dtype=torch.float64,requires_grad=True)
    def function(prior,base):
        value=decoder(tiny_ids,prior,base)
        return value['Delta_b'],value['Rmix'],value['entropy'],value['ElogM']
    assert torch.autograd.gradcheck(function,(tiny_logits,tiny_base),eps=1e-6,atol=2e-5,rtol=2e-4)
    return dict(frozen_buffers_only=True,dense_onehot_bias_zero_and_R_exact=True,
        single_M_relative_moments_correct=True,mixed_total_variance_correct=True,positive_variance=True,
        posterior_and_baseR_gradients_finite=True,impossible_support_weight_and_gradient_exact_zero=True,
        impossible_prior_raises=True,shared_logc_frame_shift_cancels=True,
        raw_wick_patterns_and_zero_range_skip=True,torch_gradcheck=True)


def score_summary(x):
    return dict(n=len(x),mean=float(np.mean(x)),standard_error=float(np.std(x,ddof=1)/np.sqrt(len(x))))


def held_diagnostics(calibration,held):
    decoder=DensityMomentDecoder(calibration);reports={};pooled={key:[] for key in ('mixture','dense','oracle','residual')}
    p=np.asarray(calibration['p_cell_given_M']);b=np.asarray(calibration['b']);R=np.asarray(calibration['R'])
    with torch.no_grad():
        for i,M in enumerate(MS):
            sample=held[M];ids_np=strata_from_sample(sample);ids=torch.tensor(ids_np)
            probabilities=p[i,ids_np];assert (probabilities>0).all()
            logits=torch.zeros((HELD_N,7),dtype=torch.float64)  # declared balanced prior, not true-M leakage
            base=decoder.variance[-1,ids]
            out=decoder(ids,logits,base)
            predicted_mean=(decoder.bias[-1,ids]+out['Delta_b']).numpy()
            variance=out['Rmix'].numpy();y=sample['y']
            mixture=np.log(variance)+(y-predicted_mean)**2/variance
            dense=np.log(R[-1,ids_np])+(y-b[-1,ids_np])**2/R[-1,ids_np]
            oracle=np.log(R[i,ids_np])+(y-b[i,ids_np])**2/R[i,ids_np]
            counts=np.bincount(ids_np,minlength=31);freq=counts/HELD_N
            cell_rows=[]
            for c in range(31):
                values=y[ids_np==c]
                cell_rows.append(dict(cell=c,held=raw_stats(values),train_b=float(b[i,c]),train_R=float(R[i,c]),
                    mean_error=None if not len(values) else float(values.mean()-b[i,c])))
            reports[str(M)]=dict(held_logP=raw_stats(y),held_mean_P=float(sample['P'].mean()),
                cell_counts=counts.tolist(),cell_probability_L1_error=float(np.abs(freq-p[i]).sum()),
                cell_cross_entropy=float(-np.log(probabilities).mean()),cell_moments=cell_rows,
                uniform_prior_posterior=dict(mean_true_M_probability=float(out['w'][:,i].mean()),
                    mean_ElogM=float(out['ElogM'].mean()),mean_entropy=float(out['entropy'].mean())),
                measurement_reconstruction=dict(mixture=score_summary(mixture),dense512=score_summary(dense),
                    known_M_moments=score_summary(oracle),residual=score_summary(y-predicted_mean)))
            for key,value in zip(('mixture','dense','oracle','residual'),(mixture,dense,oracle,y-predicted_mean)):
                pooled[key].append(value)
    balanced={key:score_summary(np.mean(value,axis=0)) for key,value in pooled.items()}
    balanced['mixture_minus_dense']=score_summary(np.mean(np.asarray(pooled['mixture'])-np.asarray(pooled['dense']),axis=0))
    balanced['standard_error_note']='same-path M samples clustered by independent fine path;20k clusters, not140k independent observations'
    return dict(per_M=reports,balanced_uniform_prior=balanced,
        note='measurement reconstruction at known unit latent variance only; no market/future forecasting outcome')


def main():
    json_path=HERE/'sampling_density_moments_calibration.json'
    npz_path=HERE/'sampling_density_moments_calibration.npz'
    for path in (json_path,npz_path):
        if path.exists():raise FileExistsError(f'Protected frozen calibration: {path}')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import sampling_density_reference_diagnostic as SIM
    assert SIM.MS==MS and SIM.FINE==512 and SIM.BATCH==1024
    simulator_path=Path(SIM.__file__).resolve();simulator_sha=hashlib.sha256(simulator_path.read_bytes()).hexdigest()
    started=time.perf_counter();edges=dict(edges_a=EDGES,edges_z=EDGES)
    train=SIM.simulate(TRAIN_N,TRAIN_SEED,edges)
    calibration,audit=fit_calibration(train)
    checks=pure_checks(calibration)
    held=SIM.simulate(HELD_N,HELD_SEED,edges)
    diagnostics=held_diagnostics(calibration,held)
    analytic=dict(train_M1=SIM.analytic_M1(train[1]),held_M1=SIM.analytic_M1(held[1]))
    assert hashlib.sha256(simulator_path.read_bytes()).hexdigest()==simulator_sha
    np.savez_compressed(npz_path,**{key:np.asarray(calibration[key]) for key in
        ('M','edges','supported','counts','p_cell_given_M','b','R')})
    result=dict(schema_version=1,status='FROZEN PURE SYNTHETIC EFFECTIVE SAMPLING CALIBRATION',
        calibration=calibration,train_audit=audit,held_diagnostics=diagnostics,pure_checks=checks,analytic_checks=analytic,
        config=dict(train_n=TRAIN_N,held_n=HELD_N,train_seed=TRAIN_SEED,held_seed=HELD_SEED,
            batch=1024,fine_increments=512,dirichlet_alpha=ALPHA,shrink_strength=SHRINK,min_pattern_group=MIN_GROUP,R_floor=R_FLOOR),
        strata='atom0; edge1+a_bin; interior6+5*a_bin+z_bin; raw wicks, no extreme-order claim; zero range invalid',
        moment_estimator='raw sample variances ddof1; shrink mean and variance+mean^2 byn/(n+50); group<20 uses global M; n<2 variance borrows anchor',
        decoder='posterior softmax(prior_logits+logp); Delta_b=E_w[b_M-b512]; Rmix=baseR*E_w[R_M/R512]+Var_w(b_M-b512)',
        sign='subtract Delta_b from baseline bias-corrected observation; sparse-density Delta_b is generally negative',
        prior_contract='liquidity/state context only; do not duplicate current geometry evidence already in likelihood',
        interpretation='effective sampling M, not real trade count or jump identification; finite-grid bias and total variance are known; future forecasting benefit untested',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),simulator_source=dict(path=str(simulator_path),sha256=simulator_sha,unchanged=True),
        npz_filename=npz_path.name,npz_sha256=hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        market_files_read=False,real2026_data_read=False,selection_file_created=False,neural_model_trained=False,
        elapsed_seconds=time.perf_counter()-started)
    validate_artifact_payload(json.loads(json.dumps(result)),npz_path)
    result['pure_checks']['JSON_NPZ_roundtrip_exact']=True
    json_path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(source_sha256=result['source_sha256'],json=str(json_path),npz=str(npz_path),
        npz_sha256=result['npz_sha256'],pure_checks=checks,
        support_cells=[int(x) for x in structural_support().sum(axis=1)],
        held_balanced_measurement=diagnostics['balanced_uniform_prior'],elapsed_seconds=result['elapsed_seconds']),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
