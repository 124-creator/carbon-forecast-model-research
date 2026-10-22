"""Synthetic raw-P Gamma observation reference; no market/model execution.

Positive latent S grid, effective M, full discrete shape plus radial likelihood.
Gamma shape is k_ref*kappa and mean is S*c_cell/c_ref: precision changes no
mean. Global dense c_ref=c512 gives mean=S. y=0 is explicitly rejected; a
future model must use its N0 hurdle/skip. All calibration remains synthetic.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import sampling_density_moments as LOGREF

HERE=Path(__file__).resolve().parent
OLD_JSON=HERE/'sampling_density_moments_calibration.json'
MS=LOGREF.MS
VAR_FLOOR=1e-12


def raw_moments(x):
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,
        variance_ddof1=float(x.var(ddof=1)) if len(x)>1 else None,
        second_moment=float(np.mean(x*x)) if len(x) else None)


def fit_raw_reference(train,old_payload):
    support=LOGREF.structural_support();counts=np.zeros((7,31),dtype=np.int64)
    means=np.empty((7,31));variances=np.empty_like(means);audit={}
    global_c=[];global_v=[]
    for i,M in enumerate(MS):
        P=train[M]['P'];cell=LOGREF.strata_from_sample(train[M])
        counts[i]=np.bincount(cell,minlength=31);assert not counts[i,~support[i]].any()
        global_stats=raw_moments(P);global_c.append(global_stats['mean']);global_v.append(global_stats['variance_ddof1'])
        groups=[]
        for pattern in range(3):
            mask=(cell==0) if pattern==0 else ((cell>=1)&(cell<=5)) if pattern==1 else cell>=6
            stats=raw_moments(P[mask])
            anchor=stats if stats['n']>=20 else global_stats
            groups.append(dict(pattern=('atom','edge','interior')[pattern],raw=stats,
                anchor=anchor,anchor_source='pattern' if stats['n']>=20 else 'global_M'))
        rows=[]
        for c in range(31):
            pattern=0 if c==0 else 1 if c<=5 else 2
            group=groups[pattern];anchor=group['anchor'];raw=raw_moments(P[cell==c]);n=raw['n'];g=anchor['n']
            if n:
                weight=n/(n+50.)
                mean=weight*raw['mean']+(1-weight)*anchor['mean']
                second=weight*raw['second_moment']+(1-weight)*anchor['second_moment']
                # Cell data overlap their pattern/global anchor. These are the
                # actual combined sample weights; correct their sum of squares.
                sum_weight_square=(n+2*n*50./g+50.**2/g)/(n+50.)**2
                variance_before=(second-mean*mean)/(1-sum_weight_square)
            else:
                weight=0.;mean=anchor['mean'];second=anchor['second_moment']
                sum_weight_square=1/g;variance_before=anchor['variance_ddof1']
            variance=max(variance_before,VAR_FLOOR)
            means[i,c],variances[i,c]=mean,variance
            rows.append(dict(cell=c,structurally_supported=bool(support[i,c]),raw=raw,
                anchor_source=group['anchor_source'],anchor_n=g,data_weight=weight,
                raw_second_shrunk=second,sum_sample_weights_squared=sum_weight_square,
                c_cell=mean,var_cell=variance,k_cell=mean*mean/variance,
                variance_floor_active=bool(variance_before<VAR_FLOOR)))
        audit[str(M)]=dict(global_raw=global_stats,groups=groups,cells=rows)
    probability=np.zeros_like(means)
    for i in range(7):
        probability[i,support[i]]=(counts[i,support[i]]+.5)/(60000+.5*support[i].sum())
    old=old_payload['calibration']
    np.testing.assert_array_equal(counts,np.asarray(old['counts']))
    np.testing.assert_array_equal(support,np.asarray(old['supported']))
    np.testing.assert_array_equal(probability,np.asarray(old['p_cell_given_M']))
    np.testing.assert_array_equal(np.asarray(global_c),np.array([old_payload['train_audit'][str(M)]['global_mean_P'] for M in MS]))
    assert np.isfinite(means).all() and (means>0).all() and np.isfinite(variances).all() and (variances>0).all()
    c_global=np.asarray(global_c);var_global=np.asarray(global_v)
    calibration=dict(M=list(MS),edges=LOGREF.EDGES.tolist(),supported=support.tolist(),counts=counts.tolist(),
        p_shape=probability.tolist(),c_cell=means.tolist(),var_cell=variances.tolist(),k_cell=(means**2/variances).tolist(),
        c_global=c_global.tolist(),var_global=var_global.tolist(),k_global=(c_global**2/var_global).tolist())
    return calibration,audit,dict(counts_exact=True,p_shape_exact=True,global_cM_exact=True,structural_support_exact=True)


class GammaObservationKernel(nn.Module):
    """Frozen buffers only. forward returns radial log density [n,grid,7].

    y:[n]>0, S_grid:[grid] or[n,grid]>0, c_ref:scalar or[n]>0,
    kappa:scalar or[n]>0. conditional=True requires cell_id:[n]. Impossible
    conditional components return -inf. global_log_prob uses unconditional
    per-M raw moments. joint_log_prob adds log p(shape|M) exactly once.
    """
    def __init__(self,calibration):
        super().__init__()
        if tuple(calibration['M'])!=MS:raise ValueError('M ordering changed')
        support=np.asarray(calibration['supported'],dtype=bool)
        if not np.array_equal(support,LOGREF.structural_support()):raise ValueError('Structural support changed')
        p=np.asarray(calibration['p_shape'],dtype=float)
        if p.shape!=(7,31) or not ((p[~support]==0).all() and (p[support]>0).all()):
            raise ValueError('Invalid shape probability support')
        np.testing.assert_allclose(p.sum(axis=1),1,rtol=0,atol=1e-14)
        self.register_buffer('M',torch.tensor(MS,dtype=torch.int64))
        self.register_buffer('p_shape',torch.tensor(p,dtype=torch.float64))
        logp=np.full_like(p,-np.inf);logp[support]=np.log(p[support])
        self.register_buffer('log_p_shape',torch.tensor(logp,dtype=torch.float64))
        for name in ('c_cell','var_cell','k_cell','c_global','var_global','k_global'):
            x=np.asarray(calibration[name],dtype=float)
            expected=(7,31) if name.endswith('cell') else (7,)
            if x.shape!=expected or not (np.isfinite(x).all() and (x>0).all()):
                raise ValueError(f'{name} must have finite positive values including unsupported fallbacks')
            self.register_buffer(name,torch.tensor(x,dtype=torch.float64))
        torch.testing.assert_close(self.k_cell,self.c_cell.square()/self.var_cell,rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(self.k_global,self.c_global.square()/self.var_global,rtol=1e-14,atol=1e-14)

    def _ids(self,cell_id,n,device):
        ids=torch.as_tensor(cell_id,device=device)
        if ids.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8) or ids.shape!=(n,):
            raise ValueError('cell_id must be integer[n]')
        if ((ids<0)|(ids>=31)).any():raise ValueError('Invalid cell, including skipped zero-range geometry')
        return ids.long()

    def shape_log_prob(self,cell_id):
        raw=torch.as_tensor(cell_id)
        if raw.ndim!=1:raise ValueError('cell_id must be[n]')
        ids=self._ids(cell_id,len(raw),self.log_p_shape.device)
        return self.log_p_shape[:,ids].T

    def shape_probability(self,cell_id):
        raw=torch.as_tensor(cell_id)
        if raw.ndim!=1:raise ValueError('cell_id must be[n]')
        ids=self._ids(cell_id,len(raw),self.p_shape.device)
        return self.p_shape[:,ids].T

    def observation_parameters(self,y,S_grid,c_ref,cell_id=None,kappa=1.,conditional=True):
        if y.ndim!=1 or not y.is_floating_point() or not (torch.isfinite(y)&(y>0)).all():
            raise ValueError('y must be finite positive[n]; y=0 requires an external N0 hurdle/skip')
        n=len(y);grid=torch.as_tensor(S_grid,dtype=y.dtype,device=y.device)
        if grid.ndim==1 and len(grid)>0:grid=grid[None,:].expand(n,-1)
        elif grid.ndim!=2 or grid.shape[0]!=n or grid.shape[1]==0:
            raise ValueError('S_grid must be[grid] or[n,grid]')
        ref=torch.broadcast_to(torch.as_tensor(c_ref,dtype=y.dtype,device=y.device),(n,))
        precision=torch.broadcast_to(torch.as_tensor(kappa,dtype=y.dtype,device=y.device),(n,))
        if not ((torch.isfinite(grid)&(grid>0)).all() and (torch.isfinite(ref)&(ref>0)).all()
                and (torch.isfinite(precision)&(precision>0)).all()):
            raise ValueError('S_grid/c_ref/kappa must be finite positive')
        if conditional:
            if cell_id is None:raise ValueError('Conditional kernel requires cell_id')
            ids=self._ids(cell_id,n,y.device)
            c=self.c_cell.to(y)[:,ids].T;k=self.k_cell.to(y)[:,ids].T
            support=self.p_shape.to(y)[:,ids].T>0
        else:
            c=self.c_global.to(y)[None,:].expand(n,-1);k=self.k_global.to(y)[None,:].expand(n,-1)
            support=torch.ones((n,7),dtype=torch.bool,device=y.device)
        mean=grid[:,:,None]*(c[:,None,:]/ref[:,None,None])
        shape=(k[:,None,:]*precision[:,None,None]).expand_as(mean)
        return dict(mean=mean,shape=shape,scale=mean/shape,variance=mean.square()/shape,supported=support)

    def forward(self,y,S_grid,c_ref,cell_id=None,kappa=1.,conditional=True):
        parameters=self.observation_parameters(y,S_grid,c_ref,cell_id,kappa,conditional)
        shape,scale=parameters['shape'],parameters['scale']
        logp=(shape-1)*torch.log(y[:,None,None])-y[:,None,None]/scale-shape*torch.log(scale)-torch.lgamma(shape)
        return torch.where(parameters['supported'][:,None,:],logp,torch.full_like(logp,-torch.inf))

    log_prob=forward

    def joint_log_prob(self,y,S_grid,c_ref,cell_id,kappa=1.):
        return self.forward(y,S_grid,c_ref,cell_id,kappa,True)+self.shape_log_prob(cell_id).to(y)[:,None,:]

    def global_log_prob(self,y,S_grid,c_ref,kappa=1.):
        return self.forward(y,S_grid,c_ref,None,kappa,False)

    def dense_global_log_prob(self,y,S_grid,c_ref,kappa=1.):
        return self.global_log_prob(y,S_grid,c_ref,kappa)[...,-1]


def analytic_m1_log_prob(y):
    """Exact unit-variance M1 atom: Gamma(1/2, scale=1/(2ln2))."""
    if not (torch.isfinite(y)&(y>0)).all():raise ValueError('Analytic M1 requires positive y')
    scale=1/(2*math.log(2))
    return -.5*math.log(math.pi*scale)-.5*torch.log(y)-y/scale


def analytic_m1_check(P):
    n=len(P);shape=.5;scale=1/(2*math.log(2));mean=shape*scale;var=shape*scale**2
    mu3=2*shape*scale**3;mu4=6*shape*scale**4+3*var**2
    var_mean=var/n;var_var=(mu4-(n-3)/(n-1)*var**2)/n;cov=mu3/n
    dk_mean=2*mean/var;dk_var=-mean**2/var**2
    dt_mean=-var/mean**2;dt_var=1/mean
    se_k=math.sqrt(dk_mean**2*var_mean+dk_var**2*var_var+2*dk_mean*dk_var*cov)
    se_scale=math.sqrt(dt_mean**2*var_mean+dt_var**2*var_var+2*dt_mean*dt_var*cov)
    actual_mean=float(P.mean());actual_var=float(P.var(ddof=1))
    actual=dict(mean=actual_mean,variance=actual_var,shape=actual_mean**2/actual_var,scale=actual_var/actual_mean)
    truth=dict(mean=mean,variance=var,shape=shape,scale=scale)
    se=dict(mean=math.sqrt(var_mean),variance=math.sqrt(var_var),shape=se_k,scale=se_scale)
    z={key:(actual[key]-truth[key])/se[key] for key in truth}
    assert all(abs(value)<6 for value in z.values()),z
    return dict(theory=truth,observed=actual,analytic_MC_SE=se,z_scores=z,
                passed_within6_SE=True,shape_not_forced_to_half=True)


def pure_checks(calibration):
    kernel=GammaObservationKernel(calibration);assert sum(p.numel() for p in kernel.parameters())==0
    y=torch.tensor([.03,.8],dtype=torch.float64);grid=torch.tensor([.2,.8,1.6],dtype=torch.float64)
    ref=kernel.c_global[-1];cells=torch.tensor([0,6])
    parameters=kernel.observation_parameters(y,grid,ref,cells)
    lp=kernel(y,grid,ref,cells)
    dist=torch.distributions.Gamma(parameters['shape'],1/parameters['scale'])
    expected=dist.log_prob(y[:,None,None])
    mask=parameters['supported'][:,None,:].expand_as(lp)
    torch.testing.assert_close(lp[mask],expected[mask],rtol=1e-12,atol=1e-12)
    assert torch.isneginf(lp[~mask]).all()
    assert torch.equal(kernel.shape_probability(torch.arange(31)).T==0,
                       torch.tensor(~LOGREF.structural_support()))
    global_params=kernel.observation_parameters(y,grid,ref,conditional=False)
    assert torch.equal(global_params['mean'][...,-1],grid[None,:].expand(2,-1))
    p1=kernel.observation_parameters(y,grid,ref,cells,kappa=.3)
    p2=kernel.observation_parameters(y,grid,ref,cells,kappa=2.7)
    assert torch.equal(p1['mean'],p2['mean'])
    torch.testing.assert_close(p1['variance']/9,p2['variance'],rtol=1e-14,atol=1e-14)
    for factor in (.1,3.7):
        scaled=kernel(y*factor,grid*factor,ref,cells)
        torch.testing.assert_close(scaled[mask],lp[mask]-math.log(factor),rtol=1e-12,atol=1e-12)
    joint=kernel.joint_log_prob(y,grid,ref,cells)
    torch.testing.assert_close(joint[mask],(lp+kernel.shape_log_prob(cells)[:,None,:])[mask],rtol=0,atol=0)
    analytic_y=torch.tensor([.0001,.01,.1,1.,4.],dtype=torch.float64)
    analytic=torch.distributions.Gamma(torch.tensor(.5,dtype=torch.float64),torch.tensor(2*math.log(2),dtype=torch.float64)).log_prob(analytic_y)
    torch.testing.assert_close(analytic_m1_log_prob(analytic_y),analytic,rtol=1e-14,atol=1e-14)
    try:kernel(torch.tensor([0.],dtype=torch.float64),grid,ref,torch.tensor([0]))
    except ValueError:pass
    else:raise AssertionError('y=0 was not rejected')
    yy=y.clone().requires_grad_(True);gg=grid.clone().requires_grad_(True)
    rr=ref.clone().requires_grad_(True);kk=torch.tensor(1.2,dtype=torch.float64,requires_grad=True)
    def function(y,S,c_ref,kappa):
        value=kernel(y,S,c_ref,cells,kappa)
        return value[mask]
    assert torch.autograd.gradcheck(function,(yy,gg,rr,kk),eps=1e-6,atol=2e-5,rtol=2e-4)
    loss=function(yy,gg,rr,kk).sum()
    gradients=torch.autograd.grad(loss,(yy,gg,rr,kk))
    assert all(torch.isfinite(value).all() for value in gradients)
    # Check the lognormal P-Jacobian independently under a unit conversion.
    mu=torch.tensor([-.4,.1],dtype=torch.float64);R=torch.tensor([.3,.5],dtype=torch.float64)
    def lognormal(y,mu,R):return -.5*(math.log(2*math.pi)+torch.log(R)+(torch.log(y)-mu).square()/R)-torch.log(y)
    torch.testing.assert_close(lognormal(3*y,mu+math.log(3),R),lognormal(y,mu,R)-math.log(3),rtol=1e-13,atol=1e-13)
    return dict(buffers_only=True,gamma_matches_torch_distribution=True,analytic_M1_density_exact=True,
        structural_impossible_loglik_minus_inf=True,dense_global_mean_equals_S_exact=True,
        kappa_changes_variance_not_mean=True,unit_transform_loglik_jacobian=True,lognormal_P_jacobian=True,
        full_radial_plus_shape=True,y_zero_rejected=True,finite_gradients=True,torch_gradcheck=True)


def summary(x):
    return dict(n=len(x),mean=float(np.mean(x)),standard_error=float(np.std(x,ddof=1)/np.sqrt(len(x))))


def held_comparison(calibration,old_payload,held):
    kernel=GammaObservationKernel(calibration);ref=kernel.c_global[-1]
    old=old_payload['calibration'];b=np.asarray(old['b']);R=np.asarray(old['R'])
    b_global=np.array([old_payload['train_audit'][str(M)]['global_logP']['mean'] for M in MS])
    R_global=np.array([old_payload['train_audit'][str(M)]['global_logP']['variance_ddof1'] for M in MS])
    pooled={key:[] for key in ('conditional_difference','global_difference','joint_mixture_difference')};reports={}
    with torch.no_grad():
        for i,M in enumerate(MS):
            P=held[M]['P'];logP=np.log(P);ids_np=LOGREF.strata_from_sample(held[M]);ids=torch.tensor(ids_np)
            y=torch.tensor(P,dtype=torch.float64);grid=ref.reshape(1)
            gamma=kernel(y,grid,ref,ids)[:,0,:]
            lognormal=-.5*(np.log(2*math.pi*R[:,ids_np].T)+(logP[:,None]-b[:,ids_np].T)**2/R[:,ids_np].T)-logP[:,None]
            support=np.asarray(calibration['p_shape'])[:,ids_np].T>0
            lognormal=np.where(support,lognormal,-np.inf)
            gc=-gamma[:,i].numpy();lc=-lognormal[:,i]
            gg=-kernel.global_log_prob(y,grid,ref)[:,0,i].numpy()
            lg=.5*(np.log(2*math.pi*R_global[i])+(logP-b_global[i])**2/R_global[i])+logP
            log_shape=kernel.shape_log_prob(ids)
            gj=-torch.logsumexp(gamma+log_shape-math.log(7),dim=-1).numpy()
            lj=-torch.logsumexp(torch.tensor(lognormal)+log_shape-math.log(7),dim=-1).numpy()
            reports[str(M)]=dict(n=len(P),conditional_true_M=dict(Gamma_NLL=summary(gc),lognormal_P_NLL=summary(lc),
                Gamma_minus_lognormal=summary(gc-lc)),global_true_M=dict(Gamma_NLL=summary(gg),
                lognormal_P_NLL=summary(lg),Gamma_minus_lognormal=summary(gg-lg)),
                joint_shape_radial_uniform_M=dict(Gamma_NLL=summary(gj),lognormal_P_NLL=summary(lj),difference=summary(gj-lj)))
            if M==1:reports[str(M)]['analytic_M1_Gamma_NLL']=summary(-analytic_m1_log_prob(y).numpy())
            for key,value in zip(pooled,(gc-lc,gg-lg,gj-lj)):pooled[key].append(value)
    return dict(per_M=reports,balanced_paired_differences={key:summary(np.mean(values,axis=0)) for key,values in pooled.items()},
        score='proper negative log density in raw P units; lognormal includes +log(P) NLL Jacobian; Gamma-minus-lognormal negative favors Gamma',
        hyperparameters='kappa1 fixed, latent unit variance represented by S=c_ref; no held tuning',
        uncertainty='M levels share the same fine path; balanced errors use20000 path clusters, not140000 independent samples',
        scope='current synthetic measurement reconstruction only; never future forecast loss')


def validate_payload(payload,npz_path):
    if hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()!=payload['npz_sha256']:raise ValueError('Gamma NPZ hash mismatch')
    with np.load(npz_path,allow_pickle=False) as arrays:
        for key,value in payload['calibration'].items():np.testing.assert_array_equal(arrays[key],np.asarray(value))
    GammaObservationKernel(payload['calibration'])
    return payload['calibration']


def load_gamma_reference(json_path=HERE/'sampling_density_gamma_calibration.json'):
    path=Path(json_path);payload=json.loads(path.read_text(encoding='utf-8'))
    return GammaObservationKernel(validate_payload(payload,path.parent/payload['npz_filename']))


def main():
    output=HERE/'sampling_density_gamma_calibration.json';npz=HERE/'sampling_density_gamma_calibration.npz'
    for path in (output,npz):
        if path.exists():raise FileExistsError(f'Protected Gamma calibration: {path}')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import sampling_density_reference_diagnostic as SIM
    old_payload=json.loads(OLD_JSON.read_text(encoding='utf-8'))
    LOGREF.validate_artifact_payload(old_payload,OLD_JSON.parent/old_payload['npz_filename'])
    paths=dict(log_module=Path(LOGREF.__file__).resolve(),log_JSON=OLD_JSON,
        log_NPZ=OLD_JSON.parent/old_payload['npz_filename'],simulator=Path(SIM.__file__).resolve())
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    started=time.perf_counter();edges=dict(edges_a=LOGREF.EDGES,edges_z=LOGREF.EDGES)
    train=SIM.simulate(60000,2026102221,edges)
    calibration,audit,exact=fit_raw_reference(train,old_payload)
    checks=pure_checks(calibration)
    held=SIM.simulate(20000,2026102222,edges)
    comparison=held_comparison(calibration,old_payload,held)
    analytic=dict(train=analytic_m1_check(train[1]['P']),held=analytic_m1_check(held[1]['P']))
    for name,path in paths.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name]
    np.savez_compressed(npz,**{key:np.asarray(value) for key,value in calibration.items()})
    result=dict(schema_version=1,status='FROZEN SYNTHETIC RAW-P GAMMA REFERENCE; not a market test',
        calibration=calibration,train_audit=audit,exact_old_reference_checks=exact,pure_checks=checks,
        analytic_M1=analytic,held_comparison=comparison,
        config=dict(train_n=60000,held_n=20000,train_seed=2026102221,held_seed=2026102222,batch=1024,fine_increments=512,
            M=list(MS),strata31='atom0,edge1+a_bin,interior6+5*a_bin+z_bin',dirichlet_alpha=.5,shrink50=50,min_group=20,raw_variance_floor=VAR_FLOOR),
        estimator='raw E[P]/E[P²] shrink to pattern (group<20 globalM); weighted Bessel correction accounts for overlapping cell/anchor samples; k=c²/var',
        API='GammaObservationKernel.log_prob(y,S_grid,c_ref,cell_id,kappa) ->[n,grid,7]; joint_log_prob adds p(shape|M); global_log_prob and dense_global_log_prob available',
        observation='Gamma(shape=k_ref*kappa,mean=S*c_cell/c_ref); no Gaussian-log/P2 approximation; y0 rejected for future N0 hurdle/skip',
        analytic_note='M1 atom exact unit Gamma(.5,scale=1/(2ln2)) is a separate analytic benchmark; fitted k not forced to .5',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frozen_inputs_unchanged=hashes,
        npz_filename=npz.name,npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
        market_files_read=False,real2026_data_read=False,selection_created=False,forecast_model_trained=False,
        interpretation='effectiveM reference only; no realM/A/IV identification; Gamma/mixture/Bayes/total variance are known; future combination benefit remains untested',
        elapsed_seconds=time.perf_counter()-started)
    validate_payload(json.loads(json.dumps(result)),npz);result['pure_checks']['JSON_NPZ_exact_roundtrip']=True
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(source_sha256=result['source_sha256'],JSON=str(output),NPZ=str(npz),npz_sha256=result['npz_sha256'],
        exact_old_reference_checks=exact,pure_checks=checks,analytic_M1=analytic,
        held_balanced=comparison['balanced_paired_differences'],
        conditional_differences_by_M={M:row['conditional_true_M']['Gamma_minus_lognormal']['mean'] for M,row in comparison['per_M'].items()},
        elapsed_seconds=result['elapsed_seconds']),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
