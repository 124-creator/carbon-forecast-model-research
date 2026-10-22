"""Post-result training-geometry audit. No optimizer, new market model or rescue search."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import core
from experiment import jwrite


def ridge_numpy(z,r):
    return np.linalg.solve(2*z.T@z/len(z)+.01*np.eye(z.shape[1]),z.T@r/len(z))


def np_gain(r,u,d):
    s=(r.T@u+u.T@r-2*u.T@u)/len(r)-.01*d.T@d
    return (s+s.T)/2


def run(out,destination):
    out=Path(out);core.configure();p,dates,_=core.load_prefix(out/'approved_prefix.csv');ref=core.reference(p)
    rows=[];max_head=0.;max_projector=0.;max_stationarity=0.;max_trace=0.
    for path in sorted((out/'fits').glob('*.npz')):
        log=json.loads(path.with_suffix('.json').read_text());mode=log['mode']
        if mode not in core.PROFILE_MODES:continue
        with np.load(path,allow_pickle=False) as a:
            state={k[7:]:torch.from_numpy(a[k].copy()) for k in a.files if k.startswith('param__')}
        m=core.make_model(mode,log['seed']);m.load_state_dict(state)
        ids,x,y,tr,va,sd=core.support(p,log['end']);tx,tq,tb=core.batch(p,ref,ids,sd)
        with torch.no_grad():z=m.features(tx[tr],tq[tr]).numpy()
        r=y[tr]/sd-tb[tr].numpy();d=ridge_numpy(z,r);P=m.P.numpy();D=m.D.numpy()
        head_error=float(np.max(abs(D-d)));max_head=max(max_head,head_error)
        normal=(2*z.T@z/len(z)+.01*np.eye(9))@D-z.T@r/len(z)
        max_stationarity=max(max_stationarity,float(abs(normal).max()))
        pairs=[(np.arange(len(z)),np.arange(len(z)))] if mode in ('IN_SAMPLE','VARPRO_FIXED') else core.folds(len(z))
        old_mats=[];full_mats=[];ds=[];objectives=[]
        for a,b in pairs:
            da=ridge_numpy(z[a],r[a]);ua=z[b]@da;ds.append(da)
            S=np_gain(r[b],ua,da);old_mats.append(S);full_mats.append(np_gain(r[b],z[b]@D,D))
            obj=(np.sum((r[b]-ua@P)**2)+np.sum((ua@P)**2))/len(b)+.01*np.sum((da@P)**2)
            closed=np.sum(r[b]**2)/len(b)-np.trace(P@S)
            max_trace=max(max_trace,abs(float(obj-closed)))
            objectives.append(float(obj/20))
        S=np.mean(old_mats,axis=0);Sfull=np.mean(full_mats,axis=0)
        eig,vec=np.linalg.eigh(S);take=np.zeros(20,dtype=bool);take[-6:]=eig[-6:]>1e-10
        npP=vec[:,take]@vec[:,take].T
        if mode in ('FWD_FIXED','VARPRO_FIXED'):
            q=np.linalg.qr(core.initial_basis(6).numpy())[0];npP=q@q.T
        pe=float(abs(P-npP).max());max_projector=max(max_projector,pe)
        cols=np.flatnonzero(take);cutgap=float(eig[cols[0]]-eig[cols[0]-1]) if len(cols) and cols[0]>0 else None
        row=dict(mode=mode,seed=log['seed'],end=log['end'],epoch=log['selected_epoch'],epochs=log['epochs'],
            support_sizes=[len(a) for a,b in pairs],query_sizes=[len(b) for a,b in pairs],rank=int(round(np.trace(P))),
            retained_eigenvalues=eig[take].tolist(),smallest_retained_gap=cutgap,decoder_error=head_error,projector_error=pe,
            original_pair_gain_per_horizon=float(np.trace(P@S)/20),same_query_gain_after_full_refit=float(np.trace(P@Sfull)/20),
            mean_support_to_full_decoder_relative_difference=float(np.mean([np.linalg.norm(da-D)/(np.linalg.norm(D)+1e-12) for da in ds])),
            fixed_encoder_pair_objective_mean=float(np.mean(objectives)))
        rows.append(row)
    summary={}
    for mode in core.PROFILE_MODES:
        rs=[r for r in rows if r['mode']==mode];g=[r['smallest_retained_gap'] for r in rs if r['smallest_retained_gap'] is not None]
        summary[mode]=dict(fits=len(rs),zero_epoch=sum(r['epoch']==0 for r in rs),total_epochs=sum(r['epochs'] for r in rs),
            ranks=dict(Counter(str(r['rank']) for r in rs)),query_n_range=[min(min(r['query_sizes']) for r in rs),max(max(r['query_sizes']) for r in rs)],
            mean_decoder_relative_shift=float(np.mean([r['mean_support_to_full_decoder_relative_difference'] for r in rs])),
            original_gain_positive=sum(r['original_pair_gain_per_horizon']>1e-10 for r in rs),
            refit_gain_negative=sum(r['same_query_gain_after_full_refit']<0 for r in rs),minimum_retained_gap=min(g) if g else None)
    all_logs=[json.loads(p.read_text()) for p in (out/'fits').glob('*.json')]
    for mode in core.MODES:
        if mode not in summary:
            rs=[r for r in all_logs if r['mode']==mode];summary[mode]=dict(fits=len(rs),zero_epoch=sum(r['selected_epoch']==0 for r in rs),total_epochs=sum(r['epochs'] for r in rs))
        summary[mode]['summed_fit_wall_seconds']=float(sum(r['seconds'] for r in all_logs if r['mode']==mode))
    passed=max_head<1e-10 and max_projector<1e-7 and max_trace<1e-10
    result=dict(scope='POST-RESULT fixed-encoder training-geometry diagnosis, no optimizer/new forecast ranking. Full refit includes the same query labels, so refit quantities are not held-out performance.',passed=passed,
        independent_numpy_head_max_error=max_head,independent_numpy_projection_max_error=max_projector,ridge_normal_equation_max_residual=max_stationarity,trace_identity_max_error=max_trace,
        summary=summary,records=rows,limitations=['Rank/gain geometry is not identified market state or proof of calibration.','Two forward pairs overlap and are not independent tasks.','Coefficient shift is a design limitation, not a proven unique cause of failure.','No next period or alternative market prediction is scored.'])
    jwrite(destination,result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    if not passed:raise ValueError('Independent linear algebra disagrees')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--destination',required=True);o=a.parse_args();run(o.out,o.destination)
