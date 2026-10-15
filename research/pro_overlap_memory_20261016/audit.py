"""Separate arithmetic implementation, saved weights and fresh causal refits.

The numerical audit is not third-party replication. Future perturbation tests
cover five representative first refits, not all 336 fresh fits a second time.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,csv,json,time
import numpy as np
import torch
import core
from experiment import fit,prediction,source_hashes,jwrite

def require(value,message):
    if not value:raise RuntimeError(message)

def raw_score(pr,p,dates,origins,ensemble=False):
    errors=[]
    if ensemble:pr=pr.mean(0,keepdims=True)
    for year in ('2020','2021'):
        ix=np.array([k for k,o in enumerate(origins) if dates[o][:4]==year and dates[o+20][:4]==year])
        for h in (1,5,20):
            target=p[origins[ix]+h]-p[origins[ix]]
            se=(pr[:,ix,h-1]-target[None])**2
            errors.append(float(se.mean()/np.square(target).mean()))
    return float(np.mean(errors)),errors

def run(out,causal=True):
    core.configure();out=Path(out);start=time.perf_counter()
    p,dates,receipt=core.load_prefix(out/'approved_prefix.csv');orig=np.arange(459,930)
    expected=np.stack([p[orig+h]-p[orig] for h in range(1,21)],axis=1)
    with np.load(out/'evaluation.npz',allow_pickle=False) as z:
        require(np.array_equal(z['origins'],orig),'Origin archive')
        require(np.array_equal(z['truth'],expected),'Target archive')
        require(int(z['valid'].sum())==451,'Common scoring support')
    scores={r['name']:r for r in csv.DictReader((out/'all_scores.csv').open())}
    errors=[];metrics=0;preds={}
    for f in sorted((out/'predictions').glob('*.npz')):
        with np.load(f,allow_pickle=False) as z:pr=z['prediction'].copy()
        preds[f.stem]=pr
        require(np.isfinite(pr[:,:,[0,4,19]]).all(),'Missing scored horizons')
        v,cells=raw_score(pr,p,dates,orig);en,_=raw_score(pr,p,dates,orig,True)
        errors.extend([abs(v-float(scores[f.stem]['mean_seed_relative_MSE'])),abs(en-float(scores[f.stem]['prediction_ensemble_relative_MSE']))]);metrics+=2
        allcells=[r for r in csv.DictReader((out/'cells.csv').open()) if r['name']==f.stem]
        for c,r in zip(cells,allcells):errors.append(abs(c-float(r['relative_MSE'])));metrics+=1
    require(max(errors)<1e-12,'Independent score mismatch')
    # Recreate issued scalar reference and dictionary from approved prices only.
    ref=core.reference(p);mem=core.MemoryData(p,ref);reconstruction=[];weight_errors=[];attention_sum=[];fits=[];negative=0;total_weights=0;max_l1=0.;overlap_fractions=[]
    expected_keys={(m,s,b) for m in core.MODES for s in core.SEEDS for b in range(12)}
    for jf in sorted((out/'fits').glob('*.json')):
        log=json.loads(jf.read_text());mode,seed,block=log['mode'],log['seed'],log['block'];end=log['end'];key=(mode,seed,block)
        require(key in expected_keys,'Unexpected or repeated fit');expected_keys.remove(key)
        require(log['source_sha256']==source_hashes(),'Executing source differs')
        require(log['max_training_target']<log['first_validation_origin'],'Purged supervised support')
        require(log['last_validation_target']<end,'Unmatured training/validation target')
        trace=np.array(log['trace']);require(trace.shape==(log['epochs'],5),'Epoch trace count');require(np.isfinite(trace).all(),'Nonfinite trace')
        best=log['initial_validation_MSE'];chosen=0
        for epoch,row in enumerate(trace,1):
            if row[3]<best-1e-6:best=float(row[3]);chosen=epoch
        require(chosen==log['selected_epoch'],'Checkpoint selection');require(abs(best-log['best_validation_MSE'])<1e-12,'Validation minimum')
        with np.load(jf.with_suffix('.npz'),allow_pickle=False) as z:
            a={k:z[k].copy() for k in z.files}
        model=core.make_model(mode,seed);model.load_state_dict({k.split('parameter::',1)[1]:torch.from_numpy(v) for k,v in a.items() if k.startswith('parameter::')})
        pr,details=prediction(model,mem,log['scale'],a['origins']);reconstruction.append(float(np.max(np.abs(pr-a['prediction']))))
        require(np.array_equal(pr,preds[mode][core.SEEDS.index(seed),a['origins']-459]),'Aggregate forecast differs')
        if 'w' in a:
            sources=a['source_origins'];mask=a['valid_mask'];pi=a['pi'];n=len(sources);omega=[]
            require(np.all((sources+20<a['origins'][:,None])[mask]),'Unmatured historical residual')
            require(np.all(sources[mask]>=60),'Invalid history key')
            for h in range(1,21):omega.append(np.maximum(0,h-abs(sources[:,:,None]-sources[:,None,:]))/h)
            om=np.stack(omega,axis=1)
            if mode=='OVERLAP':
                off=~np.eye(core.K,dtype=bool);overlap_fractions.append(float(np.mean(om[:,19,:, :][:,off]>0)))
            if mode in ('DIAGONAL','THIN','SOFTMAX'):om=np.broadcast_to(np.eye(core.K),om.shape).copy()
            if mode=='SHUFFLED':
                for j,i in enumerate(a['origins']):
                    v=int(mask[j].sum());perm=np.r_[np.random.default_rng(1009+int(i)).permutation(v),np.arange(v,core.K)]
                    om[j]=om[j][:,perm][:,:,perm]
            if mode=='SOFTMAX':nw=np.broadcast_to(pi[:,None],a['w'].shape)
            else:
                mat=om*(mask[:,None,:,None]*mask[:,None,None,:]);diag=mask/np.maximum(pi,1e-12)+(1-mask)
                mat=mat+np.eye(core.K)[None,None]*diag[:,None,:,None]
                rhs=np.broadcast_to(mask[:,None,:,None],(n,20,core.K,1)).astype(float)
                sol=np.linalg.solve(mat,rhs)[...,0];den=sol.sum(-1,keepdims=True)
                nw=np.divide(sol,den,out=np.zeros_like(sol),where=den>0)
            weight_errors.append(float(np.max(abs(nw-a['w']))));attention_sum.append(float(np.max(abs(a['w'].sum(-1)-1))))
            negative+=int((a['w']< -1e-10).sum());total_weights+=a['w'].size;max_l1=max(max_l1,float(abs(a['w']).sum(-1).max()))
        fits.append(log)
    require(not expected_keys,'Missing formal fits');require(max(reconstruction)==0,'Stored-weight predictions fail exact reconstruction')
    require(max(weight_errors)<1e-10,'Independent GLS weights mismatch')
    # Independent OLS combination reconstruction from matured issued pairs.
    g=preds['GRU'];base=ref['b'][orig];w=np.ones_like(g)*.5
    for s in range(4):
        for k,i in enumerate(orig):
            for h in range(1,21):
                eligible=np.flatnonzero(orig+h<=i)[-120:]
                if len(eligible)<30:continue
                dx=g[s,eligible,h-1]-base[eligible,h-1];target=expected[eligible,h-1]-base[eligible,h-1];den=np.dot(dx,dx)
                if den>1e-12:w[s,k,h-1]=np.clip(np.dot(dx,target)/den,0.,1.)
    with np.load(out/'COMBINATION_WEIGHTS.npz',allow_pickle=False) as z:we=float(np.max(abs(w-z['mature'])))
    require(we<1e-12,'Mature weights mismatch')
    require(np.max(abs(preds['CORE_GRU_MATURE']-(base[None]+w*(g-base[None]))))<1e-12,'Combination forecast mismatch')
    fresh=[]
    if causal:
        for mode in ('OVERLAP','DIAGONAL','SOFTMAX','THIN','FULL'):
            t0=time.perf_counter();end=459;seed=17;q=p.copy();q[end:]+=10000.
            rq=core.reference(q);mq=core.MemoryData(q,rq)
            m0,s0,l0=fit(p,ref,mem,end,mode,seed);m1,s1,l1=fit(q,rq,mq,end,mode,seed)
            same=all(torch.equal(v,m1.state_dict()[k]) for k,v in m0.state_dict().items())
            f0,_=prediction(m0,mem,s0,np.array([end-1]));f1,_=prediction(m1,mq,s1,np.array([end-1]))
            require(same and s0==s1 and np.array_equal(f0,f1),'Fresh future-perturbation refit differs')
            stem=out/'fits'/f'{mode}_s17_b00.npz'
            with np.load(stem,allow_pickle=False) as z:matches=all(np.array_equal(v.detach().numpy(),z['parameter::'+k]) for k,v in m0.state_dict().items())
            require(matches,'Fresh original-prefix run differs from first fit')
            fresh.append(dict(mode=mode,seed=seed,end=end,perturbed_from=end,issued_comparison=end-1,all_parameters_equal=True,forecast_equal=True,first_fit_reproduced=True,extra_fit_count=2,seconds=time.perf_counter()-t0))
            print('Fresh paired causal refit',mode,flush=True)
    result=dict(passed=True,formal_fits=len(fits),metrics_recomputed=metrics,max_arithmetic_error=max(errors),stored_weights_reconstructed=len(reconstruction),max_reconstruction_error=max(reconstruction),
                gls_archives_independently_checked=len(weight_errors),max_weight_error=max(weight_errors),max_weight_sum_error=max(attention_sum),negative_weight_fraction=negative/max(1,total_weights),global_max_l1_weight_norm=max_l1,
                mean_h20_overlap_pair_fraction_in_candidate_fits=float(np.mean(overlap_fractions)),combination_weight_max_error=we,all_epoch_rows_checked=sum(a['epochs'] for a in fits),
                future_refits=fresh,total_extra_fits=sum(a['extra_fit_count'] for a in fresh),data_receipt=receipt,seconds=time.perf_counter()-start,
                scope='Separate arithmetic scorer, complete saved-weight reconstruction and five representative paired fresh refits. Not third-party replication; not all fits retrained.',
                no_later_market_access=True)
    jwrite(out/'AUDIT.json',result);print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--skip-causal',action='store_true');v=a.parse_args();run(v.out,not v.skip_causal)
