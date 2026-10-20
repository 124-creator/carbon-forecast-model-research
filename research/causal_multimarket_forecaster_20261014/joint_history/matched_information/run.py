import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import importlib.util, json,csv,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
SPEC=importlib.util.spec_from_file_location('joint_original',PARENT/'run.py')
if SPEC is None or SPEC.loader is None: raise RuntimeError('parent source cannot be loaded')
j=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(j)
b=j.b
METHODS=('independent_increment','joint_matched','other_shuffled','other_shift97')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def design(p,seed,method,m):
    data,d=j.dataset(p); train=data['train'][2]; rng=np.random.default_rng(seed+9000+m)
    donors={'train':rng.permutation(len(train)), 'val':rng.integers(len(train),size=len(data['val'][2])), 'test':rng.integers(len(train),size=len(data['test'][2]))}
    windows={k:v[0].reshape(len(v[0]),31,3) for k,v in data.items()}
    out={}; other=[i for i in range(3) if i!=m]
    for k,(x,y,t) in data.items():
        own=windows[k][:,:,m]; cross=windows[k][:,:,other]
        if method=='other_shuffled': cross=windows['train'][donors[k]][:,:,other]
        if method=='other_shift97':
            ix=t[:,None]-97+np.arange(-30,1)[None,:]
            cross=d[np.maximum(ix,0)][:,:,other]; cross=np.where((ix>=0)[:,:,None],cross,0.)
        xx=own if method=='independent_increment' else np.column_stack([own,cross.reshape(len(t),-1)])
        out[k]=(xx,y[:,m*3:(m+1)*3],t)
    return out

def main():
    start=time.monotonic(); rows=[]; checks=[]
    originals={str(p.relative_to(PARENT)):sha(p) for p in PARENT.rglob('*') if p.is_file() and ROOT not in p.parents}
    provenance={'joint_source':str(PARENT/'run.py'),'joint_sha256':sha(PARENT/'run.py'),'generator_source':str(j.SOURCE.resolve()),'generator_sha256':sha(j.SOURCE),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'original_hashes':originals}
    (ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    for seed in b.SEEDS:
        for shared in (True,False):
            cond='shared' if shared else 'no_shared'; out=ROOT/f'seed_{seed}_{cond}'; out.mkdir(exist_ok=True)
            p,latent=b.generate(seed,shared)
            if not np.array_equal(p,np.load(PARENT/out.name/'observations.npz')['prices']): raise RuntimeError('generator mismatch')
            ts=b.origins(len(p))['test']; target=np.stack([p[t+b.H] for t in ts]).transpose(0,2,1)
            cut=int(ts[0]); pert=p.copy(); pert[cut+1:]+=np.random.default_rng(seed+8000).normal(10000,1000,size=pert[cut+1:].shape)
            for method in METHODS:
                pred=np.empty_like(target); histories=[]
                for m in range(3):
                    data=design(p,seed,method,m); delta,info,cp=j.ridge(data)
                    pd=design(pert,seed,method,m); delta2,info2,cp2=j.ridge(pd)
                    equal=all(np.array_equal(cp[k],cp2[k]) for k in cp) and info['lambda']==info2['lambda'] and np.array_equal(delta[0],delta2[0])
                    if not equal: raise RuntimeError('future refit leak')
                    checks.append({'seed':seed,'condition':cond,'method':method,'market':m,'future_cut':cut,'exact_refit_and_origin_prediction_equal':equal})
                    pred[:,m]=p[ts,m,None]+delta; histories.append({'market':m,**info})
                    np.savez_compressed(out/f'{method}_market{m}_checkpoint.npz',**cp)
                np.savez_compressed(out/f'{method}_predictions.npz',prediction=pred,target=target,origins=ts)
                (out/f'{method}_loss.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
                mse=((pred-target)**2).mean(0)
                for m in range(3):
                    for hi,h in enumerate(b.H): rows.append(dict(seed=seed,condition=cond,model=method,market=m,horizon=int(h),mse=float(mse[m,hi])))
                print(seed,cond,method,float(mse.mean()),flush=True)
            if time.monotonic()-start>600: raise RuntimeError('budget exceeded')
    with (ROOT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    saved=list(csv.DictReader((ROOT/'metrics.csv').open(encoding='utf-8')))
    if len(saved)!=144: raise RuntimeError('row count')
    for r in saved:
        a=np.load(ROOT/f"seed_{r['seed']}_{r['condition']}"/f"{r['model']}_predictions.npz")
        actual=float(np.mean((a['prediction'][:,int(r['market']),list(b.H).index(int(r['horizon']))]-a['target'][:,int(r['market']),list(b.H).index(int(r['horizon']))])**2))
        if not np.isclose(actual,float(r['mse']),rtol=1e-12,atol=0): raise RuntimeError('metric mismatch')
    if any(sha(PARENT/path)!=s for path,s in originals.items()): raise RuntimeError('original modified')
    aggregate={c:{m:float(np.mean([r['mse'] for r in rows if r['condition']==c and r['model']==m])) for m in METHODS} for c in ('shared','no_shared')}
    by_seed={f'{s}_{c}':{m:float(np.mean([r['mse'] for r in rows if r['seed']==s and r['condition']==c and r['model']==m])) for m in METHODS} for s in b.SEEDS for c in ('shared','no_shared')}
    verification={'wall_seconds':time.monotonic()-start,'metric_rows':len(saved),'metrics_reload_verified':True,'originals_unchanged':True,'future_refits':checks,'aggregate':aggregate,'by_seed':by_seed,'train_last_label':int(b.origins(b.N)['train'][-1]+20),'val_first_origin':int(b.origins(b.N)['val'][0]),'val_last_label':int(b.origins(b.N)['val'][-1]+20),'test_first_origin':int(ts[0]),'no_shared_population_independent':True,'independence_basis':'independent normal coordinates; entirely coordinatewise recursion; no shared preprocessing of targets/features; each market scaler is train-only'}
    (ROOT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    report='# Matched information results\n\nSecondary diagnostic; original uncertainty candidate and its failure retained unchanged. No model innovation claim; no market access.\n\n'+json.dumps({'aggregate':aggregate,'by_seed':by_seed},indent=2)+'\n\nIndependent increment ridge versus old level MLP changes both representation and estimator: attribution to representation alone is not identified. Matched joint versus independent ridge isolates added synchronized market histories, conditional on this learner/budget. Shuffled and fixed lag97 controls are negative-information checks, not proof of causal effects in actual markets. no_shared is independent in population; finite-sample gains/losses are estimation and selection noise, not true cross-market information. Shared common shocks are explicitly present, including market0 direct innovation reuse.\n\nAll 144 market/horizon metrics reloaded and verified; all 48 market-level future-perturbation full six-lambda refits exactly invariant. Original artifact hashes unchanged. See metrics.csv for every market/horizon, loss JSON for validation selection, provenance.json for absolute source/hash audit.\n'
    (ROOT/'RESULTS.md').write_text(report,encoding='utf-8'); print(json.dumps({'aggregate':aggregate,'by_seed':by_seed,'seconds':verification['wall_seconds']},indent=2))
if __name__=='__main__': main()
