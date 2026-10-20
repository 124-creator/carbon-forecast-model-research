import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent/'pilot'/'run_pilot.py'
spec=importlib.util.spec_from_file_location('pilot',PILOT); pilot=importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
CFG=json.loads((ROOT/'protocol.json').read_text()); START=time.monotonic()
def dump(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
mask=np.concatenate([np.tile(np.arange(12)<6,20),np.tile(np.arange(12)<6,4),np.tile(np.arange(12)<6,4),np.zeros(4,dtype=bool)])
def project(G,g):
    norms=np.linalg.norm(G,axis=1); A=G/np.maximum(norms[:,None],1e-30)
    lam,_=nnls(A.T,-g,maxiter=10000,atol=1e-12)
    d=g+A.T@lam
    if np.min(A@d)<-1e-7: raise RuntimeError('infeasible projection')
    return d,float(np.min(A@d)),float(np.max(np.abs(lam*(A@d))))
def gradients(p,x,y,method,groups):
    arch='independent' if method=='independent' else 'shared_private'
    gs=np.array([pilot.objective(p,x,y,arch,t)[1] for t in range(4)])
    g=gs.mean(0); diag={}
    if method=='pcgrad':
        G=gs[:,mask]; H=G.copy(); rng=groups
        for i in range(4):
            for j in rng.permutation(4):
                if j==i: continue
                dot=H[i]@G[j]
                if dot<0: H[i]-=dot/(G[j]@G[j]+1e-30)*G[j]
        g[mask]=H.mean(0)
    elif method in ['block','random_block','no_block']:
        ids=groups; G=np.array([pilot.objective(p,x[:,ix],y[:,ix],arch,t)[1][mask] for t in range(4) for ix in ids])
        d,feas,kkt=project(G,g[mask]); diag=dict(min_normalized_dot=feas,kkt_complementarity=kkt,shared_norm=float(np.linalg.norm(d)),original_shared_norm=float(np.linalg.norm(g[mask])))
        g[mask]=d
    return g,diag

def run():
    results=[]; audit=[]
    dump('start.json',dict(protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),pilot_sha256=hashlib.sha256(PILOT.read_bytes()).hexdigest(),run_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gradient_checks=pilot.gradient_check()))
    for seed in CFG['seeds']:
        x,y=pilot.data(seed); np.savez_compressed(ROOT/f'data_{seed}.npz',x=x,y=y)
        for method in CFG['methods']:
            arch='independent' if method=='independent' else 'shared_private'; p=pilot.init(seed); best=np.full(4,np.inf); bp=np.tile(p,(4,1)); bs=np.zeros(4,int); trace=[]; rng=np.random.default_rng(seed+333)
            ids=[np.arange(i*16,(i+1)*16) for i in range(4)]
            if method=='random_block': ids=list(np.array_split(rng.permutation(64),4))
            if method=='no_block': ids=[np.arange(64)]
            completed=0
            for step in range(CFG['steps']+1):
                if time.monotonic()-START>CFG['budget_seconds']: break
                g,diag=gradients(p,x[:,:64],y[:,:64],method,rng if method=='pcgrad' else ids)
                if step%20==0:
                    val=((pilot.forward(p,x[:,64:128],arch)[0]-y[:,64:128])**2).mean(1); improve=val<best; best[improve]=val[improve]; bp[improve]=p; bs[improve]=step
                    trace.append(dict(step=step,train_loss=pilot.objective(p,x[:,:64],y[:,:64],arch)[0],val=val.tolist(),**diag)); dump(f'trace_{seed}_{method}.json',trace)
                    np.savez_compressed(ROOT/f'checkpoint_{seed}_{method}.npz',p=p,best_p=bp,best_steps=bs)
                if step==CFG['steps']: completed=step; break
                new=p-CFG['learning_rate']*g
                if method in ['block','random_block','no_block']:
                    before=np.array([pilot.objective(p,x[:,ix],y[:,ix],arch,t)[0] for t in range(4) for ix in ids]); after=np.array([pilot.objective(new,x[:,ix],y[:,ix],arch,t)[0] for t in range(4) for ix in ids]); shared=p.copy(); shared[mask]=new[mask]; only=np.array([pilot.objective(shared,x[:,ix],y[:,ix],arch,t)[0] for t in range(4) for ix in ids]); audit.append(dict(seed=seed,method=method,step=step,before=before.tolist(),after_joint=after.tolist(),after_shared_only=only.tolist(),**diag))
                p=new; completed=step+1
            pred=np.array([pilot.forward(bp[t],x,arch)[0][t] for t in range(4)])
            np.savez_compressed(ROOT/f'fit_{seed}_{method}.npz',p_by_task=bp,pred=pred,best_steps=bs)
            row=dict(seed=seed,method=method,parameters=340,shared_parameters=int(mask.sum()) if arch!='independent' else 0,steps=completed,best_steps=bs.tolist(),train_mse=((pred[:,:64]-y[:,:64])**2).mean(1).tolist(),val_mse=((pred[:,64:128]-y[:,64:128])**2).mean(1).tolist(),test_mse=((pred[:,128:]-y[:,128:])**2).mean(1).tolist())
            results.append(row); dump('results.json',results); dump('actual_block_losses.json',audit); print(json.dumps(row),flush=True)
            if time.monotonic()-START>CFG['budget_seconds']: break
    dump('receipt.json',dict(seconds=time.monotonic()-START,runs=len(results),expected_runs=12,complete=len(results)==12 and all(r['steps']==CFG['steps'] for r in results),numpy=np.__version__))
if __name__=='__main__': run()
