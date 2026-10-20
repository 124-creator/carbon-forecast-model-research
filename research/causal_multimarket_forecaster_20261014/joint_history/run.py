"""Observed joint history benchmark; generator imported unchanged by absolute path."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import importlib.util,json,csv,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path('E:/碳价格/11/实验十一十一/田/第三次-tian-碳价格实验/research/causal_multimarket_forecaster_20261014/baseline/benchmark.py')
spec=importlib.util.spec_from_file_location('original',SOURCE); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

def dataset(p):
    splits=b.origins(len(p)); d=np.diff(p,axis=0,prepend=p[:1]); data={}
    for k,t in splits.items():
        x=np.stack([d[i-b.W+2:i+1].ravel() for i in t]); y=np.stack([p[i+b.H]-p[i] for i in t]).transpose(0,2,1).reshape(len(t),9)
        data[k]=(x,y,t)
    return data,d

def ridge(data,rank=None):
    x,y,_=data['train']; mu=x.mean(0); sd=np.maximum(x.std(0),1e-8)
    xs={k:np.column_stack([np.ones(len(v[0])),(v[0]-mu)/sd]) for k,v in data.items()}
    best=None; hist=[]
    for lam in [0.01,0.1,1,10,100,1000]:
        a=xs['train']; penalty=np.eye(a.shape[1])*lam; penalty[0,0]=0
        coef=np.linalg.solve(a.T@a+penalty,a.T@y)
        if rank:
            u,s,v=np.linalg.svd(coef[1:],full_matrices=False); coef[1:]=(u[:,:rank]*s[:rank])@v[:rank]
        loss=float(np.mean((xs['val']@coef-data['val'][1])**2)); hist.append({'lambda':lam,'train_mse':float(np.mean((a@coef-y)**2)),'val_mse':loss})
        if best is None or loss<best[0]: best=(loss,coef.copy(),lam)
    return xs['test']@best[1],{'history':hist,'lambda':best[2]}, {'coef':best[1],'mu':mu,'sd':sd}

def mlp(data,seed):
    x,y,_=data['train']; mu=x.mean(0); sd=np.maximum(x.std(0),1e-8)
    xs={k:(v[0]-mu)/sd for k,v in data.items()}; r=np.random.default_rng(seed)
    ps=[r.normal(0,1/np.sqrt(x.shape[1]),(x.shape[1],48)),np.zeros(48),r.normal(0,.03,(48,9)),np.zeros(9)]
    m=[np.zeros_like(v) for v in ps]; v=[z.copy() for z in m]; step=0; best=np.inf; history=[]
    for epoch in range(90):
        losses=[]
        for start in range(0,len(x),128):
            xx=xs['train'][start:start+128]; yy=y[start:start+128]; h=np.tanh(xx@ps[0]+ps[1]); e=h@ps[2]+ps[3]-yy; de=2*e/e.size; dh=(de@ps[2].T)*(1-h*h)
            gs=[xx.T@dh,dh.sum(0),h.T@de,de.sum(0)]; losses.append(float(np.mean(e*e))); step+=1
            for j,g in enumerate(gs):
                m[j]=.9*m[j]+.1*g; v[j]=.999*v[j]+.001*g*g; ps[j]-=.002*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
        val=np.tanh(xs['val']@ps[0]+ps[1])@ps[2]+ps[3]; loss=float(np.mean((val-data['val'][1])**2)); history.append({'epoch':epoch+1,'train_mse':float(np.mean(losses)),'val_mse':loss})
        if loss<best: best=loss; bp=[z.copy() for z in ps]; be=epoch+1
    pred=np.tanh(xs['test']@bp[0]+bp[1])@bp[2]+bp[3]
    return pred,{'history':history,'best_epoch':be},{**{f'p{i}':z for i,z in enumerate(bp)},'mu':mu,'sd':sd}

def factor(data,d):
    # PCA measurement equation and AR state estimated ONLY from train observations.
    end=int(data['train'][2][-1]); obs=d[1:end+1]; mu=obs.mean(0); _,_,vt=np.linalg.svd(obs-mu,full_matrices=False); load=vt[0]; scores=(obs-mu)@load
    phi=float(np.clip(scores[:-1]@scores[1:]/(scores[:-1]@scores[:-1]+1e-12),-.99,.99)); q=max(float(np.var(scores[1:]-phi*scores[:-1])),1e-6); rr=max(float(np.mean(((obs-mu)-scores[:,None]*load)**2)),1e-6)
    z=0.; cov=q/(1-phi*phi); states=[]; covs=[]
    for row in d:
        prior=phi*z; pc=phi*phi*cov+q; gain=pc/(pc+rr); z=prior+gain*((row-mu)@load-prior); cov=(1-gain)*pc; states.append(z); covs.append(cov)
    fdata={}
    for k,(x,y,t) in data.items():
        z=np.array(states)[t]; c=np.array(covs)[t]
        # sole uncertainty candidate: state mean + posterior variance interaction.
        xx=np.column_stack([x,z,c,z*c]); fdata[k]=(xx,y,t)
    return fdata,{'load':load,'phi':phi,'q':q,'r':rr,'posterior_variance_range':[float(min(covs)),float(max(covs))]}

def main():
    start=time.monotonic(); rows=[]; checks=[]
    for seed in b.SEEDS:
        for shared in [True,False]:
            name='shared' if shared else 'no_shared'; out=ROOT/f'seed_{seed}_{name}'; out.mkdir(exist_ok=True)
            p,latent=b.generate(seed,shared); data,d=dataset(p); ts=data['test'][2]; target=np.stack([p[t+b.H] for t in ts]).transpose(0,2,1)
            np.savez_compressed(out/'observations.npz',prices=p)
            results={'persistence':(np.zeros((len(ts),9)),{},{} )}
            # Exact original independent learner, freshly retrained.
            b.train(p,seed,'independent',out); a=np.load(out/'independent_predictions.npz'); results['independent']=((a['prediction']-p[ts,:,None]).reshape(len(ts),9),{}, {})
            results['joint_linear']=ridge(data); results['joint_lowrank']=ridge(data,rank=2); results['joint_mlp']=mlp(data,seed)
            fd,fa=factor(data,d); pred,info,cp=ridge(fd); info['estimated_factor']=fa; results['uncertainty_candidate']=(pred,info,cp)
            # Oracle audit ONLY: true current latent, never passed to learners.
            common=np.repeat(latent['persistent'][ts,0,None],3,axis=1) if shared else latent['persistent'][ts]
            oracle=np.empty_like(target)
            for j,h in enumerate(b.H):
                oracle[:,:,j]=p[ts]+np.array([.8,-.6,1.1])*common*(.94*(1-.94**h)/(1-.94))+(np.array([.55,-.35,.80])**h-1)*latent['transient'][ts]
            results['oracle_audit']=((oracle-p[ts,:,None]).reshape(len(ts),9),{}, {})
            for method,(delta,info,cp) in results.items():
                pred=p[ts,:,None]+delta.reshape(len(ts),3,3); mse=np.mean((pred-target)**2,axis=0)
                np.savez_compressed(out/f'{method}_predictions.npz',prediction=pred,target=target,origins=ts)
                if cp: np.savez_compressed(out/f'{method}_checkpoint.npz',**cp)
                (out/f'{method}_loss.json').write_text(json.dumps(info,default=lambda x:x.tolist(),indent=2),encoding='utf-8')
                for m in range(3):
                    for j,h in enumerate(b.H): rows.append(dict(seed=seed,condition=name,model=method,market=m,horizon=int(h),mse=float(mse[m,j])))
                print(seed,name,method,float(mse.mean()),flush=True)
            if seed==17 and shared:
                t=int(ts[0]); pert=p.copy(); pert[t+1:]+=10000; pd,_=dataset(pert); _,_,pc2=ridge(pd); pc1=results['joint_linear'][2]
                equal=all(np.array_equal(pc1[k],pc2[k]) for k in pc1)
                if not equal: raise RuntimeError('future perturbation changed fitted checkpoint')
                x1=data['test'][0][0]; x2=pd['test'][0][0]
                if not np.array_equal(x1,x2): raise RuntimeError('future feature leakage')
                checks.append({'retrained_model':'joint_linear','future_cut':t,'checkpoint_exact_equal':equal,'origin_features_exact_equal':True})
    with (ROOT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    # independent reload verification of every saved prediction MSE.
    for row in rows:
        a=np.load(ROOT/f"seed_{row['seed']}_{row['condition']}"/f"{row['model']}_predictions.npz"); j=list(b.H).index(row['horizon']); actual=np.mean((a['prediction'][:,row['market'],j]-a['target'][:,row['market'],j])**2)
        if not np.isclose(actual,row['mse'],rtol=1e-12): raise RuntimeError('metric mismatch')
    summary={'wall_seconds':time.monotonic()-start,'rows':len(rows),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'future_retrain':checks,'all_saved_metrics_verified':True,'train_last_label':int(b.origins(b.N)['train'][-1]+20),'validation_first_origin':int(b.origins(b.N)['val'][0]),'validation_last_label':int(b.origins(b.N)['val'][-1]+20),'test_first_origin':int(b.origins(b.N)['test'][0]),'aggregates':{cond:{model:float(np.mean([r['mse'] for r in rows if r['condition']==cond and r['model']==model])) for model in results} for cond in ['shared','no_shared']}}
    (ROOT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
