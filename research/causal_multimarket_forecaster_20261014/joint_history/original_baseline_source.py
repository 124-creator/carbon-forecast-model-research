"""Pure synthetic prices. NumPy analytic backprop; no latent inputs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import json,csv,time,sys
from pathlib import Path
N=1800; W=32; H=np.array([1,5,20]); SEEDS=(17,43); EPOCHS=90

def generate(seed,shared):
    r=np.random.default_rng(seed)
    innovations=r.normal(size=(N,3)); private_noise=r.normal(size=(N,3))
    z=np.zeros((N,3)); u=np.zeros((N,3)); p=np.zeros((N,3)); p[0]=[60,85,110]
    for t in range(1,N):
        z[t]=.94*z[t-1]+.13*innovations[t]
        u[t]=np.array([.55,-.35,.80])*u[t-1]+.22*private_noise[t]
        common=np.repeat(z[t,0],3) if shared else z[t]
        # Integrated level with persistent predictable innovations and transients.
        p[t]=p[t-1]+np.array([.8,-.6,1.1])*common+(u[t]-u[t-1])+.04*innovations[t]
    return p,{'persistent':z,'transient':u,'innovations':innovations}

def origins(n):
    a=int(n*.6); b=int(n*.8)
    return {'train':np.arange(W-1,a-20),'val':np.arange(a,b-20),'test':np.arange(b,n-20)}

def scaler(p):
    a=int(len(p)*.6)
    return p[:a].mean(0),np.maximum(p[:a].std(0),1e-6)

def features(p,ts,mu,sd):
    return np.stack([(p[t-W+1:t+1]-mu)/sd for t in ts])

class Net:
    def __init__(self,seed,independent=False):
        r=np.random.default_rng(seed); self.independent=independent
        self.params=[r.normal(0,1/np.sqrt(W),size=((3 if independent else 1),W,24)),np.zeros((3 if independent else 1,24)),r.normal(0,.03,size=(3,24,3)),np.zeros((3,3))]
    def forward(self,x,base):
        a,b,c,d=self.params
        hidden=np.stack([np.tanh(x[:,:,m]@a[m if self.independent else 0]+b[m if self.independent else 0]) for m in range(3)],1)
        return base[:,:,None]+np.einsum('bmk,mkh->bmh',hidden,c)+d,hidden
    def gradients(self,x,base,y):
        pred,h=self.forward(x,base); e=pred-y; grads=[]
        for m in range(3):
            g=[np.zeros_like(v) for v in self.params]; delta=2*e[:,m]/(len(x)*3)
            g[2][m]=h[:,m].T@delta; g[3][m]=delta.sum(0)
            dh=(delta@self.params[2][m].T)*(1-h[:,m]**2); j=m if self.independent else 0
            g[0][j]=x[:,:,m].T@dh; g[1][j]=dh.sum(0); grads.append(g)
        return float(np.mean(e**2)),grads

def project(gs,rng):
    out=gs.copy()
    for i in range(len(gs)):
        for j in rng.permutation(len(gs)):
            dot=out[i]@gs[j]
            if dot<0: out[i]-=dot/(gs[j]@gs[j]+1e-20)*gs[j]
    return out

def gradient_check():
    r=np.random.default_rng(9); x=r.normal(size=(4,W,3)); y=r.normal(size=(4,3,3)); base=np.zeros((4,3)); net=Net(9)
    _,g=net.gradients(x,base,y); analytic=[sum(v[k] for v in g)/3 for k in range(4)]; errors=[]
    for k,p in enumerate(net.params):
        for index in [tuple(0 for _ in p.shape),tuple(s-1 for s in p.shape)]:
            old=p[index]; eps=1e-6
            p[index]=old+eps; plus=net.gradients(x,base,y)[0]
            p[index]=old-eps; minus=net.gradients(x,base,y)[0]; p[index]=old
            errors.append(abs((plus-minus)/(2*eps)-analytic[k][index]))
    return max(errors)

def train(p,seed,kind,out):
    mu,sd=scaler(p); split=origins(len(p)); data={}
    for name,ts in split.items():
        x=features(p,ts,mu,sd); y=np.stack([p[t+H] for t in ts]).transpose(0,2,1)
        data[name]=(x,p[ts],y)
    net=Net(seed,kind=='independent'); initial=[v.copy() for v in net.params]
    rng=np.random.default_rng(seed+1000); prng=np.random.default_rng(seed+2000)
    first=[np.zeros_like(v) for v in net.params]; second=[v.copy() for v in first]; step=0; best=float('inf'); history=[]; bestparams=None
    conflicts=0; pairs=0; norms=[]
    for epoch in range(EPOCHS):
        x,base,y=data['train']; epochloss=[]
        for start in range(0,len(x),128):
            # Chronological batches, no random sample split or shuffle.
            loss,gs=net.gradients(x[start:start+128],base[start:start+128],y[start:start+128]); epochloss.append(loss)
            shared=np.stack([np.concatenate([g[0].ravel(),g[1].ravel()]) for g in gs])
            for i in range(3):
                for j in range(i+1,3): conflicts+=int(shared[i]@shared[j]<0); pairs+=1
            norms.append(np.linalg.norm(shared,axis=1).tolist())
            agg=[sum(g[k] for g in gs)/3 for k in range(4)]
            if kind=='pcgrad':
                v=project(shared,prng).mean(0); n=net.params[0].size
                agg[0]=v[:n].reshape(net.params[0].shape); agg[1]=v[n:].reshape(net.params[1].shape)
            step+=1
            for k in range(4):
                first[k]=.9*first[k]+.1*agg[k]; second[k]=.999*second[k]+.001*agg[k]**2
                net.params[k]-=.002*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
        vx,vb,vy=data['val']; vl=float(np.mean((net.forward(vx,vb)[0]-vy)**2))
        history.append({'epoch':epoch+1,'train_mse':float(np.mean(epochloss)),'val_mse':vl})
        if vl<best: best=vl; bestparams=[v.copy() for v in net.params]; bestepoch=epoch+1
    net.params=bestparams
    tx,tb,ty=data['test']; pred=net.forward(tx,tb)[0]
    np.savez_compressed(out/f'{kind}_checkpoint.npz',**{f'p{k}':v for k,v in enumerate(net.params)},mu=mu,sd=sd)
    np.savez_compressed(out/f'{kind}_predictions.npz',prediction=pred,target=ty,origins=split['test'])
    audit={'best_epoch':bestepoch,'history':history,'gradient_conflict_fraction':conflicts/max(pairs,1),'gradient_norms':norms,'initial_parameters_checksum':float(sum(v.sum() for v in initial))}
    (out/f'{kind}_training.json').write_text(json.dumps(audit),encoding='utf-8')
    return ((pred-ty)**2).mean(0),{'mse':best,'epoch':bestepoch,'conflict':audit['gradient_conflict_fraction']}

def main():
    start=time.monotonic(); root=Path(__file__).resolve().parent; rows=[]; audits=[]
    for seed in SEEDS:
        for shared in [True,False]:
            name='shared' if shared else 'no_shared'; out=root/f'seed_{seed}_{name}'; out.mkdir(exist_ok=True)
            p,latent=generate(seed,shared); np.savez_compressed(out/'prices.npz',prices=p)
            # Latents are generator audit ONLY, never passed to train.
            np.savez_compressed(out/'generator_audit_only.npz',**latent)
            split=origins(N); mu,sd=scaler(p); t=int(split['test'][0]); q=p.copy(); q[t+1:]+=10000
            audit={'seed':seed,'condition':name,'counts':{k:len(v) for k,v in split.items()},'ranges':{k:[int(v[0]),int(v[-1])] for k,v in split.items()},'train_last_label':int(split['train'][-1]+20),'val_last_label':int(split['val'][-1]+20),'maturity_train':bool(split['train'][-1]+20<split['val'][0]),'maturity_val':bool(split['val'][-1]+20<split['test'][0]),'future_feature_invariant':bool(np.array_equal(features(p,[t],mu,sd),features(q,[t],mu,sd))),'future_scaler_invariant':bool(np.array_equal(scaler(p)[0],scaler(q)[0]))}
            audits.append(audit); allmetrics={}
            for kind in ['independent','mtl','pcgrad']:
                metric,info=train(p,seed,kind,out); allmetrics[kind]=metric
                for m in range(3):
                    for j,h in enumerate(H): rows.append({'seed':seed,'condition':name,'model':kind,'market':m,'horizon':int(h),'mse':float(metric[m,j]),'rmse':float(np.sqrt(metric[m,j]))})
                print(seed,name,kind,float(metric.mean()),info,flush=True)
            ts=split['test']; truth=np.stack([p[t+H] for t in ts]).transpose(0,2,1); rw=((p[ts,:,None]-truth)**2).mean(0)
            for m in range(3):
                for j,h in enumerate(H): rows.append({'seed':seed,'condition':name,'model':'persistence','market':m,'horizon':int(h),'mse':float(rw[m,j]),'rmse':float(np.sqrt(rw[m,j]))})
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    lookup={(r['seed'],r['condition'],r['model'],r['market'],r['horizon']):r['mse'] for r in rows}
    failures=[]
    for r in rows:
        if r['model'] in ['mtl','pcgrad']:
            ref=lookup[(r['seed'],r['condition'],'independent',r['market'],r['horizon'])]
            if r['mse']>=ref: failures.append({**r,'independent_mse':ref,'relative_degradation':r['mse']/ref-1})
    result={'framework':'numpy','torch_available':False,'seeds':list(SEEDS),'epochs':EPOCHS,'wall_seconds':time.monotonic()-start,'gradient_check_max_abs_error':gradient_check(),'audits':audits,'all_shared_losses_vs_independent':failures,'metric_rows':len(rows),'budget_seconds':600}
    (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('DONE',result['wall_seconds'],'seconds; rows',len(rows),'losses',len(failures),flush=True)
if __name__=='__main__': main()
