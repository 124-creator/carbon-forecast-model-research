"""Synthetic-only NumPy MLP pilot; no dataset readers or oracle inputs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
METHODS=['independent','fully_shared','shared_private','sp_orthogonal']

def data(seed):
    r=np.random.default_rng(seed); x=np.zeros((4,512,5))
    for t in range(1,512): x[:,t]=.55*x[:,t-1]+r.normal(size=(4,5))
    common=1.3*np.tanh(x[:,:,0]*x[:,:,1])+.6*np.sin(1.5*x[:,:,2])
    private=np.stack([.9*np.tanh(x[0,:,3])+.3*x[0,:,4],-.9*np.tanh(x[1,:,3])-.3*x[1,:,4],.7*np.sin(2*x[2,:,4]),1.4*np.sin(x[3,:,3]*x[3,:,4])])
    y=common+private; y[3]=private[3]+.7*np.cos(2*x[3,:,0]); y+=r.normal(scale=.15,size=y.shape)
    mu=x[:,:64].mean(axis=1,keepdims=True); sd=x[:,:64].std(axis=1,keepdims=True)
    return (x-mu)/sd,y

def init(seed):
    r=np.random.default_rng(seed+900)
    return np.concatenate([r.normal(scale=.35,size=240),np.zeros(48),r.normal(scale=.2,size=48),np.zeros(4)])

def unpack(p): return p[:240].reshape(4,5,12),p[240:288].reshape(4,12),p[288:336].reshape(4,12),p[336:]

def routes(method):
    a=np.zeros((4,4,12))
    for t in range(4):
        if method=='independent': a[t,t,:]=1
        elif method=='fully_shared': a[t,:,:]=.5
        else: a[t,:,:6]=.5; a[t,t,6:]=1
    return a

def forward(p,x,method):
    w,b,v,c=unpack(p); h=np.tanh(np.einsum('tnd,bdh->tbnh',x,w)+b[None,:,None,:]); a=routes(method)
    pred=np.einsum('tbnh,bh,tbh->tn',h,v,a)+c[:,None]
    return pred,h

def objective(p,x,y,method,task=None):
    pred,h=forward(p,x,method); w,b,v,c=unpack(p); a=routes(method); e=pred-y; n=y.shape[1]
    mask=np.ones((4,1)) if task is None else np.eye(4)[task,:,None]*4
    loss=np.mean(e*e*mask); dp=2*e*mask/(4*n)
    dout=dp[:,None,:,None]*a[:,:,None,:]
    if method=='sp_orthogonal' and task is None:
        sa=a.copy(); sa[:,:,6:]=0; pa=a-sa
        s=np.einsum('tbnh,bh,tbh->tn',h,v,sa); q=np.einsum('tbnh,bh,tbh->tn',h,v,pa)
        sc=s-s.mean(1,keepdims=True); qc=q-q.mean(1,keepdims=True); cov=np.mean(sc*qc,axis=1)
        loss+=.1*np.mean(cov**2)
        ds=.2*cov[:,None]*qc/(4*n); dq=.2*cov[:,None]*sc/(4*n)
        dout+=ds[:,None,:,None]*sa[:,:,None,:]+dq[:,None,:,None]*pa[:,:,None,:]
    dv=np.einsum('tbnh,tbnh->bh',dout,h)
    dz=dout*v[None,:,None,:]*(1-h*h)
    dw=np.einsum('tnd,tbnh->bdh',x,dz); db=dz.sum((0,2)); dc=dp.sum(1)
    return float(loss),np.concatenate([dw.ravel(),db.ravel(),dv.ravel(),dc])

def gradient_check():
    x,y=data(17); x=x[:,:11]; y=y[:,:11]; p=init(17); out={}; r=np.random.default_rng(4)
    for m in METHODS:
        _,g=objective(p,x,y,m); errors=[]
        for j in r.choice(len(p),size=20,replace=False):
            plus=p.copy(); minus=p.copy(); plus[j]+=1e-6; minus[j]-=1e-6
            fd=(objective(plus,x,y,m)[0]-objective(minus,x,y,m)[0])/2e-6
            errors.append(abs(fd-g[j]))
        out[m]=max(errors)
        if max(errors)>1e-6: raise RuntimeError(('gradient',m,out[m]))
    return out

def run():
    start=time.monotonic(); protocol=hashlib.sha256((ROOT/'protocol.md').read_bytes()).hexdigest()
    checks=gradient_check(); results=[]
    for seed in [17,29]:
        x,y=data(seed); np.savez_compressed(ROOT/f'data_{seed}.npz',x=x,y=y)
        for method in METHODS:
            p=init(seed); mom=np.zeros_like(p); var=np.zeros_like(p); best=np.inf; bestp=None; trace=[]; ts=time.monotonic()
            for step in range(1201):
                loss,g=objective(p,x[:,:64],y[:,:64],method)
                if step%100==0:
                    val=np.mean((forward(p,x[:,64:128],method)[0]-y[:,64:128])**2)
                    gs=np.array([objective(p,x[:,:64],y[:,:64],method,t)[1] for t in range(4)])
                    norms=np.linalg.norm(gs,axis=1); cos=gs@gs.T/np.maximum(norms[:,None]*norms[None,:],1e-20)
                    trace.append(dict(step=step,objective=loss,val=float(val),gradient_norm=float(np.linalg.norm(g)),task_gradient_norms=norms.tolist(),task_gradient_cosines=cos.tolist()))
                    if val<best: best=float(val); bestp=p.copy(); beststep=step
                    (ROOT/f'trace_{seed}_{method}.json').write_text(json.dumps(trace,indent=2))
                if step==1200: break
                if time.monotonic()-start>840: raise RuntimeError('14min training stop')
                mom=.9*mom+.1*g; var=.999*var+.001*g*g
                p-=.008*(mom/(1-.9**(step+1)))/(np.sqrt(var/(1-.999**(step+1)))+1e-8)
            pred,h=forward(bestp,x,method); row=dict(seed=seed,method=method,parameters=len(p),steps=1200,best_step=beststep,seconds=time.monotonic()-ts)
            for name,sl in [('train',slice(0,64)),('validation',slice(64,128)),('test',slice(128,None))]: row[name+'_mse']=np.mean((pred[:,sl]-y[:,sl])**2,axis=1).tolist()
            if 'sp' in method or method=='shared_private':
                w,b,v,c=unpack(bestp); a=routes(method); sa=a.copy(); sa[:,:,6:]=0
                shared=np.einsum('tbnh,bh,tbh->tn',h,v,sa); private=pred-shared-c[:,None]
                row['remove_shared_test_mse']=np.mean((private[:,128:]+c[:,None]-y[:,128:])**2,axis=1).tolist()
                row['remove_private_test_mse']=np.mean((shared[:,128:]+c[:,None]-y[:,128:])**2,axis=1).tolist()
                row['train_shared_private_covariance']=np.mean((shared[:,:64]-shared[:,:64].mean(1,keepdims=True))*(private[:,:64]-private[:,:64].mean(1,keepdims=True)),axis=1).tolist()
            np.savez_compressed(ROOT/f'fit_{seed}_{method}.npz',p=bestp,pred=pred)
            results.append(row); (ROOT/'results.json').write_text(json.dumps(results,indent=2)); print(json.dumps(row),flush=True)
    receipt=dict(protocol_sha256=protocol,gradient_max_abs_error=checks,seconds=time.monotonic()-start,runs=len(results),numpy=np.__version__)
    (ROOT/'receipt.json').write_text(json.dumps(receipt,indent=2)); print(receipt)
if __name__=='__main__': run()
