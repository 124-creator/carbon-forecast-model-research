"""Observations-only direct structured response pilot. No market imports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json, time, hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
H=np.arange(1,11,dtype=float)

def features(past):
    return np.column_stack([past[:,i:i+20].mean(1) for i in (0,20,40)]+[past[:,-5:].mean(1),np.abs(past[:,-5:]).mean(1),(past[:,-5:]**2).mean(1)])

def generate(seed, process):
    # Unknown generator coefficients stay here; learners receive only measurements.
    rng=np.random.default_rng(seed+9100)
    n=P['samples']; t=np.arange(-59,1)
    drift=rng.normal(0,.10,(n,1)); amplitudes=rng.normal(0,.4,(n,2))
    past=drift+amplitudes[:,0,None]*np.exp(t/18)+amplitudes[:,1,None]*np.exp(t/5)+rng.normal(0,.15,(n,60))
    f=features(past)
    if process=='no_signal': future=rng.normal(0,.4,(n,10))
    else:
        # Strong positive controls use observable nonlinear conditional means;
        # intentionally expressible, not evidence of realistic market dynamics.
        trend=.12*np.tanh(2*f[:,0])
        a=.9*np.tanh(3*(f[:,3]-f[:,0])); b=.6*np.tanh(2*(f[:,2]-f[:,0]))
        if process=='asymmetric':
            trend=.12*np.tanh(2*f[:,0]+.5*f[:,4]-.3)
            a=.9*np.tanh(3*(f[:,3]-f[:,0])+.8*f[:,4]-.25)
            b=.6*np.tanh(2*(f[:,2]-f[:,0])-.5*f[:,5]+.2)
        future=trend[:,None]*H+a[:,None]*(1-np.exp(-H*np.log(2)/2.5))+b[:,None]*(1-np.exp(-H*np.log(2)/9))
        future+=np.cumsum(rng.normal(0,.08,(n,10)),axis=1)
    return past,future

def dictionary(raw=None):
    if raw is None: rates=np.log(2)/np.array([2.,8.]); deriv=np.zeros(2)
    else:
        low=np.log(2)/np.array([4.,16.]); high=np.log(2)/np.array([1.,5.])
        s=1/(1+np.exp(-raw)); rates=low+(high-low)*s; deriv=(high-low)*s*(1-s)
    D=np.vstack([H,1-np.exp(-rates[0]*H),1-np.exp(-rates[1]*H)])
    dd=H[None,:]*np.exp(-rates[:,None]*H)*deriv[:,None]
    return D,dd,rates

def factory(kind,X,Y,reg):
    nonlinear=kind in ('structured_rate','fixed_dictionary','mlp')
    structured=kind!='mlp' and kind!='linear'
    k=3 if structured else 10; d=X.shape[1]; width=4
    size=(d*width+width+width*k+k) if nonlinear else d*k+k
    size+=2 if kind=='structured_rate' else 0
    def calc(theta, data=X, target=Y, gradient=True):
        off=0
        if nonlinear:
            W=theta[:d*width].reshape(d,width); off=d*width
            bias=theta[off:off+width]; off+=width
            V=theta[off:off+width*k].reshape(width,k); off+=width*k
            c=theta[off:off+k]; off+=k
            z=np.tanh(data@W+bias); coeff=z@V+c
        else:
            W=theta[:d*k].reshape(d,k); off=d*k
            c=theta[off:off+k]; off+=k; coeff=data@W+c
        if structured:
            D,dd,rates=dictionary(theta[off:] if kind=='structured_rate' else None); pred=coeff@D
        else: pred=coeff
        if not gradient: return pred
        err=pred-target; loss=np.mean(err**2)+reg*np.mean(theta[:off]**2)
        gp=2*err/err.size; gc=gp@D.T if structured else gp
        if nonlinear:
            gz=(gc@V.T)*(1-z*z)
            grad=np.concatenate([(data.T@gz).ravel(),gz.sum(0),(z.T@gc).ravel(),gc.sum(0)])
        else: grad=np.concatenate([(data.T@gc).ravel(),gc.sum(0)])
        grad+=2*reg*theta[:off]/off
        if kind=='structured_rate': grad=np.r_[grad,[(gp*coeff[:,j+1,None]*dd[j]).sum() for j in range(2)]]
        return loss,grad
    return size,calc

def main():
    start=time.monotonic(); records=[]; checks=[]
    protocol_hash=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
    for process in P['processes']:
        for seed in P['seeds']:
            past,Y=generate(seed,process); F=features(past)
            mu=F[:600].mean(0); sd=F[:600].std(0); X=(F-mu)/sd
            np.savez_compressed(ROOT/f'data_{process}_{seed}.npz',past=past,targets=Y,features=X,mean=mu,std=sd)
            for reg in P['regularization']:
                for kind in P['models']:
                    if time.monotonic()-start>P['budget_seconds']: raise RuntimeError('preregistered budget exceeded')
                    rng=np.random.default_rng(seed); size,calc=factory(kind,X[:600],Y[:600],reg)
                    theta=rng.normal(0,.08,size); initial,g=calc(theta)
                    direction=rng.normal(size=size); direction/=np.linalg.norm(direction); eps=1e-5
                    fd=(calc(theta+eps*direction)[0]-calc(theta-eps*direction)[0])/(2*eps)
                    ge=abs(fd-g@direction)/max(1.,abs(fd))
                    if ge>1e-6: raise RuntimeError(f'gradient mismatch {kind} {ge}')
                    tic=time.monotonic(); accepted=[]
                    fit=minimize(calc,theta,jac=True,method='L-BFGS-B',callback=lambda v: accepted.append(float(calc(v)[0])),options={'maxiter':160,'ftol':1e-10,'gtol':1e-6})
                    pred=calc(fit.x,data=X,gradient=False); final,gg=calc(fit.x)
                    name=f'{process}_{seed}_{reg}_{kind}'
                    np.savez_compressed(ROOT/(name+'.npz'),prediction=pred,parameters=fit.x,initial_parameters=theta,accepted_objectives=accepted)
                    r=dict(process=process,seed=seed,reg=reg,model=kind,parameters=size,initial=initial,final=final,gradient_norm=float(np.linalg.norm(gg)),gradient_check=ge,success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),accepted_steps=len(accepted),seconds=time.monotonic()-tic,train_mse=float(np.mean((pred[:600]-Y[:600])**2)),validation_mse=float(np.mean((pred[600:800]-Y[600:800])**2)),test_mse=float(np.mean((pred[800:]-Y[800:])**2)),zero_test_mse=float(np.mean(Y[800:]**2)),finite=bool(np.isfinite(pred).all()))
                    records.append(r); (ROOT/'results.json').write_text(json.dumps(records,indent=2))
                    print(name,round(r['test_mse']/r['zero_test_mse'],4),fit.success,flush=True)
    primary=[r for r in records if r['model']=='structured_rate' and r['reg']==.001]
    gate=all(r['finite'] and r['final']<.99*r['initial'] and r['test_mse']<(.8 if r['process']!='no_signal' else 1.1)*r['zero_test_mse'] for r in primary)
    receipt=dict(protocol_sha256=protocol_hash,fit_count=len(records),elapsed_seconds=time.monotonic()-start,gate_passed=gate,market_run=False,market_reason='Separate market identity/source audit required even if synthetic gate passes',optimizer_success_count=sum(r['success'] for r in records))
    (ROOT/'receipt.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt))
if __name__=='__main__': main()
