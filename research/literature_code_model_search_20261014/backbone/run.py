"""Ordinary fully trainable nonlinear capability baseline; local outputs only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,zipfile,importlib.util
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'research/deep_direction_review_20261013/calibration'))
import calibrate as base
sys.path.insert(0,str(ROOT/'research/local_candidate_next_20261013'))
import run as old
base.OUT=OUT
D=60;K=24;O=3;REG=.01
START=time.monotonic()
def check(c,msg):
    if not c:raise ValueError(msg)
def dump(n,v):base.dump(n,v)
def unpack(t):
    a=D*K;b=a+K;c=b+K*O
    return t[:a].reshape(D,K),t[a:b],t[b:c].reshape(K,O),t[c:]
def initial(seed):
    r=np.random.default_rng(seed)
    return np.r_[r.normal(size=D*K)/np.sqrt(D),np.zeros(K),r.normal(size=K*O)*.05,np.zeros(O)]
def forward(t,x,loc):
    w,b,v,c=unpack(t);return loc[:,None]*(np.tanh(x@w+b)@v+c)
def objective(t,x,loc,y,hs):
    w,b,v,c=unpack(t);h=np.tanh(x@w+b);pred=loc[:,None]*(h@v+c);e=(pred-y)/hs
    val=float(np.mean(e*e)+REG*np.dot(t,t))
    dz=2*(pred-y)/(hs*hs)*loc[:,None]/y.size
    dh=(dz@v.T)*(1-h*h)
    grad=np.r_[(x.T@dh).ravel(),dh.sum(0),(h.T@dz).ravel(),dz.sum(0)]+2*REG*t
    check(np.isfinite(val) and np.isfinite(grad).all(),'nonfinite objective/gradient')
    return val,grad
def fit(x,loc,y,hs,seed):
    check(time.monotonic()-START<900,'Budget exceeded')
    t=initial(seed);first=objective(t,x,loc,y,hs)[0];hist=[first]
    def callback(q):
        check(time.monotonic()-START<900,'Budget exceeded');hist.append(objective(q,x,loc,y,hs)[0])
    f=minimize(objective,t,args=(x,loc,y,hs),jac=True,method='L-BFGS-B',callback=callback,options=dict(maxiter=200,maxfun=240,ftol=1e-9,gtol=1e-6))
    final,grad=objective(f.x,x,loc,y,hs)
    return f.x,dict(seed=seed,initial_objective=first,final_objective=final,decreased=final<first,history=hist,iterations=int(f.nit),evaluations=int(f.nfev),solver_success=bool(f.success),solver_message=str(f.message),gradient_norm=float(np.linalg.norm(grad)),parameter_movement=float(np.linalg.norm(f.x-t)))
def gradient_test():
    r=np.random.default_rng(818);x=r.normal(size=(11,D));loc=r.uniform(.3,2,size=11);y=r.normal(size=(11,3));hs=np.array([1.,2.,3.]);t=initial(7);g=objective(t,x,loc,y,hs)[1];err=[]
    for j in r.choice(len(t),size=40,replace=False):
        e=np.zeros(len(t));e[j]=1e-6
        fd=(objective(t+e,x,loc,y,hs)[0]-objective(t-e,x,loc,y,hs)[0])/2e-6
        err.append(abs(fd-g[j])/max(1e-5,abs(fd),abs(g[j])))
    for _ in range(5):
        d=r.normal(size=len(t));d/=np.linalg.norm(d);fd=(objective(t+1e-6*d,x,loc,y,hs)[0]-objective(t-1e-6*d,x,loc,y,hs)[0])/2e-6;an=g@d
        err.append(abs(fd-an)/max(1e-5,abs(fd),abs(an)))
    out=dict(max_relative_error=max(err),checks=len(err),passed=bool(max(err)<1e-5));dump('GRADIENT_CHECK.json',out);return out['passed']
def controls():
    rows=[];saved={}
    for seed in [7,17]:
        r=np.random.default_rng(1000+seed);x=r.normal(size=(2480,D));noise=r.normal(size=(2480,3))
        for kind in ['nonlinear_positive','no_signal']:
            y=np.tanh(2*x[:,0,None])*np.array([1.,2.,3.])+.1*noise if kind=='nonlinear_positive' else noise
            t,log=fit(x[:480],np.ones(480),y[:480],np.ones(3),seed);pred=forward(t,x[480:],np.ones(2000))
            a=np.column_stack([x[:480],np.ones(480)]);av=np.column_stack([x[480:],np.ones(2000)]);coef=np.linalg.solve(a.T@a+REG*480*np.eye(61),a.T@y[:480]);lin=av@coef
            mse=float(np.mean((pred-y[480:])**2));rw=float(np.mean(y[480:]**2));lm=float(np.mean((lin-y[480:])**2));ratio=mse/rw
            passed=ratio<.6 and mse/lm<.9 and log['decreased'] and np.isfinite(t).all()
            rows.append(dict(kind=kind,seed=seed,mse=mse,rw_mse=rw,ridge_mse=lm,ratio_rw=ratio,ratio_ridge=mse/lm,positive_gate=bool(passed) if kind=='nonlinear_positive' else None,training=log))
            saved[f'{kind}_{seed}_pred']=pred;saved[f'{kind}_{seed}_true']=y[480:];saved[f'{kind}_{seed}_theta']=t
    np.savez_compressed(OUT/'CONTROL_LOCAL_ONLY.npz',**saved);dump('CONTROLS.json',rows)
    return all(v['positive_gate'] for v in rows if v['kind']=='nonlinear_positive')
def prepare(p):
    raw=base.raw_features(p);loc=np.sqrt(np.mean(raw*raw,axis=1));x=np.divide(raw,loc[:,None],out=np.zeros_like(raw),where=loc[:,None]>0)
    return x,loc

def main():
    protocol_hash=hashlib.sha256((OUT/'PROTOCOL.json').read_bytes()).hexdigest()
    dump('ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,dependencies={m:bool(importlib.util.find_spec(m)) for m in ['torch','sklearn']},protocol_sha256=protocol_hash))
    with zipfile.ZipFile('D:/默认下载/carbon_stageF_20261013_complete.zip') as z:
        names=['carbon_stageF_20261013/src/model.py','carbon_stageF_20261013/src/market.py','carbon_stageF_20261013/DELIVERY_MANIFEST.json']
        reviewed={n:hashlib.sha256(z.read(n)).hexdigest() for n in names}
        manifest=json.loads(z.read(names[-1]));dump('SOURCE_REVIEW.json',dict(selected_archive_sources=reviewed,manifest=manifest,archive_code_executed=False,old_weights_loaded=False,direction_learning='Reviewed explicit trainable matrix interaction with known poles, not generic market capability proof',F='Encoder directions learned with torch, ridge head profiled; constrained interaction is not fixed RFF',old_manifest_sha256=hashlib.sha256((ROOT/'research/local_candidate_next_20261013/MANIFEST.json').read_bytes()).hexdigest()))
    g=gradient_test();positive=controls();dump('GATE.json',dict(gradient_passed=g,positive_passed=positive,market_allowed=g and positive))
    if not(g and positive):
        dump('RESULTS.json',dict(status='STOP_POSITIVE_OR_GRADIENT_FAILED',seconds=time.monotonic()-START));return
    dates,p=base.read_prefix();initial_n=int(np.sum(dates<'2020-01-01'));origins=np.array([t for t in range(initial_n,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
    check(len(origins)==451,'451 origins');years=np.array([d[:4] for d in dates[origins]]);truth=p[origins[:,None]+base.H]-p[origins,None]
    x,loc=prepare(p);hs=np.sqrt([np.mean((p[h:initial_n]-p[:initial_n-h])**2) for h in base.H]);saved=dict(origins=origins,true_changes=truth,horizon_scales=hs);logs=[];results=[]
    for seed in [7,17]:
        pred=np.full_like(truth,np.nan)
        for end in range(initial_n,len(p),40):
            pos=np.where((origins>=end)&(origins<end+40))[0]
            if not len(pos):continue
            idx=np.arange(max(60,end-20-480),end-20);check(idx[-1]+20<end,'Maturity violation');y=p[idx[:,None]+base.H]-p[idx,None]
            theta,log=fit(x[idx],loc[idx],y,hs,seed);pred[pos]=forward(theta,x[origins[pos]],loc[origins[pos]])
            log.update(fit_end_exclusive=end,max_training_target=int(idx[-1]+20),min_training_origin=int(idx[0]),n_training=len(idx));logs.append(log);saved[f'theta_{seed}_{end}']=theta
        check(np.isfinite(pred).all(),'forecast finite');saved[f'RAW_MLP_seed{seed}']=pred;s,rows=base.score(pred,truth,years);results.append(dict(name=f'RAW_MLP_seed{seed}',score=s,by_year=rows,historical_selected=False))
    for name in ['RW','MLE_window480','PROBE_trend_alpha1']:
        pred=np.full_like(truth,np.nan)
        for end in range(initial_n,len(p),40):
            pos=np.where((origins>=end)&(origins<end+40))[0]
            if not len(pos):continue
            if name=='PROBE_trend_alpha1':v,log=old.probe(p,dates,end,origins[pos],'trend',1.)
            else:v,log=base.predict(p,end,origins[pos],dict(family='RW') if name=='RW' else dict(family='MLE',window=480))
            pred[pos]=v;log.update(name=name);logs.append(log)
        saved[name]=pred;s,rows=base.score(pred,truth,years);results.append(dict(name=name,score=s,by_year=rows,historical_selected=name!='RW'))
    dump('TRAINING_LOGS.json',logs);np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved)
    mean=float(np.mean([v['score'] for v in results[:2]]));dump('RESULTS.json',dict(status='COMPLETE_DEVELOPMENT_ONLY',results=results,primary_two_seed_mean_loss_score=mean,seed_selection=False,prediction_ensemble=False,seconds=time.monotonic()-START,protocol_sha256=protocol_hash,parameter_count=len(initial(7)),training_fits=sum('seed' in v for v in logs),solver_non_success=sum(not v['solver_success'] for v in logs if 'seed' in v),train_non_decrease=sum(not v['decreased'] for v in logs if 'seed' in v)))
    print(json.dumps(dict(scores={v['name']:v['score'] for v in results},two_seed_mean=mean,seconds=time.monotonic()-START),indent=2))
if __name__=='__main__':main()
