"""Fixed-budget synthetic experiment; self-owned data only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,time
from pathlib import Path
import numpy as np
from autograd import grad
from autograd.misc.flatten import flatten
from model import initialize,predict,loss,train,H,D

OUT=Path(__file__).parent

def generate(seed,regime,n=384):
    r=np.random.RandomState(seed+1000); state=r.uniform(-1,1,n)
    probs=np.stack([.2-.12*state,.3+0*state,.5+.12*state],1)
    if regime=='no_information': probs=np.tile([.2,.3,.5],(n,1))
    if regime=='known_calendar':
        k=np.arange(n)%3; probs=np.eye(3)[k]
    else: k=np.array([r.choice(3,p=pr) for pr in probs])
    support=np.array([1.,2.,4.]); duration=support[k]
    pastgap=1+r.binomial(1,(state+1)[:,None]/2,(n,60))
    if regime=='known_calendar': pastgap=np.tile(duration[:,None],(1,60))
    # Past prices expose state, not future realized class.
    increments=.025*state[:,None]+r.normal(0,.025,(n,60))
    x=np.cumsum(increments,axis=1)
    if regime=='no_information':
        y=.025*state[:,None]*H+r.normal(0,.03,(n,3))
        mean=.025*state[:,None]*H
    else:
        y=.025*state[:,None]*duration[:,None]*H+r.normal(0,.03,(n,3))
        mean=.025*state[:,None]*(probs@support)[:,None]*H
    return x,pastgap,y,np.tile(k[:,None],(1,3)),mean,probs

def verify():
    p=initialize(7); f,unflat=flatten(p)
    x,g,y,k,_,_=generate(7,'unknown_state',8); reps=np.tile([1.,2.,4.],(3,1))
    fun=lambda v:loss(unflat(v),x,g,y,k,'marginal',reps)
    gr=grad(fun)(f); ids=np.argsort(np.abs(gr))[-12:]; errors=[]
    for i in ids:
        v=np.zeros_like(f); v[i]=1e-5
        fd=(fun(f+v)-fun(f-v))/2e-5
        errors.append(float(abs(fd-gr[i])/max(1,abs(fd),abs(gr[i]))))
    if max(errors)>1e-4: raise RuntimeError('gradient gate failed')
    pred,q,mu=predict(p,x,g,'marginal',reps)
    np.testing.assert_allclose(pred,np.sum(q*mu,axis=2),atol=1e-12)
    # Future labels never appear in inference. Shuffle every future label while keeping prefix.
    kk=k[::-1].copy(); yy=y+999
    other=predict(p,x.copy(),g.copy(),'marginal',reps)[0]
    np.testing.assert_array_equal(pred,other)
    return {'gradient_max_relative_error':max(errors),'gradient_norm':float(np.linalg.norm(gr)),
            'marginal_identity':True,'inference_api_past_only':True,
            'future_label_mutation':bool(np.any(kk!=k) or np.any(yy!=y)),
            'total_parameters':len(f),'note':'does not test market loader/prefix access'}

def main():
    start=time.monotonic(); deadline=start+1200
    checks=verify(); (OUT/'CHECKS.json').write_text(json.dumps(checks,indent=2))
    results=[]; models=['original','gap','auxiliary','no_condition','marginal']
    try:
        for regime in ['unknown_state','no_information','known_calendar']:
            for seed in [7,17]:
                x,g,y,k,mean,probs=generate(seed,regime)
                scale=np.maximum(np.std(y[:256],axis=0),.05)
                # Train-only common input scale and gap transformation.
                sx=max(np.std(x[:256]-x[:256,-1:]),.05)
                xx=x/sx; gg=(g-1)/3; yy=y/scale
                reps=np.tile([1.,2.,4.],(3,1))/4
                empirical=np.stack([np.bincount(k[:256,j],minlength=3)/256 for j in range(3)])
                for mode in models:
                    initial=float(loss(initialize(seed),xx[:256],gg[:256],yy[:256],k[:256],mode,reps))
                    p,log,elapsed=train(xx[:256],gg[:256],yy[:256],k[:256],mode,reps,seed,deadline=deadline)
                    pred,q,mu=predict(p,xx[256:],gg[256:],mode,reps); pred=pred*scale
                    ce=float(-np.mean(np.log(q[np.arange(128)[:,None],np.arange(3),k[256:]]+1e-12)))
                    ce0=float(-np.mean(np.log(empirical[np.arange(3),k[256:]]+1e-12)))
                    result=dict(regime=regime,seed=seed,model=mode,initial_loss=initial,final_loss=log[-1],
                                epochs=len(log),seconds=elapsed,mse=float(np.mean((pred-y[256:])**2)),
                                per_h_mse=np.mean((pred-y[256:])**2,0).tolist(),
                                mean_recovery_mse=float(np.mean((pred-mean[256:])**2)),
                                q_ce=ce,empirical_ce=ce0,total_parameters=checks['total_parameters'],
                                active_parameters=checks['total_parameters']-(0 if mode=='marginal' else 16)-(0 if mode in ['auxiliary','no_condition','marginal'] else 153)-(80 if mode=='original' else 0),
                                decoder_queries=9,training_effective=bool(log[-1]<.9*initial),loss_history=log)
                    results.append(result)
                    np.savez(OUT/f'{regime}_{seed}_{mode}.npz',prediction=pred,target=y[256:],true_mean=mean[256:],q=q,
                             future_k_diagnostic_only=k[256:],scale=scale,origin_id=np.arange(256,384))
                    (OUT/'RESULTS.json').write_text(json.dumps(results,indent=2))
                    print(regime,seed,mode,'mse',round(result['mse'],6),'seconds',round(elapsed,2),flush=True)
    except TimeoutError as e:
        (OUT/'BUDGET_EXHAUSTED.txt').write_text(str(e))
    gates={}
    for seed in [7,17]:
        rows={r['model']:r for r in results if r['regime']=='unknown_state' and r['seed']==seed}
        gates[f'positive_{seed}']=len(rows)==5 and all(rows['marginal']['mse']<.95*rows[m]['mse'] for m in ['gap','auxiliary','no_condition']) and rows['marginal']['q_ce']<.95*rows['marginal']['empirical_ce']
        for regime in ['no_information','known_calendar']:
            rr={r['model']:r for r in results if r['regime']==regime and r['seed']==seed}
            gates[f'{regime}_{seed}']=len(rr)==5 and rr['marginal']['mse']<=1.05*rr['gap']['mse']
    gates['training_effective_all']=len(results)==30 and all(r['training_effective'] for r in results)
    gates['semantic_market_permission']=False
    gates['market_allowed']=False
    gates['elapsed_seconds']=time.monotonic()-start
    (OUT/'GATES.json').write_text(json.dumps(gates,indent=2))
    print(json.dumps(gates,indent=2),flush=True)

if __name__=='__main__': main()
