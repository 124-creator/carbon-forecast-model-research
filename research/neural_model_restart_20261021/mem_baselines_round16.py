"""Training-only MEM baselines for the same pre2026 Parkinson proxy and origins.
Stationary and integrated positive conditional-mean recursions retain all dates.
No validation-dependent parameter selection or test-period computation.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import bridge_precision_round16 as B

HERE=Path(__file__).resolve().parent


def decode(theta,integrated):
    omega=float(np.exp(theta[0]))
    rho=1. if integrated else .999999*float(expit(theta[1]))
    share=float(expit(theta[1] if integrated else theta[2]))
    return omega,rho*share,rho*(1-share)


def forecasts(proxy,origin,parameters):
    omega,alpha,beta=parameters
    level=np.empty(len(proxy));level[0]=1.
    for t in range(1,len(proxy)):
        level[t]=omega+alpha*proxy[t-1]+beta*level[t-1]
    current=omega+alpha*proxy[origin]+beta*level[origin]
    daily=[current]
    for k in range(1,22):daily.append(omega+(alpha+beta)*daily[-1])
    cumulative=np.cumsum(np.stack(daily,1),axis=1)
    result=np.column_stack([cumulative[:,h-1]/h for h in [1,5,22]])
    assert np.isfinite(result).all() and (result>0).all()
    return result


def checks():
    p=np.ones(90);origin=np.array([20,40])
    np.testing.assert_allclose(forecasts(p,origin,(.05,.1,.85)),1,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(forecasts(p,origin,(0.,.1,.9)),1,rtol=1e-12,atol=1e-12)
    p2=p.copy();p2[41:]=100
    np.testing.assert_array_equal(forecasts(p,origin,(.05,.1,.85)),forecasts(p2,origin,(.05,.1,.85)))
    return {'constant_mean_recursion':True,'future_price_mutation_invariant':True,'integrated_keeps_finite_horizons':True}


def fit(D,integrated):
    observed_proxy=np.zeros(D['n'])
    observed_proxy[D['observed']]=np.exp(D['log_proxy'][D['observed']])
    scale=observed_proxy[:D['cut']].mean()
    tr,va=D['tr'],D['va'];origin=D['origin']
    y=D['y'][tr]/scale
    def objective(theta):
        pred=forecasts(observed_proxy[:D['cut']]/scale,origin[tr],decode(theta,integrated))
        return float((np.log(pred)+y/pred).mean())
    starts=([np.array([np.log(1e-5),np.log(a/(1-a))]) for a in [.06,.2]] if integrated else
            [np.array([np.log(1-r),np.log((r/.999999)/(1-r/.999999)),np.log(.1/.9)]) for r in [.9,.98]])
    records=[]
    for start in starts:
        bounds=[(-16,2)]+[(-8,16)]+([] if integrated else [(-8,8)])
        out=minimize(objective,start,method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':300,'ftol':1e-12,'gtol':1e-7,'maxls':50})
        assert np.isfinite(out.fun) and np.isfinite(out.x).all()
        records.append(dict(theta=out.x.tolist(),training_objective=float(out.fun),
                            converged=bool(out.success),message=str(out.message),iterations=int(out.nit)))
    best=min(records,key=lambda r:r['training_objective'])
    pars=decode(np.array(best['theta']),integrated)
    pred=forecasts(observed_proxy/scale,origin[va],pars)*scale
    return pred,dict(selected_by='training objective only',starts=records,selected=best,
                     omega_actual=pars[0]*scale,alpha=pars[1],beta=pars[2],rho=pars[1]+pars[2],
                     integrated=integrated,scale_from_pre2025_observations=scale)


def main():
    B.torch.set_num_threads(1)
    result=dict(status='DEVELOPMENT ONLY; 2026 removed before all modeling',
                target='future Parkinson proxy mean; MEM is not a return-GARCH mislabeled comparator',
                checks=checks(),markets={})
    arrays={}
    for instrument in ['KRBN','KCCA']:
        D,audit=B.prepare(instrument,512);result['markets'][instrument]=dict(audit=audit,fits={},scores={})
        y=D['y'][D['va']]
        for model,integrated in [('MEM_stationary',False),('MEM_integrated',True)]:
            pred,fit_info=fit(D,integrated)
            result['markets'][instrument]['fits'][model]=fit_info
            result['markets'][instrument]['scores'][model]={str(h):dict(n=len(y),mean_QLIKE=float((np.log(pred[:,j])+y[:,j]/pred[:,j]).mean()),
                MSE=float(((pred[:,j]-y[:,j])**2).mean())) for j,h in enumerate([1,5,22])}
            arrays[instrument+'_'+model]=pred
            print(instrument,model,'rho',round(fit_info['rho'],7),'converged',fit_info['selected']['converged'],flush=True)
        arrays[instrument+'_target']=y;arrays[instrument+'_date']=D['date'][D['origin'][D['va']]]
    result['script_sha256']=B.sha256(Path(__file__).resolve())
    result['filter_script_sha256']=B.sha256(Path(B.__file__).resolve())
    path=HERE/'mem_baselines_round16_results.json'
    assert not path.exists()
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(HERE/'mem_baselines_round16_predictions.npz',**arrays)


if __name__=='__main__':main()
