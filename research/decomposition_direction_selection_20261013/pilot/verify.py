"""Read-back scoring and causality checks; explicit errors survive python -O."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import run as m

def check(ok,msg):
    if not ok:raise ValueError(msg)

def main():
    dates,p=m.base.read_prefix();out=m.OUT
    z=np.load(out/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False)
    origins=z['origins'];years=np.array([x[:4] for x in dates[origins]])
    check(len(origins)==451,'origin count')
    check(all(dates[t][:4]==dates[t+20][:4] for t in origins),'same year')
    truth=p[origins[:,None]+m.base.H]-p[origins,None]
    np.testing.assert_allclose(z['true_changes'],truth,rtol=0,atol=0)
    results=json.loads((out/'ALL_RESULTS.json').read_text(encoding='utf-8'));logs=json.loads((out/'TRAINING_LOGS.json').read_text(encoding='utf-8'))
    for r in results:
        pr=z[r['name']];cells=[]
        price_error=(p[origins,None]+pr)-p[origins[:,None]+m.base.H]
        np.testing.assert_allclose(price_error,pr-truth,atol=1e-12)
        for y in ['2020','2021']:
            mask=years==y
            for j in range(3):cells.append(float(np.mean(price_error[mask,j]**2)/np.mean(truth[mask,j]**2)))
        check(abs(np.mean(cells)-r['score'])<1e-12,'score mismatch')
    raw=m.base.raw_features(p);banks={w:m.features(p,w)[0] for w in [5,10,20]}
    cfg={r['name']:r['configuration'] for r in results};replay_count=0
    for log in logs:
        fit=log.get('fit_end_exclusive');check(fit is not None,'fit missing')
        if 'max_training_target' in log:check(log['max_training_target']<fit,'label maturity')
        c=cfg[log['name']];pos=np.where((origins>=fit)&(origins<fit+40))[0]
        if c['family'] in ['RW','MLE']:
            pr,_=m.base.predict(p,fit,origins[pos],dict(family=c['family'],window=480))
        else:
            X=raw[:,:c['dim']] if c['family']=='RAW' else banks[c['w']][:,:16 if c['family']=='DECOMP' else 18]
            pr=((X[origins[pos]]-np.array(log['feature_mean']))/np.array(log['feature_sd']))@np.array(log['coefficients'])+np.array(log['intercept'])
            check(log['max_training_target']<fit,'maturity')
        np.testing.assert_allclose(pr,z[log['name']][pos],rtol=1e-12,atol=1e-12);replay_count+=1
    perturb=[]
    for w in [5,10,20]:
        X,rev=m.features(p,w)
        for t in [500,700,900]:
            q=p.copy();q[t+1:]+=100000
            Y,_=m.features(q,w);Z,_=m.features(p[:t+1],w)
            np.testing.assert_allclose(X[:t+1],Y[:t+1],rtol=0,atol=1e-10)
            np.testing.assert_allclose(X[t],Z[t],rtol=0,atol=1e-10)
            # A forecast with identical fit support must survive future price perturbation.
            a,_=m.ridge(p,X,t,np.array([t]),1.)
            b,_=m.ridge(q,Y,t,np.array([t]),1.)
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)
            last=t-w-1;rr=rev[last-19:last+1]
            np.testing.assert_allclose(X[t,-2:],[rr.mean(),np.sqrt(np.mean(rr**2))],atol=1e-12)
            perturb.append(dict(w=w,t=t,max_revision_source=last,max_revision_maturity=last+w))
    hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in out.iterdir() if x.is_file() and x.name not in ['VERIFICATION.json','VERIFICATION_OPTIMIZED.json']}
    result=dict(passed=True,origin_count=451,configs=len(results),replayed_fit_blocks=replay_count,score_original_price_equivalence=True,future_feature_and_forecast_perturbations=perturb,hashes=hashes)
    m.dump('VERIFICATION_OPTIMIZED.json' if __debug__ is False else 'VERIFICATION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='hashes'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
