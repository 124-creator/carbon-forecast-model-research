"""Independent scoring plus complete training replay and suffix checks."""
import json
import numpy as np
import run
from run import OUT,base
rows=json.loads((OUT/'ALL_RESULTS.json').read_text(encoding='utf-8'))
logs=json.loads((OUT/'TRAINING_LOGS.json').read_text(encoding='utf-8'))
dates,p=base.read_prefix()
z=np.load(OUT/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False)
o=z['origins'];true=p[o[:,None]+np.array([1,5,20])]-p[o,None]
if len(o)!=451 or not np.array_equal(true,z['true_changes']):raise ValueError('Protocol mismatch')
years=np.array([s[:4] for s in dates[o]])
initial=int(np.sum(dates<'2020-01-01'));count=0;future_checks=0;max_error=0.
for r in rows:
    pred=z[r['name']];loss=[]
    for y in ['2020','2021']:
        m=years==y;loss.extend((np.mean((pred[m]-true[m])**2,0)/np.mean(true[m]**2,0)).tolist())
    if abs(np.mean(loss)-r['score'])>1e-12:raise ValueError('Score mismatch')
    c=r['configuration']
    for fit in range(initial,len(p),40):
        pos=np.where((o>=fit)&(o<fit+40))[0]
        if not len(pos):continue
        if c['family']=='PROBE':a,trace=run.probe(p,dates,fit,o[pos],c['group'],c['alpha'])
        else:a,trace=base.predict(p,fit,o[pos],c)
        max_error=max(max_error,float(np.max(np.abs(a-pred[pos]))));count+=1
        if 'max_training_target' in trace and trace['max_training_target']>=fit:raise ValueError('Immature label')
        # Perturb all prices after the last requested origin, leaving dates fixed.
        q=p.copy();q[o[pos].max()+1:]+=10000
        if c['family']=='PROBE':b,_=run.probe(q,dates,fit,o[pos],c['group'],c['alpha'])
        else:b,_=base.predict(q,fit,o[pos],c)
        if not np.array_equal(a,b):raise ValueError('Future dependency')
        future_checks+=1
if max_error>1e-12 or count!=len(logs) or len(rows)!=23:raise ValueError('Replay/count mismatch')
old=base.ROOT/'research/deep_direction_review_20261013/calibration/ALL_RESULTS.json'
original=json.loads(old.read_text(encoding='utf-8'))
if any(abs(a['score']-b['score'])>1e-12 for a,b in zip(rows[:15],original)):raise ValueError('Baseline calibration changed')
result=dict(configs=len(rows),origins=len(o),fit_replays=count,future_perturbations=future_checks,max_prediction_error=max_error,baseline_score_match=True,independent_score_pass=True,optimized=not __debug__)
base.dump('VERIFICATION_OPTIMIZED.json' if not __debug__ else 'VERIFICATION.json',result)
print(json.dumps(result))
