"""Explicit exceptions survive python -O; local artifacts only."""
import json,sys
import numpy as np
import run as m

def check(ok,msg):
 if not ok:raise ValueError(msg)

def main():
 dates,p=m.base.read_prefix();o=m.OUT
 z=np.load(o/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);v=np.load(o/'VINTAGES_LOCAL_ONLY.npz',allow_pickle=False)
 results=json.loads((o/'ALL_RESULTS.json').read_text());frozen=json.loads((o/'FROZEN_CONFIG.json').read_text())
 check(len(results)==len(frozen['configurations'])==20,'configuration count')
 orig=z['origins'];years=np.array([s[:4] for s in dates[orig]]);truth=p[orig[:,None]+m.H]-p[orig,None]
 check(len(orig)==451,'origin count');check(np.array_equal(truth,z['true_changes']),'truth')
 maxerr=0.;replay=0
 for t in range(59,len(p)):
  a,b=m.feature_at(p[:t+1],t)
  check(np.allclose(a,v['RAW'][t],atol=1e-12),'raw vintage')
  check(np.allclose(b,v['SSA'][t],atol=1e-12),'SSA vintage')
  check(np.allclose(b.reshape(3,59).sum(0),a,atol=1e-10),'reconstruction')
 for r in results:
  c=r['configuration'];pred=np.full(truth.shape,np.nan)
  for fit in range(int(np.sum(dates<'2020-01-01')),len(p),40):
   pos=np.where((orig>=fit)&(orig<fit+40))[0]
   if not len(pos):continue
   if c['backbone']=='RW':a=np.zeros((len(pos),3))
   elif c['backbone']=='MLE':a,_=m.matched_mle(p,fit,orig[pos])
   else:a,_=m.learned(p,v[c['representation']],fit,orig[pos],c)
   pred[pos]=a;replay+=1
  check(np.allclose(pred,z[r['name']],rtol=1e-10,atol=1e-10),'prediction replay')
  score,_=m.base.score(pred,truth,years);check(abs(score-r['score'])<1e-12,'score mismatch')
  maxerr=max(maxerr,float(np.max(np.abs(pred-z[r['name']]))))
 # Future tail changed AFTER issuance; regenerate every usable historical feature.
 perturb=0
 for t in [500,700,900]:
  q=p.copy();q[t+1:]+=12345
  raw=np.zeros_like(v['RAW']);ssa=np.zeros_like(v['SSA'])
  for s in range(59,t+1):raw[s],ssa[s]=m.feature_at(q,s)
  for c in frozen['configurations']:
   if c['backbone']=='RW':continue
   if c['backbone']=='MLE':a,_=m.matched_mle(p,t,np.array([t]));b,_=m.matched_mle(q,t,np.array([t]))
   else:
    a,_=m.learned(p,v[c['representation']],t,np.array([t]),c)
    b,_=m.learned(q,raw if c['representation']=='RAW' else ssa,t,np.array([t]),c)
   check(np.allclose(a,b,atol=1e-10),'future leakage');perturb+=1
 # Negative controls must actually throw with optimization enabled.
 negatives=0
 for fn in [lambda:m.feature_at(p,58),lambda:m.feature_at(p,len(p)),lambda:m.indices(20),lambda:check(False,'deliberate failure')]:
  try:fn()
  except ValueError:negatives+=1
  else:raise RuntimeError('missing explicit exception')
 receipt=dict(passed=True,optimized=not __debug__,configurations=20,replayed_blocks=replay,max_forecast_error=maxerr,vintages=891,future_perturbations=perturb,explicit_negative_controls=negatives)
 m.base.dump('VERIFICATION_OPTIMIZED.json' if not __debug__ else 'VERIFICATION.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
