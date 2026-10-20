import json,sys,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
def require(c,m):
 if not c:raise ValueError(m)
a=np.load(P/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);o=a['origins'];p=a['prices'];dates=a['dates'];years=dates[o].astype('U10');years=np.array([s[:4] for s in years]);truth=p[o[:,None]+np.array([1,5,20])]-p[o,None]
require(len(o)==451 and len(p)==950 and str(dates[-1])[:4]=='2021','prefix/support');require(np.array_equal(truth,a['truth']),'truth');require(all(dates[t][:4]==dates[t+20][:4] for t in o),'sameyear');scores={}
for r in json.loads((P/'RESULTS.json').read_text()):
 pr=a[r['name']];rat=[]
 for y in sorted(set(years)):
  m=years==y;rat.extend((np.mean((pr[m]-truth[m])**2,axis=0)/np.mean(truth[m]**2,axis=0)).tolist())
 v=float(np.mean(rat));require(abs(v-r['score'])<1e-12,'score');scores[r['name']]=v
for r in json.loads((P/'ACQUISITION_RECEIPTS.json').read_text()):
 b=(P/'raw_LOCAL_ONLY'/f"{r['currency']}.csv").read_bytes();require(len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],'rawreceipt')
for r in json.loads((P/'TRAINING_LOGS.json').read_text()):require(r['inner_max_target']<r['validation_first'] and r['outer_max_target']<r['end'],'purge')
checks=json.loads((P/'FUTURE_FX_REFIT.json').read_text());require(all(r['parameter_difference']==0 and r['issuance_prediction_difference']==0 and r['alpha_same'] for r in checks),'refit')
result=dict(optimized=bool(sys.flags.optimize),scores=scores,refit_checks=len(checks),passed=True);(P/('VERIFIED_OPT.json' if sys.flags.optimize else 'VERIFIED_NORMAL.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
