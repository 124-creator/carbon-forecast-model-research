from pathlib import Path
import sys,json
import numpy as np
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
import run as r
import torch
from core import Route
from audit import engineering
results=json.loads((P/'RESULTS.json').read_text());checks=[]
for x in results:
 seed=x['seed'];mode=x['method'];y,_=r.generate('G2',seed);scale=float(np.std(np.diff(y[:1200])));cap=float(np.quantile(np.abs(np.diff(y[:1200]))/scale,.995));o,d,f,t,m=r.dataset(y,scale,cap);ms=r.masks(o,m)
 source=P.parent if x['status']=='existing_complete_replayed' else P
 ck=torch.load(source/f'G2_{seed}_{mode}.pt',weights_only=True);model=Route(mode);model.load_state_dict(ck['selected'])
 with torch.no_grad():pred=model(d,f)
 z=np.load(P/f'G2_{seed}_{mode}_predictions.npz',allow_pickle=False);delta=float(np.max(np.abs(z['pred']-pred.numpy())));score=r.metrics(pred,t,ms['test']);err=float(np.max(np.abs(np.array(score)-x['test_h'])))
 if delta>1e-12 or err>1e-12:raise RuntimeError('replay mismatch')
 # Perturb all prices after this issuance; frozen train-only scale/cap.
 origin=1810;changed=y.copy();changed[origin+1:]+=np.arange(len(y)-origin-1)*10
 _,dd,ff,_,_=r.dataset(changed,scale,cap);idx=int(np.where(o==origin)[0][0])
 with torch.no_grad():future_delta=float((model(dd[idx:idx+1],ff[idx:idx+1])-model(d[idx:idx+1],f[idx:idx+1])).abs().max())
 if future_delta!=0:raise RuntimeError('future dependency')
 checks.append({'seed':seed,'method':mode,'prediction_replay_max':delta,'metric_replay_max':err,'future_suffix_prediction_delta':future_delta})
(P/'VERIFICATION.json').write_text(json.dumps({'runs':checks,'engineering':engineering()},indent=2),encoding='utf-8')
table=[]
for seed in [7,17]:
 q={x['method']:float(np.mean(x['test_h'])) for x in results if x['seed']==seed}
 table.append({'seed':seed,'mean_mse':q,'full_vs_age_improvement_pct':100*(1-q['full']/q['age']),'full_vs_evidence_improvement_pct':100*(1-q['full']/q['evidence']),'full_vs_constant_improvement_pct':100*(1-q['full']/q['constant']),'independent_2pct_gate':q['full']/q['age']<=.98 and q['full']/q['evidence']<=.98})
(P/'DECISION.json').write_text(json.dumps(table,indent=2),encoding='utf-8');print(json.dumps(table,indent=2))
