"""Independent arithmetic, saved-state reconstruction and representative refits.
Not a third-party audit or complete independent retraining.
"""
from pathlib import Path
import json
import numpy as np
import torch
import core,experiment

def run(path,out,refit=True):
 out=Path(out);p,dates,raw=core.load_prefix(path);ref=core.reference(p)
 with np.load(out/'evaluation.npz',allow_pickle=False) as z:orig=z['origins'];y=z['truth'];years=z['years']
 truth=np.array([[p[t+h]-p[t] for h in range(1,21)] for t in orig]);np.testing.assert_array_equal(truth,y)
 expected=json.loads((out/'RESULTS.json').read_text())['scores'];maxdiff=0.;metrics=0
 for name,row in expected.items():
  with np.load(out/'predictions'/f'{name}.npz',allow_pickle=False) as z:pred=z['prediction']
  values=[];ev=[]
  for year in ['2020','2021']:
   ii=np.flatnonzero(years==year)
   for h in [0,4,19]:
    den=sum(y[i,h]**2 for i in ii)/len(ii)
    mse=sum(sum((pred[s,i,h]-y[i,h])**2 for i in ii)/len(ii) for s in range(len(pred)))/len(pred)
    em=sum((sum(pred[s,i,h] for s in range(len(pred)))/len(pred)-y[i,h])**2 for i in ii)/len(ii)
    values.append(mse/den);ev.append(em/den)
  got=sum(values)/6;ens=sum(ev)/6;maxdiff=max(maxdiff,abs(got-row['score']),abs(ens-row['ensemble']));metrics+=2
 rebuilt=0.;count=0;parameter_files=[]
 for mode in core.MODES:
  for seed in core.SEEDS:
   for block in range(12):
    f=out/'fits'/f'{mode}_s{seed}_b{block:02}.npz';info=json.loads(f.with_suffix('.json').read_text());end=459+40*block
    if info['train_max_target']>=info['validation_first_origin'] or info['validation_max_target']>=end:raise AssertionError('maturity')
    with np.load(f,allow_pickle=False) as z:
     m=core.make_model(mode,seed);m.load_state_dict({k[7:]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('state__')})
     pr=core.predict(m,p,ref,z['origins'],float(z['sd']));rebuilt=max(rebuilt,float(np.max(abs(pr-z['prediction']))));count+=1
    parameter_files.append(str(f.name))
 receipts=[]
 if refit:
  for mode in ['CAUSAL','IDENTITY','FULL_WINDOW']:
   q=p.copy();q[459:]+=10000
   a,sa,ra=experiment.fit(p,core.reference(p),459,mode,17);b,sb,rb=experiment.fit(q,core.reference(q),459,mode,17)
   equal=sa==sb and all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
   pa=core.predict(a,p,ref,[458],sa);pb=core.predict(b,q,core.reference(q),[458],sb)
   with np.load(out/'fits'/f'{mode}_s17_b00.npz',allow_pickle=False) as z:
    first=all(np.array_equal(v.detach().numpy(),z['state__'+k]) for k,v in a.state_dict().items())
   if not equal or not np.array_equal(pa,pb) or not first:raise AssertionError('fresh causal reproduction')
   receipts.append({'mode':mode,'end':459,'seed':17,'original_and_perturbed_params_equal':equal,'matches_first_run':first,'additional_fits':2})
 if maxdiff>1e-10 or rebuilt>1e-10:raise AssertionError('numeric disagreement')
 result={'passed':True,'metrics_recomputed':metrics,'max_score_error':maxdiff,'saved_models_rebuilt':count,'max_prediction_rebuild_error':rebuilt,
         'representative_refits':receipts,'source':'independent scoring implementation plus own source refits, not third-party science replication','later_period_read':False}
 (out/'AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
