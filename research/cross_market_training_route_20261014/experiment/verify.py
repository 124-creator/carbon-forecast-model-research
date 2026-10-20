"""Independent original-price scoring checks, explicit errors survive python -O."""
import json,sys
from pathlib import Path
import numpy as np
import run
R=Path(__file__).resolve().parent

def require(ok,message):
 if not ok:raise RuntimeError(message)

def main():
 data,audit=run.load();p,d=data['guangzhou'];z=np.load(R/'predictions_LOCAL_ONLY.npz',allow_pickle=False)
 origins=np.array([t for t in range(459,930) if d[t][:4]==d[t+20][:4]])
 require(len(origins)==451 and np.array_equal(origins,z['origins']),'origin identity')
 truth=np.array([p[t+1:t+21] for t in origins]);require(np.array_equal(truth,z['truth']),'original labels');require(np.array_equal(np.repeat(p[origins,None],20,1),z['rw']),'RW labels')
 scores=json.loads((R/'scores.json').read_text());receipts=json.loads((R/'fit_receipts.json').read_text());require(len(receipts)==120,'fit count')
 for rec in receipts:
  b=rec['fit_end_exclusive'];cut=d[b-1]
  require(rec['source_ends']==[b,int(np.sum(data['hubei'][1]<cut)),int(np.sum(data['eu'][1]<cut))],'source strict endpoint')
  lines=[json.loads(x) for x in (R/(rec['tag']+'_loss.jsonl')).read_text().splitlines()]
  require(lines[-1]['gradient_calls']==120,'per-market gradient objective budget')
  require((R/(rec['tag']+'_weights_LOCAL_ONLY.pt')).is_file(),'checkpoint missing')
 for row in scores['year_H20']:
  idx=z['years']==row['year'];h=row['h']-1;pred=[z[f"{row['kind']}_{s}"] for s in run.SEEDS]
  require(all(np.isfinite(a).all() for a in pred),'missing predictions')
  mse=np.mean([np.mean((a[idx,h]-truth[idx,h])**2) for a in pred]);mae=np.mean([np.mean(abs(a[idx,h]-truth[idx,h])) for a in pred]);require(abs(mse-row['mse'])<1e-12 and abs(mae-row['mae'])<1e-12,'score mismatch')
 causal=json.loads((R/'causality.json').read_text());require(len(causal)==1 and causal[0]['parameter_identical'] and causal[0]['issuance_prediction_identical'],'true-refit causality')
 result={'mode':'optimized' if sys.flags.optimize else 'normal','verified':True,'models':5,'seeds':2,'rolling_fits':len(receipts),'future_perturbed_refits':1,'year_horizon_rows':len(scores['year_H20']),'origins':len(origins),'first_pool':[379,405,400],'numeric_future_boundary_access':False}
 run.save('verify_'+result['mode']+'.json',result);print(json.dumps(result))
if __name__=='__main__':main()
