"""Post-result fixed-parameter diagnostics; never fits or selects another model."""
import argparse,json
from pathlib import Path
import numpy as np
from experiment import jwrite

def main(out):
 out=Path(out);summary={};byfit=[]
 for mode in ['MP_DIFF','TWOHOT','GAUSS','CAT_MSE','IND_DIFF']:
  sat=[];ent=[];change=[];times=[];chosen=[]
  for path in sorted((out/'fits').glob(mode+'_*.npz')):
   log=json.loads(path.with_suffix('.json').read_text());times.append(log['seconds']);chosen.append(log['selected_epoch'])
   with np.load(path,allow_pickle=False) as z:
    a=z['param__atoms'];prob=z['detail__prob'];prior=np.exp(z['param__prior']);c=z['detail__correction']
    sat.extend((np.abs(c)/np.maximum(np.abs(a).max(-1),1e-12)).ravel())
    ent.extend((-np.sum(prob*np.log(np.maximum(prob,1e-300)),-1)).ravel())
    change.extend(np.abs(prob-prior[None]).ravel())
  summary[mode]=dict(mean_abs_probability_change_from_prior=float(np.mean(change)),max_correction_fraction_of_atom_limit=float(np.max(sat)),mean_correction_fraction_of_atom_limit=float(np.mean(sat)),mean_prediction_entropy=float(np.mean(ent)),sum_fit_wall_seconds=float(np.sum(times)),selected_epoch_zero=int(np.sum(np.array(chosen)==0)))
 for path in sorted((out/'fits').glob('MP_DIFF_s17_*.json')):
  log=json.loads(path.read_text());byfit.append(dict(block=log['block'],scale=log['scale'],**log['label_diagnostics']))
 # Realized ensemble squared-error decomposition, not population bias-variance identification.
 metrics=json.loads((out/'ALL_MODEL_METRICS.json').read_text())
 result=dict(scope='Post-result fixed-parameter descriptions, no new optimizer or alternate forecast score',categorical_models=summary,unique_training_label_diagnostics=byfit,realized_seed_dispersion={k:v['score']-v['ensemble'] for k,v in metrics.items() if len(v['seed_scores'])>1},limits=['Support exceedances of realized outcomes do not imply that a conditional-mean forecast must saturate.','A numerically preserved target first moment does not guarantee lower finite-sample forecast risk.'])
 jwrite(out/'POSTHOC_DESCRIPTION.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();main(a.out)
