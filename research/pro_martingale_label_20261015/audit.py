"""Independent scoring, numeric reconstruction and representative causal refits."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import core
from experiment import jwrite,fit,predict

def run(out,refit=True):
 out=Path(out);core.configure();p,dates,_=core.load_prefix(out/'approved_prefix.csv');ref=core.reference(p)
 metrics=json.loads((out/'ALL_MODEL_METRICS.json').read_text())
 with np.load(out/'EVALUATION.npz',allow_pickle=False) as z:
  ids=z['origins'];mask=z['eligible'];truth=z['truth'][mask];years=z['years'][mask]
 maxscore=0.;count=0
 for path in sorted((out/'predictions').glob('*.npz')):
  with np.load(path,allow_pickle=False) as z:pr=z['prediction'][:,mask]
  for ensemble in [False,True]:
   q=pr.mean(0,keepdims=True) if ensemble else pr;ratios=[]
   for yy in ['2020','2021']:
    ii=np.flatnonzero(years==yy)
    for hh in [0,4,19]:
     den=sum(float(truth[j,hh])**2 for j in ii)/len(ii)
     rr=sum(sum(float(q[s,j,hh]-truth[j,hh])**2 for j in ii)/len(ii) for s in range(len(q)))/len(q)/den
     ratios.append(rr)
   score=sum(ratios)/6;key='ensemble' if ensemble else 'score'
   maxscore=max(maxscore,abs(score-metrics[path.stem][key]));count+=1
 maxpred=0.;nfit=0;epochs=0;labelmax=0.;out_of_support={};encoded_outside={};selected_zero={}
 for path in sorted((out/'fits').glob('*.npz')):
  log=json.loads(path.with_suffix('.json').read_text());mode=log['mode'];seed=log['seed'];end=log['end']
  if not (log['max_training_target']<log['first_validation_origin'] and log['max_validation_target']<end):raise AssertionError('time support')
  with np.load(path,allow_pickle=False) as z:
   sd=float(z['scale']);ori=z['origins'];old=z['prediction'];m=core.make_model(mode,seed)
   state={k[7:]:torch.as_tensor(z[k].copy()) for k in z.files if k.startswith('param__')};m.load_state_dict(state)
   new,detail=predict(m,sd,p,ref,ori);err=float(np.max(np.abs(new-old)));maxpred=max(maxpred,err)
   if err>1e-12:raise AssertionError(f'Reconstruction {path.name}:{err}')
   if mode in core.CATEGORICAL:
    support=state['atoms'].numpy();yy=np.stack([p[t+1:t+21]-p[t]-ref['b'][t] for t in ori])/sd
    outside=(yy<support[:,0])|(yy>support[:,-1]);out_of_support.setdefault(mode,[0,0]);out_of_support[mode][0]+=int(outside.sum());out_of_support[mode][1]+=int(outside.size)
    eo=(yy/2<support[:,0])|(yy/2>support[:,-1]);encoded_outside.setdefault(mode,[0,0]);encoded_outside[mode][0]+=int(eo.sum());encoded_outside[mode][1]+=int(eo.size)
   if log['selected_epoch']==0:selected_zero[mode]=selected_zero.get(mode,0)+1
  labelmax=max(labelmax,log['label_diagnostics']['target_mean_max_error']);epochs+=log['epochs'];nfit+=1
 if nfit!=432:raise AssertionError('incomplete grid')
 causal=[]
 if refit:
  for mode in ['MP_DIFF','GAUSS','CAT_MSE','RANK6']:
   t0=time.perf_counter();end=459;seed=17
   a,sa,la=fit(p,core.reference(p),end,mode,seed)
   pp=p.copy();pp[end:]+=np.linspace(1000,100000,len(p)-end)
   b,sb,lb=fit(pp,core.reference(pp),end,mode,seed)
   dif=max(float((a.state_dict()[k]-b.state_dict()[k]).abs().max()) for k in a.state_dict())
   pr,_=predict(a,sa,p,ref,[end-1]);pq,_=predict(b,sb,pp,core.reference(pp),[end-1])
   if sa!=sb or dif!=0 or not np.array_equal(pr,pq):raise AssertionError('future refit failed')
   with np.load(out/'fits'/f'{mode}_s17_b00.npz',allow_pickle=False) as z:
    first=max(float(np.max(np.abs(a.state_dict()[k].numpy()-z['param__'+k]))) for k in a.state_dict())
   causal.append(dict(mode=mode,cutoff=end,parameter_difference=dif,prediction_difference=float(np.max(np.abs(pr-pq))),first_run_parameter_difference=first,seconds=time.perf_counter()-t0))
 report=dict(scoring_verified=maxscore<1e-12,independently_recomputed_main_scores=count,max_score_error=maxscore,reconstructed_fits=nfit,max_prediction_reconstruction_error=maxpred,epochs_checked=epochs,maturity_violations=0,label_mean_error_max=labelmax,selected_initialization=selected_zero,forecast_residual_outside_atom_support=out_of_support,encoded_half_residual_outside_atom_support=encoded_outside,support_count_scope='All issued origins x20 horizons x4 seeds, not independent samples',causal_full_refit_pairs=causal,scope='Independent arithmetic implementation and representative full causal refits, not external peer review or independent time-period validation')
 jwrite(out/'AUDIT.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--skip-refits',action='store_true');o=a.parse_args();run(o.out,not o.skip_refits)
